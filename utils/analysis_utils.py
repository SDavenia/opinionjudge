import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from utils.data_utils import DATASETS, DEFAULT_GENERATOR, generator_tag
from utils.models_utils import CLOSED_MODELS, REGISTRY
from utils.paths import FEATURES_DIR, SCORING_DIR
from utils.prompts_utils import (
    ALIGNMENT_PROMPTS_BY_DATASET, CATEGORY_MAPPING, DEFAULT_PERPLEXITY_PROMPT, LABEL_MAPPING,
)

SCORING_COLUMNS = [
    "id_1", "situation_id_1", "text_id_1", "stance_1", "text_1",
    "id_2", "situation_id_2", "text_id_2", "stance_2", "text_2",
    "parsed_score_1to2", "parsed_score_2to1",
]

# Qwen2.5 judges of the model-scale analysis. Only the largest one is a main judge.
SCALE_MODELS = ["qwen2.5-05b", "qwen2.5-1b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b"]
SCALE_ONLY_MODELS = SCALE_MODELS[:-1]

DATASET_SHORT = {"valueprism": "VP", "habermas": "HB", "habermas_hybrid": "HBY"}
DATASET_NAMES = {"valueprism": "ValuePrism", "habermas": "Habermas", "habermas_hybrid": "Habermas-Hybrid"}
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
DEFAULT_REFUSAL_CLASSIFIER = "llama-3.3-70b"

# Stance label (ValuePrism, or the Habermas agreement scale) -> alignment category.
STANCE_TO_CATEGORY = {
    "Supports": "positive", "Opposes": "negative", "Either": "neutral",
    "STRONGLY_AGREE": "positive", "AGREE": "positive", "SOMEWHAT_AGREE": "positive",
    "STRONGLY_DISAGREE": "negative", "DISAGREE": "negative", "SOMEWHAT_DISAGREE": "negative",
    "NEUTRAL": "neutral",
}


def add_data_args(parser, datasets=DATASETS):
    parser.add_argument("--datasets", nargs="+", default=list(datasets), choices=DATASETS)
    parser.add_argument("--run_on", default="all", choices=["all", "open", "closed"],
                        help="Judges to include: open (local) models, closed (OpenRouter) ones, or both.")
    parser.add_argument("--scale_analysis", action="store_true",
                        help="Run on the Qwen2.5 judges from 0.5B to 72B instead of the main judges.")
    parser.add_argument("--generator_model_id", default=DEFAULT_GENERATOR)
    parser.add_argument("--scoring_prompt_version", default="0_1")
    parser.add_argument("--scoring_dir", type=Path, default=SCORING_DIR)
    parser.add_argument("--features_dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--embedding_model_id", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--refusal_classifier_id", default=DEFAULT_REFUSAL_CLASSIFIER)


@dataclass
class DatasetContext:
    dataset_id: str
    run_on: str = "all"
    scale_analysis: bool = False
    generator_model_id: str = DEFAULT_GENERATOR
    scoring_prompt_version: str = "0_1"
    scoring_dir: Path = SCORING_DIR
    features_dir: Path = FEATURES_DIR
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL
    refusal_classifier_id: str = DEFAULT_REFUSAL_CLASSIFIER

    @classmethod
    def from_args(cls, args, dataset_id):
        values = {f.name: getattr(args, f.name) for f in fields(cls) if f.name != "dataset_id" and hasattr(args, f.name)}
        return cls(dataset_id, **values)

    @property
    def generator(self):
        return generator_tag(self.dataset_id, self.generator_model_id)

    @property
    def feature_dataset(self):
        """Situation-level features of habermas_hybrid are the habermas ones."""
        return "habermas" if self.dataset_id == "habermas_hybrid" else self.dataset_id

    def feature_path(self, feature, *parts):
        return Path(self.features_dir) / feature / self.feature_dataset / Path(*parts)

    def hybrid_feature_path(self, feature, *parts):
        return Path(self.features_dir) / feature / "habermas_hybrid" / self.generator / Path(*parts)


def run_label(args):
    return "scale" if args.scale_analysis else args.run_on


def scoring_prefix(ctx):
    return "" if ctx.generator is None else f"{ctx.generator}_"


def judge_kind(judge):
    return "open" if judge in REGISTRY else "closed"


def discover_judges(ctx):
    """Judges with a scoring file for the dataset, from the open/ and closed/ folders."""
    kinds = ["open"] if ctx.scale_analysis else (["open", "closed"] if ctx.run_on == "all" else [ctx.run_on])
    known = {"open": REGISTRY, "closed": CLOSED_MODELS}
    prefix, suffix = scoring_prefix(ctx), f"_{ctx.scoring_prompt_version}.csv"
    judges = []
    for kind in kinds:
        kind_dir = Path(ctx.scoring_dir) / ctx.dataset_id / kind
        if not kind_dir.exists():
            continue
        for f in sorted(kind_dir.rglob(f"*{suffix}")):
            rel = f.relative_to(kind_dir).as_posix()
            if not rel.startswith(prefix) or len(rel) <= len(prefix) + len(suffix):
                continue
            judge = rel[len(prefix):-len(suffix)]
            if judge not in known[kind]:
                print(f"[warn] {judge} found in {kind_dir} is not a known {kind} judge (utils/models_utils.py).")
                continue
            judges.append(judge)
    if ctx.scale_analysis:
        return [m for m in SCALE_MODELS if m in judges]
    return [m for m in judges if m not in SCALE_ONLY_MODELS]


def scoring_path(ctx, judge):
    return (Path(ctx.scoring_dir) / ctx.dataset_id / judge_kind(judge)
            / f"{scoring_prefix(ctx)}{judge}_{ctx.scoring_prompt_version}.csv")


def load_scoring_df(ctx, judge, columns=SCORING_COLUMNS):
    """The judge's scores on the dataset, without the rows with an unparsed score."""
    path = scoring_path(ctx, judge)
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=columns)[columns]
    unparsed = df[["parsed_score_1to2", "parsed_score_2to1"]].isna().any(axis=1)
    if unparsed.any():
        print(f"[warn] {ctx.dataset_id}/{judge}: dropping {unparsed.sum()} of {len(df)} rows with an unparsed score.")
    return df[~unparsed].reset_index(drop=True)


# Feature enrichers: each adds to a scoring dataframe the feature of both opinions of a
# pair, as {name}_1 / {name}_2 (or a pair-level column), and returns None if the feature
# was not extracted for this dataset/judge.

def read_opinion_features(ctx, feature, rel_path, reader=pd.read_csv):
    """Per-opinion feature table. For habermas_hybrid, the habermas table (human opinions)
    plus the one of the synthetic opinions."""
    path = ctx.feature_path(feature, rel_path)
    if not path.is_file():
        return None
    df = reader(path)
    if ctx.dataset_id != "habermas_hybrid":
        return df
    synthetic_path = ctx.hybrid_feature_path(feature, rel_path)
    if not synthetic_path.is_file():
        print(f"[warn] {synthetic_path} not found: no {feature} for the synthetic opinions.")
        return None
    return pd.concat([df, reader(synthetic_path)], ignore_index=True)


def merge_per_text(df, table, columns, names=None):
    """Merge per-opinion columns (keyed by text_id) onto both opinions of each pair."""
    names = names or columns
    for i in (1, 2):
        df = df.merge(
            table[["text_id", *columns]].rename(columns={"text_id": f"text_id_{i}",
                                                         **{c: f"{n}_{i}" for c, n in zip(columns, names)}}),
            on=f"text_id_{i}", how="left")
    return df


def add_aggregates(df, name, skipna=False):
    """avg/max/min/diff_{name} over the two opinions."""
    s1, s2 = df[f"{name}_1"], df[f"{name}_2"]
    df[f"avg_{name}"] = (s1 + s2) / 2
    if skipna:
        df[f"max_{name}"] = df[[f"{name}_1", f"{name}_2"]].max(axis=1)
        df[f"min_{name}"] = df[[f"{name}_1", f"{name}_2"]].min(axis=1)
    else:
        df[f"max_{name}"] = np.maximum(s1, s2)
        df[f"min_{name}"] = np.minimum(s1, s2)
    df[f"diff_{name}"] = (s1 - s2).abs()
    return df


def enrich_text(scoring_df, ctx, judge):
    df = scoring_df.copy()
    for i in (1, 2):
        df[f"text_length_{i}"] = df[f"text_{i}"].str.split().str.len()
    return add_aggregates(df, "text_length", skipna=True)


def load_perplexity(ctx, judge):
    prompt = DEFAULT_PERPLEXITY_PROMPT[ctx.dataset_id]
    if ctx.dataset_id == "valueprism":
        paths = [ctx.feature_path("perplexity", judge, ctx.generator, f"{prompt}.csv")]
    else:
        paths = [ctx.feature_path("perplexity", judge, f"{prompt}.csv")]
        if ctx.dataset_id == "habermas_hybrid":
            paths.append(Path(ctx.features_dir) / "perplexity" / "habermas_hybrid" / judge / ctx.generator
                         / f"{prompt}.csv")
    if not all(p.exists() for p in paths):
        return None
    return pd.concat([pd.read_csv(p)[["text_id", "perplexity"]] for p in paths], ignore_index=True)


def enrich_perplexity(scoring_df, ctx, judge):
    """Log-perplexity of each opinion under the judge, z-scored over the judge's values."""
    perplexity = load_perplexity(ctx, judge)
    if perplexity is None:
        return None
    df = merge_per_text(scoring_df.copy(), perplexity, ["perplexity"])
    df["perplexity_1"] = np.log(df["perplexity_1"])
    df["perplexity_2"] = np.log(df["perplexity_2"])
    values = pd.concat([df["perplexity_1"], df["perplexity_2"]])
    mean, std = values.mean(), values.std()
    std = std if std > 0 else 1.0
    df["perplexity_1"] = (df["perplexity_1"] - mean) / std
    df["perplexity_2"] = (df["perplexity_2"] - mean) / std
    return add_aggregates(df, "perplexity", skipna=True)


def _cosine(v1, v2):
    if v1 is None or v2 is None or (not isinstance(v1, np.ndarray) and pd.isna(v1)) \
            or (not isinstance(v2, np.ndarray) and pd.isna(v2)):
        return np.nan
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.nan if norm_product == 0 else float(np.dot(v1, v2) / norm_product)


def enrich_embedding(scoring_df, ctx, judge):
    """Cosine similarity between the embeddings of the two opinions."""
    model_dir = ctx.embedding_model_id.replace("/", "__")
    embeddings = read_opinion_features(ctx, "embeddings", f"{model_dir}/opinion_embeddings.parquet",
                                       reader=pd.read_parquet)
    if embeddings is None:
        return None
    vectors = dict(zip(embeddings["text_id"], embeddings["embedding"]))
    pairs = scoring_df[["text_id_1", "text_id_2"]].drop_duplicates()
    similarity = {(a, b): _cosine(vectors.get(a), vectors.get(b))
                  for a, b in zip(pairs["text_id_1"], pairs["text_id_2"])}
    df = scoring_df.copy()
    df["cosine_similarity"] = [similarity[key] for key in zip(df["text_id_1"], df["text_id_2"])]
    return df


def enrich_sentiment(scoring_df, ctx, judge):
    scores = read_opinion_features(ctx, "sentiment", "opinion_sentiment_scores.csv")
    if scores is None:
        return None
    df = merge_per_text(scoring_df.copy(), scores, ["sentiment_score"], ["sentiment_score_opinion"])
    return add_aggregates(df, "sentiment_score_opinion", skipna=True)


VAD_DIMENSIONS = ["valence", "arousal", "dominance"]


def enrich_vad(scoring_df, ctx, judge):
    scores = read_opinion_features(ctx, "vad", "opinion_vad_scores.csv")
    if scores is None:
        return None
    names = [f"{d}_opinion" for d in VAD_DIMENSIONS]
    df = merge_per_text(scoring_df.copy(), scores, VAD_DIMENSIONS, names)
    for name in names:
        df = add_aggregates(df, name)
    return df


VADER_SCORES = ["compound", "neg", "pos"]


def enrich_vader(scoring_df, ctx, judge):
    scores = read_opinion_features(ctx, "vader", "opinion_vader_scores.csv")
    if scores is None:
        return None
    names = [f"vader_{c}_opinion" for c in VADER_SCORES]
    df = merge_per_text(scoring_df.copy(), scores, VADER_SCORES, names)
    for name in names:
        df = add_aggregates(df, name)
    return df


def enrich_safety(scoring_df, ctx, judge):
    """Llama-Guard unsafe score of each opinion."""
    scores = read_opinion_features(ctx, "safety", "opinion_scores.csv")
    if scores is None:
        return None
    df = merge_per_text(scoring_df.copy(), scores, ["unsafe_score"], ["safety_unsafe_score_opinion"])
    return add_aggregates(df, "safety_unsafe_score_opinion")


def add_stance_values(df, category_columns, name):
    """Give each opinion the situation-level value of its own stance category
    (category_columns: positive/negative/neutral -> column) as {name}_1 / {name}_2."""
    for i in (1, 2):
        category = df[f"stance_{i}"].map(STANCE_TO_CATEGORY)
        unknown = df.loc[category.isna(), f"stance_{i}"].unique()
        if len(unknown):
            print(f"[{name}] unmapped stance_{i} values {list(unknown)} -> NaN")
        df[f"{name}_{i}"] = np.select([category == c for c in category_columns],
                                      [df[col] for col in category_columns.values()], default=np.nan)
    df = df.drop(columns=["situation_id", *category_columns.values()])
    return add_aggregates(df, name)


def _evaluator_type(filename):
    matches = [k for k in LABEL_MAPPING if k in filename]
    if not matches:
        raise ValueError(f"Cannot tell the alignment prompt of {filename}.")
    return max(matches, key=len)


def load_alignment(ctx, judge):
    """Per situation, the judge's probability of a positive / negative / neutral position,
    averaged over the alignment prompts. None unless every prompt of the dataset was run."""
    align_dir = ctx.feature_path("alignment", judge)
    files = [f for f in align_dir.glob("*.csv") if "_generations" not in f.name] if align_dir.exists() else []
    if not files:
        return None
    expected = ALIGNMENT_PROMPTS_BY_DATASET[ctx.feature_dataset]
    missing = set(expected) - {_evaluator_type(f.name) for f in files}
    if missing:
        print(f"[alignment] {judge}: missing prompts {sorted(missing)} - skipped.")
        return None

    records = []
    for f in files:
        mapping = CATEGORY_MAPPING[_evaluator_type(f.name)]
        for _, row in pd.read_csv(f).iterrows():
            dist = row["alignment_distribution"]
            dist = dist if isinstance(dist, dict) else json.loads(dist)
            probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for key, value in dist.items():
                if key == "total_evaluations":
                    continue
                label = key[2:-1] if key.startswith("p(") and key.endswith(")") else key
                probs[mapping[label]] += value
            records.append({"situation_id": row["situation_id"], **probs})
    agg = pd.DataFrame.from_records(records).groupby("situation_id")[["positive", "negative", "neutral"]].mean()
    return agg.rename(columns=lambda c: f"p_{c}").reset_index()


def enrich_alignment(scoring_df, ctx, judge):
    """Each opinion gets the judge's own probability of the opinion's stance on the situation."""
    alignment = load_alignment(ctx, judge)
    if alignment is None:
        return None
    df = scoring_df.merge(alignment, left_on="situation_id_1", right_on="situation_id", how="left")
    return add_stance_values(df, {"positive": "p_positive", "negative": "p_negative", "neutral": "p_neutral"},
                             "stance_alignment")


def enrich_refusal(scoring_df, ctx, judge):
    """Each opinion gets the judge's probability of refusing to write an opinion with that
    stance on the situation (refusals re-classified by --refusal_classifier_id)."""
    path = ctx.feature_path("refusal", judge, "refusal_classified", ctx.refusal_classifier_id, "refusal_prob.csv")
    if not path.is_file():
        return None
    refusal = pd.read_csv(path)[["situation_id", "p_refusal_support", "p_refusal_oppose", "p_refusal_neutral"]]
    df = scoring_df.merge(refusal, left_on="situation_id_1", right_on="situation_id", how="left")
    return add_stance_values(df, {"positive": "p_refusal_support", "negative": "p_refusal_oppose",
                                  "neutral": "p_refusal_neutral"}, "stance_refusal")


# Significance. Trends are tested on opinion pairs (Spearman's rho between a feature and
# the pair's outcome), with uncertainty from resampling situations, since pairs of the
# same situation share opinions.

TAIL_FRACTION = 0.2
SIGNIFICANCE_STATS = ["rho", "rho_low_half", "rho_high_half", "contrast_top_bottom", "mean"]


def _rank_corr(rank_x, y):
    if len(y) < 3:
        return np.nan
    rank_y = rankdata(y)
    if rank_x.std() == 0 or rank_y.std() == 0:
        return np.nan
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _safe_mean(values):
    return float(values.mean()) if len(values) else np.nan


def bootstrap_p(boot):
    """Two-sided bootstrap p-value of statistic = 0 (at least 2 / (B + 1))."""
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return np.nan
    below = (np.sum(boot <= 0) + 1) / (boot.size + 1)
    above = (np.sum(boot >= 0) + 1) / (boot.size + 1)
    return float(min(1.0, 2 * min(below, above)))


def bh_adjust(p_values):
    """Benjamini-Hochberg q-values (NaN p-values stay NaN)."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m:
        order = np.argsort(p[finite])
        scaled = p[finite][order] * m / np.arange(1, m + 1)
        scaled = np.minimum.accumulate(scaled[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(scaled, 1.0)
        q[finite] = adjusted
    return q


def _shape(result):
    def significant(name):
        return result[f"{name}_ci95_low"] > 0 or result[f"{name}_ci95_high"] < 0

    if not np.isfinite(result["rho_ci95_low"]):
        return "too few situations"
    if result["tie_share"] >= 0.5:
        return "mostly tied x"
    if (significant("rho_low_half") and significant("rho_high_half")
            and np.sign(result["rho_low_half"]) != np.sign(result["rho_high_half"])):
        return "non-monotonic"
    if significant("rho"):
        return "increasing" if result["rho"] > 0 else "decreasing"
    return "no monotonic trend"


def significance_tests(x, outcomes, situations, n_boot, min_situations=5, seed=0):
    """Per outcome: Spearman's rho with x, the same within the lower and upper half of x,
    the mean outcome of the top minus the bottom 20% of x, and the mean outcome, each with a
    95% CI from n_boot resamples of the situations, plus bootstrap p-values and a shape label."""
    x = np.asarray(x, dtype=float)
    situations = np.asarray(situations)
    outcomes = {name: np.asarray(y, dtype=float) for name, y in outcomes.items()}
    n = len(x)
    order_rank = rankdata(x, method="ordinal")
    low_half = order_rank <= n / 2
    bottom = order_rank <= TAIL_FRACTION * n
    top = order_rank > (1 - TAIL_FRACTION) * n
    situation_codes, inverse = np.unique(situations, return_inverse=True)
    rows_by_situation = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])

    def statistics(idx):
        xb, lb = x[idx], low_half[idx]
        rank_x, rank_x_low, rank_x_high = rankdata(xb), rankdata(xb[lb]), rankdata(xb[~lb])
        out = {}
        for name, y in outcomes.items():
            yb = y[idx]
            out[name] = [_rank_corr(rank_x, yb), _rank_corr(rank_x_low, yb[lb]), _rank_corr(rank_x_high, yb[~lb]),
                         _safe_mean(yb[top[idx]]) - _safe_mean(yb[bottom[idx]]), _safe_mean(yb)]
        return out

    observed = statistics(np.arange(n))
    enough = len(situation_codes) >= max(min_situations, 2) and n >= 3
    boots = {name: np.full((n_boot, len(SIGNIFICANCE_STATS)), np.nan) for name in outcomes}
    if enough:
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            drawn = rng.integers(0, len(rows_by_situation), len(rows_by_situation))
            for name, values in statistics(np.concatenate([rows_by_situation[k] for k in drawn])).items():
                boots[name][b] = values

    tie_share = float(pd.Series(x).value_counts(normalize=True).iloc[0]) if n else np.nan
    results = {}
    for name in outcomes:
        result = {"n_pairs": n, "n_situations": len(situation_codes), "tie_share": tie_share}
        for k, stat in enumerate(SIGNIFICANCE_STATS):
            result[stat] = observed[name][k]
            column = boots[name][:, k]
            if enough and np.isfinite(column).any():
                result[f"{stat}_ci95_low"], result[f"{stat}_ci95_high"] = np.nanpercentile(column, [2.5, 97.5])
            else:
                result[f"{stat}_ci95_low"] = result[f"{stat}_ci95_high"] = np.nan
            if stat in ("rho", "contrast_top_bottom", "mean"):
                result[f"{stat}_p"] = bootstrap_p(column) if enough else np.nan
        result["shape"] = _shape(result)
        results[name] = result
    return results


def write_significance(rows, significance_dir, analysis, p_columns):
    """{analysis}.csv with q-values within the analysis, and all.csv with every analysis
    of the folder and q-values across all of them."""
    if not rows:
        return
    significance_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    for p_col in p_columns:
        table[p_col.replace("_p", "_q")] = bh_adjust(table[p_col])
    table.to_csv(significance_dir / f"{analysis}.csv", index=False)

    tables = [pd.read_csv(f) for f in sorted(significance_dir.glob("*.csv")) if f.name != "all.csv"]
    combined = pd.concat(tables, ignore_index=True)
    for p_col in p_columns:
        combined[p_col.replace("_p", "_q_all")] = bh_adjust(combined[p_col])
    combined.to_csv(significance_dir / "all.csv", index=False)


# Smoothed curves: statistics over overlapping windows of the pairs sorted by a feature,
# with 95% bands from resampling situations.

def sliding_window_starts(n, window, step):
    w = min(n, max(1, round(window * n)))
    stride = max(1, round(step * n))
    starts = list(range(0, n - w + 1, stride))
    if starts[-1] != n - w:
        starts.append(n - w)
    return starts, w


def situation_bootstrap_bands(terms, n_boot, seed=0):
    """2.5/97.5 percentiles of window statistics that are means of per-unit terms.
    terms[name] = (sums, counts), each (n_windows, n_situations): the per-situation sum and
    number of terms in each window. All windows share the same resampled situations."""
    n_situations = next(iter(terms.values()))[1].shape[1]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_situations, np.full(n_situations, 1 / n_situations), size=n_boot)
    bands = {}
    for name, (sums, counts) in terms.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            stats = (weights @ sums.T) / (weights @ counts.T)
        bands[name] = np.nanpercentile(stats, [2.5, 97.5], axis=0)
    return bands


# Joint model of all the features (robustness check of the one-feature analyses).

# name -> (per-opinion column, +1 if the opinion with the higher value is the pivot, -1 if the lower)
JOINT_FEATURES = {
    "text_length": ("text_length", 1),
    "sentiment": ("sentiment_score_opinion", 1),
    "vad_valence": ("valence_opinion", 1),
    "vad_arousal": ("arousal_opinion", 1),
    "vad_dominance": ("dominance_opinion", 1),
    "vader_compound": ("vader_compound_opinion", 1),
    "guard_safety": ("safety_unsafe_score_opinion", -1),
    "perplexity": ("perplexity", -1),
    "alignment": ("stance_alignment", 1),
}
# Refusal is left out: too sparse, and constant for some judges.
JOINT_ENRICHERS = [enrich_text, enrich_embedding, enrich_sentiment, enrich_vad, enrich_vader, enrich_safety,
                   enrich_perplexity, enrich_alignment]
PAIR_COLUMNS = ["id_1", "id_2", "situation_id_1", "text_id_1", "text_id_2", "stance_1", "stance_2", "text_1", "text_2"]


def joint_pairs(ctx, judge, scoring):
    """One row per opinion pair, with every feature available for the judge."""
    pairs = scoring[PAIR_COLUMNS].drop_duplicates(["id_1", "id_2"]).reset_index(drop=True)
    for enrich in JOINT_ENRICHERS:
        enriched = enrich(pairs, ctx, judge)
        if enriched is not None:
            pairs = enriched
    return pairs.set_index(["id_1", "id_2"])


def rank_columns(values):
    return np.column_stack([rankdata(values[:, j]) for j in range(values.shape[1])])


def situation_rows(situations):
    _, inverse = np.unique(situations, return_inverse=True)
    return np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])


def ci_p(observed, boots, names, prefix):
    low, high = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({prefix: observed, f"{prefix}_ci95_low": low, f"{prefix}_ci95_high": high,
                         f"{prefix}_p": [bootstrap_p(boots[:, j]) for j in range(boots.shape[1])]}, index=names)
