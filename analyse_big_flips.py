"""
Big score flips: judgments whose score moves by at least --threshold.

--flip_type order (default): the two opinions swap position, one instance per (pair, prompt
variation), delta = |score_1to2 - score_2to1| (what PC averages).
--flip_type prompt: the prompt wording changes, one instance per (pair, direction), delta =
max - min of the scores over the prompt variations (the spread RS penalises).
A pair is flipped when at least one of its instances is.

Per judge: how often big flips happen and the distribution of delta. Across judges: whether
the same pairs flip for many judges, compared with the number expected if the judges flipped
independently at their own rates (Poisson-binomial), and the pairwise agreement of the
flipped sets. Per situation: each judge ranks the situations by --situation_score (share of
flipped pairs, or mean delta) and the rankings are compared (Spearman, Kendall tau-b, and the
overlap of the --top_k most problematic situations).

By default the judges are the main ones; --scale_analysis uses Qwen2.5 from 7B to 72B.

Usage:
    python analyse_big_flips.py
    python analyse_big_flips.py --scale_analysis
    python analyse_big_flips.py --flip_type prompt --threshold 0.4
    python analyse_big_flips.py --plots_only
"""
import argparse
import json
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator
from scipy.stats import hypergeom, kendalltau, spearmanr

from utils.analysis_utils import DATASET_SHORT, DatasetContext, add_data_args, discover_judges, run_label, scoring_path
from utils.models_utils import REGISTRY
from utils.paths import FIGURES_DIR, RESULTS_DIR
from utils.plot_utils import (
    BOTTOM, CELL_GAP, CMAP, COLORBAR_GAP, COLORBAR_WIDTH, HEATMAP_FONT_SIZES, JUDGES, LINE_FONT_SIZES, MAIN_STYLES,
    MINUS, MUTED_TEXT, NA_COLOR, PANEL_WIDTH, PLOT_HEIGHT, RIGHT, SCALE_JUDGES, SCALE_STYLES, TEXT_COLOR, TOP,
    fmt_signed, label_width, legend_figure, save, setup_matplotlib, style_axis,
)

PAIR_KEY = ["text_id_1", "text_id_2"]
SCORE_COLUMNS = ["parsed_score_1to2", "parsed_score_2to1"]
PAIR_META_COLUMNS = ["situation_id_1", "situation", "stance_1", "stance_2", "text_1", "text_2"]
SCALE_FLIP_JUDGES = ["qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--judge_model_ids", nargs="+", default=None, help="Use exactly these judges.")
    parser.add_argument("--flip_type", default="order", choices=["order", "prompt"])
    parser.add_argument("--threshold", type=float, default=0.3, help="Smallest score change counted as a big flip.")
    parser.add_argument("--min_judges", type=int, default=None,
                        help="Pairs flipped by at least this many judges are 'shared' (default: a majority).")
    parser.add_argument("--situation_score", default="mean_delta", choices=["flip_rate", "mean_delta"],
                        help="Situation ranking by share of flipped pairs, or by mean delta (no threshold).")
    parser.add_argument("--top_k", type=int, default=50, help="Most problematic situations compared across judges.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR / "big_flips")
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR / "big_flips")
    parser.add_argument("--plots_only", action="store_true", help="Only redraw the figures from saved results.")
    return parser.parse_args()


def select_judges(args, ctx):
    judges = discover_judges(ctx)
    if args.judge_model_ids:
        missing = set(args.judge_model_ids) - set(judges)
        if missing:
            raise SystemExit(f"No scoring files for {sorted(missing)}; found {judges}")
        judges = [j for j in judges if j in args.judge_model_ids]
    elif args.scale_analysis:
        judges = [j for j in judges if j in SCALE_FLIP_JUDGES]
    size = lambda j: getattr(REGISTRY.get(j), "num_params", None)  # noqa: E731
    return sorted(judges, key=lambda j: (size(j) is None, size(j) or 0, j))


def load_scores(ctx, judge):
    path = scoring_path(ctx, judge)
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=PAIR_KEY + PAIR_META_COLUMNS + ["prompt_variation"] + SCORE_COLUMNS)


# Part 1: flips per judge

def compute_instances(df, flip_type, threshold):
    """One row per instance with its delta (rounded, so that 0.7 - 0.4 counts as 0.3)."""
    if flip_type == "order":
        inst = df.dropna(subset=SCORE_COLUMNS)[PAIR_KEY + ["prompt_variation"] + SCORE_COLUMNS].copy()
        inst["signed_delta"] = (inst["parsed_score_1to2"] - inst["parsed_score_2to1"]).round(6)
        inst["delta"] = inst["signed_delta"].abs()
    else:
        long = df.melt(id_vars=PAIR_KEY + ["prompt_variation"], value_vars=SCORE_COLUMNS,
                       var_name="direction", value_name="score").dropna(subset=["score"])
        long["direction"] = long["direction"].str.removeprefix("parsed_score_")
        inst = long.groupby(PAIR_KEY + ["direction"], as_index=False).agg(
            score_max=("score", "max"), score_min=("score", "min"), n_variations=("score", "size"))
        inst = inst[inst["n_variations"] >= 2]
        inst["delta"] = (inst["score_max"] - inst["score_min"]).round(6)
    inst["big_flip"] = inst["delta"] >= threshold
    return inst


def pair_level(inst):
    return inst.groupby(PAIR_KEY, as_index=False).agg(max_delta=("delta", "max"), mean_delta=("delta", "mean"),
                                                     n_instances=("delta", "size"), n_big_flips=("big_flip", "sum"))


def model_stats(judge, inst, pairs, flip_type):
    delta = inst["delta"]
    flipped = inst[inst["big_flip"]]
    stats = {
        "judge": judge, "n_pairs": len(pairs), "n_instances": len(inst), "n_big_flips": len(flipped),
        "pct_instances_flipped": 100 * inst["big_flip"].mean(),
        "n_pairs_flipped": int((pairs["n_big_flips"] > 0).sum()),
        "pct_pairs_flipped": 100 * (pairs["n_big_flips"] > 0).mean(),
        # flipped in most of the pair's instances: a systematic rather than one-off flip
        "pct_pairs_flipped_majority": 100 * (pairs["n_big_flips"] > pairs["n_instances"] / 2).mean(),
        "mean_delta": delta.mean(), "median_delta": delta.median(), "p90_delta": delta.quantile(0.90),
        "p99_delta": delta.quantile(0.99), "max_delta": delta.max(),
        "n_instances_at_max": int((delta == delta.max()).sum()),
        "mean_delta_of_big_flips": flipped["delta"].mean() if len(flipped) else np.nan,
    }
    if flip_type == "order":
        stats["pct_big_flips_1to2_higher"] = 100 * (flipped["signed_delta"] > 0).mean() if len(flipped) else np.nan
    return stats


def by_variation(judge, df, inst, flip_type, threshold):
    """order: flip rate per prompt variation; prompt: per pair of variations."""
    if flip_type == "order":
        out = inst.groupby("prompt_variation").agg(
            n_instances=("delta", "size"), pct_flipped=("big_flip", "mean"), mean_delta=("delta", "mean"),
            pct_1to2_higher_of_flips=("signed_delta", lambda s: (s[inst.loc[s.index, "big_flip"]] > 0).mean()))
        out["pct_flipped"] *= 100
        out["pct_1to2_higher_of_flips"] *= 100
        return out.reset_index().assign(judge=judge)
    rows = []
    for direction in SCORE_COLUMNS:
        wide = df.pivot_table(index=PAIR_KEY, columns="prompt_variation", values=direction, aggfunc="first")
        for var_a, var_b in combinations(sorted(wide.columns), 2):
            diff = (wide[var_a] - wide[var_b]).abs().round(6).dropna()
            rows.append({"variation_a": var_a, "variation_b": var_b,
                         "direction": direction.removeprefix("parsed_score_"), "n": len(diff),
                         "pct_flipped": 100 * (diff >= threshold).mean(), "mean_abs_diff": diff.mean()})
    out = pd.DataFrame(rows)
    both = (out.assign(w_flip=out["pct_flipped"] * out["n"], w_diff=out["mean_abs_diff"] * out["n"])
            .groupby(["variation_a", "variation_b"], as_index=False)[["n", "w_flip", "w_diff"]].sum())
    both = both.assign(direction="both", pct_flipped=both["w_flip"] / both["n"],
                       mean_abs_diff=both["w_diff"] / both["n"]).drop(columns=["w_flip", "w_diff"])
    return pd.concat([out, both], ignore_index=True).assign(judge=judge)


# Part 2: overlap across judges

def poisson_binomial_pmf(probs):
    pmf = np.zeros(len(probs) + 1)
    pmf[0] = 1.0
    for p in probs:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= (1 - p)
    return pmf


def cohen_kappa(a, b):
    p_obs = (a == b).mean()
    p_exp = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else np.nan


def pairwise_overlap(flags, deltas):
    n = len(flags)
    rows = []
    for a, b in combinations(flags.columns, 2):
        fa, fb = flags[a].to_numpy(), flags[b].to_numpy()
        n_a, n_b, n_both = int(fa.sum()), int(fb.sum()), int((fa & fb).sum())
        union = n_a + n_b - n_both
        rows.append({"judge_a": a, "judge_b": b, "n_a": n_a, "n_b": n_b, "n_both": n_both,
                     "jaccard": n_both / union if union else np.nan,
                     "overlap_coef": n_both / min(n_a, n_b) if min(n_a, n_b) else np.nan,
                     "lift": (n_both / n) / ((n_a / n) * (n_b / n)) if n_a and n_b else np.nan,
                     "kappa": cohen_kappa(fa, fb),
                     "spearman_max_delta": spearmanr(deltas[a], deltas[b]).statistic})
    return pd.DataFrame(rows)


def scores_json(raw_scores, pairs):
    """Per pair: JSON {judge: {prompt_variation: [score_1to2, score_2to1]}}."""
    long = (pd.concat([r.assign(judge=j) for j, r in raw_scores.items()], ignore_index=True)
            .merge(pairs[PAIR_KEY], on=PAIR_KEY))
    nested = {}
    for t1, t2, variation, s12, s21, judge in long[PAIR_KEY + ["prompt_variation"] + SCORE_COLUMNS + ["judge"]].itertuples(index=False):
        nested.setdefault((t1, t2), {}).setdefault(judge, {})[variation] = [
            None if pd.isna(s12) else float(s12), None if pd.isna(s21) else float(s21)]
    return pd.Series([json.dumps(nested.get(key, {})) for key in zip(pairs[PAIR_KEY[0]], pairs[PAIR_KEY[1]])],
                     index=pairs.index)


# Part 3: situation rankings

def situation_scores(score, flags, inst_tables, situation_of_pair):
    """situations x judges, higher = more problematic, on the pairs every judge scored."""
    if score == "flip_rate":
        return flags.astype(float).groupby(situation_of_pair.reindex(flags.index).to_numpy()).mean()
    columns = {}
    for judge, inst in inst_tables.items():
        inst = inst.join(situation_of_pair, on=PAIR_KEY, how="inner")
        inst = inst[inst.set_index(PAIR_KEY).index.isin(flags.index)]
        columns[judge] = inst.groupby("situation_id_1")["delta"].mean()
    return pd.DataFrame(columns)


def topk_membership(scores, k):
    """1 above the top-k cut, (slots left) / (situations tied) at the cut, 0 below."""
    scores = scores.round(9)
    cutoff = scores.sort_values(ascending=False).iloc[k - 1]
    above, tied = scores > cutoff, scores == cutoff
    return above.astype(float) + tied * (k - above.sum()) / tied.sum()


def ranking_similarity(scores, k):
    n = len(scores)
    k = min(k, n)
    members = {j: topk_membership(scores[j], k) for j in scores.columns}
    rows = []
    for a, b in combinations(scores.columns, 2):
        rho = spearmanr(scores[a], scores[b])
        tau = kendalltau(scores[a], scores[b])
        overlap = float((members[a] * members[b]).sum())
        rows.append({"judge_a": a, "judge_b": b, "n_situations": n,
                     "spearman": rho.statistic, "spearman_p": rho.pvalue,
                     "kendall_tau_b": tau.statistic, "kendall_p": tau.pvalue,
                     "top_k": k, "topk_overlap": overlap, "topk_overlap_frac": overlap / k, "topk_chance_frac": k / n,
                     "topk_p": hypergeom.sf(round(overlap) - 1, n, k, k)})
    return pd.DataFrame(rows)


def analyse_dataset(args, dataset, out_dir):
    ctx = DatasetContext.from_args(args, dataset)
    judges = select_judges(args, ctx)
    if not judges:
        print(f"[{dataset}] no scoring files - skipped.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{dataset}] judges: {judges}")

    stats_rows, dist_rows, variation_tables, pair_tables, raw_scores, inst_tables = [], [], [], {}, {}, {}
    pair_meta = None
    for judge in judges:
        df = load_scores(ctx, judge)
        n_nan = int(df[SCORE_COLUMNS].isna().any(axis=1).sum())
        if n_nan:
            print(f"  [{judge}] {n_nan} rows with an unparsed score, left out of their instances.")
        if pair_meta is None:
            pair_meta = df.drop_duplicates(subset=PAIR_KEY)[PAIR_KEY + PAIR_META_COLUMNS]
        raw_scores[judge] = df[PAIR_KEY + ["prompt_variation"] + SCORE_COLUMNS]
        inst = compute_instances(df, args.flip_type, args.threshold)
        inst_tables[judge] = inst[PAIR_KEY + ["delta"]]
        pairs = pair_level(inst)
        pair_tables[judge] = pairs
        stats_rows.append(model_stats(judge, inst, pairs, args.flip_type))
        counts = inst["delta"].value_counts().sort_index()
        dist_rows += [{"judge": judge, "delta": d, "count": int(c), "pct": 100 * c / len(inst)} for d, c in counts.items()]
        variation_tables.append(by_variation(judge, df, inst, args.flip_type, args.threshold))

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(out_dir / "per_model_stats.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out_dir / "delta_distribution.csv", index=False)
    pd.concat(variation_tables, ignore_index=True).to_csv(out_dir / "by_variation.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(stats.set_index("judge").drop(columns=["n_instances_at_max"]).to_string())
    if len(judges) < 2:
        return

    deltas = pd.concat({j: p.set_index(PAIR_KEY)["max_delta"] for j, p in pair_tables.items()}, axis=1)
    n_all = len(deltas)
    deltas = deltas.dropna()
    if len(deltas) < n_all:
        print(f"[warn] {n_all - len(deltas)} of {n_all} pairs are missing for some judge: "
              f"the overlap uses the {len(deltas)} pairs every judge scored.")
    flags = deltas >= args.threshold
    min_judges = args.min_judges or len(judges) // 2 + 1
    k = flags.sum(axis=1)
    k_dist = pd.DataFrame({"k": np.arange(len(judges) + 1),
                           "observed": np.bincount(k, minlength=len(judges) + 1),
                           "expected_independent": poisson_binomial_pmf(flags.mean().tolist()) * len(flags)})
    k_dist["observed_over_expected"] = k_dist["observed"] / k_dist["expected_independent"]
    k_dist.to_csv(out_dir / "k_distribution.csv", index=False)
    pairwise_overlap(flags, deltas).to_csv(out_dir / "pairwise_overlap.csv", index=False)

    pair_flags = (deltas.add_prefix("max_delta__")
                  .assign(n_judges_flipped=k, mean_max_delta=deltas.mean(axis=1),
                          flipped_by=flags.apply(lambda r: ",".join(r.index[r]), axis=1))
                  .reset_index().merge(pair_meta, on=PAIR_KEY, how="left")
                  .sort_values(["n_judges_flipped", "mean_max_delta"], ascending=False))
    front = PAIR_KEY + ["n_judges_flipped", "mean_max_delta", "flipped_by"] + PAIR_META_COLUMNS
    pair_flags = pair_flags[front + [c for c in pair_flags.columns if c not in front]]
    pair_flags.to_csv(out_dir / "pair_flags.csv", index=False)
    shared = pair_flags[pair_flags["n_judges_flipped"] >= min_judges]
    shared.assign(flip_type=args.flip_type, threshold=args.threshold, scores=scores_json(raw_scores, shared)).to_csv(
        out_dir / "shared_pairs.csv", index=False)
    (pair_flags.groupby("situation_id_1")
     .agg(situation=("situation", "first"), n_pairs=("n_judges_flipped", "size"),
          mean_judges_flipped=("n_judges_flipped", "mean"),
          pct_pairs_flipped_by_any=("n_judges_flipped", lambda s: 100 * (s > 0).mean()),
          n_shared_pairs=("n_judges_flipped", lambda s: int((s >= min_judges).sum())))
     .sort_values(["mean_judges_flipped", "n_shared_pairs"], ascending=False).reset_index()
     .to_csv(out_dir / "situation_summary.csv", index=False))
    at_least = k_dist.loc[k_dist["k"] >= min_judges, ["observed", "expected_independent"]].sum()
    print(f"Pairs flipped by >= {min_judges} judges: {int(at_least['observed'])} "
          f"(expected if independent: {at_least['expected_independent']:.1f}) of {len(flags)}.")

    score = args.situation_score
    situation_of_pair = pair_meta.set_index(PAIR_KEY)["situation_id_1"]
    sit_scores = situation_scores(score, flags, inst_tables, situation_of_pair).dropna()
    ranks = sit_scores.rank(ascending=False, method="average")
    sit_meta = pair_flags.groupby("situation_id_1").agg(situation=("situation", "first"),
                                                        n_pairs=("n_judges_flipped", "size"))
    ranking = (sit_meta.join(sit_scores.add_prefix(f"{score}__"), how="inner").join(ranks.add_prefix("rank__"))
               .assign(mean_score=sit_scores.mean(axis=1), mean_rank=ranks.mean(axis=1))
               .sort_values("mean_rank").rename_axis("situation_id").reset_index())
    front = ["situation_id", "situation", "n_pairs", "mean_rank", "mean_score"]
    ranking[front + [c for c in ranking.columns if c not in front]].to_csv(
        out_dir / f"situation_ranking_{score}.csv", index=False)
    similarity = ranking_similarity(sit_scores, args.top_k)
    similarity.to_csv(out_dir / f"ranking_similarity_{score}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        cols = ["spearman", "kendall_tau_b", "topk_overlap_frac"]
        print(f"Situation rankings ({len(sit_scores)} situations, {score}), averaged over judge pairs:")
        print(similarity[cols].agg(["mean", "min", "max"]).to_string())


# Figures

FLIP_TYPES = {
    "order": ("Order-swap score change, Δ", ("Any prompt variation", "Most prompt variations")),
    "prompt": ("Score range across prompts, Δ", ("Either direction", "Both directions")),
}
LAYOUTS = {"valueprism": "left", "habermas": "right", "habermas_hybrid": "left"}
LEFT_PAD = 0.03
OBSERVED_COLOR, INDEPENDENT_COLOR = "#5b95c7", "#dc6f5c"
LIGHT = 0.33
SEQ_CMAP = LinearSegmentedColormap.from_list("soft_blues", ["#f7f7f7", "#bcd6ea", "#5b95c7", "#1f5590"])
MATRIX_ROW_HEIGHT, MATRIX_TOP, MATRIX_BOTTOM = 0.22, 0.04, 0.27
TILT = 40
DELTA_BIN = 0.1
DELTA_CMAP = LinearSegmentedColormap.from_list("delta_blues", ["#e6eff8", "#bcd6ea", "#5b95c7", "#1f5590"])
DELTA_NORM = LogNorm(0.1, 100, clip=True)
DELTA_ROW_HEIGHT, DELTA_TOP, DELTA_BOTTOM = 0.2, 0.15, 0.33
# metric -> (CSV, value column, p-value column, colour scale, colorbar label)
METRICS = {
    "jaccard": ("pairwise_overlap.csv", "jaccard", None, "sequential", "Jaccard"),
    "lift": ("pairwise_overlap.csv", "lift", None, "log", "Lift"),
    "spearman": ("pairwise_overlap.csv", "spearman_max_delta", None, "diverging", "Spearman ρ"),
    "ranking_spearman": ("ranking_similarity_{score}.csv", "spearman", "spearman_p", "diverging", "Spearman ρ"),
    "ranking_kendall": ("ranking_similarity_{score}.csv", "kendall_tau_b", "kendall_p", "diverging", "Kendall τ$_b$"),
    "ranking_topk": ("ranking_similarity_{score}.csv", "topk_overlap_frac", "topk_p", "sequential", None),
}
ALPHA = 0.05


def present(judges, available):
    available = set(available)
    return [j for j in judges if j in available]


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def single_axes():
    fig_height = TOP + PLOT_HEIGHT + BOTTOM
    fig = plt.figure(figsize=(PANEL_WIDTH, fig_height))
    ax = fig.add_axes([0.42 / PANEL_WIDTH, BOTTOM / fig_height, (PANEL_WIDTH - 0.42 - RIGHT) / PANEL_WIDTH,
                       PLOT_HEIGHT / fig_height])
    style_axis(ax)
    return fig, ax


def fit_left(fig, ax):
    """Move the left edge so that the y label ends LEFT_PAD inches from the figure edge."""
    overhang = LEFT_PAD - ax.yaxis.get_tightbbox(fig.canvas.get_renderer()).x0 / fig.dpi
    pos = ax.get_position()
    left = pos.x0 + overhang / PANEL_WIDTH
    ax.set_position([left, pos.y0, pos.x1 - left, pos.height])
    return fig


def axis_labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LINE_FONT_SIZES["label"], color=TEXT_COLOR, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=LINE_FONT_SIZES["label"], color=TEXT_COLOR, labelpad=3)


def tint(color, weight):
    return tuple(weight * c + (1 - weight) for c in to_rgb(color))


def pct_flipped_plot(stats, styles, key_labels):
    """Pairs with a big flip in any (light) and in most (solid) of their instances, per judge
    and dataset."""
    fig, ax = single_axes()
    n_max = max(len(present(styles, s.index)) for s in stats.values())
    width = 0.84 / n_max
    for g, s in enumerate(stats.values()):
        judges = present(styles, s.index)
        for i, judge in enumerate(judges):
            x = g + (i - (len(judges) - 1) / 2) * width
            color = styles[judge][1]
            ax.bar(x, s.loc[judge, "pct_pairs_flipped"], width=0.86 * width, color=tint(color, LIGHT), linewidth=0,
                   zorder=2)
            ax.bar(x, s.loc[judge, "pct_pairs_flipped_majority"], width=0.86 * width, color=color, linewidth=0,
                   zorder=3)
    ax.set_xticks(range(len(stats)), [DATASET_SHORT[d] for d in stats])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(stats) - 0.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)
    axis_labels(ax, None, "Pairs with a big flip (%)")
    grey = "#6f6f6f"
    ax.legend([Patch(color=tint(grey, LIGHT)), Patch(color=grey)], key_labels, loc="upper left", frameon=False,
              fontsize=LINE_FONT_SIZES["region"], handlelength=1.2, handleheight=0.8, handletextpad=0.4,
              labelspacing=0.3, borderaxespad=0.2, labelcolor=TEXT_COLOR)
    return fit_left(fig, ax)


def k_distribution_plot(k_dist):
    """Pairs flipped by k judges, observed vs independent judges, with their ratio on top."""
    fig, ax = single_axes()
    k = k_dist["k"].to_numpy()
    observed, expected = k_dist["observed"].to_numpy(), k_dist["expected_independent"].to_numpy()
    width = 0.38
    ax.bar(k - width / 2, observed, width=0.9 * width, color=OBSERVED_COLOR, linewidth=0, zorder=2)
    ax.bar(k + width / 2, expected, width=0.9 * width, color=INDEPENDENT_COLOR, linewidth=0, zorder=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    bottom = 10 ** math.floor(math.log10(max(min(observed[observed > 0].min(), expected.min()), 0.5)))
    ax.set_ylim(bottom, max(observed.max(), expected.max()) * 4)
    for x, o, e in zip(k, observed, expected):
        if o > 0 and e > 0:
            ratio = o / e
            ax.annotate(f"×{ratio:.0f}" if ratio >= 100 else f"×{ratio:.2g}", xy=(x, max(o, e)), xytext=(0, 1.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=LINE_FONT_SIZES["region"] - 0.5, color=MUTED_TEXT)
    ax.set_xticks(k)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(k[0] - 0.6, k[-1] + 0.6)
    ax.grid(axis="y", which="major", color="#e8e8e8", linewidth=0.5)
    axis_labels(ax, "Judges with a big flip on the pair", "Pairs")
    ax.legend([Patch(color=OBSERVED_COLOR), Patch(color=INDEPENDENT_COLOR)], ["Observed", "Independent judges"],
              loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False, fontsize=LINE_FONT_SIZES["region"],
              handlelength=1.2, handleheight=0.8, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1,
              labelcolor=TEXT_COLOR)
    return fit_left(fig, ax)


@lru_cache(maxsize=None)
def width_pt(text, fontsize):
    return label_width([text], fontsize) * 72


@lru_cache(maxsize=None)
def bold_width_pt(text, fontsize):
    fig = plt.figure()
    width = fig.text(0, 0, text, fontsize=fontsize, fontweight="bold").get_window_extent(
        fig.canvas.get_renderer()).width / fig.dpi * 72
    plt.close(fig)
    return width


def size_column_pt(sizes):
    return max(width_pt(s, HEATMAP_FONT_SIZES["ticks"] - 1) for s in sizes)


def row_label_width(labels):
    """Room for "Family Size" row labels (the size one point smaller)."""
    families, sizes = zip(*(l.split("\n") for l in labels))
    return (max(width_pt(f, HEATMAP_FONT_SIZES["ticks"]) for f in families) + 2.5 + size_column_pt(sizes)) / 72 + 0.06


def colorbar_width(tick_labels):
    fs = HEATMAP_FONT_SIZES
    return (COLORBAR_GAP + COLORBAR_WIDTH + 0.05 + max(width_pt(t, fs["colorbar_ticks"]) for t in tick_labels) / 72
            + 0.04 + fs["colorbar"] / 72)


def colour_scale(kind, values):
    """(norm, cmap, value -> shown value, colorbar ticks, tick labels)."""
    values = np.asarray(values, dtype=float)
    if kind == "log":
        limit = max(math.ceil(np.nanmax(np.abs(np.log2(values))) / 0.25 - 1e-9) * 0.25, 0.25)
        lifts = [t for t in (0.25, 0.5, 1, 2, 4) if abs(math.log2(t)) <= limit + 1e-9]
        if len(lifts) < 3:
            lifts = [t for t in (0.67, 0.8, 1, 1.25, 1.5) if abs(math.log2(t)) <= limit + 1e-9]
        return TwoSlopeNorm(0, -limit, limit), CMAP, np.log2, [math.log2(t) for t in lifts], [f"{t:g}" for t in lifts]
    if kind == "diverging":
        limit = max(math.ceil(np.nanmax(np.abs(values)) / 0.1 - 1e-9) * 0.1, 0.1)
        step = 0.2 if limit > 0.4 else 0.1
        ticks = np.arange(-math.floor(limit / step + 1e-9), math.floor(limit / step + 1e-9) + 1) * step
        return (TwoSlopeNorm(0, -limit, limit), CMAP, lambda v: v, list(ticks),
                ["0" if abs(t) < 1e-9 else f"{t:+.1f}".replace("-", MINUS).lstrip("+") for t in ticks])
    limit = max(math.ceil(np.nanmax(values) / 0.1 - 1e-9) * 0.1, 0.1)
    step = 0.2 if limit > 0.4 else 0.1
    ticks = list(np.arange(0, limit + 1e-9, step))
    return Normalize(0, limit), SEQ_CMAP, lambda v: v, ticks, ["0" if t == 0 else f"{t:.1f}" for t in ticks]


def cell_text(value, kind, leading_zero):
    if kind == "diverging":
        return fmt_signed(value, leading_zero)
    text = f"{value:.2f}"
    return text if leading_zero or value >= 1 else text.replace("0.", ".", 1)


def cell_format(cells, kind, cell_width):
    """(leading_zero, fontsize) so that the widest cell text fits: drop the leading zero
    first, then shrink the font."""
    def widest(leading_zero, fs):
        return max(bold_width_pt(cell_text(v, kind, leading_zero) + ("*" if sig else ""), fs) for v, sig in cells)

    room = cell_width - 4
    for leading_zero in (True, False):
        if widest(leading_zero, HEATMAP_FONT_SIZES["rho"]) <= room:
            return leading_zero, HEATMAP_FONT_SIZES["rho"]
    fs = HEATMAP_FONT_SIZES["rho"]
    while fs > 5 and widest(False, fs) > room:
        fs -= 0.5
    return False, fs


def column_label_layout(judges, labels, cell_width):
    """(tilted, bottom margin): two-line labels if every family name fits its cell, else one
    tilted line."""
    fs = HEATMAP_FONT_SIZES["ticks"]
    families, sizes = zip(*(labels[j].split("\n") for j in judges))
    if max(width_pt(f, fs) for f in families) <= cell_width - 3:
        return False, MATRIX_BOTTOM
    longest = max(width_pt(f, fs) + 2.5 + width_pt(z, fs - 1) for f, z in zip(families, sizes))
    return True, (longest * math.sin(math.radians(TILT)) + fs * math.cos(math.radians(TILT)) + 6) / 72


def row_labels(ax, judges, labels, fontsize):
    size_column = size_column_pt([labels[j].split("\n")[1] for j in judges])
    for i, judge in enumerate(judges):
        family, size = labels[judge].split("\n")
        ax.annotate(size, xy=(0, i + 0.5), xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=fontsize - 1, color=MUTED_TEXT)
        ax.annotate(family, xy=(0, i + 0.5), xytext=(-5.5 - size_column, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=fontsize, color=TEXT_COLOR)


def colorbar(fig, rect, norm, cmap, ticks, tick_labels, label):
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(rect))
    bar.outline.set_visible(False)
    bar.set_ticks(ticks)
    bar.set_ticklabels(tick_labels)
    bar.ax.minorticks_off()
    fs = HEATMAP_FONT_SIZES
    bar.ax.tick_params(labelsize=fs["colorbar_ticks"], length=2, width=0.4, pad=1.5, color=MUTED_TEXT,
                       labelcolor=TEXT_COLOR)
    bar.set_label(label, fontsize=fs["colorbar"], labelpad=3, color=TEXT_COLOR)


def matrix_panel(cells, judges, labels, kind, scale, colorbar_label, with_colorbar, label_margin, colorbar_margin,
                 n_max, text_format, column_layout):
    """Judge x judge matrix with a blank diagonal; cells: {(a, b): (value, significant or None)}."""
    norm, cmap, transform, ticks, tick_labels = scale
    leading_zero, cell_fontsize = text_format
    tilted, bottom = column_layout
    n = len(judges)
    cell = (PANEL_WIDTH - label_margin - colorbar_margin) / n_max
    height = MATRIX_ROW_HEIGHT * n
    fig_height = MATRIX_TOP + height + bottom
    fig = plt.figure(figsize=(PANEL_WIDTH, fig_height))
    ax = fig.add_axes([label_margin / PANEL_WIDTH, bottom / fig_height, cell * n / PANEL_WIDTH, height / fig_height])
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.axis("off")

    fs = HEATMAP_FONT_SIZES
    gap_x, gap_y = CELL_GAP / 2 / cell, CELL_GAP / 2 / MATRIX_ROW_HEIGHT
    row_labels(ax, judges, labels, fs["ticks"])
    for i, row_judge in enumerate(judges):
        for j, col_judge in enumerate(judges):
            rect = (j + gap_x, i + gap_y), 1 - 2 * gap_x, 1 - 2 * gap_y
            if i == j:
                ax.add_patch(Rectangle(*rect, facecolor=NA_COLOR, edgecolor="none"))
                continue
            value, significant = cells.get((row_judge, col_judge)) or cells[(col_judge, row_judge)]
            shown = norm(transform(value))
            ax.add_patch(Rectangle(*rect, facecolor=cmap(shown), edgecolor="none"))
            dark = (abs(shown - 0.5) > 0.3) if kind != "sequential" else shown > 0.6
            color = "white" if dark else (TEXT_COLOR if significant is not False else "#4d4d4d")
            ax.text(j + 0.5, i + 0.5, cell_text(value, kind, leading_zero) + ("*" if significant else ""),
                    ha="center", va="center", fontsize=cell_fontsize, color=color,
                    fontweight="bold" if significant else "normal")

    angle = math.radians(TILT)
    for j, judge in enumerate(judges):
        family, size = labels[judge].split("\n")
        if tilted:
            anchor = dict(xy=(j + 0.5, 0), xycoords=("data", "axes fraction"), textcoords="offset points",
                          ha="right", va="top", rotation=TILT, rotation_mode="anchor")
            ax.annotate(size, xytext=(0, -3), fontsize=fs["ticks"] - 1, color=MUTED_TEXT, **anchor)
            shift = width_pt(size, fs["ticks"] - 1) + 2.5
            ax.annotate(family, xytext=(-shift * math.cos(angle), -3 - shift * math.sin(angle)), fontsize=fs["ticks"],
                        color=TEXT_COLOR, **anchor)
            continue
        ax.annotate(family, xy=(j + 0.5, 0), xycoords=("data", "axes fraction"), xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top", fontsize=fs["ticks"], color=TEXT_COLOR)
        ax.annotate(size, xy=(j + 0.5, 0), xycoords=("data", "axes fraction"), xytext=(0, -10),
                    textcoords="offset points", ha="center", va="top", fontsize=fs["ticks"] - 1, color=MUTED_TEXT)

    if with_colorbar:
        colorbar(fig, [(label_margin + cell * n + COLORBAR_GAP) / PANEL_WIDTH, bottom / fig_height,
                       COLORBAR_WIDTH / PANEL_WIDTH, height / fig_height], norm, cmap, ticks, tick_labels, colorbar_label)
    return fig


def delta_matrix(dist, judges):
    """judges x delta bins: % of the judge's instances in [b, b + DELTA_BIN) ([.., 1] for the last)."""
    n_bins = round(1 / DELTA_BIN)
    bins = np.minimum(np.floor(dist["delta"] / DELTA_BIN + 1e-6), n_bins - 1).astype(int)
    return (dist.assign(bin=bins).groupby(["judge", "bin"])["pct"].sum().unstack()
            .reindex(index=judges, columns=range(n_bins)).fillna(0))


def pct_text(v):
    return f"{v:.0f}" if v >= 9.95 else f"{v:.1f}" if v >= 0.05 else "<.1"


def delta_heatmap(m, labels, threshold, xlabel, with_colorbar, label_margin, colorbar_margin):
    fs = HEATMAP_FONT_SIZES
    n_rows, n_bins = m.shape
    width = PANEL_WIDTH - label_margin - colorbar_margin
    cell = width / n_bins
    height = DELTA_ROW_HEIGHT * n_rows
    fig_height = DELTA_TOP + height + DELTA_BOTTOM
    fig = plt.figure(figsize=(PANEL_WIDTH, fig_height))
    ax = fig.add_axes([label_margin / PANEL_WIDTH, DELTA_BOTTOM / fig_height, width / PANEL_WIDTH, height / fig_height])
    ax.set_xlim(0, n_bins)
    ax.set_ylim(n_rows, 0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)

    gap_x, gap_y = CELL_GAP / 2 / cell, CELL_GAP / 2 / DELTA_ROW_HEIGHT
    for i in range(n_rows):
        for j in range(n_bins):
            v = m.iat[i, j]
            shown = DELTA_NORM(v) if v > 0 else None
            ax.add_patch(Rectangle((j + gap_x, i + gap_y), 1 - 2 * gap_x, 1 - 2 * gap_y, edgecolor="none",
                                   facecolor=NA_COLOR if shown is None else DELTA_CMAP(shown)))
            if shown is not None:
                ax.text(j + 0.5, i + 0.5, pct_text(v), ha="center", va="center", fontsize=fs["ci"] - (0.5 if v < 0.05 else 0),
                        color="white" if shown > 0.6 else TEXT_COLOR)
    row_labels(ax, list(m.index), labels, fs["ticks"])

    edges = np.arange(0, n_bins + 1, 2)
    ax.set_xticks(edges, [f"{e * DELTA_BIN:g}" for e in edges])
    ax.tick_params(axis="x", length=2, width=0.5, pad=1.5, color="#444444", labelsize=fs["ticks"], labelcolor=TEXT_COLOR)
    ax.set_xlabel(xlabel, fontsize=fs["ticks"] + 0.5, color=TEXT_COLOR, labelpad=2)
    x = threshold / DELTA_BIN
    ax.plot([x, x], [-0.12, n_rows], color=TEXT_COLOR, linewidth=0.9, clip_on=False, zorder=4)
    ax.annotate(f"big flip, Δ ≥ {threshold:g} →", xy=(x, 0), xycoords="data", xytext=(2, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=fs["ci"], color=MUTED_TEXT)
    if with_colorbar:
        ticks = [0.1, 1, 10, 100]
        colorbar(fig, [(label_margin + width + COLORBAR_GAP) / PANEL_WIDTH, DELTA_BOTTOM / fig_height,
                       COLORBAR_WIDTH / PANEL_WIDTH, height / fig_height],
                 DELTA_NORM, DELTA_CMAP, ticks, [f"{t:g}" for t in ticks], "Instances (%)")
    return fig


def load_cells(path, column, p_column, judges):
    """{(judge_a, judge_b): (value, significant)}; significant is None without a test (BH q < ALPHA)."""
    d = pd.read_csv(path)
    d = d[d["judge_a"].isin(judges) & d["judge_b"].isin(judges)]
    significant = (bh(d[p_column]) < ALPHA) if p_column else [None] * len(d)
    return {(a, b): (v, None if s is None else bool(s))
            for a, b, v, s in zip(d["judge_a"], d["judge_b"], d[column], significant)}


def draw_figures(args, run_dirs, fig_dir):
    """run_dirs: dataset -> folder of the analysis outputs."""
    setup_matplotlib()
    styles, labels = (SCALE_STYLES, SCALE_JUDGES) if args.scale_analysis else (MAIN_STYLES, JUDGES)
    xlabel, key_labels = FLIP_TYPES[args.flip_type]
    stats, used = {}, set()
    for dataset, folder in run_dirs.items():
        stats[dataset] = pd.read_csv(folder / "per_model_stats.csv").set_index("judge")
        used |= set(stats[dataset].index)
        if (folder / "k_distribution.csv").exists():
            save(k_distribution_plot(pd.read_csv(folder / "k_distribution.csv")),
                 fig_dir / f"{DATASET_SHORT[dataset]}_k_distribution")
    if not stats:
        return
    save(pct_flipped_plot(stats, styles, key_labels), fig_dir / "pct_flipped")

    deltas = {d: delta_matrix(pd.read_csv(f / "delta_distribution.csv"), present(styles, stats[d].index))
              for d, f in run_dirs.items()}
    label_margin = row_label_width([labels[j] for m in deltas.values() for j in m.index])
    for dataset, m in deltas.items():
        save(delta_heatmap(m, labels, args.threshold, xlabel, LAYOUTS[dataset] == "right", label_margin,
                           colorbar_width(["100"])), fig_dir / f"{DATASET_SHORT[dataset]}_delta")

    for metric, (csv, column, p_column, kind, colorbar_label) in METRICS.items():
        panels = {}
        for dataset, folder in run_dirs.items():
            path = folder / csv.format(score=args.situation_score)
            if not path.exists():
                continue
            d = pd.read_csv(path)
            judges = present(styles, pd.concat([d["judge_a"], d["judge_b"]]).unique())
            panels[dataset] = (judges, load_cells(path, column, p_column, judges))
            if metric == "ranking_topk":
                colorbar_label = f"Top-{int(d['top_k'].iloc[0])} overlap (chance {d['topk_chance_frac'].iloc[0]:.2f})"
        if not panels:
            continue
        scale = colour_scale(kind, [v for _, cells in panels.values() for v, _ in cells.values()])
        all_judges = [j for judges, _ in panels.values() for j in judges]
        label_margin = row_label_width([labels[j] for j in all_judges])
        colorbar_margin = colorbar_width(scale[4])
        n_max = max(len(judges) for judges, _ in panels.values())
        cell_width = (PANEL_WIDTH - label_margin - colorbar_margin) / n_max * 72
        text_format = cell_format([c for _, cells in panels.values() for c in cells.values()], kind, cell_width)
        column_layout = column_label_layout(all_judges, labels, cell_width)
        for dataset, (judges, cells) in panels.items():
            save(matrix_panel(cells, judges, labels, kind, scale, colorbar_label, LAYOUTS[dataset] == "right",
                              label_margin, colorbar_margin, n_max, text_format, column_layout),
                 fig_dir / f"{DATASET_SHORT[dataset]}_{metric}")
    save(legend_figure({j: styles[j] for j in present(styles, used)}), fig_dir / "legend")


def main():
    args = parse_args()
    run = run_label(args)
    flip_dir = f"{args.flip_type}_t{args.threshold:g}"
    run_dirs = {d: args.results_dir / run / d / flip_dir for d in args.datasets}
    if not args.plots_only:
        for dataset, out_dir in run_dirs.items():
            analyse_dataset(args, dataset, out_dir)
    run_dirs = {d: f for d, f in run_dirs.items() if (f / "per_model_stats.csv").exists()}
    fig_dir = args.figures_dir / ("main" if run == "all" else run)
    if args.flip_type != "order":
        fig_dir = fig_dir / flip_dir
    draw_figures(args, run_dirs, fig_dir)


if __name__ == "__main__":
    main()
