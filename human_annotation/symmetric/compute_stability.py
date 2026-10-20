"""
Measures of the human symmetry study: is the rating of "how well does B align with A"
(1to2, A = text_1) the same as the reverse (2to1) when humans give it?

Every measure is computed over all pairs ("all") and for each dataset, with 95% CIs from
bootstrap resamples of pairs (within each dataset for "all"):
  PC                   1 - mean |r(1to2) - r(2to1)|, as for the LLM judges
  mean_abs_asym        mean |r(1to2) - r(2to1)| (= 1 - PC)
  share_symmetric      share of pairs rated the same in both orders (and within 0.1)
  signed_asym          mean r(1to2) - r(2to1) (+ Wilcoxon p); the orientation is arbitrary, so ~0
  order_effect         mean second - first exposure (+ Wilcoxon p): drift or anchoring
  retest_stability     1 - mean |r - r'| on exact repeats (only with retest items)
  asym_minus_retest    asymmetry beyond the test-retest noise
With pairs rated by two or more annotators:
  RS                   calculate_rs with the annotators as the repeated judgments
  inter_annotator_stability  1 - mean |r_a - r_b|, same pair and order
  asym_minus_between   within-person asymmetry minus between-person disagreement
  alpha_interval       Krippendorff's alpha (interval) of the ratings
  consensus_PC         PC of the ratings averaged over annotators
  alpha_signed_asym    Krippendorff's alpha of the signed asymmetry: ~0 if it is individual
                       noise, > 0 if some pairs are rated asymmetrically by everyone

Also: the rating by which stance is the reference A (stance_direction.csv), and every LLM
judge's PC / RS on the same pairs with the Spearman correlation of its scores with the
human ones (llm_comparison.csv).

Usage:
    python -m human_annotation.symmetric.compute_stability
    python -m human_annotation.symmetric.compute_stability --exclude ANN4
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from utils.evaluation_utils import calculate_pc, calculate_pc_from_df, calculate_rs, calculate_rs_from_df
from utils.paths import RESULTS_DIR, SCORING_DIR

ANNOTATION_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RESULTS_DIR / "human_annotation" / "symmetric"
GENERATION_PREFIX = {"valueprism": "llama-3.3-70b_reflective_person", "habermas": None}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation_dir", type=Path, default=ANNOTATION_DIR,
                        help="Folder with samples/, assignments/ and ratings/.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Annotator ids to leave out.")
    parser.add_argument("--scoring_dir", type=Path, default=SCORING_DIR)
    parser.add_argument("--scoring_suffix", default="_0_1.csv")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_ratings(annotation_dir=ANNOTATION_DIR, exclude=()):
    """Every returned rating joined with the assignment key (one row per assigned item)."""
    files = sorted((Path(annotation_dir) / "ratings").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No ratings in {annotation_dir}/ratings.")
    ratings = pd.concat([pd.read_csv(f).assign(ratings_file=f.name) for f in files], ignore_index=True)
    ratings = ratings[~ratings["annotator_id"].isin(exclude)]
    dup = ratings.duplicated(["annotator_id", "item_id"], keep="last")
    if dup.any():
        print(f"[warn] {dup.sum()} duplicated (annotator, item) ratings: keeping the last file's value.")
        ratings = ratings[~dup]

    key = pd.read_csv(Path(annotation_dir) / "assignments" / "key.csv")
    unknown = ratings.merge(key, on=["annotator_id", "item_id"], how="left", indicator=True)
    if (unknown["_merge"] == "left_only").any():
        raise ValueError("Some returned item_ids are not in key.csv.")
    key = key[key["annotator_id"].isin(ratings["annotator_id"].unique())]
    long = key.merge(ratings[["annotator_id", "item_id", "rating", "first_rating", "rt_seconds", "n_visits", "comment",
                              "ratings_file"]], on=["annotator_id", "item_id"], how="left")
    missing = long["rating"].isna()
    if missing.any():
        print(f"[warn] {missing.sum()} assigned items have no rating: "
              f"{long[missing].groupby('annotator_id').size().to_dict()}")
    return long


def wide_pairs(long, pairs):
    """One row per (annotator, pair): the rating in each order, by exposure, and the retest."""
    main = long[long["role"] == "main"]
    by_dir = main.pivot_table(index=["annotator_id", "pair_id"], columns="direction", values="rating")
    by_exp = main.pivot_table(index=["annotator_id", "pair_id"], columns="exposure", values="rating")
    first_dir = main[main["exposure"] == 1].set_index(["annotator_id", "pair_id"])["direction"]
    wide = pd.DataFrame({"r_1to2": by_dir.get("1to2"), "r_2to1": by_dir.get("2to1"), "first_direction": first_dir,
                         "r_first": by_exp.get(1), "r_second": by_exp.get(2)})
    retest = long[long["role"] == "retest"].set_index(["annotator_id", "pair_id"])["rating"]
    wide["r_retest"] = retest.reindex(wide.index)
    wide = wide.reset_index().dropna(subset=["r_1to2", "r_2to1"])
    wide["asym"] = wide["r_1to2"] - wide["r_2to1"]
    wide["order_diff"] = wide["r_second"] - wide["r_first"]
    wide["retest_diff"] = wide["r_retest"] - wide["r_first"]
    return wide.merge(pairs[["pair_id", "dataset", "situation_id", "stance_combo", "stance_1", "stance_2",
                             "text_id_1", "text_id_2"]], on="pair_id", how="left")


def krippendorff_alpha_interval(units):
    """units: one array of values per item; items with fewer than two values are dropped."""
    units = [np.asarray(u, dtype=float) for u in units]
    units = [u[~np.isnan(u)] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")

    def sq(u):  # sum over i != j of (x_i - x_j)^2
        return 2 * (len(u) * np.sum(u ** 2) - np.sum(u) ** 2)

    values = np.concatenate(units)
    n = len(values)
    d_observed = sum(sq(u) / (len(u) - 1) for u in units) / n
    d_expected = sq(values) / (n * (n - 1))
    return float(1 - d_observed / d_expected) if d_expected > 0 else float("nan")


def wilcoxon_p(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5 or np.all(x == 0):
        return float("nan")
    return float(wilcoxon(x, zero_method="zsplit", method="approx").pvalue)


def symmetry_stats(w):
    """Within-annotator measures."""
    stats = {
        "n_pairs": len(w),
        "PC": calculate_pc([[(a, b)] for a, b in zip(w["r_1to2"], w["r_2to1"])]) if len(w) else np.nan,
        "mean_abs_asym": w["asym"].abs().mean(),
        "share_symmetric": (w["asym"].abs() < 1e-9).mean(),
        "share_within_0.1": (w["asym"].abs() <= 0.1 + 1e-9).mean(),
        "signed_asym": w["asym"].mean(),
        "order_effect": w["order_diff"].mean(),
    }
    rt = w["retest_diff"].dropna()
    stats["n_retest"] = len(rt)
    stats["retest_stability"] = 1 - rt.abs().mean() if len(rt) else np.nan
    stats["asym_minus_retest"] = stats["mean_abs_asym"] - rt.abs().mean() if len(rt) else np.nan
    return stats


def overlap_stats(w):
    """Between-annotator measures, on the pairs rated by two or more annotators."""
    n_ann = w.groupby("pair_id")["annotator_id"].nunique()
    w = w[w["pair_id"].isin(n_ann[n_ann >= 2].index)]
    if w.empty:
        return {}
    items = {d: w.groupby("pair_id")[f"r_{d}"].apply(list) for d in ("1to2", "2to1")}
    all_items = list(items["1to2"]) + list(items["2to1"])
    between = [abs(a - b) for ratings in all_items for a, b in combinations(ratings, 2)]
    consensus = w.groupby("pair_id")[["r_1to2", "r_2to1"]].mean()
    return {
        "n_shared_pairs": w["pair_id"].nunique(),
        "RS": calculate_rs(all_items),
        "inter_annotator_stability": 1 - np.mean(between),
        "asym_minus_between": w["asym"].abs().mean() - np.mean(between),
        "alpha_interval": krippendorff_alpha_interval(all_items),
        "consensus_PC": calculate_pc([[(a, b)] for a, b in consensus.itertuples(index=False)]),
        "alpha_signed_asym": krippendorff_alpha_interval(list(w.groupby("pair_id")["asym"].apply(list))),
    }


def all_stats(w):
    return {**symmetry_stats(w), **overlap_stats(w)}


def bootstrap(w, fn, n_boot, seed):
    """Estimate and 95% CI of every statistic of fn, resampling pairs within each dataset
    (a pair's rows move together; resampled copies count as distinct pairs)."""
    point = fn(w)
    rng = np.random.default_rng(seed)
    w = w.reset_index(drop=True)
    grouped = w.groupby("pair_id")
    rows_of = list(grouped.indices.values())
    strata = [np.flatnonzero(grouped["dataset"].first().values == d) for d in w["dataset"].unique()]
    draws = []
    for _ in range(n_boot):
        sample = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        boot = w.iloc[np.concatenate([rows_of[s] for s in sample])].copy()
        boot["pair_id"] = np.repeat(np.arange(len(sample)), [len(rows_of[s]) for s in sample])
        draws.append(fn(boot))
    draws = pd.DataFrame(draws)
    rows = []
    for name, value in point.items():
        col = draws[name] if name in draws else pd.Series(dtype=float)
        is_count = name.startswith("n_")
        rows.append({"metric": name, "value": value, "ci95_low": np.nan if is_count else col.quantile(0.025),
                     "ci95_high": np.nan if is_count else col.quantile(0.975)})
    return pd.DataFrame(rows)


def stance_direction_table(w):
    """Mixed-stance pairs: the rating when A holds stance x (B holds y) vs the reverse."""
    mixed = w[w["stance_1"] != w["stance_2"]].copy()
    if mixed.empty:
        return pd.DataFrame()
    x_is_1 = mixed["stance_1"] < mixed["stance_2"]
    mixed["stance_x"] = np.where(x_is_1, mixed["stance_1"], mixed["stance_2"])
    mixed["stance_y"] = np.where(x_is_1, mixed["stance_2"], mixed["stance_1"])
    mixed["r_A_is_x"] = np.where(x_is_1, mixed["r_1to2"], mixed["r_2to1"])
    mixed["r_A_is_y"] = np.where(x_is_1, mixed["r_2to1"], mixed["r_1to2"])
    mixed["diff"] = mixed["r_A_is_x"] - mixed["r_A_is_y"]
    return (mixed.groupby(["dataset", "stance_x", "stance_y"])
            .agg(n=("diff", "size"), r_A_is_x=("r_A_is_x", "mean"), r_A_is_y=("r_A_is_y", "mean"),
                 mean_diff=("diff", "mean"), mean_abs_diff=("diff", lambda d: d.abs().mean()),
                 wilcoxon_p=("diff", wilcoxon_p))
            .reset_index())


def llm_comparison(args, w, pairs, dataset_id):
    """Humans vs every LLM judge on the dataset's rated pairs."""
    scoring_dir = args.scoring_dir / dataset_id
    if not scoring_dir.is_dir():
        print(f"Skipping the LLM comparison for {dataset_id}: no {scoring_dir}.")
        return pd.DataFrame()
    prefix = GENERATION_PREFIX[dataset_id]
    human_mean = (w.groupby("pair_id")[["r_1to2", "r_2to1"]].mean()
                  .join(pairs.set_index("pair_id")[["text_id_1", "text_id_2"]]))
    keys = set(zip(human_mean["text_id_1"], human_mean["text_id_2"]))
    rows = [{"dataset": dataset_id, "judge": "humans (pooled)", "n_pairs": w["pair_id"].nunique(),
             "PC": calculate_pc([[(a, b)] for a, b in zip(w["r_1to2"], w["r_2to1"])]),
             "RS": overlap_stats(w).get("RS", np.nan), "spearman_vs_human": np.nan, "file": ""}]
    for path in sorted(scoring_dir.rglob(f"*{args.scoring_suffix}")):
        rel = str(path.relative_to(scoring_dir))
        if prefix and f"{prefix}_" not in rel:
            continue
        df = pd.read_csv(path, usecols=["id_1", "id_2", "text_id_1", "text_id_2", "judge_model", "parsed_score_1to2",
                                        "parsed_score_2to1"])
        forward = df[[k in keys for k in zip(df["text_id_1"], df["text_id_2"])]]
        backward = df[[k in keys for k in zip(df["text_id_2"], df["text_id_1"])]].rename(columns={
            "text_id_1": "text_id_2", "text_id_2": "text_id_1",
            "parsed_score_1to2": "parsed_score_2to1", "parsed_score_2to1": "parsed_score_1to2"})
        sub = pd.concat([forward, backward]).dropna(subset=["parsed_score_1to2", "parsed_score_2to1"])
        if sub.empty:
            continue
        llm_mean = sub.groupby(["text_id_1", "text_id_2"])[["parsed_score_1to2", "parsed_score_2to1"]].mean()
        joined = human_mean.join(llm_mean, on=["text_id_1", "text_id_2"], how="inner")
        rows.append({"dataset": dataset_id, "judge": sub["judge_model"].iloc[0], "n_pairs": len(llm_mean),
                     "PC": calculate_pc_from_df(sub), "RS": calculate_rs_from_df(sub),
                     "spearman_vs_human": spearmanr(np.r_[joined["r_1to2"], joined["r_2to1"]],
                                                    np.r_[joined["parsed_score_1to2"], joined["parsed_score_2to1"]]).statistic,
                     "file": rel})
    return pd.DataFrame(rows)


def annotator_row(wa, la, args, seed):
    boot = bootstrap(wa, symmetry_stats, args.n_boot, seed).set_index("metric")
    row = boot["value"].to_dict()
    for m in ("PC", "mean_abs_asym", "retest_stability", "asym_minus_retest"):
        low, high = boot.loc[m, "ci95_low"], boot.loc[m, "ci95_high"]
        row[f"{m}_ci95"] = "" if pd.isna(low) else f"[{low:.3f}, {high:.3f}]"
    row["signed_asym_p"] = wilcoxon_p(wa["asym"])
    row["order_effect_p"] = wilcoxon_p(wa["order_diff"])
    row["rating_mean"], row["rating_sd"] = la["rating"].mean(), la["rating"].std()
    row["median_rt_s"] = la["rt_seconds"].median()
    row["n_revised"] = int((la["rating"] != la["first_rating"]).sum())
    row["n_comments"] = int(la["comment"].fillna("").astype(str).str.strip().ne("").sum())
    return row


def main():
    args = parse_args()
    pairs = pd.read_csv(args.annotation_dir / "samples" / "pairs.csv")
    long = load_ratings(args.annotation_dir, args.exclude)
    w = wide_pairs(long, pairs)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_dir / "ratings_long.csv", index=False)
    w.to_csv(out_dir / "per_pair.csv", index=False)

    scopes = {"all": w, **{d: w[w["dataset"] == d] for d in w["dataset"].unique()}}
    per_annotator, pooled, llm = [], [], []
    for s, (scope, ws) in enumerate(scopes.items()):
        ls = long if scope == "all" else long[long["dataset"] == scope]
        for k, (annotator_id, wa) in enumerate(ws.groupby("annotator_id")):
            row = annotator_row(wa, ls[ls["annotator_id"] == annotator_id], args, args.seed + 100 * s + k)
            per_annotator.append({"scope": scope, "annotator_id": annotator_id, **row})
        p = bootstrap(ws, all_stats, args.n_boot, args.seed + 100 * s)
        p.loc[len(p)] = {"metric": "signed_asym_wilcoxon_p", "value": wilcoxon_p(ws["asym"])}
        p.loc[len(p)] = {"metric": "order_effect_wilcoxon_p", "value": wilcoxon_p(ws["order_diff"])}
        pooled.append(p.assign(scope=scope))
        if scope != "all":
            llm.append(llm_comparison(args, ws, pairs, scope))

    per_annotator = pd.DataFrame(per_annotator)
    per_annotator.to_csv(out_dir / "per_annotator.csv", index=False)
    pooled = pd.concat(pooled)[["scope", "metric", "value", "ci95_low", "ci95_high"]]
    pooled.to_csv(out_dir / "pooled.csv", index=False)

    def cell(r):
        if pd.isna(r["value"]):
            return ""
        if r["metric"].startswith("n_"):
            return f"{r['value']:.0f}"
        if pd.isna(r["ci95_low"]):
            return f"{r['value']:.3f}"
        return f"{r['value']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"

    pooled_table = (pooled.assign(cell=pooled.apply(cell, axis=1))
                    .pivot(index="metric", columns="scope", values="cell")
                    .reindex(index=pooled["metric"].unique(), columns=list(scopes)))
    pooled_table.to_csv(out_dir / "pooled_table.csv")
    stance = stance_direction_table(w)
    stance.to_csv(out_dir / "stance_direction.csv", index=False)
    llm = pd.concat([d for d in llm if not d.empty]) if any(not d.empty for d in llm) else pd.DataFrame()
    if not llm.empty:
        llm.to_csv(out_dir / "llm_comparison.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\n{w['annotator_id'].nunique()} annotators, {w['pair_id'].nunique()} pairs "
          f"({w.groupby('dataset')['pair_id'].nunique().to_dict()})")
    overall = per_annotator[per_annotator["scope"] == "all"]
    show = ["annotator_id", "n_pairs", "PC", "PC_ci95", "share_symmetric", "share_within_0.1", "signed_asym",
            "signed_asym_p", "order_effect", "order_effect_p", "rating_sd", "median_rt_s"]
    print(overall[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    flags = overall[(overall["median_rt_s"] < 8) | (overall["rating_sd"] < 0.1)]
    if not flags.empty:
        print(f"[check] median time < 8 s or rating sd < 0.1: {flags['annotator_id'].tolist()}")
    print("\nPooled over annotators: value [95% CI over pairs]")
    print(pooled_table.fillna("").to_string())
    print(f"\nSaved the tables to {out_dir}")


if __name__ == "__main__":
    main()
