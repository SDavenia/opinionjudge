"""
Results of the human quality check of the llama-3.3-70b ValuePrism opinions: the share of
opinions that keep the target stance, the value and the explanation.

Each opinion gets yes_share, the share of its annotators answering yes; the rate of a group
(all, each stance, each value type, each annotator) is the mean yes_share, with a 95% Wilson
interval over opinions. Differences between groups are tested by permuting the labels (chi-
square statistic). With items shared between annotators, agreement.csv has percent agreement,
Fleiss' kappa, Krippendorff's alpha, Gwet's AC1 and pairwise Cohen's kappa.

The value type and the ValuePrism value and explanation of each opinion are not in the released
data: they are taken from the ValuePrism sample made by prepare_data.py. Without it, the value-
type breakdown is skipped and no_items.csv has no value and explanation; the other results are
unchanged.

Usage:
    python -m human_annotation.quality.compute_quality
    python -m human_annotation.quality.compute_quality --exclude ANN2
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.data_utils import with_valueprism_fields
from utils.paths import RESULTS_DIR

HERE = Path(__file__).resolve().parent
STANCES = ["Supports", "Opposes", "Either"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation_dir", type=Path, default=HERE,
                        help="Folder with samples/, assignments/ and ratings/.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Annotator ids to leave out.")
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR / "human_annotation" / "quality")
    return parser.parse_args()


def load_ratings(annotation_dir, items, exclude):
    # Partial files first, so that a final file's answer wins.
    files = sorted((annotation_dir / "ratings").glob("*.csv"), key=lambda f: ("_partial" not in f.stem, f.name))
    if not files:
        raise FileNotFoundError(f"No ratings in {annotation_dir}/ratings.")
    ratings = pd.concat([pd.read_csv(f).assign(ratings_file=f.name) for f in files], ignore_index=True)
    ratings = ratings[~ratings["annotator_id"].isin(exclude)]
    dup = ratings.duplicated(["annotator_id", "item_id"], keep="last")
    if dup.any():
        print(f"[warn] {dup.sum()} duplicated (annotator, item) ratings: keeping the final file's answer.")
        ratings = ratings[~dup]

    key = pd.read_csv(annotation_dir / "assignments" / "key.csv").drop(columns="vp_vrd", errors="ignore")
    unknown = ratings.merge(key, on=["annotator_id", "item_id"], how="left", indicator=True)
    if (unknown["_merge"] == "left_only").any():
        raise ValueError("Some returned item_ids are not in key.csv.")
    key = key[key["annotator_id"].isin(ratings["annotator_id"].unique())]
    long = key.merge(ratings[["annotator_id", "item_id", "label", "first_label", "rt_seconds", "n_visits", "comment",
                              "ratings_file"]], on=["annotator_id", "item_id"], how="left")
    long["label"] = long["label"].str.strip().str.lower()
    bad = long["label"].notna() & ~long["label"].isin(["yes", "no"])
    if bad.any():
        raise ValueError(f"Unexpected labels: {sorted(long.loc[bad, 'label'].unique())}")
    missing = long["label"].isna()
    if missing.any():
        print(f"[warn] {missing.sum()} assigned items have no answer: "
              f"{long[missing].groupby('annotator_id').size().to_dict()}")
    long = long[~missing].copy()
    long["yes"] = (long["label"] == "yes").astype(int)
    long["revised"] = long["label"] != long["first_label"].str.strip().str.lower()
    item_columns = [c for c in ["sample_id", "situation", "vp_vrd", "vp_vrd_value", "vp_explanation", "text"]
                    if c in items.columns]
    return long.merge(items[item_columns], on="sample_id", how="left")


def wilson(successes, n, level=0.95):
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(0.5 + level / 2)
    p = successes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def rate_row(per_item):
    n, k = len(per_item), per_item["yes_share"].sum()
    low, high = wilson(k, n)
    return {"n_items": n, "n_ratings": int(per_item["n_ratings"].sum()), "rate_yes": k / n if n else np.nan,
            "ci95_low": low, "ci95_high": high}


def chi2_stat(groups, labels, n_groups):
    n = np.bincount(groups, minlength=n_groups).astype(float)
    yes = np.bincount(groups, weights=labels, minlength=n_groups)
    table = np.stack([yes, n - yes], axis=1)
    expected = n[:, None] * table.sum(0)[None, :] / n.sum()
    return float(((table - expected) ** 2 / expected).sum())


def permutation_p(groups, labels, n_perm, seed):
    codes, uniques = pd.factorize(pd.Series(groups))
    labels = np.asarray(labels, dtype=float)
    if len(uniques) < 2 or len(np.unique(labels)) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    observed = chi2_stat(codes, labels, len(uniques))
    hits = sum(chi2_stat(codes, rng.permutation(labels), len(uniques)) >= observed - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def agreement_stats(long):
    """Agreement on the items rated by two or more annotators."""
    counts = long.groupby("sample_id")["yes"].agg(n="size", n_yes="sum")
    counts = counts[counts["n"] >= 2]
    if counts.empty:
        return pd.DataFrame()
    n_i, yes_i = counts["n"].to_numpy(float), counts["n_yes"].to_numpy(float)
    no_i = n_i - yes_i
    p_agree = ((yes_i * (yes_i - 1) + no_i * (no_i - 1)) / (n_i * (n_i - 1))).mean()
    pi_yes = yes_i.sum() / n_i.sum()
    p_chance = pi_yes ** 2 + (1 - pi_yes) ** 2
    fleiss = (p_agree - p_chance) / (1 - p_chance) if p_chance < 1 else np.nan
    p_chance_ac1 = 2 * pi_yes * (1 - pi_yes)
    ac1 = (p_agree - p_chance_ac1) / (1 - p_chance_ac1)
    n_total, n_yes_total = n_i.sum(), yes_i.sum()
    d_observed = (2 * yes_i * no_i / (n_i - 1)).sum() / n_total
    d_expected = 2 * n_yes_total * (n_total - n_yes_total) / (n_total * (n_total - 1))
    alpha = 1 - d_observed / d_expected if d_expected > 0 else np.nan
    rows = [{"measure": "n_shared_items", "value": len(counts)}, {"measure": "percent_agreement", "value": p_agree},
            {"measure": "fleiss_kappa", "value": fleiss}, {"measure": "krippendorff_alpha_nominal", "value": alpha},
            {"measure": "gwet_ac1", "value": ac1}, {"measure": "share_yes_on_shared", "value": pi_yes}]
    wide = long[long["sample_id"].isin(counts.index)].pivot_table(index="sample_id", columns="annotator_id", values="yes")
    for a, b in combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if both.empty:
            continue
        p_o = (both[a] == both[b]).mean()
        p_e = both[a].mean() * both[b].mean() + (1 - both[a].mean()) * (1 - both[b].mean())
        rows.append({"measure": f"cohen_kappa_{a}_{b}", "value": (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan})
        rows.append({"measure": f"percent_agreement_{a}_{b}", "value": p_o})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    items = with_valueprism_fields(pd.read_csv(args.annotation_dir / "samples" / "items.csv"))
    has_type = "vp_vrd" in items.columns and items["vp_vrd"].notna().all()
    if not has_type:
        print("[warn] No ValuePrism value types (run prepare_data.py): skipping the value-type breakdown.")
    long = load_ratings(args.annotation_dir, items, args.exclude)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_dir / "ratings_long.csv", index=False)

    per_item = (long.groupby(["sample_id", "stance"] + (["vp_vrd"] if has_type else []))
                .agg(n_ratings=("yes", "size"), n_yes=("yes", "sum"), yes_share=("yes", "mean"),
                     annotators=("annotator_id", lambda s: " ".join(sorted(s))))
                .reset_index())
    per_item.to_csv(out_dir / "per_item.csv", index=False)

    rows = [{"scope": "all", "group": "all", **rate_row(per_item)}]
    for col, scope in [("stance", "stance")] + ([("vp_vrd", "value_type")] if has_type else []):
        order = STANCES if col == "stance" else sorted(per_item[col].unique())
        rows += [{"scope": scope, "group": g, **rate_row(per_item[per_item[col] == g])} for g in order
                 if (per_item[col] == g).any()]
    for a, la in long.groupby("annotator_id"):
        rows.append({"scope": "annotator", "group": a, **rate_row(la.assign(n_ratings=1, yes_share=la["yes"]))})
    summary = pd.DataFrame(rows)
    p_values = {scope: permutation_p(long[col], long["yes"], args.n_perm, args.seed + k)
                for k, (scope, col) in enumerate([("stance", "stance"), ("value_type", "vp_vrd"),
                                                  ("annotator", "annotator_id")])
                if has_type or col != "vp_vrd"}
    summary["perm_p_between_groups"] = summary["scope"].map(p_values)
    summary.to_csv(out_dir / "summary.csv", index=False)

    per_annotator = []
    for a, la in long.groupby("annotator_id"):
        low, high = wilson(la["yes"].sum(), len(la))
        row = {"annotator_id": a, "n": len(la), "n_no": int((la["yes"] == 0).sum()), "rate_yes": la["yes"].mean(),
               "ci95": f"[{low:.3f}, {high:.3f}]"}
        row.update({f"rate_yes_{s}": la.loc[la["stance"] == s, "yes"].mean() for s in STANCES})
        row.update({"median_rt_s": la["rt_seconds"].median(), "n_revised": int(la["revised"].sum()),
                    "n_comments": int(la["comment"].fillna("").astype(str).str.strip().ne("").sum())})
        per_annotator.append(row)
    per_annotator = pd.DataFrame(per_annotator)
    per_annotator.to_csv(out_dir / "per_annotator.csv", index=False)
    agreement = agreement_stats(long)
    if not agreement.empty:
        agreement.to_csv(out_dir / "agreement.csv", index=False)
    no_ids = per_item.loc[per_item["n_yes"] < per_item["n_ratings"], "sample_id"]
    no_columns = [c for c in ["sample_id", "text_id", "annotator_id", "label", "comment", "stance", "vp_vrd",
                              "situation", "vp_vrd_value", "vp_explanation", "text"] if c in long.columns]
    (long[long["sample_id"].isin(no_ids)].sort_values(["sample_id", "annotator_id"])[no_columns]
     .to_csv(out_dir / "no_items.csv", index=False))

    pd.set_option("display.width", 220)
    fmt = lambda df: df.to_string(index=False, float_format=lambda v: f"{v:.3f}")  # noqa: E731
    print(f"{long['annotator_id'].nunique()} annotators, {len(long)} ratings, {len(per_item)} opinions")
    print("\nShare of opinions keeping the stance, the value and the explanation (95% Wilson CI):")
    print(fmt(summary))
    print("\nPer annotator:")
    print(fmt(per_annotator))
    if not agreement.empty:
        print("\nAgreement on the shared items:")
        print(fmt(agreement))
    print(f"\n{len(no_ids)} opinions with a 'no' -> {out_dir / 'no_items.csv'}")


if __name__ == "__main__":
    main()
