"""
Paper tables of the human symmetry study (run compute_stability first):
  human_symmetry.tex     the pooled human measures, for all pairs and each dataset;
  human_vs_judges.tex    per dataset, humans and each main judge on the human-rated pairs:
                         PC, and Spearman's rho between the judge's per-pair score and the
                         human one (for humans: each annotator against the mean of the others,
                         averaged over annotators); 95% CIs from bootstrap resamples of pairs,
                         the same for every scorer;
  symmetry_shares.csv    share of judgments with the same score in both orders, within 0.1,
                         and differing by 0.3 or more, per scorer.

Usage:
    python -m human_annotation.symmetric.paper_tables
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from human_annotation.symmetric.compute_stability import ANNOTATION_DIR, OUTPUT_DIR, load_ratings, wide_pairs
from utils.analysis_utils import SCALE_MODELS, DatasetContext, discover_judges, load_scoring_df
from utils.paths import SCORING_DIR
from utils.plot_utils import judge_name

DATASETS = {"valueprism": "ValuePrism", "habermas": "Habermas"}
SCOPES = {"all": "All", **DATASETS}
JUDGE_ORDER = ["gemma3", "llama-3.3-70b", "mistral-24b", *SCALE_MODELS, "openai/gpt-5.4-nano"]
# pooled.csv metric -> (row label, scale factor, signed)
SYMMETRY_ROWS = {
    "PC": ("PC", 1, False),
    "share_symmetric": ("Identical in both orders (\\%)", 100, False),
    "share_within_0.1": ("Within $0.1$ (\\%)", 100, False),
    "signed_asym": ("Signed asymmetry", 1, True),
    "order_effect": ("Order effect", 1, True),
    "inter_annotator_stability": ("Inter-annotator stability", 1, False),
    "asym_minus_between": ("Asymmetry $-$ disagreement", 1, True),
    "alpha_interval": ("$\\alpha$ (ratings)", 1, False),
    "alpha_signed_asym": ("$\\alpha$ (signed asymmetry)", 1, True),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation_dir", type=Path, default=ANNOTATION_DIR)
    parser.add_argument("--exclude", nargs="*", default=[], help="Annotator ids to leave out.")
    parser.add_argument("--scoring_dir", type=Path, default=SCORING_DIR)
    parser.add_argument("--run_on", default="all", choices=["all", "open", "closed"])
    parser.add_argument("--results_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pair_key(text_id_1, text_id_2):
    return tuple(sorted((text_id_1, text_id_2)))


def judge_dfs(args, dataset_id):
    """judge -> scoring dataframe, in the paper's judge order."""
    ctx = DatasetContext(dataset_id, run_on=args.run_on, scoring_dir=args.scoring_dir)
    found = discover_judges(ctx)
    return {judge: load_scoring_df(ctx, judge) for judge in JUDGE_ORDER if judge in found}


def human_matrices(w):
    """(pair x annotator) score (mean of both orders) and |asymmetry|."""
    w = w.assign(pair=[pair_key(a, b) for a, b in zip(w["text_id_1"], w["text_id_2"])],
                 score=w[["r_1to2", "r_2to1"]].mean(axis=1), abs_asym=w["asym"].abs())
    score = w.pivot(index="pair", columns="annotator_id", values="score")
    abs_asym = w.pivot(index="pair", columns="annotator_id", values="abs_asym")
    if score.isna().any().any():
        raise ValueError("Not every annotator rated every pair.")
    return score, abs_asym


def judge_pair_stats(args, dataset_id, pairs):
    """judge name -> per pair: mean score and mean |1to2 - 2to1| over the prompt variations."""
    out = {}
    for judge, df in judge_dfs(args, dataset_id).items():
        df["pair"] = [pair_key(a, b) for a, b in zip(df["text_id_1"], df["text_id_2"])]
        df = df[df["pair"].isin(pairs)]
        df["score"] = df[["parsed_score_1to2", "parsed_score_2to1"]].mean(axis=1)
        df["abs_asym"] = (df["parsed_score_1to2"] - df["parsed_score_2to1"]).abs()
        stats = df.groupby("pair")[["score", "abs_asym"]].mean().reindex(pairs)
        if stats.isna().any().any():
            print(f"[warn] {dataset_id}/{judge}: scored {stats['score'].notna().sum()} of {len(pairs)} pairs - skipped.")
            continue
        out[judge_name(judge)] = stats
    return out


def symmetry_shares(args, w):
    long = []
    for dataset_id in DATASETS:
        wd = w[w["dataset"] == dataset_id]
        long.append(pd.DataFrame({"dataset": dataset_id, "scorer": "Humans", "unit": wd["annotator_id"].to_numpy(),
                                  "s1": wd["r_1to2"].to_numpy(), "s2": wd["r_2to1"].to_numpy()}))
        pairs = {pair_key(a, b) for a, b in zip(wd["text_id_1"], wd["text_id_2"])}
        for judge, df in judge_dfs(args, dataset_id).items():
            df = df[[pair_key(a, b) in pairs for a, b in zip(df["text_id_1"], df["text_id_2"])]]
            long.append(pd.DataFrame({"dataset": dataset_id, "scorer": judge_name(judge),
                                      "unit": df.groupby(["id_1", "id_2"]).cumcount().to_numpy(),
                                      "s1": df["parsed_score_1to2"].to_numpy(),
                                      "s2": df["parsed_score_2to1"].to_numpy()}))
    long = pd.concat(long, ignore_index=True)
    diff = (long["s1"] - long["s2"]).abs()
    on_grid = lambda v: (v * 10 - np.round(v * 10)).abs() < 1e-6  # noqa: E731
    long = long.assign(identical=diff < 1e-9, within_01=diff <= 0.1 + 1e-9, large_flip=diff >= 0.3 - 1e-9,
                       identical_rounded=(np.round(long["s1"] * 10) == np.round(long["s2"] * 10)),
                       off_grid=~(on_grid(long["s1"]) & on_grid(long["s2"])))
    rows = []
    for scope, sub in [("all", long), *((d, long[long["dataset"] == d]) for d in DATASETS)]:
        for scorer, g in sub.groupby("scorer", sort=False):
            per_unit = g.groupby("unit")[["identical", "within_01"]].mean()
            rows.append({"scope": scope, "scorer": scorer, "n_judgments": len(g),
                         "identical": g["identical"].mean(), "within_0.1": g["within_01"].mean(),
                         "large_flip": g["large_flip"].mean(), "identical_rounded": g["identical_rounded"].mean(),
                         "off_grid": g["off_grid"].mean(),
                         "identical_unit_min": per_unit["identical"].min(),
                         "identical_unit_max": per_unit["identical"].max(),
                         "within_0.1_unit_min": per_unit["within_01"].min(),
                         "within_0.1_unit_max": per_unit["within_01"].max()})
    return pd.DataFrame(rows)


def scope_stats(score, abs_asym, judges):
    """score / abs_asym: pair x annotator; judges: name -> pair x [score, abs_asym]."""
    human_mean = score.mean(axis=1)
    human_pc = 1 - abs_asym.mean()
    loo = [spearmanr(score[:, a], np.delete(score, a, axis=1).mean(axis=1)).statistic for a in range(score.shape[1])]
    stats = {("Humans", "PC"): human_pc, ("Humans", "rho"): np.mean(loo), ("Humans", "dPC"): 0.0}
    for name, j in judges.items():
        stats[(name, "PC")] = 1 - j[:, 1].mean()
        stats[(name, "rho")] = spearmanr(j[:, 0], human_mean).statistic
        stats[(name, "dPC")] = human_pc - stats[(name, "PC")]
    return stats


def human_vs_judges(args, w):
    data = {}
    for dataset_id in DATASETS:
        score, abs_asym = human_matrices(w[w["dataset"] == dataset_id])
        judges = judge_pair_stats(args, dataset_id, score.index)
        data[dataset_id] = (score.to_numpy(), abs_asym.to_numpy(), {k: v.to_numpy() for k, v in judges.items()})
    first = next(iter(data.values()))[2]
    common = [name for name in first if all(name in d[2] for d in data.values())]

    def combine(idx):
        out = {d: scope_stats(s[idx[d]], a[idx[d]], {k: v[idx[d]] for k, v in j.items()})
               for d, (s, a, j) in data.items()}
        cat = lambda arr: np.concatenate([arr(d)[idx[d]] for d in data])  # noqa: E731
        out["all"] = scope_stats(cat(lambda d: data[d][0]), cat(lambda d: data[d][1]),
                                 {k: cat(lambda d: data[d][2][k]) for k in common})
        return out

    point = combine({d: np.arange(len(v[0])) for d, v in data.items()})
    rng = np.random.default_rng(args.seed)
    draws = [combine({d: rng.integers(0, len(v[0]), len(v[0])) for d, v in data.items()}) for _ in range(args.n_boot)]
    rows = []
    for scope, stats in point.items():
        for (scorer, metric), value in stats.items():
            boot = np.array([dr[scope][(scorer, metric)] for dr in draws])
            rows.append({"scope": scope, "scorer": scorer, "metric": metric, "value": value,
                         "ci95_low": np.nanquantile(boot, 0.025), "ci95_high": np.nanquantile(boot, 0.975),
                         "n_pairs": sum(len(data[d][0]) for d in data) if scope == "all" else len(data[scope][0])})
    return pd.DataFrame(rows)


def _num(v, signed, digits):
    return f"${v:+.{digits}f}$" if signed else f"${v:.{digits}f}$"


def _cell(value, low, high, signed=False, digits=2, bold=False, mark=""):
    v = _num(value, signed, digits)
    if bold:
        v = f"$\\mathbf{{{v.strip('$')}}}$"
    return f"{v}{mark} [{_num(low, signed, digits)}, {_num(high, signed, digits)}]"


def _tabular(n_values):
    return f"\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}l{'c' * n_values}@{{}}}}"


def human_vs_judges_tex(table, n_annotators):
    def get(scope, scorer, metric):
        return table[(table["scope"] == scope) & (table["scorer"] == scorer) & (table["metric"] == metric)].iloc[0]

    best = {(d, m): table[(table["scope"] == d) & (table["metric"] == m) & (table["scorer"] != "Humans")]["value"].max()
            for d in DATASETS for m in ("PC", "rho")}
    lines = ["\\begin{table}[ht]", "\\centering", "\\scriptsize", _tabular(2 * len(DATASETS)), "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in DATASETS.values()) + " \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Scorer" + " & PC & $\\rho$ with humans" * len(DATASETS) + " \\\\", "\\midrule"]
    for scorer in dict.fromkeys(table["scorer"]):
        cells = []
        for d in DATASETS:
            for m in ("PC", "rho"):
                r = get(d, scorer, m)
                # PC with 3 digits: the judges' PCs are too close to tell apart with 2.
                cells.append(_cell(r["value"], r["ci95_low"], r["ci95_high"], digits=3 if m == "PC" else 2,
                                   bold=scorer != "Humans" and np.isclose(r["value"], best[(d, m)]),
                                   mark="$^{\\dagger}$" if scorer == "Humans" and m == "rho" else ""))
        label = f"Humans ({n_annotators})" if scorer == "Humans" else scorer
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if scorer == "Humans":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def human_symmetry_tex(pooled):
    lines = ["\\begin{table}[ht]", "\\centering", "\\scriptsize", _tabular(len(SCOPES)), "\\toprule",
             "Measure & " + " & ".join(SCOPES.values()) + " \\\\", "\\midrule"]
    for metric, (label, scale, signed) in SYMMETRY_ROWS.items():
        cells = []
        for scope in SCOPES:
            r = pooled[(pooled["scope"] == scope) & (pooled["metric"] == metric)].iloc[0]
            digits = 0 if scale == 100 else 3
            cells.append(_cell(r["value"] * scale, r["ci95_low"] * scale, r["ci95_high"] * scale, signed, digits))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    pairs = pd.read_csv(args.annotation_dir / "samples" / "pairs.csv")
    w = wide_pairs(load_ratings(args.annotation_dir, args.exclude), pairs)
    n_annotators = w["annotator_id"].nunique()
    tables_dir = args.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table = human_vs_judges(args, w)
    table.to_csv(args.results_dir / "human_vs_judges.csv", index=False)
    (tables_dir / "human_vs_judges.tex").write_text(human_vs_judges_tex(table, n_annotators))
    (tables_dir / "human_symmetry.tex").write_text(human_symmetry_tex(pd.read_csv(args.results_dir / "pooled.csv")))
    shares = symmetry_shares(args, w)
    shares.to_csv(args.results_dir / "symmetry_shares.csv", index=False)

    show = table.assign(cell=[f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in
                              zip(table["value"], table["ci95_low"], table["ci95_high"])])
    pd.set_option("display.width", 220)
    for metric in ("PC", "rho", "dPC"):
        print(f"\n-- {metric} --")
        print(show[show["metric"] == metric].pivot(index="scorer", columns="scope", values="cell")
              .reindex(index=list(dict.fromkeys(table["scorer"])), columns=list(SCOPES)).to_string())
    print(f"\nSaved the tables to {args.results_dir}")


if __name__ == "__main__":
    main()
