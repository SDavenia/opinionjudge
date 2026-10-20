"""
Stability of the judges' scores (RS over prompt variations, PC over the order of the two
opinions) as a function of features of the opinion pairs.

For every dataset, judge and feature analysis, it tests the Spearman correlation between
each characteristic of the pair (average, maximum, minimum and absolute difference of the
feature over the two opinions) and the pair's RS/PC, with 95% CIs and p-values from
situation-level bootstrap resamples and Benjamini-Hochberg q-values within each analysis.
It also computes the smoothed RS/PC curve along the difference in the feature (sliding
windows with bootstrap bands), and the overall RS/PC of each judge. The paper figures are
then drawn from these results.

With --joint_model it instead fits all features together (partial Spearman correlations
and a linear mixed model with a random intercept per situation), as a robustness check.

Usage:
    python analyse_stability.py
    python analyse_stability.py --scale_analysis
    python analyse_stability.py --datasets habermas --analyses text perplexity
    python analyse_stability.py --plots_only
    python analyse_stability.py --joint_model
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils.analysis_utils import (
    DATASET_NAMES, DATASET_SHORT, JOINT_FEATURES, SCALE_MODELS, DatasetContext, add_data_args, bh_adjust, ci_p,
    discover_judges, enrich_alignment, enrich_embedding, enrich_perplexity, enrich_refusal, enrich_safety,
    enrich_sentiment, enrich_text, enrich_vad, enrich_vader, joint_pairs, load_scoring_df, rank_columns, run_label,
    significance_tests, situation_bootstrap_bands, situation_rows, sliding_window_starts, write_significance,
)
from utils.data_utils import DATASETS
from utils.evaluation_utils import calculate_pc_from_df, calculate_rs_from_df, get_den_normalization
from utils.paths import FIGURES_DIR, RESULTS_DIR
from utils.plot_utils import (
    HEATMAP_STEM, JUDGES, MAIN_STYLES, SCALE_JUDGES, SCALE_STYLES, heatmap_figure, judge_name, legend_figure,
    line_plot, load_curves, save, select_strongest, setup_matplotlib,
)


def characteristics(name, label):
    return {f"avg_{name}": f"Average {label}", f"max_{name}": f"Maximum {label}",
            f"min_{name}": f"Minimum {label}", f"diff_{name}": f"Absolute difference in {label}"}


# Feature analyses, in the order of the heatmap rows. "curve" is the characteristic of the
# smoothed curves (in percentiles of the pairs for the sentiment, whose difference is bimodal).
ANALYSES = {
    "text": {"enrich": enrich_text, "row": "Length",
             "characteristics": characteristics("text_length", "text length (words)"),
             "curve": "diff_text_length", "xlabel": "Length difference (words)"},
    "embedding": {"enrich": enrich_embedding, "row": "Emb. sim.",
                  "characteristics": {"cosine_similarity": "Cosine similarity between the opinion embeddings"},
                  "curve": "cosine_similarity", "xlabel": "Cosine similarity"},
    "sentiment": {"enrich": enrich_sentiment, "row": "RoBERTa",
                  "characteristics": characteristics("sentiment_score_opinion", "sentiment score"),
                  "curve": "diff_sentiment_score_opinion", "percentile": True,
                  "xlabel": "Sentiment difference (percentile)"},
    **{f"vad_{dim}": {"enrich": enrich_vad, "row": dim.capitalize(),
                      "characteristics": characteristics(f"{dim}_opinion", f"VAD {dim}"),
                      "curve": f"diff_{dim}_opinion", "xlabel": f"{dim.capitalize()} difference"}
       for dim in ("valence", "arousal", "dominance")},
    "vader": {"enrich": enrich_vader, "row": "VADER",
              "characteristics": {**characteristics("vader_compound_opinion", "VADER compound score"),
                                  **characteristics("vader_neg_opinion", "VADER negative proportion"),
                                  **characteristics("vader_pos_opinion", "VADER positive proportion")},
              "curve": "diff_vader_compound_opinion", "xlabel": "VADER compound difference"},
    "safety": {"enrich": enrich_safety, "row": "Guard",
               "characteristics": characteristics("safety_unsafe_score_opinion", "Llama-Guard unsafe score"),
               "curve": "diff_safety_unsafe_score_opinion", "xlabel": "Guard unsafe-score difference"},
    "perplexity": {"enrich": enrich_perplexity, "row": "PPL",
                   "characteristics": characteristics("perplexity", "log-perplexity under the judge (z-score)"),
                   "curve": "diff_perplexity", "xlabel": "Log-PPL difference (z-score)"},
    "refusal": {"enrich": enrich_refusal, "row": "Refusal",
                "characteristics": characteristics("stance_refusal", "P(judge refuses the opinion's stance)"),
                "curve": "diff_stance_refusal", "xlabel": "Refusal difference"},
    "alignment": {"enrich": enrich_alignment, "row": "Alignment",
                  "characteristics": characteristics("stance_alignment", "P(judge holds the opinion's stance)"),
                  "curve": "diff_stance_alignment", "xlabel": "Alignment difference"},
}
HEATMAP_ROWS = {analysis: spec["row"] for analysis, spec in ANALYSES.items()}
PAIR_LEVEL_CHARACTERISTICS = {"cosine_similarity"}
AGGREGATORS = ["diff", "avg", "max", "min"]

# Curves in the main paper, as (dataset, analysis); the appendix has the other features
# with a significant correlation.
MAIN_PLOTS = [("valueprism", "text"), ("habermas_hybrid", "text"), ("valueprism", "embedding"),
              ("valueprism", "sentiment"), ("habermas_hybrid", "perplexity"), ("valueprism", "alignment")]
SCALE_PLOTS = [("valueprism", "alignment"), ("habermas", "sentiment"), ("habermas_hybrid", "perplexity")]
PANELS = [("rs", "RS"), ("pc", "PC")]
JUDGE_ORDER = ["gemma3", "llama-3.3-70b", "mistral-24b", *SCALE_MODELS, "openai/gpt-5.4-nano"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--analyses", nargs="+", default=list(ANALYSES), choices=list(ANALYSES))
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples of the situations.")
    parser.add_argument("--min_situations", type=int, default=5,
                        help="Tests and curve windows with fewer situations get no CI.")
    parser.add_argument("--smooth_window", type=float, default=0.2, help="Share of the pairs in each window.")
    parser.add_argument("--smooth_step", type=float, default=0.02, help="Shift between windows, as a share of the pairs.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR / "stability")
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--plots_only", action="store_true", help="Only redraw the figures from saved results.")
    parser.add_argument("--joint_model", action="store_true", help="Run the joint model of all the features.")
    parser.add_argument("--min_rho", type=float, default=0.1, help="Joint model: |rho_partial| marked as relevant.")
    return parser.parse_args()


# RS / PC of opinion pairs

STABILITY_COLUMNS = ["id_1", "id_2", "parsed_score_1to2", "parsed_score_2to1"]


def rs_pc(df):
    df = df[STABILITY_COLUMNS]
    return calculate_rs_from_df(df), calculate_pc_from_df(df)


def stability_terms(df):
    """Per pair, the terms that calculate_rs / calculate_pc average: for each direction the
    normalised sum of pairwise score differences over the prompt variations (RS), and the
    mean |score_1to2 - score_2to1| (PC)."""
    keys = ["id_1", "id_2"]
    groups = df.groupby(keys)
    terms = pd.DataFrame({"situation": groups["situation_id_1"].first(), "n": groups.size()})
    terms["rs_sum"] = 0.0
    for direction in ("parsed_score_1to2", "parsed_score_2to1"):
        # sum over i < j of |x_i - x_j| = sum_k (2k - n + 1) x_(k), over the sorted x
        ordered = df[keys + [direction]].sort_values(keys + [direction])
        rank = ordered.groupby(keys).cumcount()
        size = ordered.groupby(keys)[direction].transform("size")
        pairwise = ((2 * rank - size + 1) * ordered[direction]).groupby([ordered[k] for k in keys]).sum()
        terms["rs_sum"] += pairwise / terms["n"].map(lambda n: get_den_normalization(n, 1.0) or np.nan)
    terms["rs_count"] = np.where(terms["n"] >= 2, 2, 0)
    terms["rs_sum"] = terms["rs_sum"].where(terms["n"] >= 2, 0.0)
    terms["pc_sum"] = (df["parsed_score_1to2"] - df["parsed_score_2to1"]).abs().groupby([df[k] for k in keys]).mean()
    return terms


def pair_stability_table(df):
    terms = stability_terms(df)
    return pd.DataFrame({"situation": terms["situation"],
                         "rs": 1 - terms["rs_sum"] / terms["rs_count"].replace(0, np.nan),
                         "pc": 1 - terms["pc_sum"]})


def stability_significance(enriched, spec, analysis, judge, args):
    pairs = pair_stability_table(enriched)
    rows = []
    for name, label in spec["characteristics"].items():
        table = pairs.assign(x=enriched.groupby(["id_1", "id_2"])[name].first()).dropna(subset=["x", "rs", "pc"])
        if table.empty:
            continue
        results = significance_tests(table["x"], {"RS": table["rs"], "PC": table["pc"]}, table["situation"],
                                     args.n_boot, args.min_situations)
        for metric, result in results.items():
            result = {k: v for k, v in result.items() if not k.startswith("mean")}
            rows.append({"analysis": analysis, "characteristic": name, "label": label, "judge": judge,
                         "metric": metric, **result})
    return rows


def sliding_window_stability(df, x_col, args):
    """RS/PC over overlapping windows of the pairs sorted by x_col, with 95% bands."""
    df = df.sort_values(x_col, kind="stable").reset_index(drop=True)
    starts, w = sliding_window_starts(len(df), args.smooth_window, args.smooth_step)
    situations = pd.Index(df["situation_id_1"].unique())
    sums = {"rs_sum": [], "rs_count": [], "pc_sum": [], "pc_count": []}
    records = []
    for i, start in enumerate(starts):
        window = df.iloc[start:start + w]
        rs, pc = rs_pc(window)
        records.append({"bin": i, "x_mean": window[x_col].mean(), "x_min": window[x_col].min(),
                        "x_max": window[x_col].max(), "n_pairs": len(window),
                        "n_situations": window["situation_id_1"].nunique(), "rs": rs, "pc": pc})
        if args.n_boot:
            terms = stability_terms(window)
            codes = situations.get_indexer(terms["situation"])
            for name, values in [("rs_sum", terms["rs_sum"]), ("rs_count", terms["rs_count"]),
                                 ("pc_sum", terms["pc_sum"]), ("pc_count", np.ones(len(terms)))]:
                sums[name].append(np.bincount(codes, weights=np.asarray(values, dtype=float),
                                              minlength=len(situations)))
    summary = pd.DataFrame.from_records(records)
    if args.n_boot:
        bands = situation_bootstrap_bands({"rs": (np.array(sums["rs_sum"]), np.array(sums["rs_count"])),
                                           "pc": (np.array(sums["pc_sum"]), np.array(sums["pc_count"]))},
                                          args.n_boot)
        too_few = summary["n_situations"] < args.min_situations
        for name in ("rs", "pc"):
            # The bands are on the mean term, and RS/PC = 1 - mean term.
            summary[f"{name}_ci95_low"] = np.where(too_few, np.nan, 1 - bands[name][1])
            summary[f"{name}_ci95_high"] = np.where(too_few, np.nan, 1 - bands[name][0])
    return summary


def stability_curve(enriched, spec, args):
    x_col = spec["curve"]
    df = enriched.dropna(subset=[x_col])
    if df.empty:
        return pd.DataFrame(columns=["bin", "x_mean", "x_min", "x_max", "n_pairs", "n_situations", "rs", "pc"])
    if not spec.get("percentile"):
        return sliding_window_stability(df, x_col, args)
    df = df.assign(pct=df[x_col].rank(pct=True) * 100)
    curve = sliding_window_stability(df, "pct", args)
    return curve.assign(same_polarity_share=(df[x_col] < 0.5).mean() * 100)


# Overall RS / PC per judge

def overall_rows(scoring, dataset, judge):
    overall = dict(zip(("RS", "PC"), rs_pc(scoring)))
    pairs = pair_stability_table(scoring[STABILITY_COLUMNS + ["situation_id_1"]])
    rows = []
    for metric, col in (("RS", "rs"), ("PC", "pc")):
        values = pairs[col].dropna()
        if not np.isclose(values.mean(), overall[metric]):
            print(f"[warn] {dataset}/{judge} {metric}: mean of the pair values differs from the overall value.")
        rows.append({"dataset": dataset, "judge": judge, "metric": metric, "mean": overall[metric],
                     "std": values.std(ddof=1), "n_pairs": len(values)})
    return rows


def overall_latex(results, decimals=3):
    judges = [j for j in JUDGE_ORDER if j in set(results["judge"])]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             rf"\begin{{tabular}}{{ll{'c' * len(judges)}}}", r"\toprule",
             " & ".join(["Dataset", "Metric"] + [judge_name(j) for j in judges]) + r" \\", r"\midrule"]
    for i, dataset in enumerate(dict.fromkeys(results["dataset"])):
        if i:
            lines.append(r"\midrule")
        for k, metric in enumerate(("RS", "PC")):
            sub = results[(results["dataset"] == dataset) & (results["metric"] == metric)].set_index("judge")
            best = sub["mean"].round(decimals).max()
            cells = []
            for judge in judges:
                if judge not in sub.index:
                    cells.append("--")
                    continue
                mean, std = sub.loc[judge, "mean"], sub.loc[judge, "std"]
                mean_str = f"{mean:.{decimals}f}"
                if round(mean, decimals) == best:
                    mean_str = rf"\mathbf{{{mean_str}}}"
                cells.append(rf"${mean_str}_{{\pm {std:.{decimals}f}}}$")
            label = rf"\multirow{{2}}{{*}}{{{DATASET_NAMES[dataset]}}}" if k == 0 else ""
            lines.append(" & ".join([label, metric] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Overall stability (RS) and positional consistency (PC) of each judge. Subscripts: standard "
              r"deviation across opinion pairs of the pair-level RS/PC. Best judge per row in bold.}",
              r"\label{tab:overall_stability}", r"\end{table}"]
    return "\n".join(lines)


def compute(args, results_dir):
    overall = []
    for dataset in args.datasets:
        ctx = DatasetContext.from_args(args, dataset)
        judges = discover_judges(ctx)
        if not judges:
            print(f"[{dataset}] no scoring files - skipped.")
            continue
        print(f"[{dataset}] judges: {judges}")
        rows = {analysis: [] for analysis in args.analyses}
        for judge in judges:
            scoring = load_scoring_df(ctx, judge)
            if scoring is None:
                continue
            overall += overall_rows(scoring, dataset, judge)
            for analysis in args.analyses:
                spec = ANALYSES[analysis]
                enriched = spec["enrich"](scoring, ctx, judge)
                if enriched is None:
                    print(f"[{dataset}/{analysis}] no {analysis} feature for {judge} - skipped.")
                    continue
                rows[analysis] += stability_significance(enriched, spec, analysis, judge, args)
                path = results_dir / "curves" / dataset / analysis / f"{judge}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                stability_curve(enriched, spec, args).to_csv(path, index=False)
                print(f"[{dataset}/{analysis}] {judge} done.")
        for analysis in args.analyses:
            write_significance(rows[analysis], results_dir / "significance" / dataset, analysis, ["rho_p"])

    if overall:
        overall = pd.DataFrame(overall)
        stem = results_dir / f"overall_stability_{'_'.join(args.datasets)}"
        overall.to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".tex").write_text(overall_latex(overall))
        print(overall.to_string(index=False))


# Figures

def load_significance(results_dir, dataset, metric, aggregator):
    path = results_dir / "significance" / dataset / "all.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    keep = (d["analysis"].isin(HEATMAP_ROWS) & (d["metric"] == metric)
            & (d["characteristic"].str.startswith(f"{aggregator}_") | d["characteristic"].isin(PAIR_LEVEL_CHARACTERISTICS)))
    return select_strongest(d[keep])


def draw_curve(results_dir, dataset, analysis, styles, path):
    spec = ANALYSES[analysis]
    summaries = load_curves(results_dir / "curves" / dataset / analysis, styles, "rs_ci95_low")
    if not summaries:
        print(f"[skip] {dataset}/{analysis}: no curve with bootstrap bands.")
        return
    stem = spec["curve"] + ("_pretty" if spec.get("percentile") else "")
    fig = line_plot(summaries, styles, PANELS, spec["xlabel"], spec.get("percentile", False), share_y=True,
                    left=0.30, gap=0.12, zero_line=False)
    save(fig, path / f"{DATASET_SHORT[dataset]}_{stem}")


def draw_figures(args, results_dir, figures_dir):
    setup_matplotlib()
    datasets = [d for d in DATASETS if (results_dir / "significance" / d / "all.csv").exists()]
    if args.scale_analysis:
        panels = [(load_significance(results_dir, d, metric, "diff"),
                   figures_dir / "scale_analysis_appendix" / "stability"
                   / f"{DATASET_SHORT[d]}_heatmap_{metric}_diff_{HEATMAP_STEM}", layout)
                  for d in datasets for metric, layout in [("RS", "left"), ("PC", "right")]]
        if panels:
            heatmap_figure(panels, HEATMAP_ROWS, False, SCALE_JUDGES)
        for dataset, analysis in SCALE_PLOTS:
            draw_curve(results_dir, dataset, analysis, SCALE_STYLES, figures_dir / "scale_analysis_main" / "stability")
        save(legend_figure(SCALE_STYLES), figures_dir / "scale_analysis_main" / "legend")
        return

    main_dir, appendix_dir = figures_dir / "stability_analysis_main", figures_dir / "stability_analysis_appendix"
    for dataset in datasets:
        short = DATASET_SHORT[dataset]
        heatmap_figure([(load_significance(results_dir, dataset, metric, "diff"),
                         main_dir / f"{short}_heatmap_{metric}_diff_{HEATMAP_STEM}", layout)
                        for metric, layout in [("RS", "left"), ("PC", "right")]], HEATMAP_ROWS, True, JUDGES)
        heatmap_figure([(load_significance(results_dir, dataset, metric, aggregator),
                         appendix_dir / f"{short}_heatmap_{metric}_{aggregator}_{HEATMAP_STEM}", layout)
                        for aggregator in AGGREGATORS[1:] for metric, layout in [("RS", "left"), ("PC", "right")]],
                       HEATMAP_ROWS, False, JUDGES)

    for dataset, analysis in MAIN_PLOTS:
        if dataset in datasets:
            draw_curve(results_dir, dataset, analysis, MAIN_STYLES, main_dir)
    save(legend_figure(MAIN_STYLES), main_dir / "legend")

    for dataset in datasets:
        cells = [load_significance(results_dir, dataset, metric, "diff") for metric in ("RS", "PC")]
        significant = set().union(*(set(c[c["significant"]].index.get_level_values("analysis")) for c in cells))
        for analysis in ANALYSES:
            if analysis in significant and (dataset, analysis) not in MAIN_PLOTS:
                draw_curve(results_dir, dataset, analysis, MAIN_STYLES, appendix_dir)
    save(legend_figure(MAIN_STYLES), appendix_dir / "legend")


# Joint model: are the features still associated with RS/PC once the others are held fixed?

def stability_dimensions(pairs):
    dims = {name: (pairs[f"{col}_1"] - pairs[f"{col}_2"]).abs() for name, (col, _) in JOINT_FEATURES.items()
            if f"{col}_1" in pairs and pairs[f"{col}_1"].notna().any()}
    if "cosine_similarity" in pairs:
        dims["cosine_similarity"] = pairs["cosine_similarity"]
    dims = pd.DataFrame(dims)
    return dims.loc[:, dims.nunique() > 1]


def spearman_partial(ranks):
    """Column 0 is the outcome: marginal and partial correlations of the others with it."""
    corr = np.corrcoef(ranks, rowvar=False)
    precision = np.linalg.pinv(corr)
    partial = -precision[0, 1:] / np.sqrt(precision[0, 0] * np.diag(precision)[1:])
    return corr[0, 1:], partial


def mixed_model(data, metric, names):
    """metric ~ z(features) + (1 | situation)."""
    import statsmodels.formula.api as smf

    z = (data[names] - data[names].mean()) / data[names].std()
    frame = pd.concat([data[[metric]], z, data[["situation_id_1"]]], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(f"{metric} ~ " + " + ".join(names), frame, groups=frame["situation_id_1"]).fit(reml=True)
    ci = fit.conf_int().loc[names]
    return pd.DataFrame({"lmm_beta": fit.params[names], "lmm_ci95_low": ci[0], "lmm_ci95_high": ci[1],
                         "lmm_p": fit.pvalues[names],
                         "lmm_situation_var_share": fit.cov_re.iloc[0, 0] / (fit.cov_re.iloc[0, 0] + fit.scale)},
                        index=names)


def joint_stability(pairs, n_boot, rng):
    dims = stability_dimensions(pairs)
    names = list(dims.columns)
    tables = []
    for metric in ("rs", "pc"):
        data = pd.concat([pairs[[metric, "situation_id_1"]], dims], axis=1).dropna()
        values = data[[metric] + names].to_numpy(dtype=float)
        rows_by_situation = situation_rows(data["situation_id_1"].to_numpy())
        rho, partial = spearman_partial(rank_columns(values))
        boot_rho = np.full((n_boot, len(names)), np.nan)
        boot_partial = np.full((n_boot, len(names)), np.nan)
        for b in range(n_boot):
            idx = np.concatenate([rows_by_situation[k]
                                  for k in rng.integers(0, len(rows_by_situation), len(rows_by_situation))])
            boot_rho[b], boot_partial[b] = spearman_partial(rank_columns(values[idx]))
        table = ci_p(rho, boot_rho, names, "rho").join(ci_p(partial, boot_partial, names, "rho_partial"))
        table = table.join(mixed_model(data, metric, names))
        table["metric"] = metric.upper()
        table["n_pairs"] = len(data)
        table["n_situations"] = data["situation_id_1"].nunique()
        tables.append(table)
    return pd.concat(tables).rename_axis("feature").reset_index()


def run_joint_model(args, results_dir):
    for dataset in args.datasets:
        ctx = DatasetContext.from_args(args, dataset)
        rng = np.random.default_rng(0)
        stability, correlations = [], []
        for judge in discover_judges(ctx):
            scoring = load_scoring_df(ctx, judge)
            if scoring is None:
                continue
            pairs = joint_pairs(ctx, judge, scoring).join(pair_stability_table(scoring))
            print(f"[{dataset}] {judge}: {len(pairs)} pairs")
            stability.append(joint_stability(pairs, args.n_boot, rng).assign(judge=judge))
            correlations.append(stability_dimensions(pairs).corr(method="spearman")
                                .rename_axis("feature").reset_index().assign(judge=judge))
        if not stability:
            continue
        stability = pd.concat(stability, ignore_index=True)
        for col in ("rho_p", "rho_partial_p", "lmm_p"):
            stability[col[:-len("_p")] + "_q"] = bh_adjust(stability[col])
        stability["marked"] = (stability["rho_partial_q"] < 0.05) & (stability["rho_partial"].abs() >= args.min_rho)
        out_dir = results_dir / "joint" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        stability.to_csv(out_dir / "stability.csv", index=False)
        pd.concat(correlations, ignore_index=True).to_csv(out_dir / "feature_correlations.csv", index=False)
        print(f"Saved {out_dir}/{{stability,feature_correlations}}.csv")


def main():
    args = parse_args()
    run = run_label(args)
    results_dir = args.results_dir / run
    if args.joint_model:
        run_joint_model(args, results_dir)
        return
    if not args.plots_only:
        compute(args, results_dir)
    draw_figures(args, results_dir, args.figures_dir if run in ("all", "scale") else args.figures_dir / run)


if __name__ == "__main__":
    main()
