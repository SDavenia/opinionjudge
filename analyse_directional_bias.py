"""
Directional bias of the judges: when the two opinions of a pair differ in a feature, is
the opinion on one side of it (the pivot, e.g. the longer one) consistently scored higher
or lower when it comes first?

Each pair is oriented so that the pivot is text 1, and

    score_diff = score(pivot -> other) - score(other -> pivot).

Pairs where both opinions have the same feature value are left out. For every dataset,
judge and feature, it tests the Spearman correlation between the feature gap and
score_diff (averaged over prompt variations) and the mean score_diff, with 95% CIs and
p-values from situation-level bootstrap resamples and Benjamini-Hochberg q-values within
each feature. It also computes the smoothed mean score_diff and directional consistency
mean(sign(score_diff)) along the gap. The paper figures and the appendix tables are then
drawn from these results.

With --joint_model it instead regresses score_diff on all the feature gaps at once, as a
robustness check.

Usage:
    python analyse_directional_bias.py
    python analyse_directional_bias.py --scale_analysis
    python analyse_directional_bias.py --datasets habermas --analyses text perplexity
    python analyse_directional_bias.py --plots_only
    python analyse_directional_bias.py --joint_model
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.analysis_utils import (
    DATASET_NAMES, DATASET_SHORT, JOINT_FEATURES, DatasetContext, add_data_args, bh_adjust, ci_p, discover_judges,
    enrich_alignment, enrich_perplexity, enrich_refusal, enrich_safety, enrich_sentiment, enrich_text, enrich_vad,
    enrich_vader, joint_pairs, load_scoring_df, run_label, significance_tests, situation_bootstrap_bands,
    situation_rows, sliding_window_starts, write_significance,
)
from utils.data_utils import DATASETS
from utils.paths import FIGURES_DIR, RESULTS_DIR
from utils.plot_utils import (
    ALPHA, HEATMAP_STEM, JUDGES, MAIN_STYLES, MIN_ABS_RHO, SCALE_JUDGES, SCALE_STYLES, heatmap_figure, judge_name,
    legend_figure, line_plot, load_curves, save, select_strongest, setup_matplotlib,
)

# Feature analyses, in the order of the heatmap rows. "column" is the per-opinion feature,
# "pivot" whether the pivot is the opinion with the higher or the lower value.
ANALYSES = {
    "text": {"enrich": enrich_text, "column": "text_length", "pivot": "higher", "feature": "text_length",
             "label": "Text length gap (words)", "pivot_label": "longer", "row": "Length",
             "name": "Text length", "table_pivot": "Longer text", "stem": "text_length",
             "xlabel": "Length gap (words)"},
    "sentiment": {"enrich": enrich_sentiment, "column": "sentiment_score_opinion", "pivot": "higher",
                  "feature": "sentiment_score", "label": "Sentiment score gap", "pivot_label": "more positive",
                  "row": "RoBERTa", "name": "Sentiment-Roberta", "table_pivot": "More positive",
                  "stem": "sentiment_score_pretty", "xlabel": "Sentiment gap (percentile)", "percentile": True},
    **{f"vad_{dim}": {"enrich": enrich_vad, "column": f"{dim}_opinion", "pivot": "higher", "feature": dim,
                      "label": f"{dim.capitalize()} gap", "pivot_label": f"higher-{dim}", "row": dim.capitalize(),
                      "name": f"VAD-{dim.capitalize()}", "table_pivot": f"Higher {dim}", "stem": dim,
                      "xlabel": f"{dim.capitalize()} gap"}
       for dim in ("valence", "arousal", "dominance")},
    "vader": {"enrich": enrich_vader, "column": "vader_compound_opinion", "pivot": "higher", "feature": "compound",
              "label": "VADER compound score gap", "pivot_label": "higher-compound", "row": "VADER",
              "name": "Sentiment-VADER", "table_pivot": "Higher compound", "stem": "vader_compound",
              "xlabel": "VADER compound gap"},
    "safety": {"enrich": enrich_safety, "column": "safety_unsafe_score_opinion", "pivot": "lower",
               "feature": "unsafe_score_opinion", "label": "Llama-Guard unsafe score gap", "pivot_label": "safer",
               "row": "Guard", "name": "Guard-Safety", "table_pivot": "Safer", "stem": "unsafe_score_opinion",
               "xlabel": "Guard unsafe-score gap"},
    "perplexity": {"enrich": enrich_perplexity, "column": "perplexity", "pivot": "lower", "feature": "perplexity",
                   "label": "Log-perplexity gap (z-score)", "pivot_label": "lower-perplexity", "row": "PPL",
                   "name": "Perplexity", "table_pivot": "Lower PPL", "stem": "perplexity",
                   "xlabel": "Log-PPL gap (z-score)"},
    # Each opinion gets the judge's probability of refusing to write its stance / of holding
    # its stance on the situation.
    "refusal": {"enrich": enrich_refusal, "column": "stance_refusal", "pivot": "lower", "feature": "stance_refusal",
                "label": "P(refuse other stance) - P(refuse pivot stance)",
                "pivot_label": "stance the judge refuses less", "row": "Refusal", "name": "Refusal (SIT)",
                "table_pivot": "Less-refused stance", "stem": "refusal", "xlabel": "Refusal gap"},
    "alignment": {"enrich": enrich_alignment, "column": "stance_alignment", "pivot": "higher",
                  "feature": "stance_alignment", "label": "P(majority stance) - P(minority stance)",
                  "pivot_label": "judge's majority stance", "row": "Alignment", "name": "Alignment (SIT)",
                  "table_pivot": "Majority stance", "stem": "alignment", "xlabel": "Alignment gap"},
}
HEATMAP_ROWS = {analysis: spec["row"] for analysis, spec in ANALYSES.items()}

MAIN_PLOTS = [("valueprism", "text"), ("habermas", "text"), ("valueprism", "vad_dominance"),
              ("habermas", "vad_valence"), ("habermas_hybrid", "perplexity"), ("valueprism", "alignment")]
SCALE_PLOTS = [("valueprism", "alignment"), ("habermas", "sentiment"), ("habermas_hybrid", "perplexity")]
PANELS = [("mean_diff", "Mean score difference"), ("consistency", "Directional consistency")]
# Heatmap panels of the figure: VP and HB side by side, HBY below on the left.
LAYOUTS = {"valueprism": "left", "habermas": "right", "habermas_hybrid": "left"}
SUMMARY_COLUMNS = ["bin", "x_mean", "x_min", "x_max", "n_pairs", "n_situations",
                   "mean_diff", "sem_diff", "consistency", "p_pivot_higher", "p_pivot_lower"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--analyses", nargs="+", default=list(ANALYSES), choices=list(ANALYSES))
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples of the situations.")
    parser.add_argument("--min_situations", type=int, default=5,
                        help="Tests and curve windows with fewer situations get no CI.")
    parser.add_argument("--smooth_window", type=float, default=0.2, help="Share of the pairs in each window.")
    parser.add_argument("--smooth_step", type=float, default=0.02, help="Shift between windows, as a share of the pairs.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR / "directional_bias")
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--plots_only", action="store_true", help="Only redraw the figures and tables from saved results.")
    parser.add_argument("--joint_model", action="store_true", help="Run the joint model of all the features.")
    return parser.parse_args()


def orient_by_pivot(df, col_1, col_2, pivot):
    """One row per (pair, prompt variation) with distinct feature values, with the gap
    between the two opinions and score_diff = score(pivot -> other) - score(other -> pivot)."""
    d = df[[col_1, col_2, "id_1", "id_2", "situation_id_1", "parsed_score_1to2", "parsed_score_2to1"]].dropna()
    d = d[d[col_1] != d[col_2]]
    pivot_is_1 = d[col_1] > d[col_2] if pivot == "higher" else d[col_1] < d[col_2]
    score_pivot_to_other = np.where(pivot_is_1, d["parsed_score_1to2"], d["parsed_score_2to1"])
    score_other_to_pivot = np.where(pivot_is_1, d["parsed_score_2to1"], d["parsed_score_1to2"])
    return pd.DataFrame({
        "gap": (d[col_1] - d[col_2]).abs(),
        "score_diff": score_pivot_to_other - score_other_to_pivot,
        "situation_id": d["situation_id_1"],
        "id_1": d["id_1"],
        "id_2": d["id_2"],
    }).reset_index(drop=True)


def oriented_pairs(enriched, spec):
    return orient_by_pivot(enriched, f"{spec['column']}_1", f"{spec['column']}_2", spec["pivot"])


def directional_significance(enriched, spec, n_boot, min_situations):
    oriented = oriented_pairs(enriched, spec)
    pairs = oriented.groupby(["id_1", "id_2"]).agg(gap=("gap", "first"), score_diff=("score_diff", "mean"),
                                                   situation=("situation_id", "first"))
    return significance_tests(pairs["gap"], {"score_diff": pairs["score_diff"]}, pairs["situation"],
                              n_boot, min_situations)["score_diff"]


def _summarise(window):
    diff = window["score_diff"]
    n = len(diff)
    return {"x_mean": window["gap"].mean(), "x_min": window["gap"].min(), "x_max": window["gap"].max(),
            "n_pairs": n, "n_situations": window["situation_id"].nunique(), "mean_diff": diff.mean(),
            "sem_diff": diff.std() / np.sqrt(n) if n > 1 else np.nan, "consistency": np.sign(diff).mean(),
            "p_pivot_higher": (diff > 0).mean(), "p_pivot_lower": (diff < 0).mean()}


def sliding_window_directional(df, args):
    """Mean score_diff and directional consistency over overlapping windows of the pairs
    sorted by gap, with 95% bands."""
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = df.sort_values("gap", kind="stable").reset_index(drop=True)
    starts, w = sliding_window_starts(len(df), args.smooth_window, args.smooth_step)
    summary = pd.DataFrame.from_records([{"bin": i, **_summarise(df.iloc[s:s + w])} for i, s in enumerate(starts)],
                                        columns=SUMMARY_COLUMNS)
    if args.n_boot:
        situations, codes = np.unique(df["situation_id"].to_numpy(), return_inverse=True)
        per_pair = {"mean_diff": df["score_diff"].to_numpy(), "consistency": np.sign(df["score_diff"]).to_numpy()}

        def window_sums(values):
            return np.array([np.bincount(codes[s:s + w], weights=values[s:s + w], minlength=len(situations))
                             for s in starts])

        counts = window_sums(np.ones(len(df)))
        bands = situation_bootstrap_bands({name: (window_sums(v), counts) for name, v in per_pair.items()},
                                          args.n_boot)
        too_few = summary["n_situations"] < args.min_situations
        for name in ("mean_diff", "consistency"):
            summary[f"{name}_ci95_low"] = np.where(too_few, np.nan, bands[name][0])
            summary[f"{name}_ci95_high"] = np.where(too_few, np.nan, bands[name][1])
    return summary


def directional_curve(enriched, spec, args):
    oriented = oriented_pairs(enriched, spec)
    if not spec.get("percentile"):
        return sliding_window_directional(oriented, args)
    same_polarity = (oriented["gap"] < 0.5).mean() * 100
    oriented["gap"] = oriented["gap"].rank(pct=True) * 100
    return sliding_window_directional(oriented, args).assign(same_polarity_share=same_polarity)


def compute(args, results_dir):
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
            for analysis in args.analyses:
                spec = ANALYSES[analysis]
                enriched = spec["enrich"](scoring, ctx, judge)
                if enriched is None:
                    print(f"[{dataset}/{analysis}] no {analysis} feature for {judge} - skipped.")
                    continue
                rows[analysis].append({"analysis": analysis, "feature": spec["feature"], "label": spec["label"],
                                       "pivot": spec["pivot_label"], "judge": judge,
                                       **directional_significance(enriched, spec, args.n_boot, args.min_situations)})
                path = results_dir / "curves" / dataset / analysis / f"{judge}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                directional_curve(enriched, spec, args).to_csv(path, index=False)
                print(f"[{dataset}/{analysis}] {judge} done.")
        for analysis in args.analyses:
            write_significance(rows[analysis], results_dir / "significance" / dataset, analysis, ["rho_p", "mean_p"])


# Figures and tables

def load_significance(results_dir, dataset):
    path = results_dir / "significance" / dataset / "all.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    return select_strongest(d[d["analysis"].isin(HEATMAP_ROWS)])


def draw_curve(results_dir, dataset, analysis, styles, path):
    spec = ANALYSES[analysis]
    summaries = load_curves(results_dir / "curves" / dataset / analysis, styles, "mean_diff_ci95_low")
    if not summaries:
        print(f"[skip] {dataset}/{analysis}: no curve with bootstrap bands.")
        return
    fig = line_plot(summaries, styles, PANELS, spec["xlabel"], spec.get("percentile", False), share_y=False,
                    left=0.33, gap=0.31, zero_line=True)
    save(fig, path / f"{DATASET_SHORT[dataset]}_{spec['stem']}")


def _tex_rho(row):
    star = "^{*}" if row["significant"] else ""
    return f"${row['rho']:+.2f}{star}$ [${row['rho_ci95_low']:+.2f}$, ${row['rho_ci95_high']:+.2f}$]"


def _tex_q(q):
    return "$<$0.001" if q < 0.001 else f"{q:.3f}"


def significance_table(dataset, cells, judges):
    """LaTeX table of the values of the main directional-bias heatmap of a dataset."""
    lines = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lllcc}", r"\toprule",
             r"Feature & Judge & Pivot & $\rho$ [95\% CI] & $q$ \\", r"\midrule"]
    blocks = []
    for analysis, spec in ANALYSES.items():
        rows = []
        for judge in judges:
            if (analysis, judge) not in cells.index:
                continue
            row = cells.loc[(analysis, judge)]
            rows.append([spec["name"] if not rows else "", judge_name(judge), spec["table_pivot"], _tex_rho(row),
                         _tex_q(row["rho_q"])])
        if rows:
            blocks.append(rows)
    for k, rows in enumerate(blocks):
        if k:
            lines.append(r"\midrule")
        lines += [" & ".join(cells_) + r" \\" for cells_ in rows]
    caption = (
        f"{DATASET_NAMES[dataset]}: Spearman correlation $\\rho$ between the gap in each feature between the two "
        r"instances and $\Delta_{sij}$ (score with the pivot first minus score with the pivot second), i.e.\ the "
        r"values of Figure~\ref{fig:directionality_analysis_heatmaps}. 95\% CI from situation-level bootstrap "
        r"resamples; $q$: Benjamini--Hochberg within each feature (all its judges); "
        rf"$^{{*}}$: $q < {ALPHA}$ and $|\rho| \geq {MIN_ABS_RHO:g}$.")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}",
              rf"\label{{tab:positionality_significance_{dataset}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def draw_figures(args, results_dir, figures_dir):
    setup_matplotlib()
    datasets = [d for d in DATASETS if (results_dir / "significance" / d / "all.csv").exists()]
    if args.scale_analysis:
        panels = [(load_significance(results_dir, d),
                   figures_dir / "scale_analysis_appendix" / "positional_bias" / f"{DATASET_SHORT[d]}_heatmap_{HEATMAP_STEM}",
                   LAYOUTS[d]) for d in datasets]
        if panels:
            heatmap_figure(panels, HEATMAP_ROWS, False, SCALE_JUDGES)
        for dataset, analysis in SCALE_PLOTS:
            draw_curve(results_dir, dataset, analysis, SCALE_STYLES,
                       figures_dir / "scale_analysis_main" / "positional_bias")
        save(legend_figure(SCALE_STYLES), figures_dir / "scale_analysis_main" / "legend")
        return

    main_dir = figures_dir / "directional_bias_analysis_main"
    appendix_dir = figures_dir / "directional_bias_analysis_appendix"
    cells = {d: load_significance(results_dir, d) for d in datasets}
    if cells:
        heatmap_figure([(cells[d], main_dir / f"{DATASET_SHORT[d]}_heatmap_{HEATMAP_STEM}", LAYOUTS[d])
                        for d in datasets], HEATMAP_ROWS, True, JUDGES)
    for dataset, analysis in MAIN_PLOTS:
        if dataset in datasets:
            draw_curve(results_dir, dataset, analysis, MAIN_STYLES, main_dir)
    save(legend_figure(MAIN_STYLES), main_dir / "legend")
    for dataset in datasets:
        significant = set(cells[dataset][cells[dataset]["significant"]].index.get_level_values("analysis"))
        for analysis in ANALYSES:
            if analysis in significant and (dataset, analysis) not in MAIN_PLOTS:
                draw_curve(results_dir, dataset, analysis, MAIN_STYLES, appendix_dir)
    save(legend_figure(MAIN_STYLES), appendix_dir / "legend")

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        path = tables_dir / f"positionality_significance_{dataset}.tex"
        path.write_text(significance_table(dataset, cells[dataset], JUDGES))
        print(f"Saved {path}")


# Joint model: score_diff = sum_k beta_k * g_k, with g_k the pivot-oriented gap in feature k
# scaled by its RMS. Swapping the two opinions flips the sign of score_diff and of every
# g_k, so the model has no intercept; beta_k < 0 means the pivot is scored lower when it
# comes first.

def directional_gaps(pairs):
    gaps = pd.DataFrame({name: sign * (pairs[f"{col}_1"] - pairs[f"{col}_2"])
                         for name, (col, sign) in JOINT_FEATURES.items()
                         if f"{col}_1" in pairs and pairs[f"{col}_1"].notna().any()})
    return gaps.loc[:, (gaps != 0).any()]


def joint_directional(pairs, n_boot, rng):
    gaps = directional_gaps(pairs)
    data = pd.concat([pairs[["delta", "situation_id_1"]], gaps], axis=1).dropna()
    names = list(gaps.columns)
    x = data[names].to_numpy(dtype=float)
    x = x / np.sqrt((x ** 2).mean(axis=0))
    y = data["delta"].to_numpy(dtype=float)
    rows_by_situation = situation_rows(data["situation_id_1"].to_numpy())

    def fit(idx):
        xb, yb = x[idx], y[idx]
        marginal = np.array([xb[:, j] @ yb / (xb[:, j] @ xb[:, j]) for j in range(len(names))])
        return marginal, np.linalg.lstsq(xb, yb, rcond=None)[0]

    marginal, joint = fit(np.arange(len(y)))
    boot_marginal = np.full((n_boot, len(names)), np.nan)
    boot_joint = np.full((n_boot, len(names)), np.nan)
    for b in range(n_boot):
        idx = np.concatenate([rows_by_situation[k]
                              for k in rng.integers(0, len(rows_by_situation), len(rows_by_situation))])
        boot_marginal[b], boot_joint[b] = fit(idx)
    table = ci_p(joint, boot_joint, names, "beta_joint").join(ci_p(marginal, boot_marginal, names, "beta_marginal"))
    table["n_pairs"] = len(y)
    table["n_situations"] = data["situation_id_1"].nunique()
    return table.rename_axis("feature").reset_index()


def run_joint_model(args, results_dir):
    for dataset in args.datasets:
        ctx = DatasetContext.from_args(args, dataset)
        rng = np.random.default_rng(0)
        tables = []
        for judge in discover_judges(ctx):
            scoring = load_scoring_df(ctx, judge)
            if scoring is None:
                continue
            delta = (scoring["parsed_score_1to2"] - scoring["parsed_score_2to1"]).groupby(
                [scoring["id_1"], scoring["id_2"]]).mean().rename("delta")
            pairs = joint_pairs(ctx, judge, scoring).join(delta)
            print(f"[{dataset}] {judge}: {len(pairs)} pairs")
            tables.append(joint_directional(pairs, args.n_boot, rng).assign(judge=judge))
        if not tables:
            continue
        table = pd.concat(tables, ignore_index=True)
        for col in ("beta_joint_p", "beta_marginal_p"):
            table[col[:-len("_p")] + "_q"] = bh_adjust(table[col])
        out_dir = results_dir / "joint" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "directional.csv", index=False)
        print(f"Saved {out_dir / 'directional.csv'}")


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
