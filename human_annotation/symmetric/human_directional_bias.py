"""
Directional bias of the human annotators, tested as for the judges in
analyse_directional_bias.py, on the human-rated pairs: the ratings are laid out like a
judge's scoring file, with one row per annotator in place of one per prompt variation.
Only the features of the opinions themselves are tested (perplexity, alignment and refusal
are measured under each judge). Every main judge is tested on the same pairs, so that
humans and judges can be compared on effect size. mean_abs is the mean |score_diff| over
pairs. q-values: Benjamini-Hochberg over all features and datasets, for each scorer.

Usage:
    python -m human_annotation.symmetric.human_directional_bias
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analyse_directional_bias import ANALYSES, directional_significance, oriented_pairs
from human_annotation.symmetric.compute_stability import ANNOTATION_DIR, OUTPUT_DIR, load_ratings, wide_pairs
from plot_score_distribution import GRID_COLOUR, HUMAN_COLOUR, TEXT_PRIMARY, TEXT_SECONDARY, judge_colour, set_style
from utils.analysis_utils import DATASET_NAMES, DatasetContext, bh_adjust, load_scoring_df
from utils.paths import FEATURES_DIR, FIGURES_DIR, SCORING_DIR
from utils.plot_utils import judge_name

DATASETS = ["valueprism", "habermas"]
MAIN_JUDGES = ["gemma3", "llama-3.3-70b", "mistral-24b", "qwen2.5-72b", "openai/gpt-5.4-nano"]
HUMAN_ANALYSES = ["text", "sentiment", "vader", "vad_valence", "vad_arousal", "vad_dominance", "safety"]
# Feature -> (name, pivot), in the order of the plot rows.
FEATURE_NAMES = {
    "text_length": ("Text length", "longer"),
    "sentiment_score": ("Sentiment-Roberta", "more positive"),
    "valence": ("VAD-Valence", "higher valence"),
    "arousal": ("VAD-Arousal", "higher arousal"),
    "dominance": ("VAD-Dominance", "higher dominance"),
    "compound": ("Sentiment-VADER", "higher compound"),
    "unsafe_score_opinion": ("Guard-Safety", "safer"),
}
JUDGE_MARKERS = ["o", "s", "^", "v", "P"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation_dir", type=Path, default=ANNOTATION_DIR)
    parser.add_argument("--exclude", nargs="*", default=[], help="Annotator ids to leave out.")
    parser.add_argument("--judges", nargs="*", default=MAIN_JUDGES)
    parser.add_argument("--scoring_dir", type=Path, default=SCORING_DIR)
    parser.add_argument("--features_dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--results_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR / "human_annotation")
    return parser.parse_args()


def human_scoring_df(w, pairs, dataset_id):
    """One row per (annotator, pair), with the columns of a judge's scoring file."""
    p = pairs[pairs["dataset"] == dataset_id].assign(situation_id_1=lambda d: d["situation_id"],
                                                     situation_id_2=lambda d: d["situation_id"])
    h = w[w["dataset"] == dataset_id][["pair_id", "annotator_id", "r_1to2", "r_2to1"]]
    return h.merge(p, on="pair_id").rename(columns={"r_1to2": "parsed_score_1to2", "r_2to1": "parsed_score_2to1"})


def judge_scoring_df(ctx, judge, human_df):
    """The judge's scores of the human-rated pairs (in whichever order it scored them)."""
    df = load_scoring_df(ctx, judge)
    if df is None:
        return None
    keys = {frozenset(k) for k in zip(human_df["text_id_1"], human_df["text_id_2"])}
    return df[[frozenset(k) in keys for k in zip(df["text_id_1"], df["text_id_2"])]].reset_index(drop=True)


def mean_abs_diff(enriched, spec):
    return oriented_pairs(enriched, spec).groupby(["id_1", "id_2"])["score_diff"].mean().abs().mean()


def plot_pivot_effects(table, judges, out_stem):
    """Mean score_diff per feature, every scorer on the same pairs, one panel per dataset."""
    humans = next(s for s in table["scorer"].unique() if s.startswith("Humans"))
    styles = {humans: dict(marker="D", color=HUMAN_COLOUR, s=46, zorder=4)}
    for judge, marker in zip(judges, JUDGE_MARKERS):
        styles[judge_name(judge)] = dict(marker=marker, color=judge_colour(judge), s=38, zorder=3)
    features = list(FEATURE_NAMES)
    rows = np.arange(len(features))[::-1]
    offsets = np.linspace(0.3, -0.3, len(styles))
    lim = 0.04 * np.ceil(table["mean"].abs().max() / 0.04 + 0.1)
    ticks = np.arange(-lim, lim + 1e-9, 0.04)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.2, 3.9), sharey=True, layout="constrained")
    for ax, dataset_id in zip(axes, DATASETS):
        sub = table[table["dataset"] == dataset_id]
        for offset, (scorer, style) in zip(offsets, styles.items()):
            values = sub[sub["scorer"] == scorer].set_index("feature").loc[features, "mean"]
            ax.scatter(values, rows + offset, edgecolor="white", linewidth=0.7, **style)
        ax.axvline(0, color=TEXT_SECONDARY, linestyle="--", linewidth=0.9, zorder=2)
        for y in rows[:-1] - 0.5:
            ax.axhline(y, color=GRID_COLOUR, linewidth=0.7, zorder=0)
        ax.set_xlim(-lim - 0.005, lim + 0.005)
        ax.set_xticks(ticks, [f"{t:+.2f}".replace("+0.00", "0").replace("-", "\N{MINUS SIGN}") for t in ticks])
        ax.set_ylim(-0.5, len(features) - 0.5)
        ax.set_title(DATASET_NAMES[dataset_id], loc="left", color=TEXT_PRIMARY, fontsize=10)
        ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.7, zorder=0)
        ax.tick_params(length=0, pad=4, labelsize=8.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_SECONDARY)
        ax.spines["bottom"].set_linewidth(0.8)
    axes[0].set_yticks(rows, [f"{name}\n(pivot: {pivot})" for name, pivot in FEATURE_NAMES.values()])
    fig.supxlabel(r"$\overline{\Delta}$ = rating with the pivot first $-$ rating with the pivot second",
                  fontsize=9, color=TEXT_SECONDARY)
    legend = [Line2D([], [], linestyle="", markeredgecolor="white", markerfacecolor=st["color"], marker=st["marker"],
                     markersize=7 if name == humans else 6, label=name) for name, st in styles.items()]
    fig.legend(handles=legend, loc="outside upper center", ncol=len(legend), frameon=False, fontsize=8.5,
               handletextpad=0.3, columnspacing=1.2)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_style()
    pairs = pd.read_csv(args.annotation_dir / "samples" / "pairs.csv")
    w = wide_pairs(load_ratings(args.annotation_dir, args.exclude), pairs)
    rows = []
    for dataset_id in DATASETS:
        ctx = DatasetContext(dataset_id, scoring_dir=args.scoring_dir, features_dir=args.features_dir)
        human_df = human_scoring_df(w, pairs, dataset_id)
        scorers = {f"Humans ({w['annotator_id'].nunique()})": human_df}
        for judge in args.judges:
            df = judge_scoring_df(ctx, judge, human_df)
            if df is not None and len(df):
                scorers[judge_name(judge)] = df
        for analysis in HUMAN_ANALYSES:
            spec = ANALYSES[analysis]
            for scorer, df in scorers.items():
                enriched = spec["enrich"](df, ctx, None)
                rows.append({"dataset": dataset_id, "scorer": scorer, "analysis": analysis, "feature": spec["feature"],
                             "pivot": spec["pivot_label"], "mean_abs": mean_abs_diff(enriched, spec),
                             **directional_significance(enriched, spec, args.n_boot, 5)})
    table = pd.DataFrame(rows)
    for p_col in ("rho_p", "mean_p"):
        table[p_col.replace("_p", "_q")] = table.groupby("scorer")[p_col].transform(lambda p: bh_adjust(p.to_numpy()))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results_dir / "human_directional_bias.csv", index=False)
    plot_pivot_effects(table, args.judges, args.figures_dir / "human_directional_bias")

    fmt = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_ci95_low']:+.2f}, {r[f'{s}_ci95_high']:+.2f}]"  # noqa: E731
    table["rho_cell"] = [fmt(r, "rho") + ("*" if r["rho_q"] < 0.05 else "") for _, r in table.iterrows()]
    table["mean_cell"] = [fmt(r, "mean") + ("*" if r["mean_q"] < 0.05 else "") for _, r in table.iterrows()]
    pd.set_option("display.width", 250)
    order = list(dict.fromkeys(table["scorer"]))
    for stat in ("rho_cell", "mean_cell"):
        print(f"\n-- {stat[:-5]} [95% CI] (*: q < 0.05) --")
        print(table.pivot_table(index=["dataset", "feature", "pivot"], columns="scorer", values=stat, aggfunc="first")
              .reindex(columns=order).to_string())
    print(f"\nSaved {args.results_dir / 'human_directional_bias.csv'} and "
          f"{args.figures_dir / 'human_directional_bias'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
