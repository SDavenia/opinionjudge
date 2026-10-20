"""
Distribution of the judges' alignment scores, one figure per dataset: each opinion pair
contributes its score averaged over the prompt variations and both orders, binned in steps
of 0.1, with one panel per judge.

With --run_human the judges are compared with the human annotators of the symmetric
annotation study, on the pairs they rated (each annotator's score of a pair is the mean of
their two ratings, one per order). A first panel shows the annotators (bars: mean share
over annotators, dots: each annotator), and their distribution is outlined on every judge
panel. It needs the human annotation data, downloaded from the Hugging Face Hub (see README).

Usage:
    python plot_score_distribution.py
    python plot_score_distribution.py --run_human
    python plot_score_distribution.py --scale_analysis
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from human_annotation.symmetric.compute_stability import ANNOTATION_DIR, load_ratings, wide_pairs
from utils.analysis_utils import SCALE_MODELS, DatasetContext, add_data_args, discover_judges, load_scoring_df, run_label
from utils.paths import FIGURES_DIR, RESULTS_DIR
from utils.plot_utils import judge_name

JUDGE_ORDER = ["gemma3", "llama-3.3-70b", "mistral-24b", *SCALE_MODELS, "openai/gpt-5.4-nano"]
# Fixed colour order, so that a judge keeps its colour across datasets and figures.
COLOUR_ORDER = ["gemma3", "llama-3.3-70b", "mistral-24b", "qwen2.5-72b", "openai/gpt-5.4-nano",
                "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-05b", "qwen2.5-1b", "qwen2.5-32b"]
JUDGE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
HUMAN_COLOUR = "#3d3d3a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOUR = "#e4e3df"
JUDGES_FONT_SCALE = 1.5

N_BINS = 10
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)
BIN_CENTRES = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
BIN_LABELS = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]


def set_style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser, datasets=["habermas", "valueprism"])
    parser.add_argument("--judges", nargs="+", default=None, help="Only these judges (default: every judge found).")
    parser.add_argument("--run_human", action="store_true", help="Compare the judges with the human annotators.")
    parser.add_argument("--annotation_dir", type=Path, default=ANNOTATION_DIR)
    parser.add_argument("--exclude", nargs="*", default=[], help="--run_human: annotator ids to leave out.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR / "score_distribution")
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR / "score_distribution")
    return parser.parse_args()


def ordered(values, preferred=JUDGE_ORDER):
    values = list(values)
    return [v for v in preferred if v in values] + sorted(v for v in values if v not in preferred)


def judge_colour(judge):
    order = ordered(set(COLOUR_ORDER) | {judge}, COLOUR_ORDER)
    return JUDGE_COLOURS[order.index(judge) % len(JUDGE_COLOURS)]


def pair_key(text_id_1, text_id_2):
    return tuple(sorted((text_id_1, text_id_2)))


def judge_pair_scores(args, dataset_id):
    """judge -> mean score of each pair over prompt variations and both orders."""
    ctx = DatasetContext.from_args(args, dataset_id)
    scores = {}
    for judge in ordered(discover_judges(ctx)):
        if args.judges and judge not in args.judges:
            continue
        df = load_scoring_df(ctx, judge)
        if df is None:
            continue
        df["pair"] = [pair_key(a, b) for a, b in zip(df["text_id_1"], df["text_id_2"])]
        long = df.melt(id_vars="pair", value_vars=["parsed_score_1to2", "parsed_score_2to1"])
        scores[judge] = long.groupby("pair")["value"].mean().dropna()
    return scores


def bin_shares(values):
    """Share of the values in each bin (the last one includes 1.0)."""
    idx = np.clip(np.floor(np.round(values.to_numpy(), 6) * N_BINS).astype(int), 0, N_BINS - 1)
    return np.bincount(idx, minlength=N_BINS) / len(idx)


def style_axis(ax, font_scale=1.0):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", ".2", ".4", ".6", ".8", "1"])
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.7, zorder=0)
    ax.tick_params(length=0, pad=3, labelsize=9 * font_scale)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.spines["bottom"].set_linewidth(0.8)


def titles(ax, name, mean, font_scale=1.0, stack_mean=False):
    mean_size = 8 * font_scale
    if not stack_mean:
        ax.set_title(name, loc="left", color=TEXT_PRIMARY, fontsize=10 * font_scale, pad=5)
        ax.set_title(f"mean {mean:.2f}", loc="right", color=TEXT_SECONDARY, fontsize=mean_size, pad=5)
        return
    ax.set_title(name, loc="left", color=TEXT_PRIMARY, fontsize=10 * font_scale, pad=5 + 1.3 * mean_size)
    ax.annotate(f"mean {mean:.2f}", (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                va="bottom", color=TEXT_SECONDARY, fontsize=mean_size)


def plot_panels(panels, out_stem, human_outline=None, legend=None, font_scale=1.0, stack_mean=False,
                xlabel="Mean alignment score of the pair (5 prompts x 2 directions)"):
    """One panel per scorer, sharing the y axis."""
    fig, axes = plt.subplots(1, len(panels), figsize=(1.95 * len(panels) + 0.6, 2.5), sharey=True, squeeze=False,
                             layout="constrained")
    axes = axes[0]
    for ax, panel in zip(axes, panels):
        ax.bar(BIN_CENTRES, panel["shares"] * 100, width=1 / N_BINS, color=panel["colour"], edgecolor="white",
               linewidth=1.2, zorder=2)
        if human_outline is not None and not panel.get("dots"):
            ax.stairs(human_outline * 100, BIN_EDGES, color=HUMAN_COLOUR, linewidth=1.4, zorder=3)
        for k, dots in enumerate(panel.get("dots") or []):
            offset = (k - (len(panel["dots"]) - 1) / 2) * 0.018
            ax.scatter(BIN_CENTRES + offset, dots * 100, s=14, color="white", edgecolor=HUMAN_COLOUR, linewidth=1.0,
                       zorder=4)
        titles(ax, panel["name"], panel["mean"], font_scale, stack_mean)
        style_axis(ax, font_scale)
    axes[0].set_ylabel("Share of pairs (%)", fontsize=9 * font_scale)
    fig.supxlabel(xlabel, fontsize=9 * font_scale, color=TEXT_SECONDARY)
    if legend:
        fig.legend(handles=legend, loc="outside upper right", ncol=len(legend), frameon=False, fontsize=8.5,
                   handlelength=1.4)
    fig.get_layout_engine().set(w_pad=0.08, wspace=0.04)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def shares_table(rows):
    """scorer -> (shares, n_pairs, mean score)."""
    table = pd.DataFrame({name: shares for name, (shares, _, _) in rows.items()}, index=BIN_LABELS).T
    table.insert(0, "mean_score", pd.Series({name: m for name, (_, _, m) in rows.items()}))
    table.insert(0, "n_pairs", pd.Series({name: n for name, (_, n, _) in rows.items()}))
    table.index.name = "scorer"
    return table


def judge_panels(scores):
    panels, rows = [], {}
    for judge, s in scores.items():
        shares = bin_shares(s)
        panels.append({"name": judge_name(judge), "shares": shares, "colour": judge_colour(judge), "mean": s.mean()})
        rows[judge_name(judge)] = (shares, len(s), s.mean())
    return panels, rows


def save_table(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    print(f"Saved {path}")


def run_judges(args, dataset_id):
    scores = judge_pair_scores(args, dataset_id)
    if not scores:
        print(f"[warn] No scoring files for {dataset_id}.")
        return
    panels, rows = judge_panels(scores)
    plot_panels(panels, args.figures_dir / run_label(args) / dataset_id, font_scale=JUDGES_FONT_SCALE, stack_mean=True)
    save_table(shares_table(rows), args.results_dir / run_label(args) / f"{dataset_id}.csv")


def run_human(args, dataset_id, human_wide):
    human = human_wide[human_wide["dataset"] == dataset_id].copy()
    if human.empty:
        print(f"[warn] No human ratings for {dataset_id}.")
        return
    human["pair"] = [pair_key(a, b) for a, b in zip(human["text_id_1"], human["text_id_2"])]
    human["score"] = human[["r_1to2", "r_2to1"]].mean(axis=1)
    rated = set(human["pair"])
    scores = {}
    for judge, s in judge_pair_scores(args, dataset_id).items():
        s = s[s.index.isin(rated)]
        if len(s) < len(rated):
            print(f"[warn] {dataset_id}/{judge}: scored {len(s)} of the {len(rated)} human-rated pairs.")
        if len(s):
            scores[judge] = s

    rows, annotator_shares = {}, []
    for annotator, ann in human.groupby("annotator_id"):
        shares = bin_shares(ann["score"])
        annotator_shares.append(shares)
        rows[f"Human {annotator}"] = (shares, len(ann), ann["score"].mean())
    human_shares = np.mean(annotator_shares, axis=0)
    human_mean = human.groupby("annotator_id")["score"].mean().mean()
    rows["Humans (mean)"] = (human_shares, len(rated), human_mean)
    panels, judge_rows = judge_panels(scores)
    rows.update(judge_rows)
    panels = [{"name": f"Humans ({len(annotator_shares)})", "shares": human_shares, "colour": "#b8b7b1",
               "mean": human_mean, "dots": annotator_shares}] + panels
    legend = [
        Patch(facecolor="#b8b7b1", label="Humans: mean of annotators"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=HUMAN_COLOUR, markersize=5,
               label="one annotator"),
        Line2D([], [], color=HUMAN_COLOUR, linewidth=1.4, label="humans (on judge panels)"),
    ]
    plot_panels(panels, args.figures_dir / run_label(args) / "human" / dataset_id, human_outline=human_shares,
                legend=legend,
                xlabel="Mean alignment score of the pair (judges: 5 prompts x 2 directions; annotators: 2 directions)")
    save_table(shares_table(rows), args.results_dir / run_label(args) / "human" / f"{dataset_id}.csv")


def main():
    args = parse_args()
    set_style()
    if not args.run_human:
        for dataset_id in args.datasets:
            run_judges(args, dataset_id)
        return
    pairs_csv = args.annotation_dir / "samples" / "pairs.csv"
    if not pairs_csv.exists():
        raise SystemExit(f"--run_human needs {pairs_csv}: download the human annotation data (see README).")
    pairs = pd.read_csv(pairs_csv)
    human_wide = wide_pairs(load_ratings(args.annotation_dir, args.exclude), pairs)
    print(f"{human_wide['annotator_id'].nunique()} annotators, {human_wide['pair_id'].nunique()} pairs.")
    for dataset_id in args.datasets:
        run_human(args, dataset_id, human_wide)


if __name__ == "__main__":
    main()
