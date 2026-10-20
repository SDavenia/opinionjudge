"""
Example figure of the alignment task on Habermas: for each pair (A, B) of human opinions on
the same statement, the statement, the two opinions, and the judge's score in both orders
(A first: how well does B align with A, and B first).

The default pairs contrast a stable judgment (a meat tax, scored the same in both orders by
all five main judges) with one that flips (recreational drugs, 0.4-0.9 between the orders).

Usage:
    python plot_alignment_examples.py
    python plot_alignment_examples.py --judge_model_id openai/gpt-5.4-nano --prompt_variation mean
    python plot_alignment_examples.py --pairs 1067:1069 557:561
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle

from utils.analysis_utils import DatasetContext, scoring_path
from utils.paths import FIGURES_DIR, SCORING_DIR

DEFAULT_PAIRS = ["1486:1488", "1067:1069"]
PAIR_TAGS = {"1486:1488": "stable", "1067:1069": "dangerous topic: direction flip"}
STANCE_LABELS = {
    "STRONGLY_AGREE": "strongly agree", "AGREE": "agree", "SOMEWHAT_AGREE": "somewhat agree", "NEUTRAL": "neutral",
    "SOMEWHAT_DISAGREE": "somewhat disagree", "DISAGREE": "disagree", "STRONGLY_DISAGREE": "strongly disagree",
}

INK, MUTED, TRACK = "#1f2328", "#6e7781", "#e6e8eb"
COLOR_A, COLOR_B = "#2f6db5", "#c2602a"
FILL_A, FILL_B = "#eef4fb", "#fbf1ea"
FIG_W = 7.0
MARGIN = 0.12
GAP = 0.18
PAD = 0.09
FS_TEXT, FS_SMALL, FS_STATEMENT = 7.6, 6.8, 8.4
LINE = 1.32


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge_model_id", default="llama-3.3-70b")
    parser.add_argument("--prompt_variation", default="base",
                        help="Scoring prompt variation to show, or 'mean' for the mean over all of them.")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Pairs as id_1:id_2 of the scoring file.")
    parser.add_argument("--scoring_dir", type=Path, default=SCORING_DIR)
    parser.add_argument("--scoring_prompt_version", default="0_1")
    parser.add_argument("--output_dir", type=Path, default=FIGURES_DIR / "alignment_examples")
    return parser.parse_args()


def load_pairs(args):
    ctx = DatasetContext("habermas", scoring_dir=args.scoring_dir, scoring_prompt_version=args.scoring_prompt_version)
    path = scoring_path(ctx, args.judge_model_id)
    df = pd.read_csv(path)
    examples = []
    for pair in args.pairs:
        id_1, id_2 = (int(i) for i in pair.split(":"))
        rows = df[(df.id_1 == id_1) & (df.id_2 == id_2)]
        if args.prompt_variation != "mean":
            rows = rows[rows.prompt_variation == args.prompt_variation]
        if rows.empty:
            raise ValueError(f"No rows for pair ({id_1}, {id_2}) / {args.prompt_variation} in {path}")
        r = rows.iloc[0]
        examples.append({
            "situation": r.situation, "tag": PAIR_TAGS.get(pair),
            "text_a": " ".join(r.text_1.split()), "stance_a": STANCE_LABELS.get(r.stance_1, r.stance_1),
            "text_b": " ".join(r.text_2.split()), "stance_b": STANCE_LABELS.get(r.stance_2, r.stance_2),
            "a_to_b": rows.parsed_score_1to2.mean(), "b_to_a": rows.parsed_score_2to1.mean(),
        })
    return examples


def wrap(fig, text, width_in, fontsize):
    """Greedy word wrap to a width in inches, measured with the renderer."""
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(size=fontsize)

    def fits(line):
        return renderer.get_text_width_height_descent(line, prop, ismath=False)[0] / fig.dpi <= width_in

    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and not fits(candidate):
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def line_h(fontsize):
    return fontsize * LINE / 72


def draw_opinion(ax, x, y_top, w, label, stance, text, color, fill):
    """One opinion box with its top left corner at (x, y_top); returns its height."""
    lines = wrap(ax.figure, text, w - 2 * PAD, FS_TEXT)
    header_h = line_h(FS_SMALL) + 0.04
    h = PAD + header_h + len(lines) * line_h(FS_TEXT) + PAD
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=color,
                                lw=0.9))
    ax.text(x + PAD, y_top - PAD, f"Opinion {label}", ha="left", va="top", fontsize=FS_SMALL, fontweight="bold",
            color=color)
    ax.text(x + w - PAD, y_top - PAD, f"participant: {stance}", ha="right", va="top", fontsize=FS_SMALL, color=MUTED,
            style="italic")
    ax.text(x + PAD, y_top - PAD - header_h, "\n".join(lines), ha="left", va="top", fontsize=FS_TEXT, color=INK,
            linespacing=LINE)
    return h


def draw_direction(ax, x, y, w, first, second, score):
    """'first -> second', the question asked, and the score as a bar."""
    c_first, c_second = (COLOR_A, COLOR_B) if first == "A" else (COLOR_B, COLOR_A)
    fs = FS_TEXT + 0.6
    ax.text(x, y, first, ha="left", va="center", fontsize=fs, fontweight="bold", color=c_first)
    ax.annotate("", xy=(x + 0.33, y), xytext=(x + 0.14, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(x + 0.36, y, second, ha="left", va="center", fontsize=fs, fontweight="bold", color=c_second)
    ax.text(x + 0.55, y, f"first = {first},  second = {second}:   how well does {second} align with {first}?",
            ha="left", va="center", fontsize=FS_TEXT, color=INK)
    bar_w, bar_h = 1.55, 0.085
    bar_x = x + w - 0.42 - bar_w
    ax.add_patch(Rectangle((bar_x, y - bar_h / 2), bar_w, bar_h, fc=TRACK, ec="none"))
    if score > 0:
        ax.add_patch(Rectangle((bar_x, y - bar_h / 2), bar_w * score, bar_h, fc=c_second, ec="none"))
    label = f"{score:.2f}"
    label = label[:-1] if label.endswith("0") else label
    ax.text(x + w, y, label, ha="right", va="center", fontsize=fs, fontweight="bold", color=INK)


def draw_example(ax, y_top, ex, index):
    """One example from y_top down; returns the y below it."""
    x0, w = MARGIN, FIG_W - 2 * MARGIN
    y = y_top
    x_label = x0
    letter = f"({chr(ord('a') + index)})" if index is not None else None
    if ex["tag"]:
        header = ex["tag"][0].upper() + ex["tag"][1:]
        ax.text(x0, y, f"{letter} {header}" if letter else header, ha="left", va="top", fontsize=FS_STATEMENT + 0.4,
                fontweight="bold", color=INK)
        y -= line_h(FS_STATEMENT + 0.4) + 0.04
    elif letter:
        ax.text(x0, y, letter, ha="left", va="top", fontsize=FS_STATEMENT, fontweight="bold", color=INK)
        x_label += 0.28
    ax.text(x_label, y, "Statement:", ha="left", va="top", fontsize=FS_STATEMENT, color=MUTED)
    ax.text(x_label + 0.74, y, ex["situation"], ha="left", va="top", fontsize=FS_STATEMENT, color=INK, style="italic")
    y -= line_h(FS_STATEMENT) + 0.08

    box_w = (w - GAP) / 2
    h_a = draw_opinion(ax, x0, y, box_w, "A", ex["stance_a"], ex["text_a"], COLOR_A, FILL_A)
    h_b = draw_opinion(ax, x0 + box_w + GAP, y, box_w, "B", ex["stance_b"], ex["text_b"], COLOR_B, FILL_B)
    h = max(h_a, h_b)
    for patch in ax.patches[-2:]:
        patch.set_y(y - h)
        patch.set_height(h)
    y -= h + 0.2
    draw_direction(ax, x0 + 0.04, y, w - 0.04, "A", "B", ex["a_to_b"])
    y -= 0.25
    draw_direction(ax, x0 + 0.04, y, w - 0.04, "B", "A", ex["b_to_a"])
    return y - 0.16


def main():
    args = parse_args()
    examples = load_pairs(args)
    # Draw on a tall canvas in inches, then crop to the height used.
    canvas_h = 12.0
    fig = plt.figure(figsize=(FIG_W, canvas_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, canvas_h)
    ax.axis("off")
    y = canvas_h - MARGIN
    for i, ex in enumerate(examples):
        if i > 0:
            ax.plot([MARGIN, FIG_W - MARGIN], [y + 0.04, y + 0.04], color=TRACK, lw=0.8)
            y -= 0.14
        y = draw_example(ax, y, ex, i if len(examples) > 1 else None)
    variation = "mean over prompt variations" if args.prompt_variation == "mean" else f"'{args.prompt_variation}' prompt"
    ax.text(MARGIN, y, f"Judge: {args.judge_model_id} ({variation}). Score in [0, 1]; the judge is asked how well "
            "the second opinion aligns with the first.", ha="left", va="top", fontsize=FS_SMALL, color=MUTED)
    y -= line_h(FS_SMALL) + MARGIN
    fig.set_size_inches(FIG_W, canvas_h - y)
    ax.set_ylim(y, canvas_h)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = args.output_dir / "habermas_alignment_directions"
    fig.savefig(out_stem.with_suffix(".png"), dpi=250)
    fig.savefig(out_stem.with_suffix(".pdf"))
    print(f"Saved {out_stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
