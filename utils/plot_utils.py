import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Judge -> "Family\nSize", in the order of the paper's figures.
JUDGES = {
    "gemma3": "Gemma\n27B",
    "llama-3.3-70b": "Llama\n70B",
    "mistral-24b": "Mistral\n24B",
    "qwen2.5-72b": "Qwen\n72B",
    "openai/gpt-5.4-nano": "GPT-5.4\nnano",
}
SCALE_JUDGES = {
    "qwen2.5-05b": "Qwen\n0.5B",
    "qwen2.5-1b": "Qwen\n1.5B",
    "qwen2.5-3b": "Qwen\n3B",
    "qwen2.5-7b": "Qwen\n7B",
    "qwen2.5-14b": "Qwen\n14B",
    "qwen2.5-32b": "Qwen\n32B",
    "qwen2.5-72b": "Qwen\n72B",
}
# Judge -> (name, colour) for the curves: colour-blind safe for the main judges, darker =
# smaller for the Qwen2.5 sizes.
MAIN_STYLES = {judge: (label.replace("\n", "-"), color) for (judge, label), color in
               zip(JUDGES.items(), ["#0072B2", "#E69F00", "#009E73", "#C9372C", "#7B5AA6"])}
SCALE_CMAP = LinearSegmentedColormap.from_list(
    "scale", ["#26185f", "#3a3f9e", "#2f74b5", "#1fa0a8", "#4dbb7a", "#a3c93a"])
SCALE_STYLES = {judge: (label.replace("\n", "-"), SCALE_CMAP(i / (len(SCALE_JUDGES) - 1)))
                for i, (judge, label) in enumerate(SCALE_JUDGES.items())}


def judge_name(judge):
    return {**SCALE_JUDGES, **JUDGES}.get(judge, judge).replace("\n", "-")


# Sizes in inches: panels are drawn at their printed size (0.49 of a 5.5in text width), so
# font sizes are the real point sizes in the PDF.
TEXT_WIDTH = 5.5
PANEL_WIDTH = 0.49 * TEXT_WIDTH

FONT = ["Nimbus Sans", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
CI_FONT = ["Nimbus Sans Narrow", "Liberation Sans Narrow", "Arial Narrow"] + FONT
MINUS = "−"
TEXT_COLOR, MUTED_TEXT = "#222222", "#7a7a7a"
NA_COLOR = "#efefef"
CMAP = LinearSegmentedColormap.from_list(
    "soft_rdbu", ["#a7282c", "#dc6f5c", "#f5c4b4", "#f7f7f7", "#bcd6ea", "#5b95c7", "#1f5590"])


def _installed(fonts):
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in fonts if f in available), "DejaVu Sans")


def setup_matplotlib():
    global CI_FONT
    if isinstance(CI_FONT, list):
        CI_FONT = _installed(CI_FONT)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [_installed(FONT)],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 2,
        "ytick.major.pad": 3,
    })


def save(fig, path):
    """path without suffix: saves a PDF and a 300 dpi PNG. No bbox_inches="tight", so the
    printed width is kept."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_name(path.name + suffix), dpi=300)
    plt.close(fig)
    print(f"Saved {path}.{{pdf,png}}")


# Heatmaps of Spearman's rho (features x judges)

ALPHA = 0.05
MIN_ABS_RHO = 0.1
HEATMAP_STEM = f"within_feature_minrho{MIN_ABS_RHO:g}"
ROW_HEIGHT = {True: 0.25, False: 0.165}
OUTER_GAP = 0.02
TOP_MARGIN = 0.27
BOTTOM_MARGIN = 0.05
COLORBAR_GAP, COLORBAR_WIDTH = 0.06, 0.056
CELL_GAP = 0.014
HEATMAP_FONT_SIZES = {"rho": 7, "ci": 6, "na": 6, "ticks": 7, "colorbar": 7, "colorbar_ticks": 6}
LEADING_ZERO = True
# Left panel of a pair has the feature names, right panel the colorbar.
LAYOUTS = {"left": (True, False), "right": (False, True)}


def select_strongest(significance, group="analysis"):
    """Per (judge, feature): the test with the largest |rho| (tests without a CI are left
    out, shown as n/a). significant = q < ALPHA (BH within the feature) and |rho| >= MIN_ABS_RHO."""
    d = significance[significance["rho_ci95_low"].notna()].copy()
    d["abs_rho"] = d["rho"].abs()
    selected = d.loc[d.groupby(["judge", group])["abs_rho"].idxmax()].drop(columns="abs_rho")
    selected["significant"] = (selected["rho_q"] < ALPHA) & (selected["rho"].abs() >= MIN_ABS_RHO)
    return selected.set_index([group, "judge"])


def color_limit(panels):
    largest = max(cells["rho"].abs().max() for cells in panels)
    return max(math.ceil(largest / 0.05 - 1e-9) * 0.05, 0.05)


def fmt_signed(value, leading_zero=None):
    text = f"{value:+.2f}".replace("-", MINUS)
    return text if (LEADING_ZERO if leading_zero is None else leading_zero) else text.replace("0.", ".", 1)


def _fmt_tick(value, leading_zero=None):
    return "0" if value == 0 else fmt_signed(value, leading_zero).lstrip("+")


def colorbar_ticks(limit):
    step = 0.1 if limit > 0.2 else 0.05
    n = math.floor(limit / step + 1e-9)
    return np.arange(-n, n + 1) * step


def label_width(texts, fontsize):
    fig = plt.figure()
    renderer = fig.canvas.get_renderer()
    width = max(fig.text(0, 0, t, fontsize=fontsize).get_window_extent(renderer).width for t in texts)
    plt.close(fig)
    return width / fig.dpi


def side_margin(features):
    """Room for the feature names (left) and the colorbar (right), the same on both sides so
    that all the heatmaps of a figure have identical cells."""
    labels = label_width(features.values(), HEATMAP_FONT_SIZES["ticks"]) + 0.06
    colorbar = (COLORBAR_GAP + COLORBAR_WIDTH + 0.05
                + label_width([f"{MINUS}0.20"], HEATMAP_FONT_SIZES["colorbar_ticks"])
                + 0.04 + HEATMAP_FONT_SIZES["colorbar"] / 72)
    return max(labels, colorbar)


def heatmap_panel(cells, features, judges, limit, show_ci, layout, margin):
    row_labels, colorbar = LAYOUTS[layout]
    fs = HEATMAP_FONT_SIZES
    # More than 5 judges (the Qwen2.5 sizes): "+.25*" instead of "+0.25*" to fit the cells.
    leading_zero = None if len(judges) <= 5 else False
    judges = {j: label for j, label in judges.items() if j in cells.index.get_level_values("judge")}
    row_height = ROW_HEIGHT[show_ci]
    left = margin if row_labels else OUTER_GAP
    right = margin if colorbar else OUTER_GAP
    width = PANEL_WIDTH - left - right
    height = row_height * len(features)
    fig_height = TOP_MARGIN + height + BOTTOM_MARGIN
    fig = plt.figure(figsize=(PANEL_WIDTH, fig_height))
    ax = fig.add_axes([left / PANEL_WIDTH, BOTTOM_MARGIN / fig_height, width / PANEL_WIDTH, height / fig_height])
    ax.set_xlim(0, len(judges))
    ax.set_ylim(len(features), 0)
    ax.axis("off")

    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    gap_x, gap_y = CELL_GAP / 2 / (width / len(judges)), CELL_GAP / 2 / row_height
    for top, feature in enumerate(features):
        if row_labels:
            ax.annotate(features[feature], xy=(0, top + 0.5), xytext=(-3, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=fs["ticks"], color=TEXT_COLOR)
        for j, judge in enumerate(judges):
            x, y = j + 0.5, top + 0.5
            present = (feature, judge) in cells.index
            row = cells.loc[(feature, judge)] if present else None
            ax.add_patch(Rectangle((j + gap_x, top + gap_y), 1 - 2 * gap_x, 1 - 2 * gap_y,
                                   facecolor=CMAP(norm(row["rho"])) if present else NA_COLOR, edgecolor="none"))
            if not present:
                ax.text(x, y, "n/a", ha="center", va="center", fontsize=fs["na"], color="#a0a0a0")
                continue
            dark = abs(row["rho"]) > 0.6 * limit
            color = "white" if dark else (TEXT_COLOR if row["significant"] else "#4d4d4d")
            ax.text(x, y - 0.17 if show_ci else y, fmt_signed(row["rho"], leading_zero) + ("*" if row["significant"] else ""),
                    ha="center", va="center", fontsize=fs["rho"], color=color,
                    fontweight="bold" if row["significant"] else "normal")
            if show_ci:
                ci = f"[{fmt_signed(row['rho_ci95_low'], False)}, {fmt_signed(row['rho_ci95_high'], False)}]"
                ax.text(x, y + 0.22, ci, ha="center", va="center", fontsize=fs["ci"], fontfamily=CI_FONT,
                        color="white" if dark else "#555555")

    for j, label in enumerate(judges.values()):
        family, size = label.split("\n")
        ax.annotate(family, xy=(j + 0.5, 1), xycoords=("data", "axes fraction"), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fs["ticks"], color=TEXT_COLOR)
        ax.annotate(size, xy=(j + 0.5, 1), xycoords=("data", "axes fraction"), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fs["ticks"] - 1, color=MUTED_TEXT)

    if colorbar:
        cax = fig.add_axes([(left + width + COLORBAR_GAP) / PANEL_WIDTH, BOTTOM_MARGIN / fig_height,
                            COLORBAR_WIDTH / PANEL_WIDTH, height / fig_height])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
        bar.outline.set_visible(False)
        ticks = colorbar_ticks(limit)
        bar.set_ticks(ticks)
        bar.set_ticklabels([_fmt_tick(t, leading_zero) for t in ticks])
        bar.ax.tick_params(labelsize=fs["colorbar_ticks"], length=2, width=0.4, pad=1.5, color=MUTED_TEXT,
                           labelcolor=TEXT_COLOR)
        bar.set_label("Spearman ρ", fontsize=fs["colorbar"], labelpad=3, color=TEXT_COLOR)
    return fig


def heatmap_figure(panels, features, show_ci, judges):
    """panels: [(cells, path, layout)], drawn on one shared colour scale."""
    limit = color_limit([cells for cells, _, _ in panels])
    margin = side_margin(features)
    for cells, path, layout in panels:
        save(heatmap_panel(cells, features, judges, limit, show_ci, layout, margin), path)


# Smoothed curves: two panels per plot (e.g. RS and PC), one line and band per judge.

TOP, PLOT_HEIGHT, BOTTOM = 0.19, 1.22, 0.34
RIGHT = 0.06
LINE_FONT_SIZES = {"ticks": 7, "label": 7.5, "title": 7, "legend": 8, "region": 6}
LINE_WIDTH, BAND_ALPHA = 1.1, 0.16


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color("#444444")
    ax.tick_params(labelsize=LINE_FONT_SIZES["ticks"], length=2, width=0.5, pad=1.5, colors="#444444",
                   labelcolor=TEXT_COLOR)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))


def _polarity_regions(ax, boundary):
    """x in percentiles of the sentiment gap: shade the same-polarity pairs."""
    ax.axvspan(0, boundary, color="#000000", alpha=0.045, linewidth=0, zorder=0)
    ax.axvline(boundary, color=MUTED_TEXT, linestyle=(0, (3, 2)), linewidth=0.6, zorder=1)
    for x, label in [(boundary / 2, "same\npolarity"), ((boundary + 100) / 2, "opposite\npolarity")]:
        ax.text(x, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=LINE_FONT_SIZES["region"], color=MUTED_TEXT, linespacing=1.0)


def _hide_colliding_tick(fig, axes):
    renderer = fig.canvas.get_renderer()

    def visible(ax):
        return [t for t in ax.get_xticklabels()
                if t.get_text() and ax.get_xlim()[0] <= t.get_position()[0] <= ax.get_xlim()[1]]

    left, right = visible(axes[0]), visible(axes[1])
    if left and right:
        gap = right[0].get_window_extent(renderer).x0 - left[-1].get_window_extent(renderer).x1
        if gap < 2:
            left[-1].set_visible(False)


def line_plot(summaries, styles, panels, xlabel, percentile, share_y, left, gap, zero_line):
    """summaries: judge -> smoothed curve; panels: [(column, title)] for the two panels."""
    fig_height = TOP + PLOT_HEIGHT + BOTTOM
    fig = plt.figure(figsize=(PANEL_WIDTH, fig_height))
    width = (PANEL_WIDTH - left - gap - RIGHT) / 2
    axes = []
    for k in range(2):
        rect = [(left + k * (width + gap)) / PANEL_WIDTH, BOTTOM / fig_height, width / PANEL_WIDTH,
                PLOT_HEIGHT / fig_height]
        axes.append(fig.add_axes(rect, sharey=axes[0] if (k and share_y) else None))

    for ax, (column, title) in zip(axes, panels):
        style_axis(ax)
        if zero_line:
            ax.axhline(0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.6, zorder=1)
        if percentile:
            _polarity_regions(ax, sum(s["same_polarity_share"].iloc[0] for s in summaries.values()) / len(summaries))
        for judge, summary in summaries.items():
            color = styles[judge][1]
            ax.plot(summary["x_mean"], summary[column], color=color, linewidth=LINE_WIDTH, solid_capstyle="round",
                    zorder=3)
            ax.fill_between(summary["x_mean"], summary[f"{column}_ci95_low"], summary[f"{column}_ci95_high"],
                            color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
        ax.set_title(title, fontsize=LINE_FONT_SIZES["title"], color=TEXT_COLOR, pad=3)
        if percentile:
            ax.set_xlim(0, 100)
            ax.set_xticks([25, 50, 75])
        else:
            ax.margins(x=0.02)
    if share_y:
        axes[1].tick_params(labelleft=False)
    largest = max(abs(x) for ax in axes for x in ax.get_xlim())
    if largest < 0.01:
        exponent = math.floor(math.log10(largest))
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 10 ** exponent:g}"))
        xlabel = f"{xlabel} (×10$^{{{exponent}}}$)"
    if percentile:
        for ax in (axes if not share_y else axes[:1]):
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + 0.22 * (high - low))

    fig.text((left + (PANEL_WIDTH - left - RIGHT) / 2) / PANEL_WIDTH, 0.02 / fig_height, xlabel,
             ha="center", va="bottom", fontsize=LINE_FONT_SIZES["label"], color=TEXT_COLOR)
    _hide_colliding_tick(fig, axes)
    return fig


def legend_figure(styles):
    """One row with every judge, as wide as the text."""
    fig = plt.figure(figsize=(TEXT_WIDTH, 0.2))
    handles = [Line2D([], [], color=color, linewidth=1.6, solid_capstyle="round") for _, color in styles.values()]
    renderer = fig.canvas.get_renderer()
    for spacing in (1.6, 1.3, 1.0, 0.8, 0.6):
        legend = fig.legend(handles, [name for name, _ in styles.values()], loc="center", ncol=len(styles),
                            frameon=False, fontsize=LINE_FONT_SIZES["legend"], handlelength=spacing + 0.2,
                            handletextpad=0.5, columnspacing=spacing, borderaxespad=0, labelcolor=TEXT_COLOR)
        if legend.get_window_extent(renderer).width <= 0.98 * TEXT_WIDTH * fig.dpi:
            break
        legend.remove()
    return fig


def load_curves(curve_dir, judges, band_column):
    """judge -> curve CSV of curve_dir, for the judges whose curve has a bootstrap band."""
    summaries = {}
    for judge in judges:
        path = Path(curve_dir) / f"{judge}.csv"
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        if band_column in summary and summary[band_column].notna().any():
            summaries[judge] = summary
    return summaries
