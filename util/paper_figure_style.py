"""Plot styling, fixed-size export and portable layout checks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

WIDTH_MM = 183.0
# Dark categorical colours on white; line styles and symbols also encode groups.
BLUE, VERMILION, GREEN, PURPLE = "#0072B2", "#D55E00", "#007A5E", "#AA3377"
ALPHA_COLORS = {10.0: BLUE, 1.0: GREEN, 0.1: PURPLE, 0.01: VERMILION}
ALPHA_MARKERS = {10.0: "o", 1.0: "s", 0.1: "^", 0.01: "D"}
ALPHA_LINES = {10.0: "--", 1.0: "-.", 0.1: ":", 0.01: "-"}
IID_COLOR = "#252525"
SETTING_COLORS = (BLUE, VERMILION, GREEN, PURPLE)
SETTING_MARKERS = ("o", "s", "^", "D")
TASK_COLORS = {"antiderivative": BLUE, "burgers": VERMILION, "diffusion_reaction": GREEN,
               "darcy": PURPLE, "shallow_water": GREEN, "cfd_cylinder": BLUE, "openfwi": VERMILION}


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        # A math subscript is about 70% of its parent size. This also keeps
        # the smallest label above 5 pt when a 183-mm PDF is placed at the
        # current NC author-proof width of 155 mm.
        "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
        "lines.linewidth": 1.1, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.usetex": False, "savefig.bbox": None, "figure.dpi": 150,
    })


def grid(nrows: int, ncols: int, *, height_mm: float | None = None):
    height = height_mm or (26.0 + 39.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_MM / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.977, bottom=12.0 / height, top=1-19.0 / height,
                        hspace=0.64, wspace=0.47 if ncols > 2 else 0.40)
    return fig, axes


def panel(axis, index: int, title: str = "") -> None:
    axis.set_gid(chr(97 + index))
    # Put letters above the plot, not in the vertical-axis label gutter.
    # Fixed point offsets keep headers aligned in both narrow and wide panels.
    letter = axis.annotate(chr(97 + index), xy=(0, 1), xycoords="axes fraction",
                           xytext=(0, 8), textcoords="offset points", ha="left", va="bottom",
                           fontsize=9, fontweight="bold", annotation_clip=False)
    letter.set_gid("panel-label")
    if title:
        header = axis.annotate(title, xy=(0, 1), xycoords="axes fraction",
                               xytext=(13, 8), textcoords="offset points", ha="left", va="bottom",
                               fontsize=8, annotation_clip=False)
        header.set_gid("panel-title")
    axis.tick_params(direction="out", length=2.5, width=0.6, pad=2)
    axis.grid(axis="y", color="#DDE1E5", zorder=0)
    axis.set_axisbelow(True)
    axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, min_n_ticks=3))
    formatter = mpl.ticker.ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    axis.yaxis.set_major_formatter(formatter)


def alpha_legend(fig, alphas, *, iid: bool = False, markers_only: bool = False) -> None:
    handles = []
    if iid:
        handles.append(Line2D([], [], color=IID_COLOR, marker="o", markerfacecolor="white", label="Stratified IID"))
    handles.extend(Line2D([], [], color=ALPHA_COLORS[a], marker=ALPHA_MARKERS[a],
                          linestyle="none" if markers_only else ALPHA_LINES[a],
                          markerfacecolor="white", markersize=4, label=f"α = {a:g}") for a in alphas)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.997), ncol=len(handles),
               frameon=False, columnspacing=1.5, handlelength=1.8, handletextpad=0.45)


def require_panel_label_clearance(fig, axes) -> None:
    """Reject panel letters overlapping any rendered vertical-axis label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [(axis.get_gid(), text) for axis in axes for text in axis.texts
              if text.get_gid() == "panel-label" and text.get_visible()]
    vertical_labels = [(axis.get_gid(), axis.yaxis.label) for axis in axes
                       if axis.yaxis.label.get_visible() and axis.yaxis.label.get_text()]
    for panel_id, letter in labels:
        letter_box = letter.get_window_extent(renderer)
        for axis_id, vertical_label in vertical_labels:
            if letter_box.overlaps(vertical_label.get_window_extent(renderer)):
                raise ValueError(f"Panel letter {panel_id} overlaps vertical-axis label {axis_id}")


def require_basic_alignment(fig, axes, path: Path, *, alignment_options=None) -> dict:
    """Check final plot rectangles in points without a skill installation."""
    fig.canvas.draw()
    width, height = fig.get_size_inches() * 72.0
    boxes = [np.asarray([a.get_position().x0 * width, a.get_position().y0 * height,
                         a.get_position().x1 * width, a.get_position().y1 * height]) for a in axes]
    problems = []
    def comparable(i, j, kind):
        if alignment_options is None:
            return True
        # Explicit comparable groups support mixed layouts, e.g. four small
        # images above two wide quantitative panels sharing the same left edge.
        for group in alignment_options.get(kind + '_groups', []):
            ids = group.get('panels', []) if isinstance(group, dict) else group
            if axes[i].get_gid() in ids and axes[j].get_gid() in ids:
                return True
        return False
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[:i]):
            if min(a[2], b[2]) - max(a[0], b[0]) > 1.5 and min(a[3], b[3]) - max(a[1], b[1]) > 1.5:
                problems.append(f"Plot areas {j} and {i} overlap")
            if comparable(i, j, 'row') and abs(a[1]-b[1]) <= 1.5 and abs(a[3]-b[3]) > 1.5:
                problems.append(f"Unequal row heights: {j}, {i}")
            if comparable(i, j, 'column') and abs(a[0]-b[0]) <= 1.5 and abs(a[2]-b[2]) > 1.5:
                problems.append(f"Unequal column widths: {j}, {i}")
    report = {"verdict": "PASS" if not problems else "FIX BEFORE DELIVERY", "tolerance_pt": 1.5,
              "panels": [{"id": a.get_gid(), "bbox_pt": box.tolist()} for a, box in zip(axes, boxes)],
              "problems": problems, "backend": "matplotlib", "checker": "portable plot-rectangle preflight"}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        raise ValueError("; ".join(problems))
    return report


class Exporter:
    def __init__(self, output: Path, *, svg: bool = False, previews: bool = False):
        self.output, self.svg, self.previews = output, svg, previews
        self.qa = output / "qa"
        self.qa.mkdir(parents=True, exist_ok=True)
        self.figures = []

    def save(self, fig, name: str, claim: str, caption: str, experiments, *, exclude_axes=(),
             alignment_options=None) -> Path:
        axes = [a for a in fig.axes if a not in exclude_axes]
        require_panel_label_clearance(fig, axes)
        require_basic_alignment(fig, axes, self.qa / f"{name}.layout.json",
                                alignment_options=alignment_options)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        # Axis tight bounds include only rendered ticks. Locator-generated ticks
        # outside the view can have visible=True without appearing in the PDF.
        artists = [*fig.axes, *fig.texts, *fig.legends]
        for artist in artists:
            if not artist.get_visible():
                continue
            box = artist.get_tightbbox(renderer)
            if box is None:
                continue
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
                raise ValueError(f"Clipped outer content in {name}: {type(artist).__name__}")
        pdf = self.output / f"{name}.pdf"
        fig.savefig(pdf, format="pdf", metadata={"Title": claim, "Creator": "FL-PDE manuscript plotting"})
        if self.svg:
            fig.savefig(self.output / f"{name}.svg", format="svg")
        if self.previews:
            (self.qa / "previews").mkdir(exist_ok=True)
            fig.savefig(self.qa / "previews" / f"{name}.png", dpi=300)
        self.figures.append(dict(name=name, claim=claim, caption=caption,
                                 experiments=[e.identifier for e in experiments], panels=len(axes),
                                 width_mm=round(float(fig.get_size_inches()[0]) * 25.4, 2),
                                 height_mm=round(float(fig.get_size_inches()[1]) * 25.4, 2)))
        plt.close(fig)
        print(f"Saved {pdf}")
        return pdf

