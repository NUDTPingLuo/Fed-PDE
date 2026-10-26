"""Plot paired FedAvg/SB-CFO results and resource costs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from util.paper_figure_style import (BLUE, GREEN, IID_COLOR, VERMILION,
                                    Exporter, configure_matplotlib, grid, panel)
from util.sb_cfo_figure_data import (FINAL, METHODS, CONDITIONS, ROUNDS, SEEDS,
                                    interval, load_studies)

ROOT = Path(__file__).resolve().parent
COLORS = {"fedavg": BLUE, "sb_cfo": VERMILION}
SERIES_COLORS = {("fedavg", "iid"): IID_COLOR,
                 ("fedavg", "alpha0.01"): BLUE,
                 ("sb_cfo", "iid"): GREEN,
                 ("sb_cfo", "alpha0.01"): VERMILION}
NAMES = {"fedavg": "FedAvg", "sb_cfo": "SB-CFO"}
TITLES = {"burgers": "Burgers, ν = 0.005", "darcy": "Darcy, β = 0.01"}
MARKERS = {"fedavg": "o", "sb_cfo": "s"}


def trajectory_style(method, condition):
    """One style source for trajectories and their legend handles."""
    color = SERIES_COLORS[method, condition]
    iid = condition == "iid"
    return dict(color=color, linestyle="--" if iid else "-", linewidth=1.4,
                marker=MARKERS[method], markersize=3.2, markeredgewidth=0.8,
                markerfacecolor="white" if iid else color)


def round_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.85, 1200)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Communication round")


def trajectory(ax, study, method, condition, field, geometric=False):
    rounds = ROUNDS[1:]  # round 0 is preserved in source data, undefined on log x.
    stats = np.array([interval(study.values(method, condition, r, field), geometric) for r in rounds])
    iid = condition == "iid"
    style = trajectory_style(method, condition)
    ax.fill_between(rounds, stats[:, 1], stats[:, 2], color=style["color"], alpha=0.07 if iid else 0.10,
                    linewidth=0, zorder=1)
    ax.plot(rounds, stats[:, 0], **style, zorder=3)


def paired_points(ax, arrays, colors, labels, geometric=False, connect=True):
    """Fixed seed offsets, visible pairing, and interval of seed-level quantities."""
    offset = np.linspace(-0.10, 0.10, len(SEEDS))
    if connect:
        for j in range(len(SEEDS)):
            ax.plot(np.arange(len(arrays))+offset[j], [a[j] for a in arrays],
                    color="#B7BDC1", linewidth=0.55, alpha=0.8, zorder=1)
    for i, (values, color) in enumerate(zip(arrays, colors)):
        ax.scatter(i+offset, values, s=13, facecolors="white", edgecolors=color,
                   linewidths=0.75, zorder=3)
        mean, low, high = interval(values, geometric)
        ax.errorbar(i+0.19, mean, yerr=[[mean-low], [high-mean]], fmt="D", color=color,
                    markersize=3.5, capsize=2.5, elinewidth=1.1, zorder=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.35, len(labels)-0.55)


MAIN_CAPTION = (
    "SB-CFO reduces the prediction penalty of solution-space heterogeneity in two PDE tasks. "
    "a,d, Relative L2 test error for FedAvg and SB-CFO under stratified IID and solution-bin "
    "Dirichlet partitioning (α = 0.01). Curves connect recorded checkpoints without smoothing; "
    "shading denotes pointwise 95% Student-t confidence intervals for arithmetic means across "
    "five seeds (0, 1, 42, 999 and 2026). Round 0 is omitted from the logarithmic axis only. "
    "b,e, Final heterogeneity penalty, defined for each method and seed as 100 times the "
    "difference between non-IID and that method's IID relative L2 error, in percentage points "
    "(pp). Open circles are seeds, grey lines join matched seeds, and diamonds indicate "
    "arithmetic means with two-sided 95% Student-t confidence intervals. c,f, Within SB-CFO "
    "at α = 0.01, gradient variance after correction divided by variance before correction "
    "at the same global model; the before value is normalized to one for every seed. "
    "Diamonds and intervals show geometric means and back-transformed 95% Student-t "
    "intervals calculated on log ratios. Smaller values indicate stronger correction. "
    "These diagnostics evaluate the applied correction, whose prototypes are one round old "
    "at the final checkpoint. All final comparisons use round 1000. Rows correspond to "
    "Burgers (ν = 0.005; 800 training and 500 test samples) and Darcy (β = 0.01; 2000 "
    "training and 500 test samples). Both use ten clients, full participation, five local "
    "steps, and Adam with a cosine learning-rate schedule from 0.001 to 0.0001."
)

DIAGNOSTIC_CAPTION = (
    "Update dispersion and computational cost of SB-CFO under α = 0.01. a,c, "
    "Sample-weighted local model-update variance at recorded rounds, summarized by geometric "
    "means and back-transformed pointwise 95% Student-t confidence intervals across five "
    "seeds. Variance describes updates during the preceding communication round, not the "
    "distance to an IID model. b,d, Cumulative costs through round 1000, expressed as "
    "within-seed SB-CFO/FedAvg ratios. Grad. visits includes optimizer and prototype gradient "
    "sample visits, excluding diagnostic-only visits; Comm. includes model and prototype "
    "tensor bytes; Time is recorded wall-clock time including evaluation and diagnostics. "
    "Open circles represent five seeds; diamonds and bars show arithmetic means and 95% "
    "Student-t intervals of the paired ratios. The dashed line denotes equal cost. "
    "Communication is logical, uncompressed payload accounting, not measured network "
    "traffic. Timing reflects this sequential execution and does not estimate parallel "
    "deployment latency. The implementation refreshes full-data gradient prototypes every "
    "round. Rows correspond to Burgers (ν = 0.005) and Darcy (β = 0.01)."
)


def main_figure(studies, exporter):
    fig, axes = grid(2, 3, height_mm=128)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.095, wspace=0.59, hspace=0.79)
    handles = [Line2D([], [], **trajectory_style(m, c),
                       label=NAMES[m] + (", IID" if c == "iid" else ", α = 0.01"))
               for m in METHODS for c in CONDITIONS]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.54, 0.995),
               frameon=False, columnspacing=2.0, handlelength=2.3)
    for row, study in enumerate(studies):
        a, b, c = axes[row]
        panel(a, row*3, TITLES[study.task])
        panel(b, row*3+1, "Final IID gap")
        panel(c, row*3+2, "Gradient correction")
        for method in METHODS:
            for condition in CONDITIONS:
                trajectory(a, study, method, condition, "test_error")
        round_axis(a)
        a.set_ylabel("Relative L2 test error")
        paired_points(b, [study.gap(m) for m in METHODS], [COLORS[m] for m in METHODS],
                      [NAMES[m] for m in METHODS])
        b.axhline(0, color="#93999E", linewidth=0.7, linestyle="--", zorder=0)
        b.set_ylabel("Non-IID − own IID (pp)")
        b.set_ylim(-3, 17)
        b.set_yticks([0, 5, 10, 15])
        paired_points(c, [np.ones(len(SEEDS)), study.gradient_remaining()],
                      ["#7B8289", COLORS["sb_cfo"]], ["Before", "After"], geometric=True)
        c.set_yscale("log")
        c.set_ylim(0.004, 1.8)
        c.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
        c.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        c.yaxis.set_minor_locator(NullLocator())
        c.set_ylabel("Gradient variance / before")
    exporter.save(fig, "sb_cfo_main", "SB-CFO: prediction penalty and gradient correction",
                  MAIN_CAPTION, studies)


def diagnostic_figure(studies, exporter):
    fig, axes = grid(2, 2, height_mm=122)
    fig.subplots_adjust(top=0.865, hspace=0.76)
    fig.legend(handles=[Line2D([], [], **trajectory_style(m, "alpha0.01"),
                               label=NAMES[m]) for m in METHODS], ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(0.54, 0.995))
    for row, study in enumerate(studies):
        a, b = axes[row]
        panel(a, row*2, TITLES[study.task])
        panel(b, row*2+1, "Cumulative cost")
        for method in METHODS:
            trajectory(a, study, method, "alpha0.01", "local_update_variance", geometric=True)
        round_axis(a)
        a.set_yscale("log")
        # Log-axis exponents render at 70% of this size: >= 5 pt even
        # when the 183-mm PDF is placed at the NC proof width of 155 mm.
        a.tick_params(axis="y", labelsize=8.5)
        a.set_ylabel("Local update variance")
        paired_points(b, study.cost_ratios(), [COLORS["sb_cfo"]]*3,
                      ["Grad. visits", "Comm.", "Time"], connect=False)
        b.axhline(1, color="#93999E", linewidth=0.7, linestyle="--")
        b.set_ylabel("SB-CFO / FedAvg cost")
        b.set_ylim(0, 9)
        b.set_yticks([1, 3, 5, 7, 9])
    exporter.save(fig, "sb_cfo_diagnostics", "SB-CFO: update dispersion and cost",
                  DIAGNOSTIC_CAPTION, studies)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_data(studies, output):
    source = output / "source_data"
    source.mkdir(exist_ok=True)
    write_csv(source / "rounds.csv", [r for s in studies for r in s.rows])
    effects, estimates = [], []
    for study in studies:
        metrics = {"fedavg_iid_gap_pp": study.gap("fedavg"), "sb_cfo_iid_gap_pp": study.gap("sb_cfo"),
                   "prediction_improvement_pp": 100*(study.values("fedavg", "alpha0.01", FINAL, "test_error")
                                                     - study.values("sb_cfo", "alpha0.01", FINAL, "test_error")),
                   "iid_gap_reduction_pp": study.gap("fedavg")-study.gap("sb_cfo"),
                   "gradient_remaining_ratio": study.gradient_remaining()}
        metrics.update(dict(zip(["gradient_visits_ratio", "communication_ratio", "time_ratio"], study.cost_ratios())))
        for i, seed in enumerate(SEEDS):
            effects.append(dict(task=study.task, seed=seed, round=FINAL,
                                **{k: v[i] for k, v in metrics.items()}))
        for metric, values in metrics.items():
            geo = metric == "gradient_remaining_ratio"
            mean, low, high = interval(values, geo)
            estimates.append(dict(task=study.task, metric=metric, round=FINAL, n=len(values),
                                  estimator="geometric" if geo else "arithmetic", mean=mean,
                                  ci95_low=low, ci95_high=high))
    write_csv(source / "final_paired_effects.csv", effects)
    write_csv(source / "final_estimates.csv", estimates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/sb_cfo_cosine_lr0p001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/pdf/sb_cfo")
    parser.add_argument("--previews", action="store_true")
    parser.add_argument("--svg", action="store_true")
    args = parser.parse_args()
    if (ROOT / "results").resolve() in [args.output_dir.resolve(), *args.output_dir.resolve().parents]:
        parser.error("Keep generated figures outside the raw results tree")
    studies = load_studies(args.results_root)
    configure_matplotlib()
    exporter = Exporter(args.output_dir, svg=args.svg, previews=args.previews)
    main_figure(studies, exporter)
    diagnostic_figure(studies, exporter)
    source_data(studies, args.output_dir)
    hashes = {k: v for study in studies for k, v in study.hashes.items()}
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
        raise RuntimeError("Source files changed during plotting")
    manifest = dict(final_round=FINAL, seeds=SEEDS, source_sha256=hashes, figures=exporter.figures,
                    protocol="Adam cosine 0.001 to 0.0001; K=10; local steps=5; full participation",
                    missing_seeds="error; never silently excluded", captions_embedded=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "figure_captions.md").write_text(
        "# SB-CFO figure legends\n\n" + "\n\n".join("## " + f["name"] + "\n\n" + f["caption"]
                                                     for f in exporter.figures) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
