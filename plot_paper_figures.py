"""Plot benchmark results, paired effects and diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fed_pde_paper_figure_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from util.paper_figure_data import (
    ALPHAS, FINAL_ROUND, FORMAL_ORDER, PUBLIC_ORDER, Experiment, bin_counts,
    discover_suite, export_source_data, mean_ci95, number, ordered, viscosity_experiments,
)
from util.paper_figure_style import (
    ALPHA_COLORS, ALPHA_LINES, ALPHA_MARKERS, IID_COLOR, SETTING_COLORS,
    SETTING_MARKERS, TASK_COLORS, Exporter, alpha_legend, configure_matplotlib,
    WIDTH_MM, grid, panel,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REPRESENTATIVE_SEED = 42
STATS_NOTE = "Mean and 95% Student-t CI across 5 seeds; intervals are pointwise and unadjusted."


def style_round_axis(axis, *, late=False, logarithmic=True):
    if late:
        axis.set_xscale("linear")
        axis.set_xlim(185, 1030)
        axis.set_xticks([200, 500, 1000])
        axis.set_xlabel("Communication round")
    elif logarithmic:
        axis.set_xscale("log")
        axis.set_xlim(0.85, 1200)
        axis.set_xticks([1, 10, 100, 1000])
        axis.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=(2, 5)))
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.tick_params(axis="x", which="minor", length=1.5, width=0.45)
        axis.set_xlabel("Communication round (log scale)")
    else:
        axis.set_xscale("linear")
        axis.set_xlim(-20, 1030)
        axis.set_xticks([0, 200, 400, 600, 800, 1000])
        axis.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.set_xlabel("Communication round")


def draw_curve(axis, experiment, metric, *, iid=False, late=False):
    conditions = (None, *ALPHAS) if iid else ALPHAS
    for alpha in conditions:
        rounds, mean, half = experiment.curve(metric, alpha, start=200 if late else 1)
        if not late and (np.any(rounds <= 0) or np.any(np.diff(rounds) <= 0)):
            raise ValueError("Log-round curves require positive, increasing checkpoints")
        if metric == "drift":
            mean, half = 100 * mean, 100 * half
        color = IID_COLOR if alpha is None else ALPHA_COLORS[alpha]
        axis.fill_between(rounds, mean-half, mean+half, color=color,
                          alpha=0.08 if iid else 0.10, linewidth=0, zorder=1)
        axis.plot(rounds, mean, color=color,
                  linestyle="-" if alpha is None else ALPHA_LINES[alpha],
                  linewidth=1.45 if alpha == 0.01 else (1.25 if alpha is None else 1.05),
                  marker="o" if alpha is None else ALPHA_MARKERS[alpha], markersize=3.3,
                  markerfacecolor="white", markeredgewidth=0.7, zorder=3)
    if metric == "drift":
        axis.axhline(0, color="#73797F", linestyle="--", lw=0.7, zorder=0)
    style_round_axis(axis, late=late, logarithmic=True)
    axis.margins(y=0.12)


def plot_error_curves(experiments, exporter, name):
    if name == "formal_test_error_curves":
        from plot_formal_training_examples import plot_formal_curves
        return plot_formal_curves(experiments, exporter, name)
    if name == "public_test_error_curves":
        from plot_public_training_examples import plot_public_curves
        return plot_public_curves(experiments, exporter, name)
    fig, axes = grid(len(experiments), 2)
    for row, e in enumerate(experiments):
        for col in range(2):
            axis = axes[row, col]
            panel(axis, 2*row+col, e.label if col == 0 else "Paired excess error")
            if col == 0:
                draw_curve(axis, e, "federated_error", iid=True)
                axis.set_yscale("linear")
                axis.set_ylabel(r"Relative $L_2$ error")
            else:
                draw_curve(axis, e, "drift", late=True)
                axis.set_ylabel("Excess error (pp)")
            if row < len(experiments) - 1:
                axis.set_xlabel("")
    alpha_legend(fig, ALPHAS, iid=True)
    exporter.save(fig, name, "Training convergence and paired non-IID prediction costs",
                  "Each row is one task. Left: recorded rounds 1–1000 on a logarithmic round axis "
                  "(ticks 1, 10, 100, 1000) and a linear error axis. Round 0 is retained in the "
                  "source data but cannot be displayed on a logarithmic axis. "
                  "Right: paired excess errors at rounds 200, 500 and 1000 on a linear round axis. "
                  "Each seed's non-IID error minus its matching IID error is computed before the "
                  "mean and confidence interval; the difference is multiplied by 100 and expressed "
                  "in percentage points, not as a relative percentage increase. The dashed zero "
                  "line denotes no excess error. This is a fixed window, not a best-test-error "
                  "selection. Both columns retain the complete arithmetic-mean 95% confidence intervals "
                  "on linear error axes, including negative bounds and negative paired effects. "
                  "Lines only connect recorded checkpoints. Stratified IID is same-seed FedAvg. " + STATS_NOTE,
                  experiments)


def final_forest(axis, experiment):
    positions = np.arange(len(ALPHAS))
    offsets = np.linspace(-0.13, 0.13, len(experiment.seeds))
    for y, alpha in zip(positions, ALPHAS):
        values = 100 * experiment.values("drift", alpha)
        mean, half = mean_ci95(values)
        color = ALPHA_COLORS[alpha]
        axis.scatter(values, y+offsets, color=color, s=9, alpha=0.45, edgecolors="none", zorder=2)
        axis.errorbar(mean, y, xerr=half, fmt=ALPHA_MARKERS[alpha], color=color,
                      markersize=4.3, markerfacecolor="white", markeredgewidth=0.9,
                      elinewidth=1.1, capsize=2, zorder=4)
    axis.axvline(0, color="#73797F", lw=0.7, linestyle="--", zorder=0)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"α = {a:g}" for a in ALPHAS])
    axis.set_ylim(3.5, -0.5)
    axis.set_xlabel("Excess error at round 1000 (pp)")
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, min_n_ticks=3))
    axis.grid(False, axis="y")
    axis.margins(x=0.12)


def plot_drift_curves(experiments, exporter, name):
    fig, axes = grid(len(experiments), 2)
    for row, e in enumerate(experiments):
        panel(axes[row, 0], 2*row, e.label)
        draw_curve(axes[row, 0], e, "drift")
        axes[row, 0].set_ylabel("Excess error (pp)")
        panel(axes[row, 1], 2*row+1, "Paired final effects")
        final_forest(axes[row, 1], e)
    alpha_legend(fig, ALPHAS)
    exporter.save(fig, name, "Predictive costs of non-IID are task- and training-stage-dependent",
                  "Left: mean paired excess relative L₂ error with pointwise 95% Student-t confidence "
                  "bands. Right: all five seed-level paired effects at the prespecified final round "
                  "1000, with means and 95% Student-t confidence intervals. Positive values denote "
                  "higher error than stratified IID, negative values denote lower error. Values are "
                  "100 times an error difference, not a relative percentage increase. No significance "
                  "tests, multiplicity-adjusted claims, or universal monotonic degradation are implied.", experiments)


def draw_alpha_summary(axis, experiment, metric="drift", *, color=None, marker="o", offset=0, label=None):
    scale = 100 if metric == "drift" else 1
    stats = [experiment.stats(metric, a) for a in ALPHAS]
    means = scale * np.asarray([s[0] for s in stats])
    half = scale * np.asarray([s[1] for s in stats])
    axis.errorbar(np.arange(4)+offset, means, yerr=half, color=color, marker=marker,
                  markerfacecolor="white", markeredgewidth=0.8, markersize=4,
                  lw=1, elinewidth=0.75, capsize=2, label=label, zorder=3)
    axis.set_xticks(np.arange(4))
    axis.set_xticklabels([f"{a:g}" for a in ALPHAS])
    axis.set_xlabel("Dirichlet concentration α")
    axis.margins(y=0.15, x=0.1)
    if metric == "drift":
        axis.axhline(0, color="#73797F", lw=0.65, linestyle="--", zorder=0)


def plot_seed_effects(experiments, exporter, name):
    nrows, ncols = (1, 3) if len(experiments) == 3 else (2, 2)
    fig, axes = grid(nrows, ncols, height_mm=85 if nrows == 1 else 130)
    for i, (e, axis) in enumerate(zip(experiments, axes.flat)):
        panel(axis, i, e.label)
        for seed in e.seeds:
            values = [100 * number(e.records[(a, seed, FINAL_ROUND)], "drift") for a in ALPHAS]
            axis.plot(np.arange(4), values, color="#ADB4BA", marker=".", lw=0.65, markersize=3, zorder=1)
        for x, a in enumerate(ALPHAS):
            mean, half = e.stats("drift", a)
            axis.errorbar(x, 100*mean, yerr=100*half, color=ALPHA_COLORS[a], fmt=ALPHA_MARKERS[a],
                          markerfacecolor="white", markersize=4.2, capsize=2, elinewidth=0.9, zorder=3)
        axis.axhline(0, color="#73797F", linestyle="--", lw=0.7)
        axis.set_xticks(np.arange(4))
        axis.set_xticklabels([f"{a:g}" for a in ALPHAS])
        axis.set_xlabel("Dirichlet concentration α")
        axis.set_ylabel("Final excess error (pp)")
        axis.margins(x=0.12, y=0.15)
    exporter.save(fig, name, "Seed-level effects reveal non-monotonicity and uncertainty",
                  "All seed-level round-1000 paired excess errors are displayed. Grey lines connect "
                  "the four repeated α conditions within the same seed; they are not additional "
                  "independent replicates. Colored points and error bars show means and unadjusted "
                  "95% Student-t confidence intervals across the five seeds.", experiments)


def absolute_comparison(axis, experiments, labels):
    for group, e in enumerate(experiments):
        for alpha, dx, marker in ((None, -0.13, "o"), (0.01, 0.13, "D")):
            values = e.values("federated_error", alpha)
            mean, half = mean_ci95(values)
            color = SETTING_COLORS[group]
            jitter = np.linspace(-0.045, 0.045, len(values))
            axis.scatter(group+dx+jitter, values, edgecolors=color,
                         facecolors="white" if alpha is None else color,
                         linewidths=0.5, marker=marker, s=7, alpha=0.45, zorder=1)
            axis.errorbar(group+dx, mean, yerr=half, fmt=marker, color=color,
                          markerfacecolor="white" if alpha is None else color,
                          markersize=4.1, elinewidth=0.8, capsize=2, zorder=3)
    axis.set_xticks(range(len(experiments)))
    axis.set_xticklabels(labels)
    axis.set_xlim(-0.48, len(experiments)-0.52)
    axis.set_ylabel(r"Final relative $L_2$ error")
    axis.margins(y=0.17)


def plot_ablation_family(formal, comparison, labels, exporter, name):
    fig, axes = grid(2, 3, height_mm=126)
    selected = []
    for col, task in enumerate(FORMAL_ORDER):
        paired = [formal[task], comparison[task]]
        if paired[0].seeds != paired[1].seeds:
            raise ValueError(f"Mismatched seeds in {name}/{task}")
        selected.extend(paired)
        panel(axes[0, col], col, paired[0].label)
        absolute_comparison(axes[0, col], paired, labels)
        panel(axes[1, col], 3+col)
        for i, e in enumerate(paired):
            draw_alpha_summary(axes[1, col], e, color=SETTING_COLORS[i], marker=SETTING_MARKERS[i],
                               offset=(-0.065 if i == 0 else 0.065))
        axes[1, col].set_ylabel("Final excess error (pp)")
    # Color denotes the setting in both rows. In the top row only, shape/fill
    # denotes IID versus non-IID; do not give these markers a competing meaning.
    handles = [Line2D([], [], color=SETTING_COLORS[i], label=label)
               for i, label in enumerate(labels)]
    handles.extend([
        Line2D([], [], color=IID_COLOR, linestyle="none", marker="o",
               markerfacecolor="white", markersize=4, label="IID (top)"),
        Line2D([], [], color=IID_COLOR, linestyle="none", marker="D",
               markersize=4, label="α = 0.01 (top)"),
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.997), ncol=4,
               columnspacing=1.2, handlelength=1.5, frameon=False)
    setting_note = (
        "Adam uses cosine decay (0.001 to 0.0001); SGD uses cosine decay "
        "(0.01 to 0.001) and Nesterov momentum 0.9. This compares optimizer/hyperparameter "
        "configurations rather than isolated optimizer names."
        if name == "ablation_optimizer" else
        "The comparison uses 10 and 20 clients. Fixed local steps and rounds do "
        "not equalize total compute across client counts."
    )
    exporter.save(fig, name, "Ablations distinguish absolute predictive quality from excess error",
                  "Top: absolute final prediction errors for stratified IID (open circles) and "
                  "α = 0.01 (filled diamonds), including all seed-level observations. Bottom: "
                  "same-seed non-IID minus each setting's own IID error for all four α values. "
                  "Means and 95% Student-t intervals use five seeds. " + setting_note, selected)


def plot_physical_regime(experiments, exporter):
    from plot_viscosity_examples import plot_viscosity_figure
    return plot_viscosity_figure(experiments, exporter)


def plot_mechanisms(experiments, exporter, name):
    fig, axes = grid(len(experiments), 3, height_mm=29+42*len(experiments))
    # The caption specifies initialization versus round 1000. Short metric
    # labels keep the enlarged proof-size typography clear of panel letters.
    metrics = (("gradient", "Gradient heterogeneity"),
               ("iid_weight_divergence", "Parameter deviation"), ("drift", "Excess error (pp)"))
    for row, e in enumerate(experiments):
        for col, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, col]
            panel(axis, 3*row+col, e.label if col == 0 else "")
            for seed in e.seeds:
                x = [number(e.diagnostics[(a, seed)], "distance") for a in ALPHAS]
                y = [(100 * number(e.records[(a, seed, FINAL_ROUND)], "drift") if metric == "drift"
                      else number(e.diagnostics[(a, seed)], metric)) for a in ALPHAS]
                axis.plot(x, y, color="#BCC2C7", lw=0.55, alpha=0.6, zorder=1)
            for alpha in ALPHAS:
                x = e.values("distance", alpha)
                y = e.values(metric, alpha) * (100 if metric == "drift" else 1)
                axis.scatter(x, y, s=17, marker=ALPHA_MARKERS[alpha], facecolors="none",
                             edgecolors=ALPHA_COLORS[alpha], linewidths=0.8, alpha=0.8, zorder=3)
            if metric == "drift":
                axis.axhline(0, color="#73797F", lw=0.65, linestyle="--")
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Solution-bin distance")
            axis.margins(x=0.12, y=0.17)
    alpha_legend(fig, ALPHAS, markers_only=True)
    exporter.save(fig, name, "Distribution shifts, gradient heterogeneity and prediction costs are distinct",
                  "Columns show normalized gradient variance at common initialization, final "
                  "relative parameter deviation from the same-seed IID model, and paired final "
                  "excess error against solution-bin transport distance. Every seed–α observation "
                  "is shown; grey lines connect repeated conditions within each seed. These are raw "
                  "observations, not seed means, so no aggregate error bars are added. Pooled "
                  "correlation annotations treating 20 points as independent are deliberately omitted. "
                  "Distance and parameter deviation are not bounded by one. Parameter deviation "
                  "does not establish prediction degradation or causal mediation.", experiments)


def plot_distance_overview(experiments, exporter):
    fig = plt.figure(figsize=(WIDTH_MM/25.4, 109/25.4))
    layout = fig.add_gridspec(1, 3, width_ratios=(4, 0.12, 1.65), wspace=0.50,
                             left=0.26, right=0.977, bottom=0.14, top=0.84)
    axis, color_axis, q_axis = [fig.add_subplot(layout[0, col]) for col in range(3)]
    panel(axis, 0, "Realized solution-bin distance")
    panel(q_axis, 1, "Quantization error")
    means, halves = np.asarray([[e.stats("distance", a) for a in ALPHAS] for e in experiments]).transpose(2, 0, 1)
    image = axis.imshow(means, cmap="Blues", vmin=0, aspect="auto", interpolation="nearest")
    axis.grid(False)
    axis.set_xticks(np.arange(4))
    axis.set_xticklabels([f"{a:g}" for a in ALPHAS])
    axis.set_xlabel("Dirichlet concentration α")
    axis.set_yticks(np.arange(len(experiments)))
    axis.set_yticklabels([e.label for e in experiments])
    axis.tick_params(length=0)
    for row in range(len(experiments)):
        for col in range(4):
            color = "white" if means[row, col] / np.max(means) > 0.6 else "#20262B"
            axis.text(col, row, f"{means[row,col]:.2f}\n± {halves[row,col]:.2f}",
                      fontsize=6.6, color=color, ha="center", va="center", linespacing=1.1)
    fig.colorbar(image, cax=color_axis)
    color_axis.tick_params(labelsize=6.5, length=2)
    for row, e in enumerate(experiments):
        values = e.values("response_quantization_error", ALPHAS[0])
        mean, half = mean_ci95(values)
        if e.suite == "public":
            # Public codebooks are fitted once on the fixed dataset/clustering seed.
            q_axis.plot(mean, row, "o", color=IID_COLOR, markerfacecolor="white", markersize=4)
        else:
            q_axis.errorbar(mean, row, xerr=half, fmt="o", color=TASK_COLORS[e.task],
                            markersize=4, capsize=2, elinewidth=0.9)
    q_axis.set_ylim(len(experiments)-0.5, -0.5)
    q_axis.set_yticks(np.arange(len(experiments)))
    q_axis.set_yticklabels([])
    q_axis.set_xlabel("Normalized residual")
    q_axis.grid(False, axis="y")
    q_axis.set_xlim(0, max(0.9, q_axis.get_xlim()[1]))
    q_axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    exporter.save(fig, "solution_distance_overview", "Realized severity and discretization fidelity differ across tasks",
                  "Left: means and 95% Student-t CI half-widths of realized distances across five "
                  "seeds. The common color scale does not make an identical α equivalent in "
                  "severity across PDEs. Right: normalized average within-bin residual; controlled "
                  "tasks refit the codebook for each seed and show its CI. Open symbols for public "
                  "tasks denote one fixed data split and clustering seed, not five independent "
                  "codebook fits. All tasks and α values are retained.", experiments, exclude_axes=[color_axis])


def plot_bin_distributions(experiment, exporter):
    settings = (None, *ALPHAS)
    if REPRESENTATIVE_SEED not in experiment.seeds:
        raise ValueError("The fixed representative seed 42 is not available")
    matrices = [bin_counts((experiment.references[(REPRESENTATIVE_SEED, 0)] if a is None
                            else experiment.diagnostics[(a, REPRESENTATIVE_SEED)])["client_bin_counts"]) for a in settings]
    fig, axes = grid(2, 5, height_mm=114)
    fig.subplots_adjust(left=0.07, right=0.87, top=0.82, bottom=0.14, wspace=0.20, hspace=0.64)
    maximum = max(int(m.sum(axis=1).max()) for m in matrices)
    for col, (alpha, counts) in enumerate(zip(settings, matrices)):
        axis, sample_axis = axes[0, col], axes[1, col]
        label = "Stratified IID" if alpha is None else f"α = {alpha:g}"
        panel(axis, col, label)
        shares = counts / counts.sum(axis=1, keepdims=True)
        image = axis.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        axis.grid(False)
        axis.set_xticks([0, 4, 9])
        axis.set_xticklabels([1, 5, 10])
        axis.set_yticks([0, 4, 9])
        axis.set_yticklabels([1, 5, 10] if col == 0 else [])
        axis.set_xlabel("Solution bin")
        axis.tick_params(length=0)
        if col == 0:
            axis.set_ylabel("Client")
        panel(sample_axis, 5+col)
        sample_axis.bar(np.arange(1, len(counts)+1), counts.sum(axis=1), width=0.75,
                        color=IID_COLOR if alpha is None else ALPHA_COLORS[alpha], zorder=2)
        sample_axis.axhline(counts.sum()/len(counts), color="#7D858D", lw=0.65, linestyle="--")
        sample_axis.set_ylim(0, 1.12*maximum)
        sample_axis.set_xticks([1, 5, 10])
        sample_axis.set_xlabel("Client")
        if col == 0:
            sample_axis.set_ylabel("Samples")
        else:
            sample_axis.set_yticklabels([])
    cax = fig.add_axes([0.895, 0.57, 0.012, 0.25])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Within-client proportion", fontsize=7)
    fig.text(0.47, 0.94, f"Burgers solution-bin allocation (fixed illustrative seed {REPRESENTATIVE_SEED})",
             ha="center", fontsize=8)
    exporter.save(fig, "client_bin_distributions", "The protocol changes both solution-bin mixtures and client sizes",
                  "All ten clients and ten solution bins are shown for stratified IID and all "
                  "four α values, using the fixed illustrative seed 42. Heatmap rows are "
                  "normalized separately within each client and share a fixed 0–1 color scale. "
                  "Client and bin identities are not reordered. Counts are provided in Source "
                  "Data rather than unreadably small cell annotations. The bottom panels use "
                  "one shared sample-count scale. No CI is assigned to this single partition; "
                  "across-seed variability is displayed separately.", [experiment], exclude_axes=[cax])


def plot_partition_statistics(formal, public, exporter):
    fig, axes = grid(3, 2, height_mm=153)
    metrics = (("distance", "Solution-bin distance"), ("client_samples_cv", "Sample-count CV"),
               ("partition_target_l1", r"Dirichlet realization $L_1$"))
    for col, experiments in enumerate((formal, public)):
        for row, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, col]
            panel(axis, row*2+col, ("Controlled tasks" if col == 0 else "Public tasks") if row == 0 else "")
            for i, e in enumerate(experiments):
                draw_alpha_summary(axis, e, metric, color=TASK_COLORS[e.task], marker=SETTING_MARKERS[i],
                                   offset=np.linspace(-0.10, 0.10, len(experiments))[i], label=e.label)
            axis.set_ylabel(ylabel)
        handles, labels = axes[0, col].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.315 if col == 0 else 0.77, 0.997),
                   ncol=1 if col == 0 else 2, fontsize=6.8, frameon=False,
                   handlelength=1.3, handletextpad=0.3, columnspacing=0.7)
    # Legends occupy a dedicated strip; axes remain a regular aligned grid.
    fig.subplots_adjust(top=0.83)
    exporter.save(fig, "partition_statistics", "Nominal α does not uniquely determine realized heterogeneity",
                  "Rows show realized solution-bin distance, client sample-count coefficient "
                  "of variation and deviation of realized versus target bin proportions. "
                  "Columns separate controlled and public tasks. Every α and seed is retained. "
                  "The minimum-client-size repair and finite sample allocation can make extreme "
                  "α non-monotonic in realized severity. " + STATS_NOTE, [*formal, *public])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/pdf")
    parser.add_argument("--svg", action="store_true", help="Also export editable SVG")
    parser.add_argument("--previews", action="store_true", help="Write PNG previews under output/qa, never into training results")
    parser.add_argument("--only-formal-curves", action="store_true",
                        help="Regenerate only controlled-task curves with training-sample panels")
    parser.add_argument("--only-public-curves", action="store_true",
                        help="Regenerate only public-task curves with training-sample panels")
    parser.add_argument("--only-physical-regime", action="store_true",
                        help="Regenerate only the viscosity figure with matched training responses")
    return parser.parse_args()


def main():
    args = parse_args()
    if sum((args.only_formal_curves, args.only_public_curves, args.only_physical_regime)) > 1:
        raise ValueError("Choose only one single-figure option.")
    configure_matplotlib()
    if args.only_physical_regime:
        from plot_viscosity_examples import export_viscosity_bundle
        export_viscosity_bundle(viscosity_experiments(args.results_root), args)
        return
    if args.only_public_curves:
        from plot_public_training_examples import export_public_bundle
        export_public_bundle(ordered(discover_suite(args.results_root, "public"), PUBLIC_ORDER), args)
        return
    formal_map = discover_suite(args.results_root, "formal")
    if args.only_formal_curves:
        from plot_formal_training_examples import export_formal_bundle
        export_formal_bundle(ordered(formal_map, FORMAL_ORDER), args)
        return
    public_map = discover_suite(args.results_root, "public")
    client_map = discover_suite(args.results_root, "ablation_clients")
    optimizer_map = discover_suite(args.results_root, "ablation_optimizer", "_lr0p01_m0p9_nesterov_")
    formal, public = ordered(formal_map, FORMAL_ORDER), ordered(public_map, PUBLIC_ORDER)
    clients, optimizers = ordered(client_map, FORMAL_ORDER), ordered(optimizer_map, FORMAL_ORDER)
    viscosity = viscosity_experiments(args.results_root)
    experiments = [*formal, *clients, *optimizers, *viscosity, *public]
    if any(len(e.seeds) != 5 for e in experiments):
        raise ValueError("This figure contract expects five seeds; revise labels before changing the protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validated {len(experiments)} completed experiments; preserving all original result files.")
    source = export_source_data(experiments, args.output_dir / "source_data", args.results_root)
    exporter = Exporter(args.output_dir, svg=args.svg, previews=args.previews)
    for suite, selected in (("formal", formal), ("public", public)):
        plot_error_curves(selected, exporter, f"{suite}_test_error_curves")
        plot_drift_curves(selected, exporter, f"{suite}_drift_curves")
        plot_mechanisms(selected, exporter, f"{suite}_mechanisms")
        plot_seed_effects(selected, exporter, f"{suite}_seed_effects")
    plot_ablation_family(formal_map, client_map, ("K = 10", "K = 20"), exporter, "ablation_clients")
    plot_ablation_family(formal_map, optimizer_map, ("Adam", "SGD + Nesterov"), exporter, "ablation_optimizer")
    plot_physical_regime(viscosity, exporter)
    plot_distance_overview([*formal, *public], exporter)
    plot_bin_distributions(formal_map["burgers"], exporter)
    plot_partition_statistics(formal, public, exporter)
    manifest = dict(source, figures=exporter.figures, final_round=FINAL_ROUND,
                    statistics=dict(unit="seed", repeated_conditions="alpha and round within each seed",
                                    interval="two-sided pointwise Student-t 95% CI", multiplicity_adjusted=False),
                    data_policy=dict(excluded_observations=0, round_zero="preserved in source data; displayed training curves start at round 1",
                                     illustrative_partition_seed=REPRESENTATIVE_SEED,
                                     convergence_right_panel="same-seed paired excess error at rounds 200, 500, 1000",
                                     convergence_left_scale="logarithmic communication round and linear relative L2 error",
                                     captions="external figure_captions.md only; no embedded bottom captions",
                                     shared_control="formal Burgers equals viscosity 0.1; not independent replication"),
                    software=dict(matplotlib=matplotlib.__version__, numpy=np.__version__))
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    captions = "# Figure captions and statistical conventions\n\n" + "\n\n".join(
        f"## {f['name']}\n\n{f['caption']}" for f in exporter.figures)
    (args.output_dir / "figure_captions.md").write_text(captions+"\n", encoding="utf-8")
    print(f"Generated {len(exporter.figures)} PDFs, source data, captions and reproducibility manifest.")


if __name__ == "__main__":
    main()
