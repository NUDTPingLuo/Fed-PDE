"""Three-row controlled-task figure: own training sample, learning curve, paired effect.

Contract: identify each operator, then show convergence and the same-seed late
effect. Image plate + quantitative grid, 183 x 168 mm, native Python vectors and
raw reference heatmaps, unchanged five-seed mean / Student-t 95% CI. Row letters
a/b/c identify entire tasks. Narrow sample axes intentionally reserve space for
colorbars; learning/effect axes share widths. QA compares every adjacent pair's
row edges and all three columns, excluding auxiliary colorbar axes.
"""
from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

from util.paper_figure_data import ALPHAS, FORMAL_ORDER, export_source_data
from util.paper_figure_style import Exporter, alpha_legend

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / 'output/training_sample_thumbnails'
TITLES = ('Antiderivative', 'Burgers (ν = 0.1)', 'Diffusion–reaction')


def sample_panel(fig, ax, task, y):
    with np.load(SAMPLES / 'source_data' / f'{task}.npz') as sample:
        if task == 'antiderivative':
            ax.plot(sample['sensor_x'], sample['input'], color='#0072B2', lw=1.05, label=r'$f(x)$')
            ax.plot(sample['x'], sample['response'], color='#D55E00', ls='--', lw=1.1, label=r'$u(x)$')
            ax.set_ylabel('Value', labelpad=1)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2,
                      frameon=False, fontsize=7, handlelength=1.2,
                      columnspacing=0.7, handletextpad=0.3, borderaxespad=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_xlabel('x', labelpad=1)
            ax.margins(y=0.1)
            extra = []
        else:
            field = sample['response']
            limit = float(np.max(np.abs(field)))
            artist = ax.imshow(field, origin='lower', extent=(0, 1, 0, 1),
                               aspect='auto', interpolation='nearest', cmap='RdBu_r',
                               norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
            cax = fig.add_axes([42/183, y/168, 1.5/183, 30/168])
            cb = fig.colorbar(artist, cax=cax, ticks=[-limit, 0, limit])
            cb.set_ticklabels([f'{-limit:.2f}', '0', f'{limit:.2f}'])
            cax.tick_params(length=1.5, pad=1, labelsize=7)
            cax.set_title('u', fontsize=8, pad=3)
            cb.outline.set_linewidth(0.4)
            ax.set_xlabel('t', labelpad=1)
            ax.set_ylabel('x', labelpad=1)
            ax.set_yticks([0, 0.5, 1], labels=['0', '0.5', '1'])
            extra = [cax]
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], labels=['0', '0.5', '1'])
    ax.tick_params(length=2, pad=2)
    return extra


def plot_formal_curves(experiments, exporter, name):
    from plot_paper_figures import draw_curve, STATS_NOTE
    if tuple(e.task for e in experiments) != FORMAL_ORDER:
        raise ValueError('Formal rows must use the three archived controlled tasks in order.')
    fig = plt.figure(figsize=(183/25.4, 168/25.4))
    excluded, rows, columns = [], [], [[], [], []]
    for row, e in enumerate(experiments):
        y = 119 - 50*row
        axes = [fig.add_axes([x/183, y/168, w/183, 30/168])
                for x, w in ((10, 30), (70, 47), (133, 47))]
        ids = [f'{chr(97+row)}-{kind}' for kind in ('sample', 'learning', 'effect')]
        for col, (axis, identifier) in enumerate(zip(axes, ids)):
            axis.set_gid(identifier)
            columns[col].append(identifier)
        rows.extend([[ids[0], ids[1]], [ids[1], ids[2]]])
        # One letter labels the whole row, not each of the three plot areas.
        fig.text(2/183, (y+39)/168, chr(97+row), fontsize=10, fontweight='bold', va='center')
        for x, title in ((10, TITLES[row]), (70, 'Test-error learning curves'), (133, 'Paired excess error')):
            fig.text(x/183, (y+39)/168, title, fontsize=8, va='center')
        excluded.extend(sample_panel(fig, axes[0], e.task, y))
        for col, metric in ((1, 'federated_error'), (2, 'drift')):
            axis = axes[col]
            axis.grid(axis='y', color='#DDE1E5', zorder=0)
            axis.set_axisbelow(True)
            axis.tick_params(direction='out', length=2.5, width=0.6, pad=2)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            formatter = ScalarFormatter(useOffset=False, useMathText=False)
            formatter.set_scientific(False)
            axis.yaxis.set_major_formatter(formatter)
            draw_curve(axis, e, metric, iid=(col == 1), late=(col == 2))
            # Keep the L2 subscript above 5 pt after the manuscript's 0.97-linewidth scaling.
            axis.set_ylabel(r'Relative $L_2$ error' if col == 1 else 'Excess error (pp)',
                            labelpad=2, fontsize=9)
            if row < 2:
                axis.set_xlabel('')
            elif col == 1:
                axis.set_xlabel('Communication round (log)', fontsize=8)
            else:
                axis.set_xlabel('Communication round', fontsize=8)
    alpha_legend(fig, ALPHAS, iid=True)
    exporter.save(fig, name, 'Controlled-task examples, convergence and paired prediction costs',
                  'Rows a–c: antiderivative, Burgers (ν=0.1), diffusion–reaction. '
                  'Left: fixed training example (experiment seed 42, index 0); input/integral '
                  'or numerical reference u(x,t), with separate color scales. '
                  'Middle: test-error curves on a logarithmic round axis. Right: within-seed '
                  'non-IID minus IID error at rounds 200, 500 and 1000, in percentage points. '
                  'All original observations and uncertainty are retained; round 0 is kept '
                  'in source data but cannot appear on a log axis. ' + STATS_NOTE,
                  experiments, exclude_axes=excluded, require_panel_labels=False,
                  alignment_options=dict(row_groups=rows, column_groups=columns))


def sync_sample_assets(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for task in FORMAL_ORDER:
        for ext in ('pdf', 'png', 'svg'):
            shutil.copy2(SAMPLES / f'{task}.{ext}', destination / f'training_sample_{task}.{ext}')


def export_formal_bundle(experiments, args):
    if any(len(e.seeds) != 5 for e in experiments):
        raise ValueError('Expected five seeds in every formal experiment.')
    source = export_source_data(experiments, args.output_dir / 'source_data', args.results_root)
    exporter = Exporter(args.output_dir, svg=args.svg, previews=args.previews, qa_tools=args.qa_tools_dir)
    plot_formal_curves(experiments, exporter, 'formal_test_error_curves')
    (args.output_dir / 'figure_manifest.json').write_text(
        json.dumps(dict(source, figures=exporter.figures), indent=2), encoding='utf-8')
    if args.sync_nc:
        destination = ROOT / 'latex/Fed_PDE_NC/figures'
        exporter.sync_nc(destination)
        sync_sample_assets(destination)
