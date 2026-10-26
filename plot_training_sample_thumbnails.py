"""Reconstruct fixed formal-training examples and export task thumbnails.

No training is performed. Metadata, seed rule and numerical solver are those of
the formal experiments. Burgers must be solved as the original complete batch:
its shared CFL time step depends on the maximum speed across the whole pool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from operator_tasks.tasks import OperatorTask

ROOT = Path(__file__).resolve().parent
TASKS = ('antiderivative', 'burgers', 'diffusion_reaction')
TITLES = ('Antiderivative', r'Burgers ($\nu=0.1$)', 'Diffusion–reaction')
MM = 1 / 25.4
WIDTH, HEIGHT = 54, 48


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(seed, index, output):
    examples, records = [], []
    for name in TASKS:
        path = ROOT / 'results' / 'formal' / name / f'{name}_formal_k10_s5_r1000_metadata.json'
        metadata = json.loads(path.read_text(encoding='utf-8'))
        cfg = metadata['config']
        if seed not in cfg['seeds'] or not 0 <= index < cfg['sampling_pool_size']:
            raise ValueError('Choose an experiment seed and a valid training sample index.')
        task = OperatorTask(name, **{k: cfg[k] for k in (
            'n_sensors', 'output_grid', 'condition_scale', 'n_modes',
            'diffusion', 'reaction', 'viscosity')})
        rng_seed = cfg['seed_base'] + 10000 + seed
        inputs = task.sample_conditions(cfg['sampling_pool_size'], rng_seed)
        print(f'Reconstructing {name}: {len(inputs)} training responses', flush=True)
        responses = task.solve_batch(inputs)
        centered = responses - responses.mean(axis=0)
        normalizer = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        recorded = metadata['benchmark']['solution_binning'][str(seed)]['response_normalizer']
        if not np.isclose(normalizer, recorded, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f'{name}: reconstructed pool disagrees with run metadata.')
        response = responses[index].copy()
        if name != 'antiderivative':
            response = response.reshape(len(task.space_grid), len(task.time_grid))
        sample = dict(task=name, sensor_x=task.sensor_grid, input=inputs[index].copy(),
                      x=task.space_grid, response=response)
        if name != 'antiderivative':
            sample['t'] = task.time_grid
        examples.append(sample)
        np.savez_compressed(output / 'source_data' / f'{name}.npz', **sample)
        np.savetxt(output / 'source_data' / f'{name}_input.csv',
                   np.column_stack([sample['sensor_x'], sample['input']]),
                   delimiter=',', header='x,input', comments='')
        if name == 'antiderivative':
            target = np.column_stack([sample['x'], response])
            header = 'x,u'
        else:
            xx, tt = np.meshgrid(sample['x'], sample['t'], indexing='ij')
            target = np.column_stack([xx.ravel(), tt.ravel(), response.ravel()])
            header = 'x,t,u'
        np.savetxt(output / 'source_data' / f'{name}_reference.csv', target,
                   delimiter=',', header=header, comments='')
        records.append(dict(task=name, experiment_seed=seed, training_rng_seed=rng_seed,
                            sample_index_zero_based=index, training_pool_size=len(inputs),
                            config=cfg, metadata_path=str(path.relative_to(ROOT)),
                            metadata_sha256=digest(path),
                            recorded_response_normalizer=recorded,
                            reconstructed_response_normalizer=normalizer,
                            normalizer_absolute_difference=abs(normalizer-recorded),
                            normalizer_check_passed=True,
                            sample_npz_sha256=digest(output / 'source_data' / f'{name}.npz')))
        print(f'  normalizer {normalizer:.15g}; recorded {recorded:.15g}', flush=True)
    return examples, records


def draw_panel(fig, sample, number, count):
    # Fixed physical rectangles keep comparable plotting areas equally sized.
    offset = number * WIDTH
    def rect(x, y, w, h):
        return [(offset+x)/(count*WIDTH), y/HEIGHT, w/(count*WIDTH), h/HEIGHT]
    ax = fig.add_axes(rect(9, 10, 31, 27))
    ax.set_gid(chr(97 + number))
    fig.text((offset+27)/(count*WIDTH), 44/HEIGHT, TITLES[TASKS.index(sample['task'])],
             ha='center', va='center', fontsize=8.5)
    if sample['task'] == 'antiderivative':
        ax.plot(sample['sensor_x'], sample['input'], color='#0072B2', lw=1.05,
                label=r'$f(x)$')
        ax.plot(sample['x'], sample['response'], color='#D55E00', lw=1.1,
                ls='--', label=r'$u(x)$')
        ax.set_xlabel(r'$x$', labelpad=1.5)
        ax.set_ylabel('Value', labelpad=1.5)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.015), ncol=2,
                  frameon=False, borderaxespad=0, handlelength=1.4,
                  handletextpad=0.4, columnspacing=0.8, fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=0.1)
    else:
        field = sample['response']
        limit = float(np.max(np.abs(field)))
        # x is the first field dimension; time is horizontal. No data smoothing.
        artist = ax.imshow(field, origin='lower', extent=(0, 1, 0, 1),
                           aspect='auto', interpolation='nearest', cmap='RdBu_r',
                           norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        cax = fig.add_axes(rect(42, 10, 1.5, 27), label=f'colorbar_{number}')
        colorbar = fig.colorbar(artist, cax=cax)
        colorbar.set_ticks([-limit, 0, limit])
        colorbar.set_ticklabels([f'{-limit:.2f}', '0', f'{limit:.2f}'])
        colorbar.ax.tick_params(length=1.5, pad=1, labelsize=7)
        colorbar.outline.set_linewidth(0.4)
        cax.set_title(r'$u$', fontsize=8, pad=3)
        ax.set_xlabel(r'$t$', labelpad=1.5)
        ax.set_ylabel(r'$x$', labelpad=1.5)
        ax.set_yticks([0, 0.5, 1], labels=['0', '0.5', '1'])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], labels=['0', '0.5', '1'])
    ax.tick_params(length=2, width=0.5, pad=2)
    return ax


def export(fig, stem):
    # Preserve the fixed canvas; tight cropping would change insertion scaling.
    fig.savefig(stem.with_suffix('.pdf'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=600, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'output' / 'training_sample_thumbnails')
    parser.add_argument('--qa-tools', type=Path,
                        default=None, help='Optional directory containing audit_panel_alignment.py')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for child in ('source_data', 'qa'):
        (output / child).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'axes.linewidth': 0.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none'})
    examples, records = reconstruct(args.seed, args.sample_index, output)
    for sample in examples:
        fig = plt.figure(figsize=(WIDTH*MM, HEIGHT*MM))
        draw_panel(fig, sample, 0, 1)
        export(fig, output / sample['task'])
    fig = plt.figure(figsize=(3*WIDTH*MM, HEIGHT*MM))
    axes = [draw_panel(fig, sample, i, 3) for i, sample in enumerate(examples)]
    from util.paper_figure_style import require_basic_alignment
    require_basic_alignment(fig, axes, output / 'qa' / 'overview.alignment.json',
                            alignment_options={'row_groups': [['a', 'b', 'c']]})
    if args.qa_tools is not None:
        sys.path.insert(0, str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=['a', 'b', 'c'], row_groups=[['a', 'b', 'c']],
            json_out=output / 'qa' / 'overview.external_alignment.json', strict=True)
    export(fig, output / 'overview')
    provenance = dict(
        selection='Fixed index chosen before viewing; one example per training pool.',
        meaning='Reconstructed numerical training targets, not model predictions.',
        solver_sha256=digest(ROOT / 'operator_tasks' / 'tasks.py'),
        script_sha256=digest(__file__), numpy_version=np.__version__,
        matplotlib_version=matplotlib.__version__,
        batch_rule='Solve entire original training pool; shared CFL step in Burgers.',
        verification='Pool response normalizer agrees with archived metadata to 1e-12; '
                     'this is an aggregate reproducibility check, not a bytewise original-sample comparison.',
        figures=dict(width_mm=WIDTH, height_mm=HEIGHT, png_dpi=600,
                     color_scale='Separate symmetric zero-centered raw-value ranges for the two PDEs.'),
        samples=records)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Exported three sample panels and overview to {output}', flush=True)


if __name__ == '__main__':
    main()
