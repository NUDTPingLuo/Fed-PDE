"""Plot Burgers responses and prediction errors across viscosities."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from operator_tasks.tasks import OperatorTask
from util.paper_figure_data import export_source_data
from util.paper_figure_style import (Exporter, SETTING_COLORS, SETTING_MARKERS,
                                    IID_COLOR, ALPHA_COLORS)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'output/viscosity_figure_revision'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_examples(experiments, output):
    """Reconstruct entire training pools to preserve batch-dependent CFL steps."""
    source = output / 'source_data'
    source.mkdir(parents=True, exist_ok=True)
    data_path = source / 'viscosity_training_samples.npz'
    provenance_path = output / 'sample_provenance.json'
    identity = dict(experiment_seed=42, sample_index=0,
                    solver_sha256=sha(ROOT/'operator_tasks/tasks.py'),
                    metadata_sha256=[sha(e.metadata_path) for e in experiments])
    if data_path.exists() and provenance_path.exists():
        record = json.loads(provenance_path.read_text(encoding='utf-8'))
        if record.get('identity') == identity and record.get('data_sha256') == sha(data_path):
            with np.load(data_path) as archive:
                return {key: archive[key] for key in archive.files}
    fields, inputs, records = [], [], []
    for e in experiments:
        cfg = e.config
        task = OperatorTask('burgers', **{key: cfg[key] for key in (
            'n_sensors', 'output_grid', 'condition_scale', 'n_modes',
            'diffusion', 'reaction', 'viscosity')})
        seed = cfg['seed_base'] + 10000 + 42
        conditions = task.sample_conditions(cfg['sampling_pool_size'], seed)
        print(f'Reconstructing viscosity {cfg["viscosity"]}: full {len(conditions)}-sample pool', flush=True)
        responses = task.solve_batch(conditions)
        centered = responses - responses.mean(axis=0)
        normalizer = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        archived = e.metadata['benchmark']['solution_binning']['42']['response_normalizer']
        if not np.isclose(normalizer, archived, rtol=1e-12, atol=1e-12):
            raise ValueError('Reconstructed training pool disagrees with archived run metadata.')
        inputs.append(conditions[0].copy())
        fields.append(responses[0].reshape(len(task.space_grid), len(task.time_grid)).copy())
        records.append(dict(viscosity=cfg['viscosity'], training_rng_seed=seed,
                            pool_size=len(conditions), normalizer=normalizer,
                            archived_normalizer=archived, config=cfg))
    for initial in inputs[1:]:
        np.testing.assert_array_equal(initial, inputs[0])
    samples = dict(fields=np.stack(fields), inputs=np.stack(inputs),
                   x=task.space_grid, t=task.time_grid)
    np.savez_compressed(data_path, **samples)
    provenance_path.write_text(json.dumps(dict(identity=identity, pools=records,
        data_sha256=sha(data_path),
        selection='Fixed first training sample, experiment seed 42, chosen before viewing.',
        reconstruction='Each complete pool is solved using the archived configuration; shared CFL time steps preserved.',
        interpretation='Numerical references; same initial function; all heatmaps share one symmetric raw-value colour scale.',
        verification='Aggregate normalizers match archived runs; not a bytewise comparison with unsaved original labels.'
        ), indent=2), encoding='utf-8')
    return samples


def plot_viscosity_figure(experiments, exporter):
    from plot_paper_figures import draw_alpha_summary
    output = getattr(exporter, 'output', DEFAULT_OUTPUT)
    samples = training_examples(experiments, output)
    fig = plt.figure(figsize=(183/25.4, 142/25.4))
    fig.text(5/183, 138/142, 'a', fontsize=10, fontweight='bold', va='center')
    fig.text(16/183, 138/142, 'Reference solutions from the same training input', fontsize=8.5, va='center')
    norm = TwoSlopeNorm(vmin=-float(np.abs(samples['fields']).max()), vcenter=0,
                        vmax=float(np.abs(samples['fields']).max()))
    heatmaps = []
    for i, (e, field) in enumerate(zip(experiments, samples['fields'])):
        x = 16 + 39*i
        ax = fig.add_axes([x/183, 96/142, 30/183, 29/142])
        ax.set_gid(f'a-{i}')
        image = ax.imshow(field, origin='lower', extent=(0,1,0,1), aspect='auto',
                          interpolation='nearest', cmap='RdBu_r', norm=norm)
        ax.set_title(f'ν = {e.config["viscosity"]:g}', fontsize=8, pad=5)
        ax.set_xticks([0,0.5,1], labels=['0','0.5','1'])
        ax.set_yticks([0,0.5,1], labels=['0','0.5','1'])
        ax.set_xlabel('t', fontsize=8, labelpad=1)
        if i==0:
            ax.set_ylabel('x', fontsize=8, labelpad=1)
        ax.tick_params(length=2,pad=2)
        heatmaps.append(ax)
    cax = fig.add_axes([172/183,96/142,1.8/183,29/142])
    cb = fig.colorbar(image,cax=cax,ticks=[norm.vmin,0,norm.vmax])
    cb.set_ticklabels([f'{norm.vmin:.2f}','0',f'{norm.vmax:.2f}'])
    cax.set_title('u',fontsize=8,pad=5)
    cax.tick_params(length=1.5,pad=1,labelsize=7)
    cb.outline.set_linewidth(0.4)
    left=fig.add_axes([16/183,14/142,66/183,47/142])
    right=fig.add_axes([110/183,14/142,66/183,47/142])
    for ax,letter,x,title in ((left,'b',16,'Paired final excess error'),
                              (right,'c',110,'Absolute final test error')):
        ax.set_gid(letter)
        fig.text((x-11)/183,81/142,letter,fontsize=10,fontweight='bold',va='center')
        fig.text(x/183,81/142,title,fontsize=8.5,va='center')
        ax.tick_params(length=2.5,pad=2)
        ax.grid(axis='y',color='#DDE1E5',zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
    for i,e in enumerate(experiments):
        draw_alpha_summary(left,e,color=SETTING_COLORS[i],marker=SETTING_MARKERS[i],
                           offset=np.linspace(-0.10,0.10,4)[i],label=f'ν = {e.config["viscosity"]:g}')
    left.set_ylabel('Final excess error (pp)',fontsize=9)
    left.legend(loc='lower left',bbox_to_anchor=(0,1.08),ncol=2,borderaxespad=0,
                columnspacing=1.2,handlelength=1.5,frameon=False)
    for alpha,color,marker,label in ((None,IID_COLOR,'o','IID'),(0.01,ALPHA_COLORS[0.01],'D','α = 0.01')):
        stats=[e.stats('federated_error',alpha) for e in experiments]
        right.errorbar(np.arange(4),[s[0] for s in stats],yerr=[s[1] for s in stats],
                       color=color,marker=marker,markerfacecolor='white',markersize=4,capsize=2,label=label)
    right.set_xticks(np.arange(4),labels=[f'{e.config["viscosity"]:g}' for e in experiments])
    right.set_xlabel('Viscosity ν (categorical settings)',fontsize=8.5)
    right.set_ylabel(r'Final relative $L_2$ error',fontsize=9)
    right.legend(loc='lower left',bbox_to_anchor=(0,1.08),ncol=2,borderaxespad=0,
                 frameon=False,columnspacing=1.5,handlelength=1.5)
    exporter.save(fig,'ablation_physical_regime',
        'Burgers viscosity changes solution structure, baseline difficulty and non-IID cost',
        'a, Four numerical training responses to the same initial input (seed 42, first sample) '
        'at ν=0.1,0.05,0.01,0.005, with a shared colour scale. b, Original paired excess errors '
        'for all four concentrations. c, Original absolute IID and α=0.01 errors. '
        'Means and pointwise 95% Student-t intervals use five seeds. '
        'Viscosities are categorical; ν=0.1 reuses the formal control. Round 1000, ten clients.',
        experiments,exclude_axes=[cax],
        alignment_options=dict(row_groups=[[ax.get_gid() for ax in heatmaps],['b','c']]))


def export_viscosity_bundle(experiments,args):
    if len(experiments)!=4 or any(len(e.seeds)!=5 for e in experiments):
        raise ValueError('Expected four viscosity settings with five seeds each.')
    source=export_source_data(experiments,args.output_dir/'source_data',args.results_root)
    exporter=Exporter(args.output_dir,svg=args.svg,previews=args.previews)
    plot_viscosity_figure(experiments,exporter)
    (args.output_dir/'figure_manifest.json').write_text(json.dumps(dict(source,figures=exporter.figures),indent=2),encoding='utf-8')
