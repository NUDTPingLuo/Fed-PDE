"""Plot public training examples and learning curves."""
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
import numpy as np

from real_world.datasets import (_chosen_indices, _cylinder_case, _cylinder_windows,
                                 _cylinder_anchors, _annotation_pairs)
from util.paper_figure_data import ALPHAS, PUBLIC_ORDER, export_source_data
from util.paper_figure_style import Exporter, alpha_legend

ROOT = Path(__file__).resolve().parent
TITLES = ('Darcy (β = 0.01)', 'Shallow water', 'Cylinder flow', 'CurveFault-B')
UNIT_SOURCES = {
    'pdebench_axes': 'https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py',
    'cfd_cylinder': 'https://arxiv.org/html/2310.05963v2#S3.SS5',
    'cylinder_axes': 'https://raw.githubusercontent.com/luo-yining/CFDBench/main/src/dataset/cylinder.py',
    'openfwi': 'https://arxiv.org/html/2111.02926v3#A4',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extent(x, y):
    dx, dy = (x[-1]-x[0])/(len(x)-1), (y[-1]-y[0])/(len(y)-1)
    return [float(x[0]-dx/2), float(x[-1]+dx/2),
            float(y[0]-dy/2), float(y[-1]+dy/2)]


def extract_samples(experiments, output):
    """Read only the selected case using the exact archived split/order rules."""
    source = output / 'sample_source_data'
    source.mkdir(parents=True, exist_ok=True)
    samples, provenance = {}, []
    for e in experiments:
        cfg = e.config['dataset']
        task = e.task
        record = dict(task=task, training_index_zero_based=0, config=cfg,
                      metadata_sha256=sha(e.metadata_path))
        if task in ('darcy', 'shallow_water'):
            path = ROOT / cfg['path']
            stride = cfg['spatial_stride']
            with h5py.File(path, 'r') as f:
                if task == 'darcy':
                    index = int(_chosen_indices(len(f['nu']), cfg['n_train'], cfg['n_test'], cfg['split_seed'])[0])
                    field = np.asarray(f['tensor'][index, -1, ::stride, ::stride], dtype=np.float32).T
                    x, y = np.array(f['x-coordinate'])[::stride], np.array(f['y-coordinate'])[::stride]
                    record.update(source_index=index, dataset_key='tensor',
                                  target_frame=int(f['tensor'].shape[1]-1))
                    label = 'Pressure\n(dataset units)'
                else:
                    keys = [k for k in sorted(f) if isinstance(f[k], h5py.Group) and 'data' in f[k]]
                    assert len(keys) == e.metadata['benchmark']['dataset']['available_trajectories']
                    index = int(_chosen_indices(len(keys), cfg['n_train'], cfg['n_test'], cfg['split_seed'])[0])
                    key, frame = keys[index], cfg['target_time_indices'][-1]
                    field = np.asarray(f[key]['data'][frame, ::stride, ::stride, 0], dtype=np.float32).T
                    x, y = np.array(f[key]['grid/x'])[::stride], np.array(f[key]['grid/y'])[::stride]
                    record.update(source_index=index, dataset_key=key+'/data', target_frame=frame,
                                  target_time=float(f[key]['grid/t'][frame]))
                    label = 'Height\n(dataset units)'
            sample = dict(field=field, extent=extent(x,y), origin='lower', label=label,
                          xlabel='x (dataset coord.)' if task=='darcy' else 'x (m)',
                          ylabel='y (dataset coord.)' if task=='darcy' else 'y (m)', cmap='viridis',
                          source_axis_order='x,y', display_axis_order='y,x',
                          display_transform='transpose source spatial axes once',
                          coordinate_units='dataset coordinates' if task=='darcy' else 'm',
                          field_units='dataset units')
            if task == 'shallow_water':
                sample['display_title'] = f'Shallow water (t = {record["target_time"]:g} s)'
                record['unit_evidence'] = 'Embedded HDF5 simulation config specifies dx/dy in m and t/dt in s; no explicit height unit attribute.'
            else:
                record['unit_evidence'] = 'No physical unit attribute found for pressure or coordinates; original dataset scale retained.'
            record['source_file'] = str(path.relative_to(ROOT))
            record['axes_evidence'] = dict(url=UNIT_SOURCES['pdebench_axes'],
                verified_blob_sha='d5bc69db79639ab2dc5e2c7fa3ca286eade8ea55',
                convention='Raw spatial axes x,y; official grids use meshgrid indexing=ij')
        elif task == 'cfd_cylinder':
            cases = []
            for subset in cfg['subsets']:
                for case in sorted((ROOT / cfg['path'] / subset).glob('case*')):
                    item = _cylinder_case(case, subset)
                    if item is not None:
                        cases.append(item)
            assert len(cases) == e.metadata['benchmark']['dataset']['available_cases']
            index = int(_chosen_indices(len(cases), cfg['n_train'], cfg['n_test'], cfg['split_seed'])[0])
            chosen = cases[index]
            # Existing loader constructs this case's windows in original order.
            inputs, targets = _cylinder_windows([chosen], cfg, cfg['spatial_stride'])
            u, v = targets[0, -2], targets[0, -1]
            # CFDBench stores velocity as [time, height(y), width(x)].
            # Preserve that YX order for imshow; do not transpose the field.
            field = np.hypot(u, v)
            p = chosen['parameters']
            display_x = np.linspace(p['x_min'],p['x_max'],field.shape[1],dtype=np.float32)
            display_y = np.linspace(p['y_min'],p['y_max'],field.shape[0],dtype=np.float32)
            gx, gy = np.meshgrid(display_x,display_y,indexing='xy')
            display_mask = gx**2+gy**2 <= p['radius']**2
            sample = dict(field=field, extent=[p['x_min'],p['x_max'],p['y_min'],p['y_max']],
                          origin='lower', label='Speed\n(m/s)', xlabel='x (m)', ylabel='y (m)', cmap='viridis',
                          radius=p['radius'], solid_mask=display_mask,
                          velocity_x=u.copy(), velocity_y=v.copy(),
                          source_axis_order='y,x', display_axis_order='y,x',
                          display_transform='none; raw speed is already image YX',
                          coordinate_units='m',field_units='m/s')
            length = np.load(chosen['u_path'], mmap_mode='r').shape[0]
            anchor = int(_cylinder_anchors(length, cfg['history_frames'], cfg['history_stride'],
                         cfg['target_offsets'], cfg['window_stride'], cfg.get('max_windows_per_case'))[0])
            record.update(source_index=index, source_case=str(Path(chosen['case']).relative_to(ROOT)),
                          window_index=0, anchor_frame=anchor,
                          target_frame=anchor+cfg['target_offsets'][-1],
                          derivation='sqrt(u**2+v**2) from last two raw target channels',
                          case_parameters=p, case_metadata_sha256=sha(Path(chosen['case'])/'case.json'))
            sample['display_title'] = f'Cylinder flow (frame {record["target_frame"]})'
            record.update(unit_evidence=UNIT_SOURCES['cfd_cylinder'],
                          axes_evidence=UNIT_SOURCES['cylinder_axes'],
                          time_label='zero-based frame only; no physical time inferred',
                          mask_scope='Plot-only YX geometry from case domain and radius; training mask unchanged',
                          display_mask_equals_existing_mask_transposed=bool(np.array_equal(display_mask,inputs[0,-5].T.astype(bool))))
        else:
            annotation = ROOT / cfg['train_annotation']
            data_path, model_path = _annotation_pairs(annotation, ROOT / cfg['data_root'])[0]
            velocity = np.load(model_path, mmap_mode='r')
            field = np.array(velocity[0,0], dtype=np.float32)
            sample = dict(field=field, extent=[-0.5,field.shape[1]-0.5,field.shape[0]-0.5,-0.5],
                          origin='upper', label='Velocity\n(m/s)', xlabel='x index', ylabel='Depth index', cmap='viridis',
                          source_axis_order='depth,x',display_axis_order='depth,x',
                          display_transform='none; depth rows increase downward',
                          coordinate_units='zero-based grid index',field_units='m/s')
            record.update(source_file=str(model_path.relative_to(ROOT)), source_index=0,
                          annotation_sha256=sha(annotation), annotation_row=0,
                          unit_evidence=UNIT_SOURCES['openfwi'])
        if not np.isfinite(sample['field']).all():
            raise ValueError(f'Non-finite target: {task}')
        np.savez_compressed(source / f'{task}.npz', **sample)
        record.update(sample_npz_sha256=sha(source / f'{task}.npz'),
                      displayed_array_sha256=hashlib.sha256(sample['field'].tobytes()).hexdigest(),
                      shape=list(sample['field'].shape), minimum=float(sample['field'].min()),
                      maximum=float(sample['field'].max()),
                      source_axis_order=sample['source_axis_order'],
                      displayed_array_axis_order=sample['display_axis_order'],
                      display_transform=sample['display_transform'],
                      coordinate_units=sample['coordinate_units'],field_units=sample['field_units'])
        provenance.append(record)
        samples[task] = sample
        print(f'{task}: first training sample; range {record["minimum"]:.5g} .. {record["maximum"]:.5g}', flush=True)
    (output / 'sample_provenance.json').write_text(json.dumps(dict(
        rule='First training sample before viewing; final configured target horizon where applicable.',
        preprocessing='Raw targets before training normalization; no smoothing, clipping or rescaling.',
        cylinder='Speed derived from velocity components; geometric solid shown in grey; equal spatial aspect.',
        orientation='PDEBench XY fields transposed to image YX; CFDBench raw YX velocity fields preserved; OpenFWI depth rows preserved.',
        display_revision='2026-10-27. Cylinder illustration orientation and unit/coordinate labels updated; no training data or results changed.',
        loader_sha256=sha(ROOT/'real_world/datasets.py'), samples=provenance),indent=2),encoding='utf-8')
    return samples


def draw_sample(fig, ax, sample, cax):
    field = sample['field']
    if 'solid_mask' in sample:
        field = np.ma.array(field, mask=sample['solid_mask'])
    cmap = plt.get_cmap(sample['cmap']).copy()
    cmap.set_bad('#B8BDC4')
    image = ax.imshow(field, extent=sample['extent'], origin=sample['origin'],
                      aspect='equal', interpolation='nearest', cmap=cmap)
    if 'radius' in sample:
        ax.add_patch(Circle((0,0),sample['radius'],facecolor='#B8BDC4',edgecolor='#555555',lw=0.45))
    cb = fig.colorbar(image,cax=cax)
    lo,hi=float(field.min()),float(field.max())
    cb.set_ticks([lo,hi])
    cb.set_ticklabels([f'{v:.0f}' if abs(v)>=1000 else f'{v:.3g}' for v in (lo,hi)])
    cb.outline.set_linewidth(0.4)
    cax.set_title(sample['label'],fontsize=6.5,pad=5.5)
    cax.tick_params(length=1.5,pad=1,labelsize=7)
    ax.set_xlabel(sample['xlabel'],fontsize=7.5,labelpad=1)
    ax.set_ylabel(sample['ylabel'],fontsize=7.5,labelpad=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
    if 'radius' in sample:
        # Compact decimal labels fit the narrow prompt column without clipping.
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, position: f'{value:g}'.replace('-0.', '−.').replace('0.', '.')))
    ax.tick_params(length=2,pad=2,labelsize=7)


def plot_public_curves(experiments, exporter, name):
    from plot_paper_figures import draw_curve, STATS_NOTE
    if tuple(e.task for e in experiments) != PUBLIC_ORDER:
        raise ValueError('Expected the four public tasks in manuscript order.')
    samples = extract_samples(experiments, exporter.output)
    fig=plt.figure(figsize=(183/25.4,218/25.4))
    excluded,rows,columns=[],[],[[],[],[]]
    for row,e in enumerate(experiments):
        y=169-50*row
        axes=[fig.add_axes([x/183,y/218,w/183,30/218]) for x,w in ((10,30),(70,47),(133,47))]
        ids=[f'{chr(97+row)}-{kind}' for kind in ('sample','learning','effect')]
        for col,(ax,identifier) in enumerate(zip(axes,ids)):
            ax.set_gid(identifier);columns[col].append(identifier)
        rows.append([ids[1],ids[2]])
        if e.task!='cfd_cylinder':
            rows.append([ids[0],ids[1]])
        fig.text(2/183,(y+39)/218,chr(97+row),fontsize=10,fontweight='bold',va='center')
        for x,title in ((10,samples[e.task].get('display_title',TITLES[row])),(70,'Test-error learning curves'),(133,'Paired excess error')):
            fig.text(x/183,(y+39)/218,title,fontsize=8,va='center')
        cax=fig.add_axes([42/183,y/218,1.5/183,30/218]);excluded.append(cax)
        draw_sample(fig,axes[0],samples[e.task],cax)
        for col,metric in ((1,'federated_error'),(2,'drift')):
            ax=axes[col]; ax.grid(axis='y',color='#DDE1E5',zorder=0);ax.set_axisbelow(True)
            ax.tick_params(direction='out',length=2.5,width=0.6,pad=2)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            formatter=ScalarFormatter(useOffset=False,useMathText=False);formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            draw_curve(ax,e,metric,iid=col==1,late=col==2)
            ax.set_ylabel(r'Relative $L_2$ error' if col==1 else 'Excess error (pp)',fontsize=9,labelpad=2)
            ax.set_xlabel('' if row<3 else ('Communication round (log)' if col==1 else 'Communication round'),fontsize=8)
    alpha_legend(fig,ALPHAS,iid=True)
    exporter.save(fig,name,'Public-task samples, convergence and paired prediction costs',
                  'Rows a–d: Darcy, shallow water, cylinder flow, CurveFault-B. '
                  'Left: raw reference field from first training sample; last configured target '
                  'horizon for time-dependent tasks. Cylinder shows speed magnitude with the solid in grey. '
                  'Middle: test error over communication rounds (log scale). Right: within-seed '
                  'non-IID minus IID error at rounds 200, 500 and 1000 (percentage points). '+STATS_NOTE,
                  experiments,exclude_axes=excluded,
                  alignment_options=dict(row_groups=rows,column_groups=columns,
                      exemptions=[dict(panels=columns[0],checks=['vertical-gutter'],
                          reason='Cylinder retains its physical aspect ratio, so sample-image vertical gutters intentionally differ; column edges/widths and all quantitative grids remain checked.')]))
    # Standalone dataset prompts, also copied into the LaTeX figures directory.
    for task,sample in samples.items():
        fig=plt.figure(figsize=(54/25.4,54/25.4))
        ax=fig.add_axes([10/54,10/54,29/54,29/54])
        cax=fig.add_axes([41/54,10/54,1.5/54,29/54])
        draw_sample(fig,ax,sample,cax)
        fig.text(0.5,49/54,sample.get('display_title',TITLES[PUBLIC_ORDER.index(task)]),fontsize=8,ha='center',va='center')
        stem=exporter.output/f'training_sample_{task}'
        fig.savefig(stem.with_suffix('.pdf'))
        fig.savefig(stem.with_suffix('.svg'))
        fig.savefig(stem.with_suffix('.png'),dpi=600)
        plt.close(fig)




def export_public_bundle(experiments,args):
    if any(len(e.seeds)!=5 for e in experiments):
        raise ValueError('Expected five seeds.')
    source=export_source_data(experiments,args.output_dir/'source_data',args.results_root)
    exporter=Exporter(args.output_dir,svg=args.svg,previews=args.previews)
    # The static validator cannot follow imported style or exporter functions.
    modules=(ROOT/'util/paper_figure_data.py',ROOT/'util/paper_figure_style.py',
             ROOT/'plot_paper_figures.py',Path(__file__))
    snapshot='from __future__ import annotations\n'+'\n'.join(
        p.read_text(encoding='utf-8').replace('from __future__ import annotations\n','')
        for p in modules)
    (exporter.qa/'plotting_source_preflight.py').write_text(snapshot,encoding='utf-8')
    plot_public_curves(experiments,exporter,'public_test_error_curves')
    (args.output_dir/'figure_manifest.json').write_text(json.dumps(dict(source,figures=exporter.figures),indent=2),encoding='utf-8')
