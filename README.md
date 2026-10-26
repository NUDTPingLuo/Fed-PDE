# Fed-PDE

Code and recorded results for **Solution-space heterogeneity shapes federated
learning dynamics across partial differential equations**. Includes solution-space
Dirichlet partitioning, FedAvg and SB-CFO for seven operator-learning tasks.

**License:** [MIT](LICENSE) · **Archived release:** [v1.0](https://github.com/NUDTPingLuo/Fed-PDE/releases/tag/v1.0) · **DOI:** [10.5281/zenodo.23662112](https://doi.org/10.5281/zenodo.23662112)

## Install

Use Python 3.10+ and a PyTorch build compatible with your CPU/CUDA environment.
Run all commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

For analysis without PyTorch, use Python 3.12 and `requirements-analysis.txt`.
Recorded SB-CFO runs used Python 3.10.22, PyTorch 2.7.1+cu128 and NumPy 1.26.4.

## Train

```bash
python main.py --study all --device cuda:0
python real_world_main.py --task all --device cuda:0
python sb_cfo_main.py --task all --device cuda:0 --output-dir output/retrained_sb_cfo
```

Use `--device cpu` for CPU execution, `--dry-run` to inspect plans, or `--smoke`
for a small integration run. Controlled data are generated automatically;
public data setup is in [data/README.md](data/README.md). Configurations are in
`config/`. Main runs use five seeds, 1,000 rounds and five local steps.

Completed matching results are skipped. For new controlled/public runs, change
the output paths in the configurations; keep archived results for comparison.
SB-CFO checks source/runtime fingerprints, so use a separate output directory.

## Reproduce results

```bash
python tools/audit_manuscript.py --skip-theory
python plot_sb_cfo_figures.py
```

The audit recreates the seven-task final table and source data from 17 benchmark
configurations and two SB-CFO studies. Outputs go to `output/reproducibility/`;
plots go to `output/pdf/`. See [results/README.md](results/README.md) for fields
and units. The audit takes a few seconds on a CPU and needs no raw-data download.

Generate controlled-task figures, including representative training samples:

```bash
python plot_training_sample_thumbnails.py
python plot_paper_figures.py --only-formal-curves
python plot_paper_figures.py --only-physical-regime
```

After preparing the public datasets in [data/README.md](data/README.md), run
`python plot_paper_figures.py` for all benchmark figures, or add
`--only-public-curves` for the public-task curves alone. The full figure set
requires the controlled samples generated above and the public datasets.

## Tests

```bash
python -m unittest discover -s tests
python tools/audit_manuscript.py
python tools/check_notation_geometry.py
```

## Structure

- `config/`: experiment settings
- `operator_tasks/`: controlled data and numerical solvers
- `real_world/`: public-data loaders, models and training
- `sb_cfo/`: bin-conditioned gradient correction
- `util/`: shared training, partitioning, statistics and plotting utilities
- `results/`: recorded CSV/JSON results and checksums
- `tools/`, `tests/`: reproduction and verification

## Citation

Use [CITATION.cff](CITATION.cff) to cite the software. The DOI identifies the
archived `v1.0` release (software version 1.0.0). Third-party datasets and
dependencies retain their own licenses.
