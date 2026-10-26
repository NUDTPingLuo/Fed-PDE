# Reproduce the recorded analysis and training

## Analysis environment

The submission-package check on 26 October 2026 used Windows 11 Home
(build 10.0.26200), an Intel Core i9-12900H CPU and Python 3.12.14.
No GPU or PyTorch was used for the recorded-result demo or mathematical checks.
The tested analysis dependencies are pinned in `requirements-analysis.txt`:

| Package | Version |
| --- | --- |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| h5py | 3.16.0 |

Create a Python 3.12 environment and install these dependencies with
`python -m pip install -r requirements-analysis.txt`. Installation time depends
on wheel availability and download speed; a clean-install timing was not
measured in this audit. Linux/macOS paths are supported by the scripts, but
these operating systems were not validated in this packaging run.

## Quick demo

From the repository root:

```bash
python tools/audit_manuscript.py --skip-theory
```

The bundled CSV/JSON files are the demo inputs. This command reads all 17
benchmark configurations and the two paired algorithm studies, validates their
contracts, and writes:

```text
output/reproducibility/
├── audit.json
├── final_results_rows.tex
├── final_metrics.json
├── round_observations.csv
├── seed_diagnostics.csv
├── round_summary.csv
└── sb_cfo/                 # paired effects, variance/cost data and provenance
```

Expected totals are 4,250 benchmark checkpoint observations, 340 seed-level
diagnostics, 1,530 summary rows and 400 SB-CFO observations. The final task
effects at alpha = 0.01, round 1,000 should round to:

| Task | Excess error (percentage points) | 95% CI |
| --- | ---: | --- |
| Antiderivative | 0.097 | [-1.017, 1.211] |
| Burgers, viscosity 0.1 | 0.741 | [0.239, 1.243] |
| Diffusion–reaction | 0.180 | [-0.397, 0.758] |
| Darcy | 9.900 | [5.161, 14.639] |
| Shallow water | 0.012 | [0.006, 0.018] |
| Cylinder flow | 0.608 | [0.349, 0.867] |
| CurveFault-B | -0.253 | [-0.528, 0.022] |

The command took about **3.4 seconds wall time** in one local run, including
Python/import startup; its internally timed analysis reported 1.179 seconds.
These measurements describe this CPU and dataset bundle, not a hardware-independent
runtime guarantee. The generated table was checked against the manuscript table.

## Mathematical checks

```bash
python tools/audit_manuscript.py
python tools/check_adam_theory.py
python tools/check_notation_geometry.py
```

The full audit took about **4.2 seconds wall time** in the same local check
(1.953 seconds internally timed). Its 13 check groups cover allocation moments,
mass normalization, prototype centering, Adam moment weights, dispersion,
aggregation and conditional descent. The geometry script passed four further
groups: weighted centering, integer rounding, restricted bin-gradient bounds
and transport bounds. Its wall time was about **1.4 seconds**.

These scripts are the manuscript's numerical-check implementations, with paths
adapted to this repository. The optional command
`python tools/check_notation_geometry.py --manuscript-dir /path/to/Fed_PDE_NC`
also checks LaTeX notation if manuscript sources are available separately.
Numerical checks complement the stated assumptions and proofs; they do not
establish convergence of the recorded neural-network runs.

The earlier upload validation passed 53 existing tests for plans, controlled
operators, partitions, dataset adapters and statistical/plotting contracts.
The three training entry points passed `--dry-run`, and the three controlled
training-sample panels were reconstructed. Full PyTorch tests and 1,000-round
GPU training were not rerun in the packaging environment.

## Training environment and full experiments

Use `requirements.txt` and the training commands in the main README. Public
experiments require the original datasets and exact configuration paths in
`data/README.md`. Configurations specify seeds, data splits, preprocessing,
architecture, local updates, learning-rate schedules and checkpoints.

The recorded SB-CFO metadata report **Python 3.10.22, PyTorch 2.7.1+cu128 and
NumPy 1.26.4**. These are observed versions for those recorded studies, not a
complete environment lock for every benchmark. The CUDA build suffix does not
identify a GPU model or driver. A complete historical OS/GPU/driver/dependency
inventory and clean-install timing are not available from this packaging audit.
Choose a PyTorch build compatible with the machine used for rerunning training.

The baseline runs and ablations use five seeds, 1,000 rounds and five local
steps. `--smoke` runs are short integration checks and do not reproduce the
reported final metrics. New runs should use a separate output location; SB-CFO
checks runtime/source fingerprints and can reject reuse of archived results
after an environment or source change. Never overwrite the reference bundle
when performing a new reproduction.

Recorded seed-level elapsed times are available in SB-CFO `rounds.csv`. They
include evaluation and diagnostics in a sequential simulator; they are not
benchmarks of parallel federated deployment or predictions of a new machine's
training time. No trained checkpoints are included; the training commands are
the route for producing new models.

## Result integrity and figures

`results/SHA256SUMS` lists hashes of the 91 archived CSV/JSON files. Original
training code, experiment configurations and those result files are unchanged
by this submission preparation. See `results/README.md` for units and figure
mapping, and the main README for plotting commands. The sample-enriched public
figures require raw public data; quantitative plots and the table demo do not.

The source-data exporter preserves the seed observations and paired IID
references. Confidence intervals are descriptive, pointwise Student-t intervals.
No best-test-checkpoint selection or observations filtered by effect sign are
introduced by this release preparation.
