# Recorded results

Current CSV/JSON outputs used in the paper; raw public fields and model weights
are obtained or generated separately. `SHA256SUMS` lists file hashes.

| Directory | Experiment |
| --- | --- |
| `formal/` | Three controlled tasks, ten clients |
| `public/` | Darcy, shallow water, cylinder flow and CurveFault-B |
| `ablation_clients/` | Twenty-client runs |
| `ablation_optimizer/` | Nesterov SGD runs |
| `ablation_viscosity/` | Burgers viscosity study |
| `sb_cfo_cosine_lr0p001/` | Paired FedAvg/SB-CFO comparisons |

Benchmark files contain checkpoint observations (`*_alpha_round_seed_raw.csv`),
seed diagnostics (`*_alpha_seed_raw.csv`), IID references (`*_iid.csv`),
summaries and configuration metadata. SB-CFO uses `rounds.csv`, `summary.csv`
and `metadata.json`.

- Errors are relative L2 fractions; multiply by 100 for percentages.
- `drift` is non-IID minus IID error; multiply by 100 for percentage points.
- Fields ending in `_pp` are already in percentage points.
- `distance` is mean pairwise response-bin transport; `gradient` is normalized
  initial gradient variance; `iid_weight_divergence` is final relative parameter deviation.
- Count matrices use `|` between clients and `;` between bins.
- SB-CFO times are cumulative seconds, communication is tensor bytes, and
  sample-visit counters distinguish optimizer, prototype and diagnostic work.
  Unavailable round-zero update/prototype diagnostics may be blank.

Differences are paired within seed (0, 1, 42, 999, 2026). Intervals are pointwise
95% Student-t intervals; positive variance ratios use log-space intervals.
Concentrations share IID references, public tasks share a fixed split/codebook,
and viscosity 0.1 reuses the main Burgers control.

Run `python tools/audit_manuscript.py --skip-theory` to validate and export the
recorded observations and the final seven-task table.

## Cylinder update (2026-10-27)

The cylinder results replace the earlier runs after aligning the geometry-mask
channel with the native velocity arrays (rows y, columns x). All 25 trajectories
were rerun: five seeds × (stratified IID + four concentrations), 1,000 rounds.
Case splits, windows, response bins and allocation settings are unchanged.
At alpha=0.01, final mean errors are 6.878% (IID) and 7.492% (non-IID);
the paired excess is 0.614 percentage points (95% CI: 0.343–0.885).

This directory contains 72 CSV and 19 JSON files across 17 benchmark
configurations and two SB-CFO studies. The audit exports 4,250 benchmark
checkpoint observations, 340 seed diagnostics, 1,530 summary rows and
400 SB-CFO observations. Files are copied unchanged from the completed runs.
SB-CFO metadata retain original source/runtime fingerprints, so a rerun under
a different code snapshot should use a new output directory.

The viscosity-0.1 condition shares the principal Burgers control. The remaining
16 benchmark configurations are not all independent task replications.
No raw third-party fields, model weights or smoke outputs are included.
