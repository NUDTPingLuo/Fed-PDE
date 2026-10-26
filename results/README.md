# Recorded results

Archived CSV/JSON outputs used in the paper; raw public fields and model weights
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
