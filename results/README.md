# Recorded results

These CSV/JSON files are the archived experiment outputs used by the paper.
They contain observations and metadata, not raw public fields or model weights.
Training code and numerical observations are unchanged by the submission packaging.

## Layout and manuscript mapping

| Directory | Experiment | Main figure outputs |
| --- | --- | --- |
| `formal/` | Three controlled tasks, ten clients | `formal_test_error_curves`, `formal_mechanisms`, `client_bin_distributions` |
| `public/` | Darcy, shallow water, cylinder flow, CurveFault-B | `public_test_error_curves`, `public_mechanisms` |
| `ablation_clients/` | Twenty-client controlled runs | `ablation_clients` |
| `ablation_optimizer/` | Nesterov SGD configuration | `ablation_optimizer` |
| `ablation_viscosity/` | Burgers viscosity configurations | `ablation_physical_regime` |
| `sb_cfo_cosine_lr0p001/` | Paired FedAvg/SB-CFO, Burgers and Darcy | `sb_cfo_main`, `sb_cfo_diagnostics` |

The viscosity 0.1 comparison reuses the principal Burgers control. It is not an
additional independent experiment. Public tasks share one fixed data split and
response codebook across training/partition seeds.

## Benchmark CSV families

- `*_alpha_round_seed_raw.csv`: one task/concentration/seed/checkpoint observation.
- `*_alpha_seed_raw.csv`: seed-level partition, gradient and final-parameter diagnostics.
- `*_iid.csv`: the paired stratified reference trajectories.
- `*_alpha.csv` and other summary CSVs: aggregated results derived from observations.
- `*_metadata.json`: resolved settings, completeness markers and partition information.

| Field | Meaning / units |
| --- | --- |
| `severity_value` | Dirichlet concentration alpha |
| `seed`, `round` | Integer replicate identifier and communication checkpoint |
| `iid_error`, `federated_error` | Relative L2 error as a fraction; multiply by 100 for percent |
| `drift` | `federated_error - iid_error` as a fraction; multiply by 100 for percentage points |
| `distance` | Realized mean pairwise response-bin transport distance |
| `gradient` | Normalized sample-weighted gradient variance at initialization |
| `iid_weight_divergence` | Final relative parameter deviation from the paired IID model |
| `client_bin_counts`, `iid_client_bin_counts` | Client rows separated by `|`, bin counts separated by `;` |
| `response_quantization_error` | Mean normalized distance from response to assigned bin centroid |

## SB-CFO tables

`rounds.csv` contains observations indexed by method, condition, seed and round;
`summary.csv` contains derived summaries. `metadata.json` includes source/runtime
fingerprints, dataset settings and partition information.

`test_error` is relative L2 error as a fraction. Fields ending in `_pp` are
percentage points. `elapsed_seconds` and `prototype_seconds` are cumulative
seconds; communication fields are logical uncompressed tensor bytes. Example
visit counters measure optimizer, prototype or diagnostic sample visits.
`local_update_variance` describes the preceding round. Gradient variances before
and after correction are computed at the same current model. Some round-zero
update/prototype diagnostics are unavailable and may be blank; the plotting
contracts explicitly select the checkpoints appropriate to those quantities.

## Aggregation

The five seeds are 0, 1, 42, 999 and 2026. Excess errors and algorithm comparisons
are calculated within seed, then averaged. Confidence intervals are pointwise,
two-sided 95% Student-t intervals with four degrees of freedom. Positive variance
ratios use geometric means and back-transformed log-space intervals. Sharing a
seed or IID trajectory across concentrations does not create independent replicates.

Run `python tools/audit_manuscript.py --skip-theory` from the repository root to
validate pairing, coverage and count contracts and regenerate the final table.
No observations are discarded during this export. Source file hashes accompany
the generated analysis outputs.
