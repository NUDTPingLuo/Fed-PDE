"""Rebuild manuscript source data; check allocation laws and local-Adam bounds.
Run with Python, NumPy and SciPy. No training or experiment-file edits.
"""
from pathlib import Path
import argparse
import time
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.paper_figure_data import (ALPHAS, FORMAL_ORDER, PUBLIC_ORDER,
    discover_suite, export_source_data, ordered, viscosity_experiments)
from check_adam_theory import adam_checks
from util.sb_cfo_figure_data import load_studies
from plot_sb_cfo_figures import source_data as export_sb_cfo


def theory_checks():
    rng = np.random.default_rng(20261013)
    K, sizes, replicates = 5, np.array([18, 27, 35]), 40000
    N = sizes.sum()
    pi = sizes/N
    checks = []
    for alpha in (0.05, 0.5, 10.0):
        R = rng.dirichlet(np.full(K, alpha), size=(replicates, len(sizes)))
        raw = K*pi[None, :, None]*R
        H = np.mean(np.sum((raw-pi[None, :, None])**2, axis=1), axis=1)
        expected = (K-1)/(K*alpha+1)*sum(pi*pi)
        counts = np.array([[rng.multinomial(int(n), p) for n, p in zip(sizes, row)] for row in R])
        empirical = K*counts/N
        Hm = np.mean(np.sum((empirical-pi[None, :, None])**2, axis=1), axis=1)
        expected_m = (K-1)/(K*alpha+1)*(sum(pi*pi)+K*alpha/N)
        for name, values, exact in (("population", H, expected), ("finite_multinomial", Hm, expected_m)):
            se = values.std(ddof=1)/np.sqrt(replicates)
            assert abs(values.mean()-exact) < 6*se, (name, alpha)
            checks.append(dict(check=name, alpha=alpha, empirical=float(values.mean()), exact=float(exact), monte_carlo_se=float(se)))
    wide_mass_cases = 0
    for trial in range(1000):
        concentration = (20., 1., .1, .02)[trial % 4]
        R = rng.dirichlet(np.ones(K)*concentration, size=len(pi)).T
        u = K*R*pi
        s = u.sum(axis=1)
        # The lemma concerns nonempty clients; guard floating-point underflow.
        while s.min() <= 0:
            R = rng.dirichlet(np.ones(K)*concentration, size=len(pi)).T
            u = K*R*pi
            s = u.sum(axis=1)
        delta = max(abs(s-1))
        rho = np.sqrt(np.mean((s-1)**2))
        q, p = u/s[:, None], s/K
        Hu = np.mean(np.sum((u-pi)**2, axis=1))
        Hq = np.sum(p*np.sum((q-pi)**2, axis=1))
        lo = max(np.sqrt(Hu)-rho*np.linalg.norm(pi), 0)**2/s.max()
        hi = (np.sqrt(Hu)+rho*np.linalg.norm(pi))**2/s.min()
        assert lo-1e-12 <= Hq <= hi+1e-12
        if delta < 1:
            old_lo = max(np.sqrt(Hu)-delta*np.linalg.norm(pi), 0)**2/(1+delta)
            old_hi = (np.sqrt(Hu)+delta*np.linalg.norm(pi))**2/(1-delta)
            assert old_lo <= lo+1e-12 and hi <= old_hi+1e-12
        else:
            wide_mass_cases += 1
    assert wide_mass_cases > 0
    checks.append(dict(check="mass_normalization_sandwich", trials=1000,
                       outside_old_delta_condition=wide_mass_cases, passed=True))
    checks.extend(adam_checks())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output/reproducibility')
    parser.add_argument('--skip-theory', action='store_true',
                        help='Fast recorded-result demo; omit Monte Carlo/theory checks')
    args = parser.parse_args()
    started = time.perf_counter()
    root = ROOT/"results"
    formal = ordered(discover_suite(root, "formal"), FORMAL_ORDER)
    public = ordered(discover_suite(root, "public"), PUBLIC_ORDER)
    clients = ordered(discover_suite(root, "ablation_clients"), FORMAL_ORDER)
    optimizer = ordered(discover_suite(root, "ablation_optimizer"), FORMAL_ORDER)
    all_experiments = formal+public+clients+optimizer+viscosity_experiments(root)
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    provenance = export_source_data(all_experiments, destination, ROOT)
    provenance["theory_checks"] = [] if args.skip_theory else theory_checks()
    provenance["theory_checks_executed"] = not args.skip_theory
    sb_studies = load_studies(ROOT/"results/sb_cfo_cosine_lr0p001")
    sb_destination = destination/"sb_cfo"
    sb_destination.mkdir(exist_ok=True)
    export_sb_cfo(sb_studies, sb_destination)
    provenance["sb_cfo"] = dict(rows=sum(len(s.rows) for s in sb_studies),
                                source_sha256={p: h for s in sb_studies for p, h in s.hashes.items()})
    (destination/"audit.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    labels = {"antiderivative": "Antiderivative", "burgers": r"Burgers ($\nu=0.1$)", "diffusion_reaction": "Diffusion--reaction", "darcy": r"Darcy ($\beta=0.01$)", "shallow_water": "Shallow water", "cfd_cylinder": "Cylinder flow", "openfwi": "CurveFault-B"}
    rows = []
    for e in formal+public:
        iid, _ = e.stats("federated_error", None)
        fed, _ = e.stats("federated_error", .01)
        mean, half = e.stats("drift", .01)
        rows.append(f"{labels[e.task]} & {100*iid:.3f} & {100*fed:.3f} & {100*mean:.3f} [{100*(mean-half):.3f}, {100*(mean+half):.3f}] " + r"\\")
    table = [r"\begin{tabular}{lrrl}", r"\toprule",
        r"Task & IID error (\%) & Non-IID error (\%) & Excess error [95\% CI] (pp)\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (destination/"final_results_rows.tex").write_text("% Generated by tools/audit_manuscript.py.\n"+"\n".join(table)+"\n", encoding="utf-8")
    summary = []
    for e in all_experiments:
        values = {str(a): {m: e.stats(m, a) for m in ("federated_error", "drift", "distance", "gradient", "iid_weight_divergence")} for a in ALPHAS}
        summary.append(dict(experiment=e.identifier, iid=e.stats("federated_error", None), values=values))
    (destination/"final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(dict(experiments=len(all_experiments), source_rows=provenance["row_counts"], theory_checks=len(provenance["theory_checks"]), sb_cfo_rows=sum(len(s.rows) for s in sb_studies), elapsed_seconds=round(time.perf_counter()-started, 3)), indent=2))


if __name__ == "__main__":
    main()
