"""Read-only data contracts for reproducible manuscript figures."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import t

ALPHAS = (10.0, 1.0, 0.1, 0.01)
FINAL_ROUND = 1000
FORMAL_ORDER = ("antiderivative", "burgers", "diffusion_reaction")
PUBLIC_ORDER = ("darcy", "shallow_water", "cfd_cylinder", "openfwi")
TASK_LABELS = {
    "antiderivative": "Antiderivative", "burgers": "Burgers",
    "diffusion_reaction": "Diffusion–reaction", "darcy": "Darcy (β = 0.01)",
    "shallow_water": "Shallow water", "cfd_cylinder": "Cylinder flow",
    "openfwi": "OpenFWI CurveFault-B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty result file: {path}")
    return rows


def number(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Missing or invalid metric {key!r}: {row.get(key)!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite metric {key!r}: {value}")
    return value


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    """Mean and two-sided Student-t CI half-width; never discard observations."""
    array = np.asarray(list(values), dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("Confidence intervals require nonempty finite observations")
    if len(array) == 1:
        return float(array[0]), 0.0
    half = t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array))
    return float(array.mean()), float(half)


def bin_counts(value: str) -> np.ndarray:
    rows = [row.split(";") for row in value.split("|")]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Malformed client-by-bin count matrix")
    result = np.asarray(rows, dtype=int)
    if np.any(result < 0) or np.any(result.sum(axis=1) == 0):
        raise ValueError("Bin counts must be nonnegative with nonempty clients")
    return result


def unique_index(rows: list[dict], keys: tuple[str, ...]) -> dict:
    result = {}
    for row in rows:
        key = tuple(number(row, field) for field in keys)
        if key in result:
            raise ValueError(f"Duplicate result key {keys}: {key}")
        result[key] = row
    return result


@dataclass
class Experiment:
    suite: str
    task: str
    path: Path
    static_path: Path
    iid_path: Path
    metadata_path: Path
    config: dict
    metadata: dict
    rows: list[dict]
    static_rows: list[dict]
    iid_rows: list[dict]
    seeds: tuple[int, ...]
    rounds: tuple[int, ...]
    records: dict
    diagnostics: dict
    references: dict

    @property
    def label(self) -> str:
        return TASK_LABELS[self.task]

    @property
    def identifier(self) -> str:
        return f"{self.suite}/{self.path.stem.replace('_alpha_round_seed_raw', '')}"

    @classmethod
    def load(cls, path: Path, suite: str) -> "Experiment":
        prefix = path.name[:-len("_alpha_round_seed_raw.csv")]
        static_path = path.with_name(prefix + "_alpha_seed_raw.csv")
        iid_path = path.with_name(prefix + "_iid.csv")
        metadata_path = path.with_name(prefix + "_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        benchmark = metadata["benchmark"]
        if not benchmark.get("complete"):
            raise ValueError(f"Experiment is not marked complete: {metadata_path}")
        config = benchmark["config"] if "config" in benchmark else metadata["config"]
        seeds = tuple(config.get("seeds", config.get("evaluation", {}).get("seeds", [])))
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid seed configuration: {metadata_path}")
        rows, static_rows, iid_rows = read_csv(path), read_csv(static_path), read_csv(iid_path)
        task = rows[0]["task"]
        rounds = tuple(sorted({int(number(row, "round")) for row in iid_rows}))
        result = cls(
            suite, task, path, static_path, iid_path, metadata_path, config,
            metadata, rows, static_rows, iid_rows, seeds, rounds,
            unique_index(rows, ("severity_value", "seed", "round")),
            unique_index(static_rows, ("severity_value", "seed")),
            unique_index(iid_rows, ("seed", "round")),
        )
        result.validate()
        return result

    def validate(self) -> None:
        training = self.config.get('training', self.config)
        is_sgd = training.get('optimizer') == 'sgd'
        initial_lr, final_lr = (0.01, 0.001) if is_sgd else (0.001, 0.0001)
        if (training.get('lr_schedule') != 'cosine'
                or training.get('lr') != initial_lr or training.get('min_lr') != final_lr):
            raise ValueError(f"Expected cosine {initial_lr} to {final_lr}: {self.path}")
        if self.task in FORMAL_ORDER and self.config.get('training_batch_size') != 16:
            raise ValueError(f"Expected controlled-task batch size 16: {self.path}")
        if is_sgd and (training.get('momentum') != 0.9 or training.get('nesterov') is not True):
            raise ValueError(f"Expected SGD Nesterov momentum 0.9: {self.path}")
        if FINAL_ROUND not in self.rounds:
            raise ValueError(f"Round {FINAL_ROUND} missing: {self.path}")
        expected_rounds = self.config.get(
            "round_checkpoints", self.config.get("evaluation", {}).get("round_checkpoints"))
        if expected_rounds is not None and set(self.rounds) != set(expected_rounds):
            raise ValueError(f"Checkpoint coverage does not match metadata: {self.path}")
        expected = {(a, s, r) for a in ALPHAS for s in self.seeds for r in self.rounds}
        if set(self.records) != expected:
            raise ValueError(f"Incomplete or unexpected alpha/seed/round coverage: {self.path}")
        if set(self.diagnostics) != {(a, s) for a in ALPHAS for s in self.seeds}:
            raise ValueError(f"Incomplete alpha/seed diagnostics: {self.static_path}")
        if set(self.references) != {(s, r) for s in self.seeds for r in self.rounds}:
            raise ValueError(f"Incomplete IID seed/round coverage: {self.iid_path}")
        for (alpha, seed, round_index), row in self.records.items():
            baseline = number(self.references[(seed, round_index)], "test_error")
            if not np.isclose(number(row, "iid_error"), baseline, rtol=1e-7, atol=1e-10):
                raise ValueError(f"Mismatched IID reference: {self.path}, {(alpha, seed, round_index)}")
            difference = number(row, "federated_error") - baseline
            if not np.isclose(number(row, "drift"), difference, rtol=1e-7, atol=1e-10):
                raise ValueError(f"Incorrect paired error difference: {self.path}")
        for seed in self.seeds:
            iid = bin_counts(self.references[(seed, self.rounds[0])]["client_bin_counts"])
            if np.ptp(iid.sum(axis=1)) > 1 or np.any(np.ptp(iid, axis=0) > 1):
                raise ValueError(f"IID allocation is not bin-stratified and size-balanced: {self.iid_path}")
            for alpha in ALPHAS:
                diagnostic = self.diagnostics[(alpha, seed)]
                counts = bin_counts(diagnostic["client_bin_counts"])
                sizes = np.asarray(diagnostic["client_samples"].split(";"), dtype=int)
                if not np.array_equal(counts.sum(axis=1), sizes):
                    raise ValueError(f"Client size/count mismatch: {self.static_path}")
                if not np.array_equal(counts.sum(axis=0), iid.sum(axis=0)):
                    raise ValueError(f"IID and non-IID do not contain the same bin population: {self.static_path}")
                for metric in ("distance", "gradient", "iid_weight_divergence", "client_samples_cv", "partition_target_l1"):
                    number(diagnostic, metric)

    def values(self, metric: str, alpha: float | None, round_index: int = FINAL_ROUND) -> np.ndarray:
        if alpha is None:
            field = "test_error" if metric == "federated_error" else metric
            return np.asarray([number(self.references[(s, round_index)], field) for s in self.seeds])
        if metric in ("drift", "federated_error", "iid_error"):
            return np.asarray([number(self.records[(alpha, s, round_index)], metric) for s in self.seeds])
        return np.asarray([number(self.diagnostics[(alpha, s)], metric) for s in self.seeds])

    def stats(self, metric: str, alpha: float | None, round_index: int = FINAL_ROUND) -> tuple[float, float]:
        return mean_ci95(self.values(metric, alpha, round_index))

    def curve(self, metric: str, alpha: float | None, start: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rounds = np.asarray([r for r in self.rounds if start <= r <= FINAL_ROUND])
        stats = [self.stats(metric, alpha, int(r)) for r in rounds]
        return rounds, np.asarray([s[0] for s in stats]), np.asarray([s[1] for s in stats])


def discover_suite(root: Path, suite: str, fragment: str | None = None) -> dict[str, Experiment]:
    found = {}
    for path in sorted((root / suite).glob("*/*_alpha_round_seed_raw.csv")):
        if "_smoke" in path.name or (fragment and fragment not in path.name):
            continue
        experiment = Experiment.load(path, suite)
        if experiment.task in found:
            raise ValueError(f"Ambiguous result selection for {suite}/{experiment.task}: {path}")
        found[experiment.task] = experiment
    return found


def ordered(experiments: dict[str, Experiment], order: tuple[str, ...]) -> list[Experiment]:
    missing = set(order) - set(experiments)
    if missing:
        raise FileNotFoundError(f"Missing completed task results: {sorted(missing)}")
    return [experiments[task] for task in order]


def viscosity_experiments(root: Path) -> list[Experiment]:
    result = [Experiment.load(path, "ablation_viscosity") for path in
              sorted((root / "ablation_viscosity/burgers").glob("*_alpha_round_seed_raw.csv"))
              if "_smoke" not in path.name]
    result.sort(key=lambda e: float(e.config["viscosity"]), reverse=True)
    if [float(e.config["viscosity"]) for e in result] != [0.1, 0.05, 0.01, 0.005]:
        raise ValueError("Expected exactly the four viscosity settings 0.1, 0.05, 0.01, 0.005")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_source_data(experiments: list[Experiment], output: Path, root: Path) -> dict:
    """Export figure inputs and summaries separately from unmodified results."""
    observations, diagnostics, summaries, provenance = [], [], [], []
    for e in experiments:
        identity = dict(experiment=e.identifier, suite=e.suite)
        observations.extend(dict(**identity, condition="non_iid", **row) for row in e.rows)
        observations.extend(dict(**identity, condition="iid", **row) for row in e.iid_rows)
        diagnostics.extend(dict(**identity, **row) for row in e.static_rows)
        for alpha in (None, *ALPHAS):
            for r in e.rounds:
                for metric in (("federated_error",) if alpha is None else ("federated_error", "drift")):
                    mean, half = e.stats(metric, alpha, r)
                    summaries.append(dict(**identity, task=e.task, condition="iid" if alpha is None else "non_iid",
                                          alpha="" if alpha is None else alpha, round=r, metric=metric,
                                          mean=mean, ci95_low=mean-half, ci95_high=mean+half, n_seeds=len(e.seeds)))
        files = [e.path, e.static_path, e.iid_path, e.metadata_path,
                 e.path.with_name(e.path.name.replace("_alpha_round_seed_raw", "_alpha"))]
        provenance.append(dict(**identity, seeds=list(e.seeds), checkpoints=list(e.rounds),
                               files=[dict(path=p.relative_to(root).as_posix(),
                                           sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
    write_csv(output / "round_observations.csv", observations)
    write_csv(output / "seed_diagnostics.csv", diagnostics)
    write_csv(output / "round_summary.csv", summaries)
    return dict(experiments=provenance, row_counts=dict(
        round_observations=len(observations), seed_diagnostics=len(diagnostics), round_summary=len(summaries)))
