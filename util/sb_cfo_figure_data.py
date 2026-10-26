"""Read-only, seed-paired source data for the completed cosine SB-CFO study."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t

SEEDS = (0, 1, 42, 999, 2026)
ROUNDS = (0, 1, 5, 10, 20, 50, 100, 200, 500, 1000)
METHODS = ("fedavg", "sb_cfo")
CONDITIONS = ("iid", "alpha0.01")
FINAL = 1000


def interval(values, geometric=False):
    """Mean and two-sided 95% Student-t limits; optionally in log space."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite, independent seed observations required")
    if geometric:
        if np.any(x <= 0):
            raise ValueError("Geometric intervals require strictly positive data")
        x = np.log(x)
    m = x.mean()
    h = t.ppf(0.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    result = np.array([m, m-h, m+h])
    return np.exp(result) if geometric else result


@dataclass
class Study:
    task: str
    directory: Path
    rows: list
    metadata: dict
    index: dict
    hashes: dict

    @property
    def identifier(self):
        return "sb_cfo_cosine_lr0p001/" + self.task

    def values(self, method, condition, round_, field):
        values = np.array([float(self.index[(method, condition, s, round_)][field]) for s in SEEDS])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {self.task}/{method}/{condition}/{round_}/{field}")
        return values

    def gap(self, method, round_=FINAL):
        return 100 * (self.values(method, "alpha0.01", round_, "test_error")
                      - self.values(method, "iid", round_, "test_error"))

    def gradient_remaining(self):
        return (self.values("sb_cfo", "alpha0.01", FINAL, "gradient_variance_after")
                / self.values("sb_cfo", "alpha0.01", FINAL, "gradient_variance_before"))

    def cost_ratios(self):
        costs = {}
        for method in METHODS:
            get = lambda field: self.values(method, "alpha0.01", FINAL, field)
            costs[method] = [get("optimizer_examples") + get("prototype_examples"),
                             get("model_communication_bytes") + get("prototype_communication_bytes"),
                             get("elapsed_seconds")]
        return np.array(costs["sb_cfo"]) / np.array(costs["fedavg"])


def load_study(directory, task):
    directory = Path(directory)
    paths = [directory / f for f in ("rounds.csv", "summary.csv", "metadata.json")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata = json.loads(paths[2].read_text(encoding="utf-8-sig"))
    if metadata.get("complete") is not True:
        raise ValueError(f"Incomplete study: {directory}")
    cfg = metadata["specification"]["config"]
    base = metadata['specification']['base']
    batch = (base['training_batch_size'] if task == 'burgers'
             else base['training']['batch_size'])
    if batch != 16:
        raise ValueError(f"Expected batch size 16: {directory}")
    expected = dict(optimizer="adam", lr=0.001, lr_schedule="cosine", min_lr=0.0001,
                    local_steps=5, n_rounds=1000, participation_rate=1.0)
    if any(cfg["training"].get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unexpected training protocol in {directory}")
    if (sorted(cfg["seeds"]) != sorted(SEEDS) or cfg["n_clients"] != 10
            or cfg["n_bins"] != 10 or cfg["alpha_values"] != [0.01]):
        raise ValueError(f"Unexpected partition protocol in {directory}")
    with paths[0].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    index = {}
    for row in rows:
        key = (row["method"], row["condition"], int(row["seed"]), int(row["round"]))
        if key in index:
            raise ValueError(f"Duplicate observation: {key}")
        if row["task"] != task:
            raise ValueError("Task mismatch")
        index[key] = row
    expected_keys = {(m, c, s, r) for m in METHODS for c in CONDITIONS for s in SEEDS for r in ROUNDS}
    if set(index) != expected_keys:
        raise ValueError(f"Missing or extra method/condition/seed/round observations: {directory}")
    study = Study(task, directory, rows, metadata, index, hashes)
    for method in METHODS:
        for condition in CONDITIONS:
            for round_ in ROUNDS:
                err = study.values(method, condition, round_, "test_error")
                base = study.values("fedavg", "iid", round_, "test_error")
                saved = study.values(method, condition, round_, "excess_vs_fedavg_iid_pp")
                if not np.allclose(saved, 100*(err-base), atol=1e-7, rtol=1e-7):
                    raise ValueError("Recorded paired differences do not match raw errors")
            if not np.allclose(study.values(method, condition, 0, "test_error"),
                               study.values("fedavg", "iid", 0, "test_error"), atol=1e-7):
                raise ValueError("Initial models are not seed-paired")
    return study


def load_studies(root):
    root = Path(root)
    return [load_study(root / "burgers/nu0p005/adam", "burgers"),
            load_study(root / "darcy/adam", "darcy")]
