#!/usr/bin/env python3
"""Paired system simulation varying published PQC operation profiles.

The experiment preserves ML-KEM-768 byte sizes and independently scales the
published client operation energy or operation latency. It is an operating
profile sensitivity, not a hardware measurement or learning-accuracy run.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from femnist_sim import OUT, FEMNISTNet
from sensitivity import energy_budget, run_one

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/raw/sensitivity/pqc_cost_sensitivity.csv"
LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)
SEEDS = tuple(20261011 + 1009 * rep for rep in range(5))
POLICIES = ("utility_blind", "cross_layer")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest_path = OUT / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("The audited full-writer campaign manifest is required")
    manifest = json.loads(manifest_path.read_text())
    clients = manifest["optimization_train"]["clients"]
    rows = []
    rounds = 50
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    fixed_budget, _ = energy_budget(clients, model_bytes, 10.0, 100.0, .85, 50)
    for axis in ("energy", "latency"):
        for level in LEVELS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                rate_trace = rng.uniform(10.0, 100.0, size=(rounds, len(clients)))
                order_rng = np.random.default_rng(seed + 104729)
                orders = np.stack([order_rng.permutation(len(clients)) for _ in range(rounds)])
                for policy in POLICIES:
                    rows.append(run_one(
                        clients, rate_trace, orders, policy, 8, 5.0, .85,
                        (10.0, 100.0), seed, f"kem_{axis}_{level:g}",
                        kem_energy_multiplier=level if axis == "energy" else 1.0,
                        kem_latency_multiplier=level if axis == "latency" else 1.0,
                        energy_budget_override_j=fixed_budget,
                    ))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "experiment": "paired ML-KEM operation-profile sensitivity",
        "input_manifest": str(manifest_path.relative_to(ROOT)),
        "input_manifest_sha256": sha256(manifest_path),
        "source_script_sha256": sha256(Path(__file__)),
        "source_sensitivity_sha256": sha256(Path(__file__).with_name("sensitivity.py")),
        "clients": len(clients), "rounds_per_run": rounds,
        "fixed_nominal_energy_budget_j": fixed_budget,
        "independent_seeds": list(SEEDS), "policies": list(POLICIES),
        "levels": list(LEVELS),
        "energy_axis": "multiplies client key-generation and decapsulation energy and server encapsulation energy; bytes and latency fixed",
        "latency_axis": "multiplies client key-generation and decapsulation latency; operation energy and bytes fixed",
        "controlled_inputs": "paired rate matrices, random client order, full optimization writer population, fixed 10-100 Mb/s scenario, 5 s deadline, fixed nominal 0.85 energy budget in joules across KEM multipliers, and 8-round session cap",
        "interpretation": "system-level sensitivity only; no retraining or accuracy claim",
        "rows": len(rows),
    }
    (OUTPUT.parent / "pqc_cost_sensitivity_config.json").write_text(
        json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(rows)} paired runs to {OUTPUT}")


if __name__ == "__main__":
    main()
