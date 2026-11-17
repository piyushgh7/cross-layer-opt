#!/usr/bin/env python3
"""Replay the executed scheduler and quantify selected-data representation."""
from __future__ import annotations

import csv
import hashlib
import json
import pandas as pd
from pathlib import Path

import numpy as np

from femnist_sim import (OUT, CACHE, POLICIES, FEMNISTNet, make_costs,
                         choose_clients)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/raw/femnist_campaign/selection_diagnostics.csv"


def normalized_entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(62))


def jsd_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float) / p.sum()
    q = q.astype(float) / q.sum()
    m = .5 * (p + q)
    pz, qz = p > 0, q > 0
    return float(.5 * np.sum(p[pz] * np.log2(p[pz] / m[pz])) +
                 .5 * np.sum(q[qz] * np.log2(q[qz] / m[qz])))


def main() -> None:
    config = json.loads((OUT / "config.json").read_text())
    manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    meta = manifest["optimization_train"]
    clients = meta["clients"]
    sizes = np.asarray([c["n"] for c in clients], dtype=np.int64)
    labels = np.load(CACHE / "train_y.npy", mmap_mode="r")
    local_hist = np.zeros((len(clients), 62), dtype=np.int64)
    for i, c in enumerate(clients):
        start, end = int(c["offset"]), int(c["offset"] + c["n"])
        local_hist[i] = np.bincount(np.asarray(labels[start:end], dtype=np.int64), minlength=62)
    global_hist = np.asarray(meta["class_counts"], dtype=np.int64)
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    median_n = float(np.median(sizes))
    budget = float(config["budget_j_per_round"])
    policies = ["random_fresh", "random_ttl8", "utility_blind", "cross_layer",
                "cross_layer_no_debt", "cross_layer_debt_only"]
    rows = []
    for rep in range(config["repetitions"]):
        seed = config["seed"] + 1009 * rep
        rates = np.random.default_rng(seed).uniform(
            config["rate_min_mbps"], config["rate_max_mbps"],
            size=(config["rounds"], len(clients)))
        rng = np.random.default_rng(seed + 104729)
        orders = np.stack([rng.permutation(len(clients)) for _ in range(config["rounds"])])
        for policy_name in policies:
            policy = POLICIES[policy_name]
            count = np.zeros(len(clients), dtype=np.int32)
            refresh = np.full(len(clients), -100000, dtype=np.int32)
            updates = 0
            selected_examples = 0
            selected_rates = []
            selected_writer_entropy = []
            round_entropy = []
            round_jsd = []
            trace = []
            for r in range(config["rounds"]):
                cost = make_costs({"clients": clients}, rates[r], count, refresh,
                                  r, policy["ttl"], model_bytes, config["deadline_s"])
                chosen = choose_clients({"clients": clients}, cost, count, r,
                                        budget, config["kmax"], policy, orders[r], median_n)
                cohort_hist = np.zeros(62, dtype=np.int64)
                for i in chosen:
                    count[i] += 1
                    updates += 1
                    selected_examples += int(sizes[i])
                    selected_rates.append(float(rates[r, i]))
                    selected_writer_entropy.append(normalized_entropy(local_hist[i]))
                    cohort_hist += local_hist[i]
                    if cost["refresh"][i]:
                        refresh[i] = r
                round_entropy.append(normalized_entropy(cohort_hist))
                round_jsd.append(jsd_bits(global_hist, cohort_hist))
                trace.append(",".join(map(str, sorted(chosen))))
            trace_hash = hashlib.sha256("\n".join(trace).encode()).hexdigest()
            rows.append({
                "rep": rep, "seed": seed, "policy": policy_name,
                "rounds": int(config["rounds"]), "completed_updates": updates,
                "unique_writers": int(np.count_nonzero(count)),
                "jain_participation": float(count.sum()**2 /
                    (len(clients) * np.square(count).sum())) if count.any() else 0.0,
                "mean_examples_per_selected_update": selected_examples / max(1, updates),
                "mean_selected_link_rate_mbps": float(np.mean(selected_rates)),
                "mean_selected_writer_entropy_norm": float(np.mean(selected_writer_entropy)),
                "mean_round_cohort_entropy_norm": float(np.mean(round_entropy)),
                "mean_round_class_jsd_bits": float(np.mean(round_jsd)),
                "selection_trace_sha256": trace_hash,
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tie_audit = pd.read_csv(ROOT / "evidence/tiebreak_audit.csv")
    audit_hashes = {(int(r.seed), r.policy): r.old_trace_sha256
                    for r in tie_audit.itertuples(index=False)}
    matched = all(audit_hashes.get((int(r["seed"]), r["policy"])) ==
                  r["selection_trace_sha256"] for r in rows)
    if not matched:
        raise RuntimeError("Selection-diagnostic replay differs from the audited executed trace")
    report = {
        "runs": len(rows), "seed_count": config["repetitions"],
        "policy_count": len(policies), "source_manifest": "dataset_manifest.json",
        "source_commit": manifest["source_repo_commit"],
        "selection_trace_hashes_match_main": matched,
        "metrics": "post-simulation data-profile diagnostics; labels characterize optimization writers only and never enter scheduling",
    }
    (OUTPUT.parent / "selection_diagnostics_config.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"Replayed and profiled {len(rows)} policy-seed scheduler traces.")


if __name__ == "__main__":
    main()
