#!/usr/bin/env python3
"""Replay scheduler-only trajectories to detect client-index tie bias."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from femnist_sim import OUT, POLICIES, FEMNISTNet, make_costs, choose_clients

ROOT = Path(__file__).resolve().parents[1]


def fair_tie_choose(meta, cost, count, round_id, budget, kmax, policy,
                    random_order, median_n):
    if policy["kind"] == "random":
        return choose_clients(meta, cost, count, round_id, budget, kmax,
                              policy, random_order, median_n)
    feasible = np.flatnonzero(cost["feasible"])
    n = cost["n"].astype(np.float64)
    expected = (round_id + 1) * kmax / len(n)
    debt = np.clip(expected - count, -1.0, 1.0)
    kind = policy.get("utility", "sample_and_debt")
    if kind == "sample_only":
        value = np.sqrt(n / max(1.0, median_n))
    elif kind == "debt_only":
        value = 1.0 + .5 * debt
    else:
        value = np.sqrt(n / max(1.0, median_n)) * (1.0 + .5 * debt)
    denom = cost["train_j"] + cost["radio_payload_j"]
    if policy["crypto_aware"]:
        denom = cost["total_j"]
    score = value / np.maximum(denom, 1e-12)
    tie_rank = np.empty(len(n), dtype=np.int32)
    tie_rank[random_order] = np.arange(len(n), dtype=np.int32)
    order = sorted((int(i) for i in feasible), key=lambda i: (-score[i], tie_rank[i]))
    chosen, spent = [], 0.0
    for i in order:
        if len(chosen) >= kmax:
            break
        if spent + float(cost["total_j"][i]) <= budget:
            chosen.append(i)
            spent += float(cost["total_j"][i])
    return chosen


def replay(meta, rates, orders, policy_name, budget, deadline, fair):
    nclients = len(meta["clients"])
    ns = np.asarray([c["n"] for c in meta["clients"]])
    median_n = float(np.median(ns))
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    counts = np.zeros(nclients, dtype=np.int32)
    refresh = np.full(nclients, -100000, dtype=np.int32)
    selected_count = np.zeros(nclients, dtype=np.int32)
    updates = 0
    trace = []
    for r in range(len(rates)):
        ttl = POLICIES[policy_name]["ttl"]
        cost = make_costs({"clients": meta["clients"]}, rates[r], counts,
                          refresh, r, ttl, model_bytes, deadline)
        if fair:
            chosen = fair_tie_choose(meta, cost, counts, r, budget, 50,
                                     POLICIES[policy_name], orders[r], median_n)
        else:
            chosen = choose_clients(meta, cost, counts, r, budget, 50,
                                     POLICIES[policy_name], orders[r], median_n)
        for i in chosen:
            counts[i] += 1
            selected_count[i] += 1
            if cost["refresh"][i]:
                refresh[i] = r
        updates += len(chosen)
        trace.append(",".join(map(str, sorted(chosen))))
    jain = counts.sum()**2 / (nclients * np.square(counts).sum()) if counts.any() else 0.0
    return {"updates": updates, "unique": int(np.count_nonzero(counts)),
            "jain": float(jain), "max_count": int(counts.max()),
            "top_index_share": float(counts[:100].sum()/max(1, counts.sum())),
            "trace_sha256": hashlib.sha256("\n".join(trace).encode()).hexdigest()}


def main():
    manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    config = json.loads((OUT / "config.json").read_text())
    meta = manifest["optimization_train"]
    nclients = meta["num_clients"]
    n = np.asarray([c["n"] for c in meta["clients"]])
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    mean_rate = (config["rate_min_mbps"] + config["rate_max_mbps"]) / 2
    radio = (1.789 * model_bytes * 8 / (mean_rate*1e6) +
             (0.966*1184 + .823*1088)*8/(mean_rate*1e6))
    budget = config["budget_fraction"]*config["kmax"]*(np.median(n)/5*.3 + radio + .015007 + .018016)
    rows = []
    for rep in range(config["repetitions"]):
        seed = config["seed"] + 1009*rep
        rates = np.random.default_rng(seed).uniform(
            config["rate_min_mbps"], config["rate_max_mbps"],
            size=(config["rounds"], nclients))
        rng = np.random.default_rng(seed+104729)
        orders = np.stack([rng.permutation(nclients) for _ in range(config["rounds"])])
        for policy in POLICIES:
            old = replay(meta, rates, orders, policy, budget, config["deadline_s"], False)
            new = replay(meta, rates, orders, policy, budget, config["deadline_s"], True)
            rows.append({"seed": seed, "policy": policy,
                         **{f"old_{k}": v for k, v in old.items()},
                         **{f"seeded_{k}": v for k, v in new.items()},
                         "unique_delta": new["unique"]-old["unique"],
                         "jain_delta": new["jain"]-old["jain"],
                         "selection_trace_same": old["trace_sha256"] == new["trace_sha256"],
                         "selection_total_changed": old["updates"] != new["updates"]})
    out = ROOT / "evidence/tiebreak_audit.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {}
    for p in POLICIES:
        q = [r for r in rows if r["policy"] == p]
        summary[p] = {
            "old_unique_mean": float(np.mean([r["old_unique"] for r in q])),
            "seeded_unique_mean": float(np.mean([r["seeded_unique"] for r in q])),
            "old_jain_mean": float(np.mean([r["old_jain"] for r in q])),
            "seeded_jain_mean": float(np.mean([r["seeded_jain"] for r in q])),
            "max_count_old_mean": float(np.mean([r["old_max_count"] for r in q])),
            "max_count_seeded_mean": float(np.mean([r["seeded_max_count"] for r in q])),
            "top100_share_old_mean": float(np.mean([r["old_top_index_share"] for r in q])),
            "top100_share_seeded_mean": float(np.mean([r["seeded_top_index_share"] for r in q])),
        }
    (ROOT / "evidence/tiebreak_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
