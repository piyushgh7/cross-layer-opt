#!/usr/bin/env python3
"""Validation-only FEMNIST pilot to check whether 50 rounds are sufficient.

The pilot uses the complete optimization-writer population and cross-layer
admission. It does not load or score final-test writers and is excluded from
the reported multi-seed campaign.
"""
from __future__ import annotations
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from femnist_sim import (
    ROOT, POLICIES, FEMNISTNet, ensure_data, make_costs, choose_clients,
    evaluate, local_train, CPU_BATCH_J, KEM768_KEYGEN_J, KEM768_DECAP_J,
    WIFI_RX_W, WIFI_TX_W, MLKEM_PK_B, MLKEM_CT_B,
)


def main():
    torch.set_num_threads(1)
    seed, rounds, kmax = 20261011, 50, 50
    train, val, _test, x, y, _tx, _ty = ensure_data(load_test=False)
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    n = train["num_clients"]
    ns = np.asarray([c["n"] for c in train["clients"]], dtype=np.int64)
    median_n = float(np.median(ns))
    mean_rate = 55.0
    one_way_s = model_bytes * 8 / (mean_rate * 1e6)
    median_radio_j = (WIFI_RX_W + WIFI_TX_W) * one_way_s + (
        WIFI_TX_W * MLKEM_PK_B * 8 / (mean_rate * 1e6) +
        WIFI_RX_W * MLKEM_CT_B * 8 / (mean_rate * 1e6))
    budget = .85 * kmax * (median_n / 5 * CPU_BATCH_J + median_radio_j +
                            KEM768_KEYGEN_J + KEM768_DECAP_J)

    rate_rng = np.random.default_rng(seed)
    order_rng = np.random.default_rng(seed + 104729)
    rates = rate_rng.uniform(10.0, 100.0, size=(rounds, n))
    orders = np.stack([order_rng.permutation(n) for _ in range(rounds)])
    policy = POLICIES["cross_layer"]
    participation = np.zeros(n, dtype=np.int32)
    last_refresh = np.full(n, -100000, dtype=np.int32)
    torch.manual_seed(seed)
    model = FEMNISTNet()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rows = []
    start = time.perf_counter()
    acc, user_acc, class_acc = evaluate(state, val, x, y)
    rows.append({"round": 0, "validation_accuracy": acc,
                 "validation_user_macro_accuracy": user_acc,
                 "validation_class_macro_accuracy": class_acc,
                 "selected": 0, "client_energy_j": 0.0,
                 "kem_refreshes": 0})
    pool = ThreadPoolExecutor(max_workers=8)
    for r in range(rounds):
        costs = make_costs(train, rates[r], participation, last_refresh, r,
                           policy["ttl"], model_bytes, 5.0)
        selected = choose_clients(train, costs, participation, r, budget, kmax,
                                  policy, orders[r], median_n)
        futures = []
        for i in selected:
            client = train["clients"][i]
            client_seed = (seed * 1_000_003 + r * 9_176 + i * 37) % (2**31 - 1)
            futures.append(pool.submit(local_train, state, x, y, client,
                                       client_seed, 0.9))
        updates = [f.result() for f in futures]
        weights = [train["clients"][i]["n"] for i in selected]
        if selected:
            denom = float(sum(weights))
            new_state = {}
            for key in state:
                total = torch.zeros_like(state[key], dtype=torch.float64)
                for local_state, weight in zip(updates, weights):
                    total.add_(local_state[key].to(torch.float64), alpha=weight / denom)
                new_state[key] = total.to(state[key].dtype)
            state = new_state
        round_energy = float(sum(costs["total_j"][i] for i in selected))
        refreshes = 0
        completion = 0.0
        for i in selected:
            participation[i] += 1
            completion = max(completion, float(costs["latency"][i]))
            if costs["refresh"][i]:
                last_refresh[i] = r
                refreshes += 1
        if (r + 1) % 5 == 0:
            acc, user_acc, class_acc = evaluate(state, val, x, y)
            rows.append({"round": r + 1, "validation_accuracy": acc,
                         "validation_user_macro_accuracy": user_acc,
                         "validation_class_macro_accuracy": class_acc,
                         "selected": len(selected), "client_energy_j": round_energy,
                         "kem_refreshes": refreshes,
                         "round_completion_s": completion})
            print(f"round={r+1}/{rounds} validation_accuracy={acc:.4f} "
                  f"selected={len(selected)}", flush=True)
    pool.shutdown(wait=True)

    out_dir = ROOT / "results/raw/validation_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "validation_pilot.csv").open("w", newline="") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    config = {
        "purpose": "validation-only learning-horizon pilot; excluded from final campaign",
        "final_test_arrays_opened_or_evaluated": False,
        "seed": seed, "rounds": rounds, "clients_per_round_cap": kmax,
        "policy": "cross_layer", "local_epochs": 1, "batch_size": 5,
        "learning_rate": .004, "momentum": .9,
        "optimization_writers": train["num_clients"],
        "optimization_examples": train["num_examples"],
        "validation_writers": val["num_clients"],
        "validation_examples": val["num_examples"],
        "wall_seconds": time.perf_counter() - start,
        "source": "results/raw/validation_pilot/validation_pilot.csv",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
