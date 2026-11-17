#!/usr/bin/env python3
"""System-level admission sensitivity using full FEMNIST client sizes.

This is not a learning-accuracy experiment: it varies operating conditions and
measures admission, modeled energy, handshake count, and completion time. The
input client population comes from the audited full writer split.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from femnist_sim import (
    DATA, OUT, CACHE, POLICIES, WIFI_RX_W, WIFI_TX_W, CPU_BATCH_J,
    KEM768_KEYGEN_J, KEM768_DECAP_J, KEM768_KEYGEN_S, KEM768_DECAP_S,
    MLKEM_PK_B, MLKEM_CT_B,
    FEMNISTNet, make_costs, choose_clients,
)

ROOT = Path(__file__).resolve().parents[1]
SENS_OUT = ROOT / "results/raw/sensitivity"


def energy_budget(client_meta, model_bytes, rate_low, rate_high, fraction, kmax,
                  radio_multiplier=1.0, kem_energy_multiplier=1.0):
    ns = np.asarray([c["n"] for c in client_meta], dtype=np.int64)
    median_n = float(np.median(ns))
    batch_equiv = median_n / 5.0
    mean_rate = (rate_low + rate_high) / 2.0
    one_way_s = model_bytes * 8.0 / (mean_rate * 1e6)
    e_radio = (WIFI_RX_W + WIFI_TX_W) * one_way_s
    e_radio += (WIFI_TX_W * MLKEM_PK_B + WIFI_RX_W * MLKEM_CT_B) * 8.0 / (mean_rate * 1e6)
    e_cpu = batch_equiv * CPU_BATCH_J
    e_kem = KEM768_KEYGEN_J + KEM768_DECAP_J
    return fraction * kmax * (e_cpu + radio_multiplier * e_radio +
                               kem_energy_multiplier * e_kem), median_n


def run_one(client_meta, rates_by_round, orders, policy_name, ttl, deadline,
            budget_fraction, rate_bounds, seed, condition, radio_multiplier=1.0,
            kmax=50, kem_energy_multiplier=1.0,
            kem_latency_multiplier=1.0, energy_budget_override_j=None):
    cost_meta = {"clients": client_meta}
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    n_clients = len(client_meta)
    budget, median_n = energy_budget(client_meta, model_bytes,
                                      rate_bounds[0], rate_bounds[1],
                                      budget_fraction, kmax, radio_multiplier,
                                      kem_energy_multiplier)
    if energy_budget_override_j is not None:
        budget = float(energy_budget_override_j)
    p = dict(POLICIES[policy_name])
    p["ttl"] = ttl
    p["crypto_aware"] = policy_name == "cross_layer"
    participation = np.zeros(n_clients, dtype=np.int32)
    last_refresh = np.full(n_clients, -100000, dtype=np.int32)
    values = {"selected": 0, "energy": 0.0, "server_energy": 0.0,
              "refreshes": 0, "handshake_bytes": 0, "model_bytes": 0,
              "round_time": [], "eligible": 0}
    for r, rates in enumerate(rates_by_round):
        # Radio, KEM operation energy, and KEM operation time are varied as
        # separate dimensions; payload sizes and link rates remain fixed.
        cost = make_costs(cost_meta, rates, participation, last_refresh,
                          r, ttl, model_bytes, deadline)
        cost["radio_payload_j"] *= radio_multiplier
        cost["radio_j"] *= radio_multiplier
        cost["client_kem_j"] *= kem_energy_multiplier
        cost["server_crypto_j"] *= kem_energy_multiplier
        cost["latency"] += cost["refresh"].astype(np.float64) * (
            (kem_latency_multiplier - 1.0) * (KEM768_KEYGEN_S + KEM768_DECAP_S))
        cost["feasible"] = cost["latency"] <= deadline
        cost["total_j"] = cost["train_j"] + cost["radio_j"] + cost["client_kem_j"]
        selected = choose_clients(client_meta, cost, participation, r, budget,
                                  kmax, p, orders[r], median_n)
        e_round = 0.0
        server_round = 0.0
        refreshed = 0
        t_round = 0.0
        for i in selected:
            e_round += float(cost["total_j"][i])
            server_round += float(cost["server_crypto_j"][i])
            t_round = max(t_round, float(cost["latency"][i]))
            participation[i] += 1
            if cost["refresh"][i]:
                last_refresh[i] = r
                refreshed += 1
        values["selected"] += len(selected)
        values["energy"] += e_round
        values["server_energy"] += server_round
        values["refreshes"] += refreshed
        values["handshake_bytes"] += refreshed * (MLKEM_PK_B + MLKEM_CT_B)
        values["model_bytes"] += len(selected) * model_bytes * 2
        values["round_time"].append(t_round)
        values["eligible"] += int(cost["feasible"].sum())
    n_rounds = len(rates_by_round)
    fairness = float((participation.sum() ** 2) /
                     (n_clients * np.square(participation).sum())) if participation.any() else 0.0
    return {
        "condition": condition, "seed": seed, "policy": policy_name,
        "ttl_rounds": ttl, "rounds": n_rounds, "kmax": kmax,
        "deadline_s": deadline, "energy_budget_fraction": budget_fraction,
        "energy_budget_j": budget,
        "rate_min_mbps": rate_bounds[0], "rate_max_mbps": rate_bounds[1],
        "radio_energy_multiplier": radio_multiplier,
        "kem_energy_multiplier": kem_energy_multiplier,
        "kem_latency_multiplier": kem_latency_multiplier,
        "avg_selected": values["selected"] / n_rounds,
        "total_selected": values["selected"],
        "eligible_client_rounds": values["eligible"],
        "mean_client_energy_j_per_round": values["energy"] / n_rounds,
        "total_client_energy_j": values["energy"],
        "server_kem_energy_j": values["server_energy"],
        "kem_refreshes": values["refreshes"],
        "handshake_bytes": values["handshake_bytes"],
        "model_bytes_up_down": values["model_bytes"],
        "mean_round_completion_s": float(np.mean(values["round_time"])),
        "p95_round_completion_s": float(np.quantile(values["round_time"], .95)),
        "unique_clients": int(np.count_nonzero(participation)),
        "jain_participation_fairness": fairness,
    }


def main():
    manifest_path = OUT / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run the full FEMNIST campaign first; its dataset manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    client_meta = manifest["optimization_train"]["clients"]
    n_clients = len(client_meta)
    rounds, reps = 50, 5
    base_seed = 20261011
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    rows = []

    # Session-lifetime sweep on the main 10--100 Mb/s / 5-s operating envelope.
    for ttl in (1, 2, 4, 8, 16, 32, 64):
        for rep in range(reps):
            seed = base_seed + 1009 * rep
            rate_rng = np.random.default_rng(seed)
            order_rng = np.random.default_rng(seed + 104729)
            rates = rate_rng.uniform(10.0, 100.0, size=(rounds, n_clients))
            orders = np.stack([order_rng.permutation(n_clients) for _ in range(rounds)])
            for policy in ("random_ttl8", "cross_layer"):
                rows.append(run_one(client_meta, rates, orders, policy, ttl, 5.0,
                                    .85, (10.0, 100.0), seed,
                                    f"ttl_{ttl}", 1.0))

    # Joint rate/deadline boundary analysis; low-rate links receive a longer
    # deadline so the experiment distinguishes slow-but-feasible from infeasible.
    rate_cases = [
        (1.0, 5.0, 40.0, "constrained_link"),
        (10.0, 20.0, 10.0, "moderate_link"),
        (10.0, 100.0, 5.0, "nominal_link"),
        (50.0, 100.0, 5.0, "fast_link"),
    ]
    for low, high, deadline, name in rate_cases:
        for rep in range(reps):
            seed = base_seed + 1009 * rep
            # Common random numbers pair link envelopes by quantile.
            rate_rng = np.random.default_rng(seed)
            order_rng = np.random.default_rng(seed + 104729)
            rates = rate_rng.uniform(low, high, size=(rounds, n_clients))
            orders = np.stack([order_rng.permutation(n_clients) for _ in range(rounds)])
            for policy in ("random_ttl8", "cross_layer"):
                rows.append(run_one(client_meta, rates, orders, policy, 8,
                                    deadline, .85, (low, high), seed,
                                    name, 1.0))

    # Radio-energy and aggregate-budget sensitivity on the same channel scenario.
    for radio_scale in (0.5, 1.0, 2.0):
        for budget in (0.70, 0.85, 1.00):
            for rep in range(reps):
                seed = base_seed + 1009 * rep
                # Hold channel rates and candidate order fixed across profile/budget settings.
                rate_rng = np.random.default_rng(seed)
                order_rng = np.random.default_rng(seed + 104729)
                rates = rate_rng.uniform(10.0, 100.0, size=(rounds, n_clients))
                orders = np.stack([order_rng.permutation(n_clients) for _ in range(rounds)])
                rows.append(run_one(client_meta, rates, orders, "cross_layer", 8,
                                    5.0, budget, (10.0, 100.0), seed,
                                    f"radio{radio_scale}_budget{budget}", radio_scale))

    # Population/round-size scaling uses the complete client roster and unchanged
    # per-client data; only the maximum simultaneous cohort changes.
    for kmax in (20, 50, 100):
        for rep in range(reps):
            seed = base_seed + 1009 * rep
            # Compare cohort caps against the same channel and tie-breaking trace.
            rate_rng = np.random.default_rng(seed)
            order_rng = np.random.default_rng(seed + 104729)
            rates = rate_rng.uniform(10.0, 100.0, size=(rounds, n_clients))
            orders = np.stack([order_rng.permutation(n_clients) for _ in range(rounds)])
            rows.append(run_one(client_meta, rates, orders, "cross_layer", 8,
                                5.0, .85, (10.0, 100.0), seed,
                                f"cohort_{kmax}", 1.0, kmax=kmax))

    SENS_OUT.mkdir(parents=True, exist_ok=True)
    with (SENS_OUT / "admission_sensitivity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config = {
        "clients": n_clients, "rounds": rounds, "independent_seeds": reps,
        "max_clients_per_round": 50, "source_manifest": str(manifest_path.relative_to(ROOT)),
        "model_bytes": model_bytes,
        "sweeps": ["session lifetime", "channel-rate/deadline envelope", "radio-energy multiplier", "aggregate energy budget", "maximum cohort size"],
        "accuracy_claims": "none; system-level admission/cost sensitivities only",
    }
    (SENS_OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(rows)} sensitivity rows to {SENS_OUT}")


if __name__ == "__main__":
    main()
