#!/usr/bin/env python3
"""Traceability and consistency audit for executed FL and system sweeps."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from femnist_sim import ROOT, POLICIES


def main():
    base = ROOT / "results/raw/femnist_campaign"
    runs = pd.read_csv(base / "runs.csv")
    rounds = pd.read_csv(base / "rounds.csv")
    config = json.loads((base / "config.json").read_text())
    manifest = json.loads((base / "dataset_manifest.json").read_text())
    dataset_audit = json.loads((ROOT / "evidence/dataset_audit.json").read_text())
    dataset_profile = json.loads((ROOT / "evidence/dataset_profile.json").read_text())
    sens_path = ROOT / "results/raw/sensitivity/admission_sensitivity.csv"
    sens = pd.read_csv(sens_path)
    pqc_path = ROOT / "results/raw/sensitivity/pqc_cost_sensitivity.csv"
    pqc = pd.read_csv(pqc_path)
    pqc_config = json.loads((ROOT / "results/raw/sensitivity/pqc_cost_sensitivity_config.json").read_text())
    tiebreak = pd.read_csv(ROOT / "evidence/tiebreak_audit.csv")
    selection_diag_path = base / "selection_diagnostics.csv"
    selection_diag = pd.read_csv(selection_diag_path)
    selection_diag_config = json.loads((base / "selection_diagnostics_config.json").read_text())
    pilot = pd.read_csv(ROOT / "results/raw/validation_pilot/validation_pilot.csv")
    seed_values = sorted(runs.seed.unique().tolist())
    expected_policies = set(POLICIES)
    failures = []

    if len(runs) != len(seed_values) * len(expected_policies):
        failures.append("Unexpected count of completed policy/seed runs")
    if set(runs.policy) != expected_policies:
        failures.append("One or more policy runs are missing")
    if len(rounds) != len(runs) * (config["rounds"] + 1):
        failures.append("Unexpected count of round-level records")
    if {"final_test_accuracy", "final_test_user_macro_accuracy"} & set(rounds.columns):
        failures.append("Final-test metric leaked into round-level/checkpoint outputs")
    if manifest.get("writer_overlap") != 0:
        failures.append("Dataset manifest reports writer overlap")
    if any(dataset_audit["writer_identity_intersections"].values()):
        failures.append("Dataset audit reports writer identity overlap")
    if dataset_audit["pixel_integrity"]["repeated_images_across_partitions"] != 0:
        failures.append("Dataset audit reports exact-image duplication across partitions")
    if len(manifest["optimization_train"]["clients"]) != manifest["optimization_train"]["num_clients"]:
        failures.append("Optimization-client manifest count mismatch")
    if len(manifest["validation"]["clients"]) != manifest["validation"]["num_clients"]:
        failures.append("Validation-client manifest count mismatch")
    if len(manifest["final_test"]["clients"]) != manifest["final_test"]["num_clients"]:
        failures.append("Final-test client manifest count mismatch")
    if dataset_profile["optimization_writers"] != manifest["optimization_train"]["num_clients"] or \
            dataset_profile["optimization_examples"] != manifest["optimization_train"]["num_examples"]:
        failures.append("Dataset-profile counts disagree with the campaign manifest")
    if dataset_profile["source_commit"] != manifest["source_repo_commit"]:
        failures.append("Dataset-profile source commit disagrees with the campaign manifest")
    audited_examples = sum(dataset_audit["partitions"][name]["examples"] for name in
                           ("optimization", "validation", "final_test"))
    manifest_examples = sum(manifest[key]["num_examples"] for key in
                            ("optimization_train", "validation", "final_test"))
    if audited_examples != manifest_examples or audited_examples != dataset_audit["total_examples"]:
        failures.append("Dataset partition totals disagree between audit and run manifest")
    partition_ids = {
        "optimization": {c["id"] for c in manifest["optimization_train"]["clients"]},
        "validation": {c["id"] for c in manifest["validation"]["clients"]},
        "final_test": {c["id"] for c in manifest["final_test"]["clients"]},
    }
    for group in manifest.get("exact_duplicate_writer_groups", []):
        expected = group["assigned_partition"]
        ids = set(group["writers"])
        if not ids.issubset(partition_ids[expected]):
            failures.append(f"Exact-duplicate writer group crosses its assigned partition: {ids}")
        if any(ids & members for name, members in partition_ids.items() if name != expected):
            failures.append(f"Exact-duplicate writer group occurs outside {expected}: {ids}")

    run_lookup = runs.set_index(["seed", "policy"])
    worst_energy_reconstruction = 0.0
    for (seed, policy), row in run_lookup.iterrows():
        d = rounds[(rounds.seed == seed) & (rounds.policy == policy)].sort_values("round")
        if d["round"].tolist() != list(range(config["rounds"] + 1)):
            failures.append(f"Round index gap for {seed}/{policy}")
            continue
        steps = d[d["round"] > 0]
        if (steps.selected > steps.max_clients).any():
            failures.append(f"Client cap violated for {seed}/{policy}")
        if (steps.client_energy_j > steps.budget_j + 1e-7).any():
            failures.append(f"Energy budget violated for {seed}/{policy}")
        if (steps.round_completion_s > config["deadline_s"] + 1e-9).any():
            failures.append(f"Deadline violated for {seed}/{policy}")
        totals = {
            "total_updates": int(steps.selected.sum()),
            "key_refreshes": int(steps.kem_refreshes.sum()),
            "client_energy_j": float(steps.client_energy_j.sum()),
            "training_energy_j": float(steps.training_energy_j.sum()),
            "radio_energy_j": float(steps.radio_energy_j.sum()),
            "client_kem_compute_energy_j": float(steps.client_kem_compute_energy_j.sum()),
        }
        for metric, value in totals.items():
            expected = float(row[metric])
            if not np.isclose(value, expected, rtol=1e-9, atol=1e-6):
                failures.append(f"Run/round mismatch {metric} for {seed}/{policy}")
        reconstructed = (totals["training_energy_j"] + totals["radio_energy_j"] +
                         totals["client_kem_compute_energy_j"])
        err = abs(reconstructed - totals["client_energy_j"])
        worst_energy_reconstruction = max(worst_energy_reconstruction, err)
        if not np.isclose(reconstructed, totals["client_energy_j"], rtol=1e-9, atol=1e-6):
            failures.append(f"Energy components do not sum for {seed}/{policy}")
        last_validation = steps.loc[steps.validation_accuracy.notna(), "validation_accuracy"].iloc[-1]
        if not np.isclose(last_validation, row.final_validation_accuracy, rtol=0, atol=1e-12):
            failures.append(f"Run/checkpoint validation mismatch for {seed}/{policy}")

    # The system-only nominal condition is independently re-executed and should
    # exactly reproduce the main cross-layer scheduler's admitted-cost trace.
    max_nominal_delta = {"admitted": 0.0, "client_energy_j": 0.0, "kem_refreshes": 0.0}
    max_pilot_delta = {"validation_accuracy": 0.0, "selected": 0.0,
                       "client_energy_j": 0.0, "kem_refreshes": 0.0}
    for seed in seed_values:
        main = run_lookup.loc[(seed, "cross_layer")]
        main_first50 = rounds[(rounds.seed == seed) & (rounds.policy == "cross_layer") &
                              (rounds["round"] > 0) & (rounds["round"] <= 50)]
        s = sens[(sens.seed == seed) & (sens.condition == "nominal_link") &
                 (sens.policy == "cross_layer")]
        if len(s) != 1:
            failures.append(f"Missing nominal paired sensitivity run for seed {seed}")
            continue
        s = s.iloc[0]
        pairs = [("admitted", float(main_first50.selected.mean()), float(s.avg_selected)),
                 ("client_energy_j", float(main_first50.client_energy_j.sum()), float(s.total_client_energy_j)),
                 ("kem_refreshes", float(main_first50.kem_refreshes.sum()), float(s.kem_refreshes))]
        for name, a, b in pairs:
            delta = abs(a-b)
            max_nominal_delta[name] = max(max_nominal_delta[name], delta)
            if not np.isclose(a, b, rtol=1e-9, atol=1e-6):
                failures.append(f"Main/sensitivity nominal mismatch {name} for seed {seed}")

    # The validation-only 50-round pilot and the predeclared final campaign share
    # the same seed and cross-layer configuration; their first 50 round outputs
    # must reproduce exactly. No final-test data are involved in this check.
    pilot_main = rounds[(rounds.seed == seed_values[0]) &
                        (rounds.policy == "cross_layer")].set_index("round")
    for _, p in pilot.iterrows():
        r = int(p["round"])
        if r not in pilot_main.index:
            failures.append(f"Pilot checkpoint round {r} is absent from main campaign")
            continue
        q = pilot_main.loc[r]
        values = [("validation_accuracy", float(p.validation_accuracy),
                   float(q.validation_accuracy))]
        if r > 0:
            values.extend([
                ("selected", float(p.selected), float(q.selected)),
                ("client_energy_j", float(p.client_energy_j), float(q.client_energy_j)),
                ("kem_refreshes", float(p.kem_refreshes), float(q.kem_refreshes)),
            ])
        for name, a, b in values:
            delta = abs(a - b)
            max_pilot_delta[name] = max(max_pilot_delta[name], delta)
            if not np.isclose(a, b, rtol=1e-10, atol=1e-9):
                failures.append(f"Validation pilot/main mismatch in {name} at round {r}")

    expected_pqc_seeds = sorted(seed_values)
    expected_pqc_levels = {0.0, 0.5, 1.0, 2.0, 4.0}
    if len(pqc) != 100 or len(pqc_config.get("levels", [])) != 5:
        failures.append("PQC profile sensitivity does not contain the complete 100-run design")
    if set(pqc.policy) != {"cross_layer", "utility_blind"}:
        failures.append("PQC profile sensitivity is missing a paired scheduler")
    if set(pqc.seed.unique()) != set(expected_pqc_seeds):
        failures.append("PQC profile sensitivity seed set differs from the main campaign")
    if set(pqc.kem_energy_multiplier.unique()) != expected_pqc_levels or \
            set(pqc.kem_latency_multiplier.unique()) != expected_pqc_levels:
        failures.append("PQC profile sensitivity multiplier levels are incomplete")
    if not np.isfinite(pqc.select_dtypes(include=[np.number]).to_numpy()).all():
        failures.append("PQC profile sensitivity contains non-finite numeric outputs")
    if pqc.duplicated(["condition", "seed", "policy"]).any():
        failures.append("PQC profile sensitivity contains duplicate run keys")
    if len(tiebreak) != len(runs) or not tiebreak.selection_trace_same.astype(bool).all():
        failures.append("Seeded tie-break replay does not reproduce every executed client-selection trace")
    if set(zip(tiebreak.seed, tiebreak.policy)) != set(zip(runs.seed, runs.policy)):
        failures.append("Tie-break replay does not cover the full executed policy-seed design")
    tie_hashes = {(int(r.seed), r.policy): r.old_trace_sha256
                  for r in tiebreak.itertuples(index=False)}
    if len(selection_diag) != len(runs) or not selection_diag_config.get("selection_trace_hashes_match_main"):
        failures.append("Selected-data diagnostics do not cover the executed campaign traces")
    for r in selection_diag.itertuples(index=False):
        if tie_hashes.get((int(r.seed), r.policy)) != r.selection_trace_sha256:
            failures.append(f"Selected-data diagnostic trace mismatch for {r.seed}/{r.policy}")
    # Unit multipliers reproduce the primary cross-layer trace exactly because
    # the profile, trace, budget, TTL, and admission rule are the same.
    for seed in expected_pqc_seeds:
        sens_nominal = sens[(sens.seed == seed) & (sens.condition == "nominal_link") &
                            (sens.policy == "cross_layer")]
        pqc_nominal = pqc[(pqc.seed == seed) & (pqc.condition == "kem_energy_1") &
                          (pqc.policy == "cross_layer")]
        if len(sens_nominal) != 1 or len(pqc_nominal) != 1:
            failures.append(f"Missing matched unit-profile KEM sensitivity trace for seed {seed}")
            continue
        for metric in ("avg_selected", "total_client_energy_j", "kem_refreshes",
                       "p95_round_completion_s"):
            if not np.isclose(float(sens_nominal.iloc[0][metric]),
                              float(pqc_nominal.iloc[0][metric]), rtol=1e-9, atol=1e-7):
                failures.append(f"Unit-profile KEM sensitivity mismatch in {metric} for seed {seed}")

    report = {
        "completed_runs": len(runs), "policies": sorted(expected_policies),
        "independent_seeds": seed_values,
        "round_rows": len(rounds), "sensitivity_rows": len(sens),
        "pqc_profile_sensitivity_runs": len(pqc),
        "pqc_profile_sensitivity_levels": sorted(expected_pqc_levels),
        "tie_break_sequences_reproduced": int(tiebreak.selection_trace_same.astype(bool).sum()),
        "selected_data_diagnostic_runs": len(selection_diag),
        "dataset_examples": sum(manifest[k]["num_examples"] for k in
                                 ("optimization_train", "validation", "final_test")),
        "published_leaf_reference_examples": 805263,
        "dataset_writers": sum(manifest[k]["num_clients"] for k in
                                ("optimization_train", "validation", "final_test")),
        "optimization_writer_size_gini": dataset_profile["writer_size_gini"],
        "optimization_writer_label_entropy_median": dataset_profile["writer_normalized_entropy_quantiles"]["50"],
        "optimization_writer_size_entropy_spearman": dataset_profile["writer_size_entropy_spearman"],
        "dataset_exact_duplicate_groups": manifest.get("exact_duplicate_writer_groups", []),
        "cross_partition_exact_duplicate_occurrences": dataset_audit["pixel_integrity"]["repeated_images_across_partitions"],
        "final_test_metrics_in_round_logs": False,
        "max_energy_reconstruction_error_j": worst_energy_reconstruction,
        "nominal_main_vs_sensitivity_max_abs_delta": max_nominal_delta,
        "validation_pilot_vs_main_first50_max_abs_delta": max_pilot_delta,
        "failures": failures,
    }
    out = ROOT / "evidence/experiment_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Executed-results audit failed; see evidence/experiment_audit.json")


if __name__ == "__main__":
    main()
