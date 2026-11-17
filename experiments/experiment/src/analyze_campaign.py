#!/usr/bin/env python3
"""Aggregate executed raw outputs into boxed IEEE tables and one-column plots."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/raw/femnist_campaign/runs.csv"
ROUNDS = ROOT / "results/raw/femnist_campaign/rounds.csv"
SENS = ROOT / "results/raw/sensitivity/admission_sensitivity.csv"
KEM = ROOT / "results/raw/pqc_benchmark.json"
KEM_RAW = ROOT / "results/raw/pqc_benchmark.csv"
TABLES = ROOT / "results/tables"
FIG = ROOT / "figures"

POLICY_ORDER = ["random_fresh", "random_ttl8", "utility_blind",
                "cross_layer_no_debt", "cross_layer_debt_only", "cross_layer"]
PRIMARY_ORDER = ["random_fresh", "random_ttl8", "utility_blind", "cross_layer"]
POLICY_LABELS = {
    "random_fresh": "Random, fresh KEM",
    "random_ttl8": "Random, 8-round session",
    "utility_blind": "Utility, crypto-blind",
    "cross_layer_no_debt": "Cross-layer, no debt",
    "cross_layer_debt_only": "Cross-layer, debt only",
    "cross_layer": "Cross-layer",
}
COLORS = {
    "random_fresh": "#4C78A8",
    "random_ttl8": "#F58518",
    "utility_blind": "#54A24B",
    "cross_layer_no_debt": "#B279A2",
    "cross_layer_debt_only": "#E45756",
    "cross_layer": "#72B7B2",
}
T95_N5 = 2.7764451051977987


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    mean = float(np.mean(x)) if len(x) else float("nan")
    if len(x) < 2:
        return mean, 0.0
    sd = float(np.std(x, ddof=1))
    critical = T95_N5 if len(x) == 5 else 1.96
    return mean, critical * sd / math.sqrt(len(x))


def cell(mean, half_ci, scale=1.0, digits=1):
    if not np.isfinite(mean):
        return "--"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean * scale)} $\\pm$ {fmt.format(half_ci * scale)}"


def boxed_table(caption, label, headings, rows, colspec):
    out = [
        r"\begin{table}[!ht]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(h.replace(r"\\", r"\newline{}") for h in headings) + r" \\",
        r"\hline",
    ]
    for row in rows:
        out.append(" & ".join(map(str, row)) + r" \\")
        out.append(r"\hline")
    out.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(out) + "\n"


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RUNS)
    rounds = pd.read_csv(ROUNDS)
    sens = pd.read_csv(SENS)
    pqc_sens = pd.read_csv(ROOT / "results/raw/sensitivity/pqc_cost_sensitivity.csv")
    pqc_raw = pd.read_csv(KEM_RAW)
    selection_diag = pd.read_csv(ROOT / "results/raw/femnist_campaign/selection_diagnostics.csv")

    # Main held-out quality table: every cell is aggregated from five run files.
    quality_rows = []
    cost_rows = []
    for policy in POLICY_ORDER:
        d = runs[runs.policy == policy]
        if d.empty:
            raise ValueError(f"No run summaries for policy {policy}")
        metrics = ["final_test_accuracy", "final_test_user_macro_accuracy", "final_test_class_macro_accuracy"]
        acc = [mean_ci(d[x]) for x in metrics]
        quality_rows.append([
            POLICY_LABELS[policy],
            *(cell(m, ci, 100.0, 1) for m, ci in acc),
        ])
        metrics2 = ["client_energy_j", "key_refreshes", "avg_selected_per_round", "jain_participation_fairness"]
        c = [mean_ci(d[x]) for x in metrics2]
        cost_rows.append([
            POLICY_LABELS[policy],
            cell(*c[0], scale=1e-3, digits=3),
            cell(*c[1], digits=1),
            cell(*c[2], digits=1),
            cell(*c[3], digits=3),
        ])

    (TABLES / "femnist_quality.tex").write_text(boxed_table(
        "Held-out-writer classification outcomes over five independent runs (mean $\\pm$ 95\\% confidence-interval half-width).",
        "tab:quality", ["Policy", "Sample accuracy (\\%)", "Writer-macro accuracy (\\%)", "Class-macro accuracy (\\%)"],
        quality_rows, "|p{0.25\\columnwidth}|p{0.22\\columnwidth}|p{0.22\\columnwidth}|p{0.21\\columnwidth}|"))
    (TABLES / "femnist_cost.tex").write_text(boxed_table(
        "Cumulative modeled energy across participating clients and scheduling outcomes over five independent runs (mean $\\pm$ 95\\% confidence-interval half-width).",
        "tab:cost", ["Policy", "Cumulative client energy (kJ)", "KEM refreshes", "Clients\\\\per round", "Jain fairness"],
        cost_rows, "|p{0.21\\columnwidth}|p{0.18\\columnwidth}|p{0.18\\columnwidth}|p{0.16\\columnwidth}|p{0.17\\columnwidth}|"))

    overhead_rows = []
    for policy in POLICY_ORDER:
        d = runs[runs.policy == policy]
        metrics = ["unique_participants", "model_bytes_up_down", "handshake_bytes", "server_crypto_energy_j"]
        vals = [mean_ci(d[c]) for c in metrics]
        overhead_rows.append([
            POLICY_LABELS[policy],
            cell(*vals[0], digits=1),
            cell(*vals[1], scale=1e-9, digits=2),
            cell(*vals[2], scale=1e-6, digits=2),
            cell(*vals[3], digits=2),
        ])
    (TABLES / "communication_overhead.tex").write_text(boxed_table(
        "Participation coverage and communication/cryptographic overhead per 100-round run (mean $\\pm$ 95\\% confidence-interval half-width over five seeds).",
        "tab:communication-overhead",
        ["Policy", "Unique\\\\writers", "Model traffic\\\\(GB)", "KEM handshake\\\\(MB)", "Server KEM\\\\energy (J)"],
        overhead_rows,
        "|p{0.18\\columnwidth}|p{0.18\\columnwidth}|p{0.18\\columnwidth}|p{0.18\\columnwidth}|p{0.18\\columnwidth}|"))

    # Normalize energy by completed client-round updates using the run-level
    # ratio as the statistical unit.
    update_rows = []
    for policy in POLICY_ORDER:
        d = runs[runs.policy == policy].copy()
        per_update = d.client_energy_j / d.total_updates
        update_rows.append([POLICY_LABELS[policy], cell(*mean_ci(per_update), digits=2)])
    (TABLES / "energy_per_update.tex").write_text(boxed_table(
        "Modeled client energy per completed client-round update (mean $\\pm$ 95\\% interval across five seeds; the ratio is computed per run).",
        "tab:energy-per-update", ["Policy", r"Client energy\\per update (J)"],
        update_rows, r"|p{0.52\columnwidth}|p{0.38\columnwidth}|"))

    # Post-hoc profile of the exact selected-client traces; local labels are
    # used only to measure representation, never by the scheduler.
    selection_rows = []
    for policy in PRIMARY_ORDER:
        d = selection_diag[selection_diag.policy == policy]
        metrics = [
            mean_ci(d.mean_examples_per_selected_update),
            mean_ci(d.mean_selected_link_rate_mbps),
            mean_ci(d.mean_round_cohort_entropy_norm),
            mean_ci(d.mean_round_class_jsd_bits),
        ]
        selection_rows.append([
            {"random_fresh": "Fresh random", "random_ttl8": "TTL-8 random",
             "utility_blind": "Utility blind", "cross_layer": "Cross-layer"}[policy],
            cell(*metrics[0], digits=1),
            cell(*metrics[1], digits=1),
            cell(*metrics[2], digits=4),
            cell(*metrics[3], digits=4),
        ])
    resource_rows = [row[:3] for row in selection_rows]
    label_rows = [row[0:1] + row[3:5] for row in selection_rows]
    (TABLES / "selection_diagnostics.tex").write_text(
        boxed_table(
            "Post-simulation client resources in the exact scheduler traces (mean $\\pm$ 95\\% interval across five seeds).",
            "tab:selection-resources",
            ["Scheduler", "Examples\\\\per update", "Link rate\\\\(Mb/s)"],
            resource_rows,
            "|p{0.30\\columnwidth}|p{0.30\\columnwidth}|p{0.30\\columnwidth}|"
        ) +
        boxed_table(
            "Post-simulation selected-cohort label profile in the exact scheduler traces (mean $\\pm$ 95\\% interval across five seeds).",
            "tab:selection-labels",
            ["Scheduler", "Normalized\\\\entropy", "Class JSD\\\\(bits)"],
            label_rows,
            "|p{0.30\\columnwidth}|p{0.30\\columnwidth}|p{0.30\\columnwidth}|"
        ))
    jsd_means, jsd_halfs = [], []
    for policy in PRIMARY_ORDER:
        values = selection_diag.loc[selection_diag.policy == policy,
                                    "mean_round_class_jsd_bits"].to_numpy(float)
        jsd_means.append(mean_ci(values)[0])
        jsd_halfs.append(mean_ci(values)[1])
    fig, ax = plt.subplots(figsize=(3.48, 2.55))
    labels = ["Fresh\nrandom", "TTL-8\nrandom", "Utility\nblind", "Full\ncross-layer"]
    colors = [COLORS[p] for p in PRIMARY_ORDER]
    ax.bar(np.arange(len(labels)), jsd_means, color=colors, width=.62,
           yerr=jsd_halfs, capsize=2, edgecolor="white", linewidth=.6)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_ylabel("Mean per-round class JSD (bits)")
    ax.set_ylim(0, max(jsd_means[i] + jsd_halfs[i] for i in range(4)) * 1.22)
    ax.grid(axis="y", color="#D9DDE2", linewidth=.5, alpha=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.1)
    fig.subplots_adjust(bottom=.22, left=.19, right=.98, top=.98)
    save_fig(fig, "selection_class_divergence.pdf")

    # Energy decomposition is generated from the per-run component accumulators.
    energy_rows = []
    for policy in POLICY_ORDER:
        d = runs[runs.policy == policy]
        cols = ["training_energy_j", "radio_energy_j", "client_kem_compute_energy_j", "client_energy_j"]
        vals = [mean_ci(d[c]) for c in cols]
        energy_rows.append([POLICY_LABELS[policy], *(cell(m, h, scale=1e-3, digits=2) for m, h in vals)])
    (TABLES / "energy_components.tex").write_text(boxed_table(
        "Run-level client-energy components in kJ (mean $\\pm$ 95\\% confidence-interval half-width); radio includes model and KEM-handshake traffic.",
        "tab:energy-components", ["Policy", "Training", "Radio", "KEM compute", "Total"],
        energy_rows, "|p{0.22\\columnwidth}|p{0.18\\columnwidth}|p{0.18\\columnwidth}|p{0.16\\columnwidth}|p{0.16\\columnwidth}|"))

    # Paired contrasts use the matched seed as the independent unit.
    contrast_rows = []
    lookup = runs.set_index(["seed", "policy"])
    for baseline in ["utility_blind", "random_ttl8", "random_fresh",
                     "cross_layer_no_debt", "cross_layer_debt_only"]:
        pairs = []
        for seed in sorted(runs.seed.unique()):
            a = lookup.loc[(seed, "cross_layer")]
            b = lookup.loc[(seed, baseline)]
            pairs.append({
                "accuracy_pp": 100.0 * (a.final_test_accuracy - b.final_test_accuracy),
                "energy_saving_pct": 100.0 * (b.client_energy_j - a.client_energy_j) / b.client_energy_j,
                "admitted": a.avg_selected_per_round - b.avg_selected_per_round,
            })
        contrasts = {}
        for metric in ("accuracy_pp", "energy_saving_pct", "admitted"):
            m, h = mean_ci([p[metric] for p in pairs])
            contrasts[metric] = cell(m, h, digits=2)
        contrast_rows.append([f"Cross-layer vs. {POLICY_LABELS[baseline]}",
                              contrasts["accuracy_pp"], contrasts["energy_saving_pct"],
                              contrasts["admitted"]])
    (TABLES / "paired_contrasts.tex").write_text(boxed_table(
        "Paired seed-level differences (cross-layer minus comparator for accuracy and admissions; positive energy denotes savings). Intervals use five independent paired seeds.",
        "tab:paired", ["Contrast", "Test accuracy (pp)", "Client energy saved (\\%)", "Clients\\\\per round"],
        contrast_rows, "|p{0.30\\columnwidth}|p{0.21\\columnwidth}|p{0.22\\columnwidth}|p{0.17\\columnwidth}|"))

    # Primitive table is calculated from the saved raw batch summary.
    kem = json.loads(KEM.read_text())
    op_keys = [
        ("ML-KEM-768 key generation", "mlkem768_keygen"),
        ("ML-KEM-768 encapsulation", "mlkem768_encapsulate"),
        ("ML-KEM-768 decapsulation", "mlkem768_decapsulate"),
        ("X25519 key generation", "x25519_keygen"),
        ("X25519 exchange", "x25519_exchange"),
    ]
    sizes = {
        "mlkem768_keygen": "PK 1,184 B",
        "mlkem768_encapsulate": "CT 1,088 B",
        "mlkem768_decapsulate": "SS 32 B",
        "x25519_keygen": "PK 32 B",
        "x25519_exchange": "SS 32 B",
    }
    kem_rows = []
    for label, key in op_keys:
        s = kem["summary"][key]["pooled"]
        kem_rows.append([label, f"{s['median_ms']:.4f}", f"{s['p95_ms']:.4f}", sizes[key]])
    (TABLES / "kem_microbenchmark.tex").write_text(boxed_table(
        "Host-measured cryptographic operation latency and serialized material. Times are pooled over five batches of 2,000 trials.",
        "tab:kem", ["Operation", "Median (ms)", "95th percentile (ms)", "Serialized material"],
        kem_rows, "|p{0.27\\columnwidth}|p{0.18\\columnwidth}|p{0.21\\columnwidth}|p{0.24\\columnwidth}|"))

    # Trial-level host primitive timing distributions with robust p5--p95
    # whiskers; system interruptions remain in the raw CSV and are not plotted.
    operation_order = ["mlkem768_keygen", "mlkem768_encapsulate",
                       "mlkem768_decapsulate", "x25519_keygen", "x25519_exchange"]
    operation_labels = ["ML-KEM\nkeygen", "ML-KEM\nencap.",
                        "ML-KEM\ndecap.", "X25519\nkeygen",
                        "X25519\nexchange"]
    timing = [pqc_raw.loc[pqc_raw.operation == op, "elapsed_ms"].to_numpy(float)
              for op in operation_order]
    fig, ax = plt.subplots(figsize=(3.48, 2.7))
    bp = ax.boxplot(timing, whis=(5, 95), showfliers=False, patch_artist=True,
                    widths=.54, medianprops={"color": "#28323C", "linewidth": 1.05},
                    whiskerprops={"color": "#65717D", "linewidth": .9},
                    capprops={"color": "#65717D", "linewidth": .9})
    for patch, color in zip(bp["boxes"], ["#4C78A8"] * 3 + ["#F58518"] * 2):
        patch.set_facecolor(color)
        patch.set_alpha(.72)
        patch.set_edgecolor("#3F4A55")
        patch.set_linewidth(.8)
    ax.set_xticks(np.arange(1, 6), labels=operation_labels)
    ax.set_ylabel("Host operation latency (ms)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9DDE2", linewidth=.5, alpha=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.0, pad=2)
    ax.tick_params(axis="y", labelsize=7.2)
    fig.subplots_adjust(left=.19, right=.98, bottom=.27, top=.98)
    save_fig(fig, "pqc_host_timing.pdf")

    # Learning curves use validation writers; final test writers are evaluated once per run.
    checkpoints = sorted(int(x) for x in rounds["round"].dropna().unique())
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    for policy in PRIMARY_ORDER:
        d = rounds[rounds.policy == policy]
        xs, means, lows, highs = [], [], [], []
        for r in checkpoints:
            vals = d.loc[d["round"] == r, "validation_accuracy"].dropna().to_numpy(float)
            if len(vals) < 2:
                continue
            m, half = mean_ci(vals)
            xs.append(r); means.append(100 * m); lows.append(100 * (m - half)); highs.append(100 * (m + half))
        if xs:
            ax.plot(xs, means, color=COLORS[policy], lw=1.55, marker="o", ms=2.5,
                    label=POLICY_LABELS[policy])
            ax.fill_between(xs, lows, highs, color=COLORS[policy], alpha=0.13, linewidth=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation-writer accuracy (%)")
    ax.grid(True, color="#D9DDE2", linewidth=0.5, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3,
              frameon=False, fontsize=6.5, handlelength=1.5, columnspacing=.8)
    fig.subplots_adjust(bottom=0.34, left=0.17, right=0.98, top=0.98)
    save_fig(fig, "femnist_learning_curve.pdf")

    # Utility-component ablation on the same paired validation-writer trajectory.
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    for policy in ["cross_layer_no_debt", "cross_layer_debt_only", "cross_layer"]:
        d = rounds[rounds.policy == policy]
        xs, means, lows, highs = [], [], [], []
        for r in checkpoints:
            vals = d.loc[d["round"] == r, "validation_accuracy"].dropna().to_numpy(float)
            if len(vals) < 2:
                continue
            m, half = mean_ci(vals)
            xs.append(r); means.append(100*m); lows.append(100*(m-half)); highs.append(100*(m+half))
        ax.plot(xs, means, color=COLORS[policy], lw=1.6, marker="o", ms=2.8,
                label=POLICY_LABELS[policy])
        ax.fill_between(xs, lows, highs, color=COLORS[policy], alpha=.14, linewidth=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation-writer accuracy (%)")
    ax.grid(True, color="#D9DDE2", linewidth=.5, alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.25), ncol=2,
              frameon=False, fontsize=6.7, handlelength=1.5, columnspacing=.8)
    fig.subplots_adjust(bottom=.30,left=.17,right=.98,top=.98)
    save_fig(fig,"utility_ablation_curve.pdf")

    # Paired cumulative modeled-energy savings by round, computed from executed logs.
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    savings_labels = {
        "random_fresh": "vs. fresh random",
        "random_ttl8": "vs. TTL-8 random",
        "utility_blind": "vs. crypto-blind utility",
    }
    indexed = rounds.set_index(["seed", "policy", "round"]).sort_index()
    seed_values = sorted(runs.seed.unique())
    for comparator, label in savings_labels.items():
        traces = []
        for seed in seed_values:
            proposed = indexed.loc[(seed, "cross_layer")].sort_index()
            baseline = indexed.loc[(seed, comparator)].sort_index()
            proposed = proposed.loc[proposed.index > 0, "client_energy_j"].fillna(0).to_numpy(float)
            baseline = baseline.loc[baseline.index > 0, "client_energy_j"].fillna(0).to_numpy(float)
            p_cum = np.cumsum(proposed)
            b_cum = np.cumsum(baseline)
            traces.append(b_cum - p_cum)
        a = np.vstack(traces)
        x = np.arange(1, a.shape[1] + 1)
        means = np.mean(a, axis=0)
        half = T95_N5 * np.std(a, axis=0, ddof=1) / math.sqrt(a.shape[0])
        color = COLORS[comparator]
        ax.plot(x, means, color=color, lw=1.55, label=label)
        ax.fill_between(x, means - half, means + half, color=color, alpha=0.13, linewidth=0)
    ax.axhline(0.0, color="#7A828B", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative modeled energy saved (J)")
    ax.grid(True, color="#D9DDE2", linewidth=0.5, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3,
              frameon=False, fontsize=6.5, handlelength=1.5, columnspacing=.8)
    fig.subplots_adjust(bottom=0.34, left=0.19, right=0.98, top=0.98)
    save_fig(fig, "femnist_energy_curve.pdf")

    # Freshness sensitivity: report handshake operations per admitted client-round.
    ttl_data = sens[sens.condition.astype(str).str.startswith("ttl_")]
    ttl_data = ttl_data[ttl_data.policy == "cross_layer"].copy()
    ttl_data["ttl"] = ttl_data.condition.str.removeprefix("ttl_").astype(int)
    ttl_data["refresh_per_admission"] = ttl_data["kem_refreshes"] / ttl_data["total_selected"].clip(lower=1)
    grouped_rows, grouped = [], []
    ttl_table_rows = []
    for ttl, g in ttl_data.groupby("ttl"):
        msel, hsel = mean_ci(g["avg_selected"])
        mrefresh, hrefresh = mean_ci(g["refresh_per_admission"])
        menc, henc = mean_ci(g["total_client_energy_j"])
        grouped.append((ttl, mrefresh, hrefresh))
        ttl_table_rows.append([str(ttl), cell(msel, hsel, digits=1),
                               cell(mrefresh, hrefresh, digits=3),
                               cell(menc, henc, scale=1e-3, digits=2)])
    (TABLES / "session_sensitivity.tex").write_text(boxed_table(
        "Cross-layer session-lifetime sensitivity over five independent seeds; energy is cumulative modeled client energy in kJ.",
        "tab:session-sensitivity", ["Maximum age (rounds)", "Clients\\\\per round", "Refreshes\\\\per admission", "Energy (kJ)"],
        ttl_table_rows, "|p{0.16\\columnwidth}|p{0.20\\columnwidth}|p{0.31\\columnwidth}|p{0.23\\columnwidth}|"))
    fig, ax = plt.subplots(figsize=(3.48, 2.55))
    ttl_x = [x[0] for x in grouped]
    ttl_y = [x[1] for x in grouped]
    ttl_h = [x[2] for x in grouped]
    ax.errorbar(ttl_x, ttl_y, yerr=ttl_h, color="#4C78A8",
                marker="o", ms=3.5, lw=1.5, capsize=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ttl_x, labels=[str(x) for x in ttl_x])
    ax.set_xlabel("Maximum session age (rounds)")
    ax.set_ylabel("Refreshes per admitted client-round")
    ax.grid(True, color="#D9DDE2", linewidth=0.5, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(bottom=0.22, left=0.20, right=0.98, top=0.98)
    save_fig(fig, "session_ttl_sensitivity.pdf")

    # Rate/deadline failure boundary: admissions are the observed system output.
    rate_data = sens[sens.condition.isin(["constrained_link", "moderate_link", "nominal_link", "fast_link"])]
    rate_data = rate_data[rate_data.policy == "cross_layer"]
    categories = ["constrained_link", "moderate_link", "nominal_link", "fast_link"]
    labels = ["1--5 Mb/s\n40 s", "10--20 Mb/s\n10 s", "10--100 Mb/s\n5 s", "50--100 Mb/s\n5 s"]
    means, halfs = [], []
    for c in categories:
        g = rate_data[rate_data.condition == c]["avg_selected"].to_numpy(float)
        m, h = mean_ci(g); means.append(m); halfs.append(h)
    fig, ax = plt.subplots(figsize=(3.48, 2.6))
    ax.bar(np.arange(len(labels)), means, color="#4C78A8", width=0.62,
           yerr=halfs, capsize=2, edgecolor="white", linewidth=0.6)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_ylabel("Admitted clients per round")
    ax.set_ylim(0, max(55, max(means + halfs) * 1.15))
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.0)
    fig.subplots_adjust(bottom=0.25, left=0.18, right=0.98, top=0.98)
    save_fig(fig, "admission_boundary.pdf")

    # Quantitative operating-boundary table: both scheduling policies use the
    # same per-seed channel trace and deadline in each condition.
    rate_data = sens[sens.condition.isin(["constrained_link", "moderate_link", "nominal_link", "fast_link"])]
    rate_labels = {
        "constrained_link": "1--5 Mb/s; 40 s",
        "moderate_link": "10--20 Mb/s; 10 s",
        "nominal_link": "10--100 Mb/s; 5 s",
        "fast_link": "50--100 Mb/s; 5 s",
    }
    rate_rows = []
    for cond in ["constrained_link", "moderate_link", "nominal_link", "fast_link"]:
        for policy in ["random_ttl8", "cross_layer"]:
            g = rate_data[(rate_data.condition == cond) & (rate_data.policy == policy)]
            rate_rows.append([rate_labels[cond], POLICY_LABELS[policy],
                              cell(*mean_ci(g.avg_selected), digits=1),
                              cell(*mean_ci(g.p95_round_completion_s), digits=2),
                              cell(*mean_ci(g.eligible_client_rounds / g.rounds), digits=1)])
    (TABLES / "admission_boundary.tex").write_text(boxed_table(
        "Admission and completion-time outcomes under literature-grounded link-rate envelopes and fixed deadlines (mean $\\pm$ 95\\% interval over five seeds).",
        "tab:admission-boundary", ["Rate range; deadline", "Policy", "Admitted\\\\per round", "95th-percentile completion (s)", "Eligible\\\\per round"],
        rate_rows, "|p{0.17\\columnwidth}|p{0.18\\columnwidth}|p{0.16\\columnwidth}|p{0.22\\columnwidth}|p{0.17\\columnwidth}|"))

    # Cohort-cap scale and radio/budget robustness tables come from the same
    # complete writer population, with no learning-accuracy inference.
    cohort = sens[sens.condition.astype(str).str.startswith("cohort_")]
    cohort_rows = []
    for cond in ["cohort_20", "cohort_50", "cohort_100"]:
        g = cohort[cohort.condition == cond]
        cap = int(cond.removeprefix("cohort_"))
        cohort_rows.append([str(cap), cell(*mean_ci(g.avg_selected), digits=1),
                            cell(*mean_ci(g.jain_participation_fairness), digits=3),
                            cell(*mean_ci(g.total_client_energy_j), scale=1e-3, digits=2)])
    (TABLES / "cohort_scale.tex").write_text(boxed_table(
        "Cohort-cap sensitivity over the complete optimization-writer population (mean $\\pm$ 95\\% interval over five seeds).",
        "tab:cohort-scale", ["Maximum cohort size", "Admitted\\\\per round", "Jain fairness", "Energy (kJ)"],
        cohort_rows, "|p{0.23\\columnwidth}|p{0.22\\columnwidth}|p{0.23\\columnwidth}|p{0.22\\columnwidth}|"))
    rb = sens[sens.condition.astype(str).str.startswith("radio")]
    radio_rows = []
    for scale in (0.5, 1.0, 2.0):
        for budget in (0.70, 0.85, 1.00):
            cond = f"radio{scale}_budget{budget}"
            g = rb[rb.condition == cond]
            radio_rows.append([f"{scale:.1f}", f"{budget:.2f}",
                               cell(*mean_ci(g.avg_selected), digits=1),
                               cell(*mean_ci(g.total_client_energy_j), scale=1e-3, digits=2)])
    (TABLES / "radio_budget_sensitivity.tex").write_text(boxed_table(
        "Joint radio-energy multiplier and aggregate-budget sensitivity for the cross-layer scheduler (mean $\\pm$ 95\\% interval over five seeds).",
        "tab:radio-budget", ["Radio multiplier", "Budget fraction", "Admitted\\\\per round", "Energy (kJ)"],
        radio_rows, "|p{0.20\\columnwidth}|p{0.20\\columnwidth}|p{0.25\\columnwidth}|p{0.25\\columnwidth}|"))

    # Paired profile sensitivity: compare only the same utility rule with and
    # without KEM terms in its ranking, using each seed's common trace.
    pqc_rows = {"energy": [], "latency": []}
    pqc_plot = {"energy": [], "latency": []}
    for axis in ("energy", "latency"):
        for level in (0.0, 0.5, 1.0, 2.0, 4.0):
            g = pqc_sens[(pqc_sens.condition == f"kem_{axis}_{level:g}")]
            cross = g[g.policy == "cross_layer"].set_index("seed").sort_index()
            blind = g[g.policy == "utility_blind"].set_index("seed").sort_index()
            if list(cross.index) != list(blind.index) or len(cross) != 5:
                raise ValueError(f"Unpaired or incomplete KEM profile sensitivity: {axis}/{level}")
            admitted_delta = cross.avg_selected.to_numpy() - blind.avg_selected.to_numpy()
            feasible_per_round = cross.eligible_client_rounds.to_numpy() / cross.rounds.to_numpy()
            energy_saved = 100 * (blind.total_client_energy_j.to_numpy() -
                                  cross.total_client_energy_j.to_numpy()) / blind.total_client_energy_j.to_numpy()
            p95_saved = (blind.p95_round_completion_s.to_numpy() -
                         cross.p95_round_completion_s.to_numpy())
            refresh_saved = (blind.kem_refreshes.to_numpy() - cross.kem_refreshes.to_numpy())
            md, hd = mean_ci(admitted_delta)
            mf, hf = mean_ci(feasible_per_round)
            me, he = mean_ci(energy_saved)
            mt, ht = mean_ci(p95_saved)
            mr, hr = mean_ci(refresh_saved)
            pqc_plot[axis].append((level, md, hd))
            row = [f"{level:g}", cell(mf, hf, digits=1), cell(md, hd, digits=2),
                   cell(me, he, digits=2), cell(mr, hr, digits=1)]
            if axis == "latency":
                row = [f"{level:g}", cell(mf, hf, digits=1), cell(md, hd, digits=2),
                       cell(mt, ht, digits=2), cell(mr, hr, digits=1)]
            pqc_rows[axis].append(row)
    (TABLES / "pqc_energy_sensitivity.tex").write_text(boxed_table(
        "Under each KEM energy scale, deadline-feasible candidates are for cross-layer scheduling; admission is cross-layer minus crypto-blind, while energy and refresh savings are crypto-blind minus cross-layer (five paired seeds).",
        "tab:pqc-energy-sensitivity",
        [r"KEM energy\\multiplier", r"Eligible\\per round", r"Admission\\difference", r"Client energy\\saved (\%)", r"Refreshes\\saved"],
        pqc_rows["energy"],
        "|p{0.13\\columnwidth}|p{0.21\\columnwidth}|p{0.19\\columnwidth}|p{0.20\\columnwidth}|p{0.17\\columnwidth}|"))
    (TABLES / "pqc_latency_sensitivity.tex").write_text(boxed_table(
        "Under each KEM latency scale, deadline-feasible candidates are for cross-layer scheduling; admission difference is cross-layer minus crypto-blind, while p95-time and refresh savings are crypto-blind minus cross-layer (five paired seeds).",
        "tab:pqc-latency-sensitivity",
        [r"KEM latency\\multiplier", r"Eligible\\per round", r"Admission\\difference", r"p95 completion\\saved (s)", r"Refreshes\\saved"],
        pqc_rows["latency"],
        "|p{0.13\\columnwidth}|p{0.21\\columnwidth}|p{0.19\\columnwidth}|p{0.20\\columnwidth}|p{0.17\\columnwidth}|"))
    fig, ax = plt.subplots(figsize=(3.48, 2.55))
    for axis, label, color, marker in (
            ("energy", "Operation energy", "#4C78A8", "o"),
            ("latency", "Operation latency", "#E45756", "s")):
        vals = pqc_plot[axis]
        x = np.array([v[0] for v in vals])
        group_means, group_halfs = [], []
        for v in vals:
            g = pqc_sens[pqc_sens.condition == f"kem_{axis}_{v[0]:g}"]
            g = g[g.policy == "cross_layer"]
            mean, half = mean_ci(g.eligible_client_rounds / g.rounds)
            group_means.append(mean); group_halfs.append(half)
        y = np.asarray(group_means)
        h = np.asarray(group_halfs)
        ax.errorbar(x, y, yerr=h, color=color, marker=marker, ms=3.5,
                    lw=1.45, capsize=2, label=label)
    ax.axhline(0.0, color="#7A828B", lw=.8, ls="--", zorder=0)
    ax.set_xlabel("Multiplier on published client KEM profile")
    ax.set_ylabel("Deadline-feasible clients per round")
    ax.set_xticks((0, .5, 1, 2, 4), labels=("0", "0.5", "1", "2", "4"))
    ax.grid(True, color="#D9DDE2", linewidth=.5, alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.3)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.25), ncol=2,
              frameon=False, fontsize=6.8, handlelength=1.4)
    fig.subplots_adjust(bottom=.31, left=.20, right=.98, top=.98)
    save_fig(fig, "pqc_cost_sensitivity.pdf")

    # Experiment-derived client-energy composition (one-column figure).
    component_cols = ["training_energy_j", "radio_energy_j", "client_kem_compute_energy_j"]
    component_labels = ["Local training", "Model and handshake radio", "ML-KEM compute"]
    component_colors = ["#4C78A8", "#F58518", "#54A24B"]
    means, errors = [], []
    for p in POLICY_ORDER:
        d = runs[runs.policy == p]
        means.append([np.mean(d[c]) / 1000.0 for c in component_cols])
        errors.append([np.std(d[c], ddof=1) / np.sqrt(len(d)) / 1000.0 for c in component_cols])
    fig, ax = plt.subplots(figsize=(3.48, 2.65))
    bottom = np.zeros(len(POLICY_ORDER))
    for j, (label, color) in enumerate(zip(component_labels, component_colors)):
        vals = np.asarray([m[j] for m in means])
        ax.bar(np.arange(len(POLICY_ORDER)), vals, bottom=bottom, color=color,
               edgecolor="white", linewidth=.5, width=.64, label=label)
        bottom += vals
    ax.set_xticks(np.arange(len(POLICY_ORDER)),
                  labels=["Fresh\nrandom", "TTL-8\nrandom", "Utility\nblind",
                          "No\ndebt", "Debt\nonly", "Full\ncross-layer"])
    ax.set_ylabel("Modeled client energy (kJ)")
    ax.grid(axis="y", color="#D9DDE2", linewidth=.5, alpha=.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.0)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.28), ncol=1,
              frameon=False, fontsize=6.7, handlelength=1.4, labelspacing=.2)
    fig.subplots_adjust(bottom=.43,left=.18,right=.98,top=.98)
    save_fig(fig,"energy_components.pdf")

    # Average admissions across radio multiplier and energy budget.
    piv = np.zeros((3,3))
    scales, budgets = (.5,1.0,2.0), (.70,.85,1.00)
    for i,scale in enumerate(scales):
        for j,budget in enumerate(budgets):
            cond=f"radio{scale}_budget{budget}"
            piv[i,j]=rb[rb.condition==cond].avg_selected.mean()
    fig, ax = plt.subplots(figsize=(3.48,2.55))
    im=ax.imshow(piv, cmap="Blues", aspect="auto", vmin=0, vmax=max(50,float(piv.max())))
    ax.set_xticks(range(3),labels=["0.70","0.85","1.00"])
    ax.set_yticks(range(3),labels=["0.5","1.0","2.0"])
    ax.set_xlabel("Energy-budget fraction"); ax.set_ylabel("Radio-energy multiplier")
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04); cb.set_label("Admitted clients/round",fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=.22,right=.89,bottom=.20,top=.98)
    save_fig(fig,"radio_budget_heatmap.pdf")

    # Client-cohort-cap scalability, with run-level seed uncertainty.
    fig, ax = plt.subplots(figsize=(3.48,2.55))
    xscale, ymean, yhalf = [], [], []
    for cap in (20,50,100):
        g=cohort[cohort.condition==f"cohort_{cap}"]
        m,h=mean_ci(g.avg_selected)
        xscale.append(cap); ymean.append(m); yhalf.append(h)
    ax.errorbar(xscale,ymean,yerr=yhalf,color="#4C78A8",marker="o",ms=4,
                lw=1.6,capsize=2,label="Observed admissions")
    ax.plot(xscale,xscale,color="#8B949E",lw=.9,ls="--",label="Admission cap")
    ax.set_xlabel("Maximum clients per round")
    ax.set_ylabel("Admitted clients per round")
    ax.set_xticks(xscale)
    ax.grid(True,color="#D9DDE2",linewidth=.5,alpha=.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.25),frameon=False,
              fontsize=6.8,ncol=2,handlelength=1.5)
    fig.subplots_adjust(bottom=.30,left=.18,right=.98,top=.98)
    save_fig(fig,"cohort_scalability.pdf")

    print("Generated IEEE tables and experiment-derived figures from saved raw outputs.")


if __name__ == "__main__":
    main()
