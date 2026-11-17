#!/usr/bin/env python3
"""Produce a compact profile of the complete audited FEMNIST writer split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/raw/femnist_campaign/dataset_manifest.json"


def boxed_table(rows: list[list[str]]) -> str:
    lines = [
        r"\begin{table}[!ht]",
        r"\caption{Empirical heterogeneity in the complete optimization partition.}",
        r"\label{tab:client-profile}",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{|p{0.48\columnwidth}|r|}",
        r"\hline",
        r"Statistic & Value \\",
        r"\hline",
    ]
    for name, value in rows:
        lines.extend([f"{name} & {value} " + chr(92) * 2, r"\hline"])
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    doc = json.loads(MANIFEST.read_text())
    part = doc["optimization_train"]
    sizes = np.asarray([c["n"] for c in part["clients"]], dtype=np.int64)
    class_counts = np.asarray(part["class_counts"], dtype=np.int64)
    y_store = np.load(ROOT / "data/processed/femnist/train_y.npy", mmap_mode="r")
    per_writer_class_counts = np.zeros((len(sizes), 62), dtype=np.int64)
    for i, client in enumerate(part["clients"]):
        start, end = client["offset"], client["offset"] + client["n"]
        per_writer_class_counts[i] = np.bincount(
            np.asarray(y_store[start:end], dtype=np.int64), minlength=62)
    probabilities = per_writer_class_counts / sizes[:, None]
    entropies = np.zeros(len(sizes), dtype=float)
    for i, p in enumerate(probabilities):
        nz = p > 0
        entropies[i] = -np.sum(p[nz] * np.log(p[nz])) / np.log(62)
    support = np.count_nonzero(per_writer_class_counts, axis=1)
    dominant_fraction = probabilities.max(axis=1)
    gini = float(np.abs(sizes[:, None] - sizes[None, :]).mean() / (2 * sizes.mean()))
    class_ratio = float(class_counts.max() / class_counts.min())
    rows = [
        ["Writers", f"{len(sizes):,}"],
        ["Examples per writer, minimum", f"{sizes.min():,}"],
        ["Examples per writer, 5th percentile", f"{np.percentile(sizes, 5):.0f}"],
        ["Examples per writer, 25th percentile", f"{np.percentile(sizes, 25):.2f}"],
        ["Examples per writer, median", f"{np.median(sizes):.0f}"],
        ["Examples per writer, 75th percentile", f"{np.percentile(sizes, 75):.0f}"],
        ["Examples per writer, 95th percentile", f"{np.percentile(sizes, 95):.0f}"],
        ["Examples per writer, maximum", f"{sizes.max():,}"],
        ["Mean examples per writer (SD)", f"{sizes.mean():.2f} ({sizes.std(ddof=1):.2f})"],
        ["Writer-size Gini coefficient", f"{gini:.3f}"],
        ["Class examples, minimum / median / maximum", f"{class_counts.min():,} / {np.median(class_counts):,.0f} / {class_counts.max():,}"],
        ["Maximum-to-minimum class-count ratio", f"{class_ratio:.2f}"],
        ["Classes per writer, 5th / median / 95th percentiles", f"{np.percentile(support, 5):.0f} / {np.median(support):.0f} / {np.percentile(support, 95):.0f}"],
        ["Normalized label entropy, 5th / median / 95th percentiles", f"{np.percentile(entropies, 5):.3f} / {np.median(entropies):.3f} / {np.percentile(entropies, 95):.3f}"],
        ["Dominant class fraction, median / 95th percentile", f"{np.median(dominant_fraction):.3f} / {np.percentile(dominant_fraction, 95):.3f}"],
        ["Spearman correlation: writer size vs. label entropy", f"{spearmanr(sizes, entropies).statistic:.3f}"],
    ]
    table_path = ROOT / "results/tables/femnist_client_profile.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(boxed_table(rows))
    x = np.sort(sizes)
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(3.48, 2.5))
    ax.step(x, y, where="post", color="#4C78A8", lw=1.5)
    ax.axvline(np.median(sizes), color="#E45756", lw=1.0, ls="--")
    ax.axhline(.5, color="#8B949E", lw=.8, ls=":")
    ax.set_xlabel("Training examples held by one writer")
    ax.set_ylabel("Cumulative fraction of writers")
    ax.set_xlim(0, sizes.max())
    ax.set_ylim(0, 1)
    ax.grid(True, color="#D9DDE2", linewidth=.5, alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=.20, right=.98, bottom=.20, top=.98)
    fig.savefig(ROOT / "figures/femnist_writer_sizes.pdf", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    x = np.sort(entropies)
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(3.48, 2.5))
    ax.step(x, y, where="post", color="#54A24B", lw=1.5)
    ax.axvline(np.median(entropies), color="#E45756", lw=1.0, ls="--")
    ax.axhline(.5, color="#8B949E", lw=.8, ls=":")
    ax.set_xlabel("Per-writer label entropy (normalized)")
    ax.set_ylabel("Cumulative fraction of writers")
    ax.set_xlim(.5, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#D9DDE2", linewidth=.5, alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=.20, right=.98, bottom=.20, top=.98)
    fig.savefig(ROOT / "figures/femnist_label_entropy.pdf", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    summary = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": digest,
        "source_commit": doc["source_repo_commit"],
        "optimization_writers": len(sizes),
        "optimization_examples": int(sizes.sum()),
        "writer_size_gini": gini,
        "writer_size_quantiles": {str(q): float(np.percentile(sizes, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "class_counts": class_counts.tolist(),
        "class_imbalance_ratio": class_ratio,
        "writer_class_support_quantiles": {str(q): float(np.percentile(support, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "writer_normalized_entropy_quantiles": {str(q): float(np.percentile(entropies, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "dominant_class_fraction_quantiles": {str(q): float(np.percentile(dominant_fraction, q)) for q in (0, 50, 95, 100)},
        "writer_size_entropy_spearman": float(spearmanr(sizes, entropies).statistic),
    }
    (ROOT / "evidence/dataset_profile.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Profiled all {len(sizes)} optimization writers and {int(sizes.sum())} examples.")


if __name__ == "__main__":
    main()
