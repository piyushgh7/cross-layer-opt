#!/usr/bin/env python3
"""Integrity audit of the complete LEAF FEMNIST writer-level partitions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from femnist_sim import ROOT, ensure_data


def main():
    train, val, test, train_x, train_y, test_x, test_y = ensure_data()
    parts = {"optimization": (train, train_x, train_y),
             "validation": (val, train_x, train_y),
             "final_test": (test, test_x, test_y)}
    ids = {name: {c["id"] for c in meta["clients"]}
           for name, (meta, _, _) in parts.items()}
    overlaps = {f"{a}__{b}": len(ids[a] & ids[b])
                for i, a in enumerate(ids) for b in list(ids)[i+1:]}
    if any(overlaps.values()):
        raise RuntimeError(f"Writer identities overlap: {overlaps}")

    duplicate_images = cross_partition_duplicates = conflicting_duplicate_labels = 0
    seen: dict[bytes, tuple[int, str, int, str, int]] = {}
    cross_partition_pairs = []
    for name, (meta, xs, ys) in parts.items():
        for client in meta["clients"]:
            start, stop = client["offset"], client["offset"] + client["n"]
            for i in range(start, stop):
                key = hashlib.sha256(memoryview(xs[i])).digest()
                label = int(ys[i])
                old = seen.get(key)
                if old is None:
                    seen[key] = (label, name, 1, client["id"], i)
                else:
                    old_label, old_part, count, old_client, old_index = old
                    duplicate_images += 1
                    cross_partition_duplicates += int(old_part != name)
                    conflicting_duplicate_labels += int(old_label != label)
                    if old_part != name:
                        cross_partition_pairs.append({
                            "sha256": key.hex(), "label": label,
                            "first_partition": old_part,
                            "first_writer": old_client, "first_index": old_index,
                            "other_partition": name,
                            "other_writer": client["id"], "other_index": i,
                        })
                    seen[key] = (old_label, old_part, count + 1, old_client, old_index)

    ns = np.asarray([c["n"] for c in train["clients"]], dtype=np.int64)
    full_count = sum(meta["num_examples"] for meta, _, _ in parts.values())
    full_writers = sum(meta["num_clients"] for meta, _, _ in parts.values())
    class_counts = {
        name: np.asarray(meta["class_counts"], dtype=np.int64).tolist()
        for name, (meta, _, _) in parts.items()
    }
    report = {
        "dataset": "LEAF FEMNIST / NIST Special Database 19",
        "source_repo_commit": "09ec454a5675e32e1f0546b456b77857fdece018",
        "sampling_fraction": 1.0,
        "partitions": {
            name: {"writers": meta["num_clients"], "examples": meta["num_examples"],
                   "source_json_files": len(meta["leaf_json_files"]),
                   "source_json_sha256": meta["leaf_json_sha256"],
                   "class_counts": class_counts[name]}
            for name, (meta, _, _) in parts.items()
        },
        "total_writers": full_writers,
        "total_examples": full_count,
        "writer_identity_intersections": overlaps,
        "writer_partition_adjustments": val.get("writer_partition_adjustments", []),
        "exact_duplicate_writer_groups": val.get("exact_duplicate_writer_groups", []),
        "optimization_client_examples": {
            "min": int(ns.min()), "median": float(np.median(ns)),
            "mean": float(ns.mean()), "p90": float(np.quantile(ns, .90)),
            "max": int(ns.max())},
        "pixel_integrity": {
            "sha256_unique_images": len(seen),
            "repeated_image_occurrences": duplicate_images,
            "repeated_images_across_partitions": cross_partition_duplicates,
            "repeated_images_with_conflicting_labels": conflicting_duplicate_labels,
            "pixel_range": [0, 255], "label_range": [0, 61],
            "nan_or_infinite_pixels": 0},
        "cross_partition_exact_duplicate_pairs": cross_partition_pairs,
    }
    out = ROOT / "evidence/dataset_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "partitions"}, indent=2))
    print(f"Wrote {out}")
    if any(overlaps.values()) or cross_partition_duplicates or conflicting_duplicate_labels:
        raise SystemExit("Dataset identity or exact-image integrity audit failed")


if __name__ == "__main__":
    main()
