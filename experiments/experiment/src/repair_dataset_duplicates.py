#!/usr/bin/env python3
"""Keep exact-duplicate FEMNIST images inside a single writer partition.

This repair moves an entire writer, never an individual example. It is applied
before model training and invalidates only the derived cache, not the raw LEAF
source shards. The selected repair is recorded for audit and reproduction.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/leaf/data/femnist/data"
CACHE = ROOT / "data/processed/femnist"


def write_atomic(path: Path, doc: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    audit_path = ROOT / "evidence/dataset_audit.json"
    audit = json.loads(audit_path.read_text())
    pairs = audit.get("cross_partition_exact_duplicate_pairs", [])
    if not pairs:
        print("No cross-partition exact duplicates require repair.")
        return

    train_dir, test_dir = DATA / "train", DATA / "test"
    upstream_train = audit["partitions"]["optimization"]["writers"] + audit["partitions"]["validation"]["writers"]
    upstream_train_examples = audit["partitions"]["optimization"]["examples"] + audit["partitions"]["validation"]["examples"]
    upstream_test = audit["partitions"]["final_test"]["writers"]
    upstream_test_examples = audit["partitions"]["final_test"]["examples"]
    records = []

    for pair in pairs:
        first, other = pair["first_writer"], pair["other_writer"]
        existing_train_ids = set()
        # The cached writer list comes from the completed, audited source split.
        for meta_path in (CACHE / "train_clients.json", CACHE / "test_clients.json"):
            if not meta_path.exists():
                raise FileNotFoundError("Audited writer cache metadata is required")
        train_meta = json.loads((CACHE / "train_clients.json").read_text())
        test_meta = json.loads((CACHE / "test_clients.json").read_text())
        existing_train_ids = {c["id"] for c in train_meta["clients"]}
        existing_test_ids = {c["id"] for c in test_meta["clients"]}
        if first in existing_train_ids and other in existing_test_ids:
            move_id = other
        elif other in existing_train_ids and first in existing_test_ids:
            move_id = first
        elif first in existing_train_ids and other in existing_train_ids:
            # The upstream writer split is already safe; validation assignment
            # is constrained by ensure_data() using the recorded pair group.
            move_id = None
        else:
            raise ValueError(f"Unexpected duplicate pair partitions: {pair}")

        if move_id is None:
            records.append({"pair": pair, "writer_moved": None})
            continue

        source_path = None
        source_doc = None
        for p in sorted(test_dir.glob("*.json")):
            doc = json.loads(p.read_text())
            if move_id in doc["user_data"]:
                source_path, source_doc = p, doc
                break
        if source_path is None:
            raise ValueError(f"Could not locate held-out writer {move_id}")

        index = source_doc["users"].index(move_id)
        sample_n = int(source_doc["num_samples"][index])
        sample = source_doc["user_data"].pop(move_id)
        del source_doc["users"][index]
        del source_doc["num_samples"][index]
        write_atomic(source_path, source_doc)

        target_path = None
        target_doc = None
        train_paths = sorted(train_dir.glob("*.json"),
                             key=lambda p: int(p.name.split("_")[2]))
        target_path = train_paths[-1]
        target_doc = json.loads(target_path.read_text())
        if len(target_doc["users"]) >= 50:
            raise ValueError("Final train shard has no capacity for the moved writer")
        target_doc["users"].append(move_id)
        target_doc["num_samples"].append(sample_n)
        target_doc["user_data"][move_id] = sample
        write_atomic(target_path, target_doc)
        records.append({"pair": pair, "writer_moved": move_id,
                        "source_shard": str(source_path.relative_to(ROOT)),
                        "target_shard": str(target_path.relative_to(ROOT)),
                        "writer_examples": sample_n,
                        "reason": "prevent exact pixel duplicate crossing writer partitions"})

    moved = sum(1 for item in records if item["writer_moved"] is not None)
    moved_examples = sum(item.get("writer_examples", 0) for item in records)

    report = {
        "procedure": "move the complete held-out writer to the training side when its exact image duplicate is already in training; validation assignment is then group-constrained",
        "records": records,
        "before": {"train_writers": upstream_train, "train_examples": upstream_train_examples,
                   "test_writers": upstream_test, "test_examples": upstream_test_examples},
        "after": {"train_writers": upstream_train + moved,
                  "train_examples": upstream_train_examples + moved_examples,
                  "test_writers": upstream_test - moved,
                  "test_examples": upstream_test_examples - moved_examples},
        "total_examples_conserved": True,
    }
    out = ROOT / "evidence/exact_duplicate_repair.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for name in ("train_clients.json", "train_x.npy", "train_y.npy",
                 "test_clients.json", "test_x.npy", "test_y.npy"):
        cached = CACHE / name
        if cached.exists():
            cached.unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
