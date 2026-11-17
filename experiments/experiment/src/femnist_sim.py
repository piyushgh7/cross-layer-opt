#!/usr/bin/env python3
"""Full-client FEMNIST FedAvg simulator with a PQC-aware admission layer.

The program deliberately distinguishes measured host ML-KEM latency from the
literature-calibrated edge energy/latency model used by the system simulation.
All dataset clients and all samples in the LEAF writer split are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import cryptography
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/leaf/data/femnist/data"
CACHE = ROOT / "data/processed/femnist"
OUT = ROOT / "results/raw/femnist_campaign"

POLICIES = {
    "random_fresh": {"kind": "random", "ttl": 1, "crypto_aware": False},
    "random_ttl8": {"kind": "random", "ttl": 8, "crypto_aware": False},
    "utility_blind": {"kind": "utility", "ttl": 8, "crypto_aware": False},
    "cross_layer_no_debt": {"kind": "utility", "ttl": 8, "crypto_aware": True,
                             "utility": "sample_only"},
    "cross_layer_debt_only": {"kind": "utility", "ttl": 8, "crypto_aware": True,
                               "utility": "debt_only"},
    "cross_layer": {"kind": "utility", "ttl": 8, "crypto_aware": True,
                     "utility": "sample_and_debt"},
}

# Literature calibration profile. These are not measurements on the present host.
CPU_BATCH_J = 0.300          # FEMNIST/LeNet-5, Xiaomi 10 CPU FP32, batch=5
CPU_BATCH_S = 0.058
SGD_MOMENTUM = 0.9            # Literature-grounded FEMNIST local optimizer setting
KEM768_ENCAP_J = 0.016090    # ESP32-P4, peer-reviewed 2026 hardware table
KEM768_ENCAP_S = 0.205441
KEM768_DECAP_J = 0.018016
KEM768_DECAP_S = 0.230032
KEM768_KEYGEN_J = 0.015007
KEM768_KEYGEN_S = 0.191612
MLKEM_PK_B = 1184
MLKEM_CT_B = 1088
WIFI_RX_W = 0.823
WIFI_TX_W = 0.966


class FEMNISTNet(nn.Module):
    """Compact two-convolution model (~1.7 MB) matching the study's profile."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, 62),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_files(split: str) -> list[Path]:
    folder = DATA / split
    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No LEAF {split} JSON files under {folder}")
    return files


def cache_split(split: str) -> dict:
    """Convert LEAF JSON shards to memory-mapped uint8 arrays without sampling."""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / f"{split}_clients.json"
    x_path, y_path = CACHE / f"{split}_x.npy", CACHE / f"{split}_y.npy"
    if meta_path.exists() and x_path.exists() and y_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta

    files = source_files(split)
    client_meta = []
    seen = set()
    total = 0
    labels = np.zeros(62, dtype=np.int64)
    for path in files:
        with path.open() as f:
            doc = json.load(f)
        if len(doc["users"]) != len(doc["num_samples"]):
            raise ValueError(f"LEAF metadata mismatch in {path}")
        for uid, n in zip(doc["users"], doc["num_samples"]):
            if uid in seen:
                raise ValueError(f"Duplicate writer {uid!r} within {split}")
            seen.add(uid)
            client_meta.append({"id": uid, "n": int(n), "offset": total})
            total += int(n)

    x_mm = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8,
                                     shape=(total, 784))
    y_mm = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.uint8,
                                     shape=(total,))
    cursor = 0
    for path in files:
        with path.open() as f:
            doc = json.load(f)
        for uid, declared_n in zip(doc["users"], doc["num_samples"]):
            row = doc["user_data"][uid]
            xs = np.asarray(row["x"], dtype=np.float32)
            ys = np.asarray(row["y"], dtype=np.int64)
            if xs.ndim != 2 or xs.shape[1] != 784 or len(xs) != len(ys):
                raise ValueError(f"Invalid image/label dimensions for writer {uid}")
            declared_n = int(declared_n)
            if len(ys) != declared_n:
                raise ValueError(f"LEAF num_samples mismatch for writer {uid}: "
                                 f"metadata={declared_n}, observed={len(ys)}")
            if not np.isfinite(xs).all() or xs.min() < -1e-6 or xs.max() > 1.000001:
                raise ValueError(f"Invalid pixel values for writer {uid}")
            if ys.min(initial=0) < 0 or ys.max(initial=0) >= 62:
                raise ValueError(f"Invalid class label for writer {uid}")
            count = len(ys)
            x_mm[cursor:cursor + count] = np.rint(xs * 255).astype(np.uint8)
            y_mm[cursor:cursor + count] = ys.astype(np.uint8)
            labels += np.bincount(ys, minlength=62)
            cursor += count
        del doc
    x_mm.flush(); y_mm.flush()
    if cursor != total:
        raise RuntimeError(f"Cache row mismatch: wrote {cursor}, expected {total}")
    meta = {
        "split": split,
        "clients": client_meta,
        "num_clients": len(client_meta),
        "num_examples": total,
        "class_counts": labels.tolist(),
        "leaf_json_files": [str(p.relative_to(ROOT)) for p in files],
        "leaf_json_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    del x_mm, y_mm
    return meta


def ensure_data(load_test: bool = True) -> tuple[dict, dict, dict, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    source_train_meta = cache_split("train")
    test_meta = cache_split("test")
    train_ids = {x["id"] for x in source_train_meta["clients"]}
    test_ids = {x["id"] for x in test_meta["clients"]}
    if train_ids & test_ids:
        raise ValueError("Writer leakage: train/test users overlap")
    # Reserve complete writers for validation before any optimization. All native
    # samples remain assigned to optimization, validation, or final test.
    rng = np.random.default_rng(20261012)
    order = rng.permutation(len(source_train_meta["clients"]))
    n_val = max(1, round(0.10 * len(order)))
    val_idx = set(order[:n_val].tolist())
    # Exact duplicate images linking different writers are treated as a group
    # for the validation boundary. This is driven by the pre-training integrity
    # audit, not model outcomes; it prevents duplicate pixels crossing splits.
    repair_path = ROOT / "evidence/exact_duplicate_repair.json"
    partition_adjustments = []
    duplicate_writer_groups = []
    if repair_path.exists():
        repair = json.loads(repair_path.read_text())
        id_to_index = {c["id"]: i for i, c in enumerate(source_train_meta["clients"])}
        grouped_indices = set()
        for record in repair.get("records", []):
            pair = record["pair"]
            group_ids = {pair["first_writer"], pair["other_writer"]}
            if not group_ids.issubset(id_to_index):
                raise ValueError(f"Exact-duplicate writer group is not wholly in training: {group_ids}")
            group = {id_to_index[uid] for uid in group_ids}
            grouped_indices.update(group)
            val_members = group & val_idx
            if val_members and val_members != group:
                val_idx.difference_update(group)
                partition_adjustments.append({
                    "writers": sorted(group_ids),
                    "assigned_partition": "optimization",
                    "reason": "keep exact duplicate images within one writer partition",
                })
            if group.issubset(val_idx):
                assigned = "validation"
            elif not (group & val_idx):
                assigned = "optimization"
            else:
                raise RuntimeError("Exact-duplicate writers remain split across validation")
            duplicate_writer_groups.append({"writers": sorted(group_ids),
                                            "assigned_partition": assigned})
        for candidate in order:
            if len(val_idx) >= n_val:
                break
            candidate = int(candidate)
            if candidate not in val_idx and candidate not in grouped_indices:
                val_idx.add(candidate)
    train_clients = [c for i, c in enumerate(source_train_meta["clients"]) if i not in val_idx]
    val_clients = [c for i, c in enumerate(source_train_meta["clients"]) if i in val_idx]
    train_meta = dict(source_train_meta) | {"clients": train_clients,
        "num_clients": len(train_clients),
        "num_examples": sum(c["n"] for c in train_clients)}
    val_meta = dict(source_train_meta) | {"clients": val_clients,
        "num_clients": len(val_clients),
        "num_examples": sum(c["n"] for c in val_clients)}
    val_meta["writer_partition_adjustments"] = partition_adjustments
    val_meta["exact_duplicate_writer_groups"] = duplicate_writer_groups
    train_ids = {x["id"] for x in train_clients}
    val_ids = {x["id"] for x in val_clients}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Writer identity leakage across optimization/validation/test")
    train_x = np.load(CACHE / "train_x.npy", mmap_mode="r")
    train_y = np.load(CACHE / "train_y.npy", mmap_mode="r")
    if load_test:
        test_x = np.load(CACHE / "test_x.npy", mmap_mode="r")
        test_y = np.load(CACHE / "test_y.npy", mmap_mode="r")
    else:
        test_x = test_y = None
    for meta in (train_meta, val_meta):
        counts = np.zeros(62, dtype=np.int64)
        for client in meta["clients"]:
            s, e = client["offset"], client["offset"] + client["n"]
            counts += np.bincount(np.asarray(train_y[s:e], dtype=np.int64), minlength=62)
        meta["class_counts"] = counts.tolist()
    return train_meta, val_meta, test_meta, train_x, train_y, test_x, test_y


def make_costs(meta: dict, rates: np.ndarray, participation: np.ndarray,
               last_refresh: np.ndarray, round_id: int, ttl: int,
               model_bytes: int, deadline_s: float) -> dict:
    n = np.asarray([c["n"] for c in meta["clients"]], dtype=np.int64)
    batch_equivalent = n.astype(np.float64) / 5.0
    train_j = batch_equivalent * CPU_BATCH_J
    train_s = batch_equivalent * CPU_BATCH_S
    # The independently published Wi-Fi model is converted from % battery/hour
    # using the HTC Desire HD 4.55-Wh pack documented in its device specification.
    one_way_s = (model_bytes * 8.0) / (rates * 1e6)
    transfer_s = 2.0 * one_way_s
    radio_j = (WIFI_RX_W + WIFI_TX_W) * one_way_s
    expired = (last_refresh < 0) | ((round_id - last_refresh) >= ttl)
    # Client creates an ephemeral KEM key pair and decapsulates the server's
    # response; server encapsulation proceeds concurrently. Delete old ephemeral
    # private material after the handshake and ratchet traffic keys per round.
    client_kem_j = expired.astype(np.float64) * (KEM768_KEYGEN_J + KEM768_DECAP_J)
    client_handshake_s = expired.astype(np.float64) * (KEM768_KEYGEN_S + KEM768_DECAP_S)
    handshake_s = expired.astype(np.float64) * (
        (MLKEM_PK_B + MLKEM_CT_B) * 8.0 / (rates * 1e6)
    )
    handshake_radio_j = expired.astype(np.float64) * (
        WIFI_TX_W * MLKEM_PK_B * 8.0 / (rates * 1e6) +
        WIFI_RX_W * MLKEM_CT_B * 8.0 / (rates * 1e6)
    )
    latency = train_s + transfer_s + client_handshake_s + handshake_s
    total_j = train_j + radio_j + handshake_radio_j + client_kem_j
    feasible = latency <= deadline_s
    return {
        "n": n, "train_j": train_j, "train_s": train_s,
        "radio_j": radio_j + handshake_radio_j,
        "radio_payload_j": radio_j, "transfer_s": transfer_s,
        "refresh": expired, "client_kem_j": client_kem_j,
        "server_crypto_j": expired.astype(np.float64) * KEM768_ENCAP_J,
        "latency": latency, "total_j": total_j, "feasible": feasible,
        "handshake_bytes": expired.astype(np.int64) * (MLKEM_PK_B + MLKEM_CT_B),
    }


def choose_clients(meta: dict, cost: dict, count: np.ndarray, round_id: int,
                   budget_j: float, kmax: int, policy: dict,
                   random_order: np.ndarray, median_n: float) -> list[int]:
    feasible = np.flatnonzero(cost["feasible"])
    if policy["kind"] == "random":
        order = [int(i) for i in random_order if cost["feasible"][i]]
    else:
        # Proxy value: square-root sample mass (diminishing returns) multiplied by
        # a bounded participation debt. It requires no local labels or losses.
        n = cost["n"].astype(np.float64)
        expected = (round_id + 1) * kmax / len(n)
        debt = np.clip(expected - count, -1.0, 1.0)
        utility_kind = policy.get("utility", "sample_and_debt")
        if utility_kind == "sample_only":
            value = np.sqrt(n / max(1.0, median_n))
        elif utility_kind == "debt_only":
            value = 1.0 + 0.5 * debt
        else:
            value = np.sqrt(n / max(1.0, median_n)) * (1.0 + 0.5 * debt)
        denom = cost["train_j"] + cost["radio_payload_j"]
        if policy["crypto_aware"]:
            denom = cost["total_j"]
        score = value / np.maximum(denom, 1e-12)
        # Reproducible, seed-shared random ranks break equal-score ties without
        # systematically preferring low client identifiers.
        tie_rank = np.empty(len(n), dtype=np.int32)
        tie_rank[random_order] = np.arange(len(n), dtype=np.int32)
        order = sorted((int(i) for i in feasible),
                       key=lambda i: (-score[i], int(tie_rank[i])))
    selected = []
    spent = 0.0
    for i in order:
        e = float(cost["total_j"][i])
        if len(selected) >= kmax:
            break
        if spent + e <= budget_j:
            selected.append(i)
            spent += e
    return selected


def load_global_state(model: nn.Module, state: dict[str, torch.Tensor]):
    model.load_state_dict(state, strict=True)


def local_train(global_state: dict[str, torch.Tensor], x_store: np.ndarray,
                y_store: np.ndarray, client: dict, rng_seed: int,
                momentum: float = SGD_MOMENTUM) -> dict[str, torch.Tensor]:
    start, end = client["offset"], client["offset"] + client["n"]
    x = torch.from_numpy(np.asarray(x_store[start:end]).copy()).float().view(-1, 1, 28, 28)
    x.div_(255.0)
    y = torch.from_numpy(np.asarray(y_store[start:end]).astype(np.int64, copy=True))
    model = FEMNISTNet()
    load_global_state(model, global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.004, momentum=momentum)
    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    order = torch.randperm(len(y), generator=gen)
    for begin in range(0, len(order), 5):
        idx = order[begin:begin + 5]
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(state: dict[str, torch.Tensor], eval_meta: dict, x: np.ndarray,
             y: np.ndarray) -> tuple[float, float, float]:
    model = FEMNISTNet()
    load_global_state(model, state)
    model.eval()
    pred_blocks, label_blocks, client_acc = [], [], []
    for client in eval_meta["clients"]:
        s, e = client["offset"], client["offset"] + client["n"]
        client_preds = []
        for begin in range(s, e, 512):
            end = min(e, begin + 512)
            xb = torch.from_numpy(np.asarray(x[begin:end]).copy()).float().view(-1, 1, 28, 28).div_(255)
            client_preds.append(model(xb).argmax(1).numpy())
        preds_i = np.concatenate(client_preds)
        labels_i = np.asarray(y[s:e], dtype=np.uint8)
        pred_blocks.append(preds_i)
        label_blocks.append(labels_i)
        client_acc.append(float((preds_i == labels_i).mean()))
    preds = np.concatenate(pred_blocks)
    labels = np.concatenate(label_blocks)
    total_n = len(labels)
    total_correct = int((preds == labels).sum())
    macro_user = float(np.mean(client_acc))
    # Macro class accuracy makes imbalanced writer/client contributions visible.
    class_acc = []
    for cls in range(62):
        mask = labels == cls
        if mask.any():
            class_acc.append(float((preds[mask] == labels[mask]).mean()))
    return total_correct / total_n, macro_user, float(np.mean(class_acc))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp_path, path)


def run_campaign(args):
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_meta, val_meta, test_meta, train_x, train_y, test_x, test_y = ensure_data()
    OUT.mkdir(parents=True, exist_ok=True)
    model_bytes = sum(p.numel() for p in FEMNISTNet().parameters()) * 4
    n_clients = train_meta["num_clients"]
    kmax = min(args.clients_per_round, n_clients)
    train_n = np.asarray([x["n"] for x in train_meta["clients"]])
    median_n = float(np.median(train_n))
    # A common, predeclared budget tied to the complete-population median profile.
    median_batches = median_n / 5.0
    median_train_j = median_batches * CPU_BATCH_J
    mean_rate = (args.rate_min_mbps + args.rate_max_mbps) / 2.0
    median_one_way_s = model_bytes * 8 / (mean_rate * 1e6)
    median_radio_j = (WIFI_RX_W + WIFI_TX_W) * median_one_way_s + (
        WIFI_TX_W * MLKEM_PK_B * 8 / (mean_rate * 1e6) +
        WIFI_RX_W * MLKEM_CT_B * 8 / (mean_rate * 1e6)
    )
    median_kem_j = KEM768_KEYGEN_J + KEM768_DECAP_J
    per_round_budget = args.budget_fraction * kmax * (median_train_j + median_radio_j + median_kem_j)

    (OUT / "dataset_manifest.json").write_text(json.dumps({
        "optimization_train": train_meta, "validation": val_meta,
        "final_test": test_meta,
        "writer_overlap": 0,
        "writer_partition_adjustments": val_meta.get("writer_partition_adjustments", []),
        "exact_duplicate_writer_groups": val_meta.get("exact_duplicate_writer_groups", []),
        "split": "LEAF user-level; whole writers divided into optimization, validation, and final test partitions",
        "sampling_fraction": 1.0,
        "source_repo_commit": "09ec454a5675e32e1f0546b456b77857fdece018",
        "classes": 62,
        "model_parameters": sum(p.numel() for p in FEMNISTNet().parameters()),
        "model_bytes_fp32": model_bytes,
    }, indent=2) + "\n")

    all_rounds, all_runs = [], []
    campaign_started = time.perf_counter()
    trainer_pool = ThreadPoolExecutor(max_workers=args.workers)
    for rep in range(args.repetitions):
        seed = args.seed + 1009 * rep
        rate_rng = np.random.default_rng(seed)
        order_rng = np.random.default_rng(seed + 104729)
        rates = rate_rng.uniform(args.rate_min_mbps, args.rate_max_mbps,
                                 size=(args.rounds, n_clients))
        candidate_orders = np.stack([order_rng.permutation(n_clients)
                                     for _ in range(args.rounds)])
        for policy_name, policy in POLICIES.items():
            run_started = time.perf_counter()
            torch.manual_seed(seed)
            model = FEMNISTNet()
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            participation = np.zeros(n_clients, dtype=np.int32)
            last_refresh = np.full(n_clients, -100000, dtype=np.int32)
            local_count = 0
            total_client_j = total_server_j = 0.0
            total_train_j = total_radio_j = total_client_kem_j = 0.0
            total_update_bytes = total_hs_bytes = 0
            eval_acc, eval_user, eval_class = evaluate(state, val_meta, train_x, train_y)
            run_metrics = {"rep": rep, "seed": seed, "policy": policy_name,
                           "round": 0, "validation_accuracy": eval_acc,
                           "validation_user_macro_accuracy": eval_user,
                           "validation_class_macro_accuracy": eval_class}
            all_rounds.append(run_metrics)
            for r in range(args.rounds):
                round_started = time.perf_counter()
                costs = make_costs(train_meta, rates[r], participation,
                                   last_refresh, r, policy["ttl"], model_bytes,
                                   args.deadline_s)
                selected = choose_clients(train_meta, costs, participation, r,
                                          per_round_budget, kmax, policy,
                                          candidate_orders[r], median_n)
                futures = []
                for i in selected:
                    client = train_meta["clients"][i]
                    client_seed = (seed * 1_000_003 + r * 9_176 + i * 37) % (2**31 - 1)
                    futures.append(trainer_pool.submit(local_train, state, train_x, train_y,
                                                      client, client_seed, args.momentum))
                updates = [future.result() for future in futures]
                weights = [train_meta["clients"][i]["n"] for i in selected]
                round_client_j = round_server_j = 0.0
                round_train_j = round_radio_j = round_client_kem_j = 0.0
                round_hs = 0
                round_refresh_count = 0
                round_complete_s = 0.0
                for i in selected:
                    client = train_meta["clients"][i]
                    local_count += 1
                    participation[i] += 1
                    round_client_j += float(costs["total_j"][i])
                    round_server_j += float(costs["server_crypto_j"][i])
                    round_train_j += float(costs["train_j"][i])
                    round_radio_j += float(costs["radio_j"][i])
                    round_client_kem_j += float(costs["client_kem_j"][i])
                    refresh = bool(costs["refresh"][i])
                    if refresh:
                        last_refresh[i] = r
                        round_refresh_count += 1
                        round_hs += MLKEM_PK_B + MLKEM_CT_B
                    total_update_bytes += model_bytes * 2
                    round_complete_s = max(round_complete_s, float(costs["latency"][i]))
                if selected:
                    denom = float(sum(weights))
                    new_state = {}
                    for key in state:
                        accumulator = torch.zeros_like(state[key], dtype=torch.float64)
                        for local_state, n_i in zip(updates, weights):
                            accumulator.add_(local_state[key].to(torch.float64), alpha=n_i / denom)
                        new_state[key] = accumulator.to(state[key].dtype)
                    state = new_state
                total_client_j += round_client_j
                total_server_j += round_server_j
                total_train_j += round_train_j
                total_radio_j += round_radio_j
                total_client_kem_j += round_client_kem_j
                total_hs_bytes += round_hs
                if (r + 1) % args.eval_every == 0 or r + 1 == args.rounds:
                    eval_acc, eval_user, eval_class = evaluate(state, val_meta, train_x, train_y)
                else:
                    eval_acc = eval_user = eval_class = float("nan")
                all_rounds.append({
                    "rep": rep, "seed": seed, "policy": policy_name, "round": r + 1,
                    "selected": len(selected), "max_clients": kmax,
                    "validation_accuracy": eval_acc,
                    "validation_user_macro_accuracy": eval_user,
                    "validation_class_macro_accuracy": eval_class,
                    "client_energy_j": round_client_j,
                    "training_energy_j": round_train_j,
                    "radio_energy_j": round_radio_j,
                    "client_kem_compute_energy_j": round_client_kem_j,
                    "server_crypto_energy_j": round_server_j,
                    "model_bytes_up_down": model_bytes * 2 * len(selected),
                    "kem_refreshes": round_refresh_count,
                    "kem_handshake_bytes": round_hs,
                    "round_completion_s": round_complete_s,
                    "simulator_wall_s": time.perf_counter() - round_started,
                    "budget_j": per_round_budget,
                    "eligible_clients": int(costs["feasible"].sum()),
                    "unique_participants": int(np.count_nonzero(participation)),
                })
                if (r + 1) % 10 == 0:
                    print(f"rep={rep+1}/{args.repetitions} policy={policy_name} "
                          f"round={r+1}/{args.rounds} selected={len(selected)} "
                          f"acc={eval_acc:.4f} energy={round_client_j:.3f}J", flush=True)
            fairness = float((participation.sum() ** 2) /
                             (n_clients * np.square(participation).sum()))
            # Final test writers are scored once after the run and never enter
            # scheduling, hyperparameter selection, or checkpoint selection.
            test_acc, test_user, test_class = evaluate(state, test_meta, test_x, test_y)
            all_runs.append({
                "rep": rep, "seed": seed, "policy": policy_name,
                "rounds": args.rounds, "kmax": kmax,
                "final_test_accuracy": test_acc,
                "final_test_user_macro_accuracy": test_user,
                "final_test_class_macro_accuracy": test_class,
                "final_validation_accuracy": eval_acc,
                "simulator_wall_s": time.perf_counter() - run_started,
                "client_energy_j": total_client_j,
                "training_energy_j": total_train_j,
                "radio_energy_j": total_radio_j,
                "client_kem_compute_energy_j": total_client_kem_j,
                "server_crypto_energy_j": total_server_j,
                "total_updates": local_count,
                "avg_selected_per_round": local_count / args.rounds,
                "key_refreshes": int(sum(x.get("kem_refreshes", 0) for x in all_rounds
                                          if x.get("rep") == rep and x.get("policy") == policy_name)),
                "model_bytes_up_down": total_update_bytes,
                "handshake_bytes": total_hs_bytes,
                "jain_participation_fairness": fairness,
                "unique_participants": int(np.count_nonzero(participation)),
                "deadline_s": args.deadline_s,
                "budget_j_per_round": per_round_budget,
                "ttl_rounds": policy["ttl"],
            })
            print(f"completed rep={rep+1} policy={policy_name} "
                  f"test_acc={test_acc:.4f} energy={total_client_j:.2f}J", flush=True)
            # Atomic evidence checkpoints preserve completed run-level artifacts
            # if a later campaign stage is interrupted.
            write_csv(OUT / "rounds.csv", all_rounds)
            write_csv(OUT / "runs.csv", all_runs)
    trainer_pool.shutdown(wait=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "rounds.csv", all_rounds)
    write_csv(OUT / "runs.csv", all_runs)
    config = vars(args) | {
        "campaign_wall_s": time.perf_counter() - campaign_started,
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": sys.version,
            "torch": torch.__version__, "cryptography": cryptography.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "torch_intraop_threads": args.torch_threads,
            "parallel_client_trainers": args.workers,
        },
        "n_train_clients": train_meta["num_clients"],
        "n_validation_clients": val_meta["num_clients"],
        "n_test_clients": test_meta["num_clients"],
        "n_train_examples": train_meta["num_examples"],
        "n_validation_examples": val_meta["num_examples"],
        "n_test_examples": test_meta["num_examples"],
        "kmax": kmax, "model_bytes": model_bytes,
        "budget_j_per_round": per_round_budget,
        "sgd_momentum": args.momentum,
        "tie_breaking": "shared seed-derived per-round permutation rank for exact utility-score ties",
        "profile_provenance": {
            "cpu_training": "Yuan et al., WWW 2024, Table 2; Xiaomi 10, FEMNIST, LeNet-5, batch 5, 58 ms and 0.3 J per batch",
            "kem": "Alrowaili et al., IET Smart Cities 2026, Table 8; ESP32-P4 ML-KEM-768 keygen/encapsulation/decapsulation",
            "wifi_power": "Arouj and Abdelmoniem, Computer Communications 2024, Wi-Fi upload/download battery slopes; HTC Desire HD battery 4.55 Wh",
            "host_measurement": "results/raw/pqc_benchmark.json; timing only, not used as embedded energy",
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(all_runs)} run summaries and {len(all_rounds)} round rows to {OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--clients-per-round", type=int, default=50)
    p.add_argument("--budget-fraction", type=float, default=0.85)
    p.add_argument("--deadline-s", type=float, default=5.0)
    p.add_argument("--rate-min-mbps", type=float, default=10.0)
    p.add_argument("--rate-max-mbps", type=float, default=100.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--momentum", type=float, default=SGD_MOMENTUM)
    run_campaign(p.parse_args())


if __name__ == "__main__":
    main()
