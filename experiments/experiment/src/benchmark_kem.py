#!/usr/bin/env python3
"""Reproducible timing benchmark for ML-KEM-768 and X25519 operations.

This records operation latency and serialized public/ciphertext sizes only.
It does not infer joules from CPU time; the host has no exposed energy counter.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cryptography
from cryptography.hazmat.primitives.asymmetric import mlkem, x25519


def openssl_version() -> str:
    try:
        return subprocess.check_output(
            ["openssl", "version"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def timed_samples(fn, warmups: int, trials: int) -> list[float]:
    for _ in range(warmups):
        fn()
    out = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - start) / 1e6)
    return out


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "sd_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/raw/pqc_benchmark"))
    args = parser.parse_args()
    if args.trials < 100 or args.batches < 5 or args.warmups < 0:
        raise SystemExit("Use at least 100 trials and five independent batches.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    private = mlkem.MLKEM768PrivateKey.generate()
    public = private.public_key()
    shared, ciphertext = public.encapsulate()
    assert private.decapsulate(ciphertext) == shared
    server_x = x25519.X25519PrivateKey.generate()
    client_x = x25519.X25519PrivateKey.generate()
    assert server_x.exchange(client_x.public_key()) == client_x.exchange(server_x.public_key())

    operations = {
        "mlkem768_keygen": lambda: mlkem.MLKEM768PrivateKey.generate(),
        "mlkem768_encapsulate": lambda: public.encapsulate(),
        "mlkem768_decapsulate": lambda: private.decapsulate(ciphertext),
        "x25519_keygen": lambda: x25519.X25519PrivateKey.generate(),
        "x25519_exchange": lambda: server_x.exchange(client_x.public_key()),
    }
    records = []
    for batch in range(args.batches):
        for operation, fn in operations.items():
            for trial, elapsed in enumerate(timed_samples(fn, args.warmups, args.trials)):
                records.append({
                    "batch": batch,
                    "operation": operation,
                    "trial": trial,
                    "elapsed_ms": elapsed,
                })

    sizes = {
        "mlkem768_public_key_bytes": len(public.public_bytes_raw()),
        "mlkem768_ciphertext_bytes": len(ciphertext),
        "mlkem768_shared_secret_bytes": len(shared),
        "x25519_public_key_bytes": len(client_x.public_key().public_bytes_raw()),
        "x25519_shared_secret_bytes": len(client_x.exchange(server_x.public_key())),
    }
    summaries = {}
    for operation in operations:
        per_batch = []
        for batch in range(args.batches):
            vals = [r["elapsed_ms"] for r in records
                    if r["operation"] == operation and r["batch"] == batch]
            per_batch.append(summarize(vals))
        summaries[operation] = {
            "batches": per_batch,
            "pooled": summarize([r["elapsed_ms"] for r in records
                                 if r["operation"] == operation]),
        }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "cryptography": cryptography.__version__,
        "openssl": openssl_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "trials_per_batch": args.trials,
        "independent_batches": args.batches,
        "warmups_per_batch": args.warmups,
        "energy_measurement": "unavailable; no host energy counter exposed",
        "serialized_sizes": sizes,
        "summary": summaries,
        "raw_samples": records,
    }
    json_path = args.out.with_suffix(".json")
    csv_path = args.out.with_suffix(".csv")
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path),
                      "sizes": sizes, "summary_ms": {
                          k: v["pooled"] for k, v in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
