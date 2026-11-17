# Final research design and execution record

## Scope and research questions

The study tests whether ML-KEM-768 session-refresh state changes client admission in cross-device federated learning when local training, communication, deadlines, and a shared client-energy budget are modeled jointly. The literature review recognizes existing post-quantum FL protocols and key-reuse schemes; the contribution is limited to the controlled scheduling comparison and its measured operating boundary.

- **RQ1:** Under what link, deadline, and session conditions does charging for KEM refresh alter client feasibility or scheduling order?
- **RQ2:** Does the cross-layer ordering preserve held-out writer accuracy while changing modeled energy or participation relative to random and crypto-blind policies?
- **RQ3:** How do data heterogeneity, participation debt, and cohort constraints delimit the observed behavior?

## Data and execution

The full LEAF FEMNIST corpus was retained: 2,914 optimization writers (661,619 images), 324 validation writers (72,463 images), and 359 final-test writers (83,769 images), for 817,851 images total. An exact pixel duplicate crossing the initial writer partitions was repaired by moving the complete 293-image writer into optimization; the final partitions contain no cross-partition exact duplicates. The test partition is evaluated once after training and does not guide method selection.

The primary campaign executed six policies × five paired seeds × 100 rounds. It uses one compact 62-class CNN, complete local writer partitions, one local epoch, a maximum cohort of 50, a 5-s client deadline, a 0.85 nominal energy-budget fraction, and independently drawn 10–100 Mb/s client-round rates. A 170-row system-level sensitivity grid and 100 additional paired KEM-profile runs evaluate admission and cost only, without retraining or accuracy claims. The ML-KEM-768 host benchmark records five independent batches of 2,000 trials for each operation and verifies matching derived secrets.

## Evidence and limits

The energy and latency model combines published training, Wi-Fi, and embedded KEM profiles from different hardware; it is explicitly modeled and not presented as a single-device measurement. Host cryptographic timings are kept separate from the embedded profile, and the host exposed no power counter. The simulator charges KEM refresh overhead in its admission model but does not implement an authenticated secure-aggregation transport or protect model tensors. Claims are therefore limited to this simulator, hardware-profile assumptions, and writer-partitioned FEMNIST experiment.

Raw campaign files and per-run configuration are in `results/raw/`; verified dataset metadata and reference evidence are in this directory. `src/audit_results.py` confirms the executed campaign, no test metrics in round logs, energy reconstruction, sensitivity reproduction, and unchanged scheduler traces under the final tie-break rule. `src/audit_refs.py` verifies 50 cited journal DOIs against Crossref metadata.
