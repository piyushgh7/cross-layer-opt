# Experiment bundle

This folder is a snapshot of the files used to run, analyze, and audit the experiments.

- `src/` — simulator, ML-KEM benchmark, sensitivity studies, diagnostics, analysis, and
  audit scripts.
- `results/` — saved raw outputs, configurations, logs, and generated LaTeX tables.
- `figures/` — generated plots and the methodology figure.
- `evidence/` — experiment design, execution audits, tie-break checks, and run logs.

The dataset itself is not duplicated here. Its location, construction procedure, seeds,
partition counts, and integrity checks are documented in `../dataset.md`; the original
dataset remains under `../data/`.

The primary campaign can be reproduced from the project root with the commands in
`../README.md`, beginning with:

```bash
python src/femnist_sim.py --rounds 100 --repetitions 5 --clients-per-round 50 \
  --budget-fraction 0.85 --deadline-s 5 --rate-min-mbps 10 --rate-max-mbps 100 \
  --eval-every 5 --seed 20261011 --torch-threads 1 --workers 8
```
