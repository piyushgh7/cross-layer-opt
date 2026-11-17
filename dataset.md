# Dataset documentation

## Dataset used

The study uses the LEAF FEMNIST dataset generated from **NIST Special Database 19 (SD-19)**.
The data are partitioned by writer, so each writer is treated as one federated-learning
client. There are 62 character classes (digits, uppercase letters, and lowercase letters).

The final audited corpus contains:

| Partition | Writers | Images |
|---|---:|---:|
| Optimization/training | 2,914 | 661,619 |
| Validation | 324 | 72,463 |
| Final test | 359 | 83,769 |
| **Total** | **3,597** | **817,851** |

Images are converted to 28×28 grayscale vectors and stored as normalized values in the
LEAF JSON files. The processed cache stores the same pixels as `uint8` arrays with shape
`(n, 784)` and labels in the range 0–61.

The audited source LEAF commit is:

```text
09ec454a5675e32e1f0546b456b77857fdece018
```

## Where the dataset files are

- `data/leaf/data/femnist/data/raw_data/` — downloaded SD-19 images, arranged in
  `by_class/` and `by_write/` layouts.
- `data/leaf/data/femnist/data/intermediate/` — file lists, hashes, class labels, and
  writer-grouping pickle files created during preprocessing.
- `data/leaf/data/femnist/data/all_data/` — writer-grouped LEAF JSON shards before the
  train/test split.
- `data/leaf/data/femnist/data/train/` and `test/` — generated LEAF JSON shards used by
  the simulator.
- `data/processed/femnist/` — memory-mapped NumPy caches and writer metadata generated
  by `src/femnist_sim.py`.

The scripts that explain and reproduce the construction are under:

- `data/leaf/data/femnist/preprocess/`
- `data/leaf/data/femnist/preprocess.sh`
- `data/leaf/data/utils/`
- `src/femnist_sim.py`
- `src/audit_dataset.py`
- `src/analyze_dataset.py`
- `src/repair_dataset_duplicates.py`

The recorded metadata and audits are in `data/leaf/data/femnist/meta/`,
`results/raw/femnist_campaign/dataset_manifest.json`, and `evidence/`.

## How it was created

1. The SD-19 `by_class.zip` and `by_write.zip` archives were downloaded using
   `data/leaf/data/femnist/preprocess/get_data.sh`.
2. `data_to_json.sh` created file-directory lists, calculated image hashes, matched
   writer images to their character classes, grouped images by writer, and converted
   the result into LEAF JSON shards.
3. The full corpus was retained (`sampling_fraction = 1.0`). A writer-level train/test
   split was generated with the recorded split seed `20261012`; the sampling seed is
   `20261011`. The resulting source split contains 3,238 train-side writers and 359
   test-side writers.
4. `src/femnist_sim.py` reads the LEAF JSON shards, validates their metadata, and creates
   the processed NumPy cache. It reserves complete train-side writers for validation
   using NumPy seed `20261012`, leaving the final optimization partition writer-disjoint
   from validation and test.
5. An exact-pixel-duplicate audit was run before training. One 293-image writer was
   moved to the training side so duplicate pixels could not cross a partition boundary.
   The final audit reports zero writer overlap and zero repeated images across the three
   final partitions.

The main integrity record is `evidence/dataset_audit.json`; the duplicate-handling
record is `evidence/exact_duplicate_repair.json`; and the complete writer/image manifest
used by the campaign is `results/raw/femnist_campaign/dataset_manifest.json`.

## Re-running preprocessing

From the project root, the LEAF preprocessing entry point is:

```bash
cd data/leaf/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t user \
  --smplseed 20261011 --spltseed 20261012
```

Do not delete the existing source or processed files unless a fresh regeneration is
intended. The current experiment uses the saved JSON shards and audited cache exactly as
described above.
