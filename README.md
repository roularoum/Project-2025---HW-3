# Assignment 3 — Remote Homolog Search with ESM‑2 + ANN + BLAST

This folder contains a complete implementation of the Assignment 3 pipeline:

- **Step 1**: Generate protein embeddings using **ESM‑2** (`facebook/esm2_t6_8M_UR50D`, layer 6, mean pooling).
- **Step 2**: Run **Approximate Nearest Neighbor (ANN)** search on the embedding space using:
  - Euclidean LSH
  - Hypercube
  - IVF‑Flat
  - IVF‑PQ
  - Neural LSH
- **Step 3**: Compare ANN neighbors with **BLAST** results (same database), computing **Recall@N** and **QPS**.

Spec references: `assignment3/project.pdf`, `assignment3/reference.pdf`

---

## Contents

### Scripts
- `protein_embed.py`: generates database embeddings (`vectors.dat` + `ids.txt`).
- `protein_search.py`: runs ANN methods, BLAST baseline, and produces `results.txt` in the required 2-level format.
- `protein_grid_search.py`: grid search for hyperparameters → CSV for Recall vs QPS curves.

### Library code
- `protein_ann/`: reusable modules:
  - `protein_ann/esm2.py`: ESM‑2 embedding (mean pooling over residues, last layer).
  - `protein_ann/ann/`: ANN index implementations (LSH, Hypercube, IVF‑Flat, IVF‑PQ, Neural LSH).
  - `protein_ann/blast.py`: BLAST+ DB creation, search execution, parsing.
  - `protein_ann/output_format.py`: output formatting utilities.

### Data
- `datasets/swissprot_50k.fasta`: database proteins (50k sequences).
- `datasets/targets.fasta`: query proteins.
- `datasets/targets.pfam_map.tsv`: helper mapping for report/bio evaluation.

---

## Requirements

- Python **3.10+**
- Linux environment (as per assignment)
- **NCBI BLAST+** installed and in `PATH`:
  - required executables: `makeblastdb`, `blastp`

Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Generate embeddings

Command (as in the spec, adapted to the included dataset paths):

```bash
python protein_embed.py \
  -i datasets/swissprot_50k.fasta \
  -o protein_vectors.dat \
  -model esm2_t6_8M_UR50D
```

Outputs:
- `protein_vectors.dat`: binary NumPy array (content is `.npy`) with shape `(N, 320)`.
- `ids.txt`: one protein ID per line, aligned with rows of `protein_vectors.dat`.

---

## Step 2/3 — Search benchmark + BLAST comparison

Run all methods:

```bash
python protein_search.py \
  -d protein_vectors.dat \
  -q datasets/targets.fasta \
  -o results.txt \
  -method all \
  --blast-fasta datasets/swissprot_50k.fasta
```

Run a single method (examples):

```bash
python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method lsh --blast-fasta datasets/swissprot_50k.fasta
python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method hypercube --blast-fasta datasets/swissprot_50k.fasta
python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method ivf --blast-fasta datasets/swissprot_50k.fasta
python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method neural --blast-fasta datasets/swissprot_50k.fasta
```

Notes:
- BLAST DB is cached under `.blast_db_cache/` (created on demand).
- The output `results.txt` follows the **plain-text, 2-level** format shown in `project.txt`:
  - [1] summary (Time/query, QPS, Recall@N vs BLAST Top‑N)
  - [2] Top‑K neighbors per method (Neighbor ID, L2, BLAST identity, In BLAST Top‑N?, Bio comment)

---

## Grid search (Recall vs QPS curves)

Example for LSH:

```bash
python protein_grid_search.py \
  -d protein_vectors.dat \
  -q datasets/targets.fasta \
  -o grid_lsh.csv \
  --method lsh \
  --recall-n 50 \
  --lsh-k-grid 2,4,6 \
  --lsh-L-grid 5,10 \
  --lsh-w-grid 2.0,4.0,6.0 \
  --blast-fasta datasets/swissprot_50k.fasta
```

The CSV can be plotted to produce trade-off curves and to justify hyperparameter choices in the report.

---

## Report

The report is in `report.md`. It follows the recommended structure from `reference.pdf` and the requirements of `project.txt`.

