# DSCI 552 — Homework 7 (Alan Nhan)

Multi-class/multi-label SVM classification and K-means clustering on the UCI
[Anuran Calls (MFCCs)](https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs) dataset, plus ISLR 12.6.2.

## Contents
| Path | Description |
|---|---|
| `Nhan_Alan_HW7.ipynb` | Main notebook (executed, with outputs) |
| `data/Frogs_MFCCs.csv` | Dataset (downloaded from UCI automatically if missing) |
| `results/` | Tables (CSV) and figures (PNG) written when the notebook runs (not tracked) |

## Setup and run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Nhan_Alan_HW7.ipynb
```
All paths are relative to the project root. Random seeds are fixed (`SEED = 552`), so results are reproducible.
A full run takes about 11 minutes on an 8-core laptop (mostly the SMOTE grid search and the 50 K-means runs).

## Methods summary
- **Problem 1:** 70/30 random split, stratified on `Species`, seed 552. Each label (Family, Genus, Species) gets its own classifier
  (binary relevance): an RBF SVM (one-vs-rest), an L1-penalized `LinearSVC`, and an L1 `LinearSVC` with SMOTE.
  Hyperparameters are tuned with stratified 10-fold CV on the training set only. Scaling and SMOTE run inside the CV pipelines.
- **Problem 2:** 50 Monte Carlo runs of K-means on the full data, with k ∈ {2,…,50} chosen by maximum silhouette
  (undefined at k = 1). Each cluster gets its majority label triplet, and Hamming distance, score, and loss are reported as mean ± SD.
- **Problem 3:** Complete- and single-linkage dendrograms for ISLR 12.6.2, worked by hand and checked with SciPy.

Optional parts 1(b)v–vi and Problem 4 are not included.
