# SiKDD 2026 Article Status

This file replaces the earlier planning notes. The article is implemented in [`main.tex`](main.tex); current technical details are documented in the repository [`README.md`](../README.md) and [`srcML/README.md`](../srcML/README.md).

## Paper

- **Title:** Combining Supervised and Unsupervised Learning on SMART Data for Hard Drive Failure Assessment
- **Authors:** Luka Petek and Vili Podgorelec
- **Conference:** Information Society 2026 / SiKDD 2026
- **Conference date:** 7 October 2026
- **DOI in the paper:** `10.70314/is.2026.sikdd.97`

## Current technical scope

The paper presents a current-state hard-drive assessment pipeline with four components:

1. Random forest on 19 processed SMART features plus manufacturer encoding.
2. Healthy-only anomaly autoencoder with a 12-dimensional bottleneck.
3. Dense classifier on an 8-dimensional representation learned by a separate healthy-only autoencoder.
4. HDBSCAN cluster risk in the same 8-dimensional representation used by the bottleneck classifier.

The four component scores are combined with a weighted RMS formula into the Aggregated Health Index (AHI). The bottleneck classifier and HDBSCAN share an encoder and are therefore not independent.

## Data used

- Source corpus: more than 32 million Backblaze daily records in 365 CSV files.
- Prepared period: 1 October 2024 to 30 September 2025.
- Failure-day records: 4,414.
- Supervised and clustering set: 4,414 failure-day + 4,414 healthy records.
- Healthy-only autoencoder split: 289,982 training + 72,481 validation records.

Each labelled instance is one daily SMART snapshot. A positive label means that the drive was marked as failed on that day. The work therefore evaluates current-state separation rather than a future prediction horizon or remaining useful life.

## Reported common evaluation

The paper reports a balanced, serial-disjoint Q1 2026 cohort with 1,018 failure-day and 1,018 healthy records. AHI achieved:

- ROC-AUC: 0.923
- PR-AUC: 0.936
- Recall at AHI 45: 0.783
- False-positive rate at AHI 45: 0.091
- Precision at AHI 45: 0.896
- F1 at AHI 45: 0.835

The random forest has the highest reported PR-AUC (0.938), precision (0.924) and F1 (0.865). AHI has the highest ROC-AUC by a small margin over the random forest (0.923 versus 0.922). The results do not show that AHI is better on every metric.

## Interpretation limits

- The evaluation is balanced and does not represent natural fleet failure prevalence.
- Reported precision must not be interpreted as expected production precision.
- AHI is a risk index, not a calibrated probability.
- The evaluation detects failure-day state; it does not measure how early a future failure can be predicted.
- Training data comes from one storage provider.
- AHI weights are manually selected.
- The raw Q1 2026 CSV files and generated per-record output CSVs are not tracked in this repository.
- NAS and edge-device performance has not been tested.

## Main source files

- Paper: [`main.tex`](main.tex)
- AHI inference: [`../srcML/ahi_final.py`](../srcML/ahi_final.py)
- Common evaluation: [`../srcML/evaluate_2026.py`](../srcML/evaluate_2026.py)
- ML engineering notes: [`../srcML/README.md`](../srcML/README.md)
- Reported evaluation plot: [`../Graphs/ahi_2026.png`](../Graphs/ahi_2026.png)
