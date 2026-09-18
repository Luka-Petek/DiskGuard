# DiskData2026 (Backblaze Q1 2026) — Inventory & Evaluation Cohorts

Measured on 2026-09-18 by chunked pandas scan over all CSVs in `DiskData2026/data_Q1_2026/`.
All numbers below are verified measurements, not estimates.

## 1. Basic inventory

| Metric | Value |
|---|---:|
| CSV files | 90 (one per day) |
| Date range | 2026-01-01 → 2026-03-31 |
| Total rows | 30,597,484 |
| `failure=1` rows | 1,030 |
| `failure=0` rows | 30,596,454 |
| Unique serial numbers | 351,095 |
| Unique drive models | 80 |
| Schema variants | 1 (uniform columns across all files) |
| All required 19 features present | yes |

## 2. Row cohorts

Each Q1 row belongs to exactly one cohort:

| Cohort | Definition | Rows | Unique serials |
|---|---|---:|---:|
| `failure_day` | `failure=1` — the day the drive died | 1,030 | 1,030 |
| `pre_failure_healthy_day` | `failure=0` row of a serial that fails later in Q1 | 42,123 | 1,005 |
| `never_failed_healthy_day` | `failure=0` row of a serial that never fails in Q1 | 30,554,331 | 350,065 |

Notes:
- Every failed serial has exactly **one** failure-day row (no serial has 2+ failure rows in Q1).
- `pre_failure_healthy_day` rows are NOT "healthy drives" — they are snapshots of drives already failing. They must not be used as clean negatives; they are the interesting lead-time analysis set.
- Clean primary evaluation: positives = `failure_day`, negatives = `never_failed_healthy_day`.

## 3. Serial-number overlap with prepared 2025 data

Baseline: `csv/koncniPodatkiZaModel.csv` (8,828 rows, 8,759 unique serials, spans 2024-10-01 → 2025-09-30).

| Metric | Count |
|---|---:|
| 2025/2026 overlapping serials (any cohort) | 4,336 |
| 2026 serials absent from prepared 2025 | 346,759 |
| `failure_day` serials seen in prepared 2025 | **12 of 1,030** |
| `failure_day` serials absent from 2025 | **1,018** |
| `failure_day` serials that also failed in 2025 | 1 |
| `pre_failure_healthy_day` serials seen in 2025 | 11 |
| `never_failed_healthy_day` serials seen in 2025 | 4,324 |

Interpretation:
- 98.8% of the 2026 failed drives were not in the prepared 2025 training data at all.
- Report two evaluation subsets: **all Q1 2026** (later-period transfer) and **2026 serials absent from 2025** (strict unseen-drive).

## 4. Missingness (whole quarter)

Missing values in the 14 SMART raw features used by the models:

| Feature | Missing rows | % of 30.6M |
|---|---:|---:|
| smart_1_raw | 67,994 | 0.22% |
| smart_3_raw | 306,463 | 1.00% |
| smart_4_raw | 306,463 | 1.00% |
| smart_5_raw | 206,428 | 0.67% |
| smart_7_raw | 306,463 | 1.00% |
| smart_9_raw | 39,418 | 0.13% |
| smart_12_raw | 39,418 | 0.13% |
| **smart_187_raw** | **20,443,161** | **66.8%** |
| **smart_188_raw** | **20,470,056** | **66.9%** |
| **smart_191_raw** | **19,001,120** | **62.1%** |
| smart_192_raw | 139,453 | 0.46% |
| smart_193_raw | 311,563 | 1.02% |
| smart_197_raw | 766,680 | 2.51% |
| smart_198_raw | 235,118 | 0.77% |

## 5. Missingness is model/vendor-dependent — NOT "measured zeros"

Per-model missingness for the three heavily-missing features (top 15 models by row count):

| Model | Rows | Failures | 187 missing | 188 missing | 191 missing |
|---|---:|---:|---:|---:|---:|
| WDC WUH722222ALE6L4 | 3,992,942 | 42 | 100% | 100% | 100% |
| TOSHIBA MG08ACA16TA | 3,548,250 | 102 | 100% | 100% | 0% |
| TOSHIBA MG07ACA14TA | 3,343,724 | 94 | 100% | 100% | 0% |
| ST16000NM001G | 3,098,105 | 44 | 0% | 0% | 100% |
| WDC WUH721816ALE6L4 | 2,366,036 | 56 | 100% | 100% | 100% |
| TOSHIBA MG10ACA20TE | 1,665,263 | 42 | 100% | 100% | 0% |
| ST12000NM0008 | 1,664,558 | 129 | 0% | 0% | 100% |
| ST12000NM001G | 1,185,545 | 33 | 0% | 0% | 100% |
| HGST HUH721212ALE604 | 1,183,774 | 86 | 100% | 100% | 100% |
| ST8000NM0055 | 1,176,619 | 39 | 0.003% | 0.003% | 0.003% |
| ST14000NM001G | 944,400 | 21 | 0% | 0% | 100% |
| HGST HUH721212ALN604 | 870,252 | 95 | 100% | 100% | 100% |
| ST24000NM002H | 820,090 | 74 | 0.08% | 0.08% | 100% |
| WDC WUH721414ALE6L4 | 776,557 | 7 | 100% | 100% | 100% |
| ST8000DM002 | 639,473 | 25 | 0.007% | 0.007% | 0.007% |

Pattern: whether SMART 187/188/191 are reported is essentially a **drive-model property** (median model: 100% missing; other models: 0% missing). These are absent attributes, not unrecorded zero counters.

Consequences for the paper and pipeline:
- Do not write "missing counters were filled with zero" — imputed values are model-input representations, not evidence of zero errors.
- Recommend adding missingness indicators (`smart_187_missing`, …) or at minimum documenting model-aware imputation as a limitation.
- Missingness correlates with manufacturer → it can leak model identity into predictions; acknowledge in limitations.

## 6. How to rebuild the deleted working files

The large intermediate manifests were deleted (repo hygiene). To rebuild any of them:

```python
import pandas as pd, glob
files = sorted(glob.glob("DiskData2026/**/*.csv", recursive=True))
cols = ["date", "serial_number", "model", "failure"]
manifest = pd.concat(
    (pd.read_csv(f, usecols=[c for c in cols if c in pd.read_csv(f, nrows=0).columns])
     for f in files), ignore_index=True)
```

Cohort labels are derived as:

```python
failed_serials = set(manifest.loc[manifest.failure == 1, "serial_number"])
is_failed_serial = manifest.serial_number.isin(failed_serials)
cohort = "never_failed_healthy_day"
cohort[is_failed_serial & (manifest.failure == 0)] = "pre_failure_healthy_day"
cohort[manifest.failure == 1] = "failure_day"
```

Scan takes ~4–5 minutes over the full 12 GB directory with chunked reads (250k rows/chunk).
