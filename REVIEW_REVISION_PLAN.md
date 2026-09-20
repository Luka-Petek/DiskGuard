# SiKDD 2026 Conditional-Acceptance Revision Plan

## Submission information

- Submission ID: **97**
- Title: **Combining Supervised and Unsupervised Learning on SMART Data for Hard Drive Failure Assessment**
- Revision deadline: **23 September 2026**
- Required DOI: `https://doi.org/10.70314/is.2026.sikdd.97`
- LaTeX value: `\acmDOI{10.70314/is.2026.sikdd.97}`
- Submission method: EasyChair **Update file** on the original submission

## Revision objective

Revise the implementation, evaluation, paper, and public documentation so the evidence supports a modest claim of **current-state hard-drive failure assessment**. The revised paper must not claim calibrated failure probability, advance warning, remaining useful life, or ensemble superiority unless the new experiments directly support those claims.

The primary external evaluation will use **Backblaze Q1 2026**, which is later than the 2025 training period. This is the correct temporal direction and can support later-period/prospective transfer of a current-state assessment model. Because positive examples are failure-day snapshots, it still does not establish early prediction.

## Measured Q1 2026 inventory

Full details (cohort definitions, missingness per model, rebuild instructions) are in `DISKDATA2026_INVENTORY.md`. The local `DiskData2026` directory contains 90 CSV files, one per day from 2026-01-01 through 2026-03-31.

| Metric | Q1 2026 |
|---|---:|
| Rows | 30,597,484 |
| Failure-day rows | 1,030 |
| Rows from never-failed serials | 30,554,331 |
| Earlier healthy rows from failed serials | 42,123 |
| Unique serials | 351,095 |
| Unique failed serials | 1,030 |
| Unique never-failed serials | 350,065 |
| Unique models | 80 |

Serial overlap with the prepared 2025 balanced dataset:

| Metric | Count |
|---|---:|
| Prepared 2025 serials | 8,759 |
| 2025/2026 overlapping serials | 4,336 |
| 2026 failed serials present in 2025 | 12 |
| 2026 failed serials absent from 2025 | 1,018 |

This is enough for the required same-record comparison without using the old 100-record sample. The cleanest primary cohort is:

- positives: `failure_day`;
- negatives: `never_failed_healthy_day`.

`pre_failure_healthy_day` records must be reported separately because a healthy-day label on a drive that later fails is not equivalent to a never-failed drive. Failure-day records whose serials were seen in the prepared 2025 dataset should be reported as a separate overlap subset.

Missingness is strongly model-dependent. SMART 187, 188, and 191 are 100% missing for many models and 0% missing for others, so absent values must not be described as measured zeros.

## Non-negotiable experimental rules

1. Freeze all 2025-fitted artifacts before final 2026 testing.
2. Never fit or tune a scaler, encoder, model, cluster risk, weight, or threshold using 2026 labels.
3. Split 2025 data by `serial_number`, not by row.
4. Keep training, validation, and internal-test serial-number sets disjoint.
5. Evaluate all components and AHI variants on exactly the same records.
6. Report rows, unique drives, dates, and class counts at every stage.
7. Report Q1 2026 as a partial-year dataset with its exact observed date range.
8. Save redesigned artifacts separately before replacing any current artifact.
9. Treat AHI as a 0–100 risk index in **points**, not as a percentage or calibrated probability.
10. Report results honestly even if AHI does not outperform the best component.

# Part I — Major reviewer comments

## M0 — Add the required DOI

### Work

- Set `\acmDOI{10.70314/is.2026.sikdd.97}` in `paper/main.tex`.
- Compile the paper and verify that the DOI appears in the first-page footnote.
- Verify the final page count, bibliography, figure descriptions, and template compliance.
- Prepare the PDF for EasyChair, but do not upload it without explicit user instruction.

### Done when

The compiled first page displays the correct DOI ending in submission ID 97.

## M1 — Reframe the work as preliminary current-state assessment

> The current evidence fits preliminary health assessment better than demonstrated early warning.

### Interpretation

The label identifies a failure-day record. Therefore, the evaluated task is discrimination of a current SMART snapshot, not prediction several days before failure.

### Work

- Keep **Failure Assessment** in the title.
- Search the paper, README, frontend, and backend for claims involving prediction, forecasting, early warning, imminent failure, future failure, and “before they die.”
- Replace unsupported claims with terms such as current-state assessment, failure-day discrimination, current SMART condition, and risk-index assessment.
- State explicitly that the system does not estimate time to failure or the probability that a healthy-day drive will fail later.
- Rewrite the abstract, contribution list, discussion, and conclusion conservatively.

### Done when

No text presents the current failure-day experiment as demonstrated advance warning.

## M2 — Compare AHI and every component on the same records

> AHI versus its components. The main missing experiment is a direct comparison on the same records.

### Evaluation set

Use the untouched Q1 2026 evaluation records. Do not evaluate AHI on the 2025 records used to train or derive its components.

### Methods to compare

1. Random forest score.
2. Bottleneck classifier score.
3. Autoencoder anomaly score.
4. HDBSCAN cluster-risk score.
5. Simple weighted arithmetic mean.
6. Current weighted RMS AHI.
7. RMS AHI without the anomaly term.
8. If space and time permit, RMS AHI without clustering.

### Metrics

- ROC-AUC.
- PR-AUC.
- Recall/sensitivity.
- Specificity.
- False-positive rate.
- Precision.
- F1.
- Confusion matrix.
- Evaluation prevalence and class counts.
- Bootstrap confidence intervals where feasible.

Threshold-free metrics are the primary comparison. Any thresholds used for classification metrics must be fixed using 2025 validation data before inspecting 2026 labels.

### Paper output

Replace the mismatched component table and AHI-only sample result with a same-record comparison table. Keep a score-distribution figure only if space permits.

### Done when

Every method in the main result table was evaluated on identical untouched 2026 records.

## M3 — Measure error overlap and complementarity

> Error overlap would also show whether the models are actually complementary.

### Meaning

Error overlap measures whether two methods fail on the same records or whether one method correctly handles records missed by another.

### Work

For AHI versus the bottleneck classifier, and optionally all method pairs, calculate separately for failure and healthy records:

- both correct;
- both wrong;
- AHI correct and component wrong;
- component correct and AHI wrong;
- prediction disagreement rate;
- unique true positives;
- unique false positives;
- pairwise score correlation;
- optional Jaccard similarity of error sets.

### Paper output

Add a compact overlap table or a short factual summary. Claim complementarity only if AHI or another component produces meaningful unique correct decisions without an unacceptable increase in false positives.

### Done when

The ensemble-complementarity claim is directly supported, weakened, or removed according to the observed overlap.

## M4 — Make every evaluation split explicit and consistent

> The split description is currently inconsistent.

### Current inconsistency

- RF: five-fold row-level cross-validation and a separate held-out split appear in different places.
- Bottleneck classifier: 70/15/15 row-level split.
- Anomaly detector: 80/20 healthy row split and all failure rows for evaluation.
- Clustering: complete balanced dataset used to fit clusters and derive label-based cluster risks.
- AHI: separate 100-record 2023 sample.

### Work

Create a dataset-flow table containing, for each stage:

- source period and exact date range;
- row count;
- unique serial-number count;
- unique date count;
- healthy and failure counts;
- split unit;
- scaler-fitting set;
- encoder-fitting set;
- classifier-fitting set;
- HDBSCAN-fitting set;
- cluster-risk estimation set;
- weight-selection set;
- threshold-selection set;
- final test set.

Correct the current claim that the prepared 2025 data covers a calendar year. The existing balanced CSV spans 2024-10-01 through 2025-09-30.

### Done when

A reader can identify which rows and labels influenced every artifact and reported result.

## M5 — Redesign splitting around serial numbers and time

> Backblaze contains repeated daily snapshots of the same drives, so serial-number overlap matters.

### Known current-data facts

The existing 8,828-row balanced dataset has:

- 4,414 failure and 4,414 healthy rows;
- 8,759 unique serial numbers;
- 69 rows beyond one row per serial number;
- 27 serial numbers represented in both classes;
- row-level rather than drive-level classifier splits.

### Work

1. Build a manifest preserving date, serial number, model, failure label, and source period.
2. Create serial-grouped 2025 training, validation, and internal-test partitions.
3. Assert that their serial sets are pairwise disjoint.
4. Fit scalers and encoders using training records only.
5. Fit supervised models using training records only.
6. Fit HDBSCAN using training records only.
7. Derive cluster-risk fractions using training labels only.
8. Select thresholds and any optimized weights using validation only.
9. Evaluate once on the grouped 2025 internal test.
10. Evaluate once on untouched Q1 2026.

### Q1 2026 checks

- Exact CSV count and date range.
- Healthy and failure row counts.
- Unique healthy and failed serial numbers.
- Repeated snapshots.
- Serial overlap with 2025.
- Model/vendor and HDD/SSD composition.
- Feature availability and missingness.

Report two external results if serial overlap exists:

1. all eligible Q1 2026 records;
2. Q1 2026 records from serial numbers absent from 2025.

### Terminology

- Grouped 2025 test: unseen-drive generalization within the source period.
- Q1 2026 test: later-period/prospective transfer evaluation.
- 2023 test: retrospective transfer; remove it from the primary evidence or clearly label it exploratory.

### Done when

No final-test serial number is used during fitting or validation, and temporal claims match the actual periods.

## M6 — Define the prediction target precisely

> It should be clearer whether positive examples are failure-day records or earlier snapshots.

### Work

- Verify Backblaze’s exact definition of `failure=1` from the official documentation.
- State that positives are failure-day snapshots if confirmed.
- State how healthy-day snapshots are sampled.
- Do not call healthy-day records permanently healthy drives.
- State whether the unit of analysis is a daily row or one selected record per drive.
- State that no lead time is defined.
- Leave 1-day, 7-day, or remaining-useful-life prediction for future work unless a separate experiment is implemented.

### Done when

The target, unit of analysis, and limitations are unambiguous.

## M7 — Treat AHI as an index, not a probability

> AHI is better described as a risk index than a failure probability.

### Work

- Describe AHI as a **0–100 current-state risk index**.
- Use “AHI points,” not `%` or “risk percentage.”
- State that higher values mean a more failure-like current SMART state.
- State that AHI is not calibrated and is not a failure probability.
- State that AHI is not a time-to-failure estimate.
- Update plots, captions, paper text, README, API descriptions, and dashboard labels.
- Evaluate scientific results using the raw 0–100 score where possible.
- If `[3,97]` clipping remains in the application, describe it only as a display choice and not a calibration method.

### Done when

No AHI value is presented as a percentage probability of failure.

## M8 — Address balanced-test precision and real prevalence

> Balanced-test precision also needs care.

### Work

Use all Q1 2026 failure-day records where feasible. For healthy records:

- run a balanced same-record comparison for method analysis;
- also run a larger or naturally imbalanced evaluation for operational metrics where feasible;
- if healthy sampling is required, document it and do not treat sampled precision as fleet precision.

Add prevalence-scenario calculations:

`PPV = sensitivity * prevalence / (sensitivity * prevalence + FPR * (1 - prevalence))`

Suggested scenarios: 0.1%, 0.5%, and 1%, clearly labeled as hypothetical.

Interpretability can help operators investigate alarms, but it does not mathematically solve low precision at low prevalence. Do not claim otherwise.

### Done when

Precision is always accompanied by the evaluation prevalence or clearly identified as scenario-based.

## M9 — Reconcile thresholds and explain weights honestly

> The text uses 45/65, while Figure 1 uses 40/75. Section 3 says weights came from validation; Section 6 describes them as manually set.

### Current implementation

- Verdict bands: 45 and 65.
- Weights: RF 0.30, bottleneck classifier 0.40, anomaly 0.20, clustering 0.10.
- Current weights are hard-coded; no reproducible data-driven optimization is present.

### Recommended deadline-safe approach

- Call the weights heuristic/manual and fixed before the external test.
- Treat Healthy/Warning/Critical bands as display categories unless validated.
- Avoid presenting confusion-matrix performance based on arbitrary bands as the main scientific result.
- Search all figures, code, frontend, backend, README files, and the paper for 40/45/65/75 and reconcile them.

### Optional stronger approach

Optimize nonnegative weights summing to one using only the 2025 validation set and a predeclared objective. Freeze the result before 2026 evaluation.

### Done when

The paper and implementation use one consistent set of choices and accurately describe how they were selected.

## M10 — Analyze the RMS anomaly edge case

> Anomaly=1 with all other scores=0 gives 44.7, still Healthy under the threshold of 45.

### Work

Document that `100 * sqrt(0.20) = 44.72`. Evaluate:

1. current weighted RMS;
2. weighted arithmetic mean;
3. RMS without anomaly;
4. optionally, a validation-defined anomaly override.

Do not raise the anomaly weight or lower a threshold after inspecting 2026 test labels. Any change must be selected on 2025 validation data.

### Paper output

Explain that RMS emphasizes large values but does not guarantee that one maximal low-weight component controls the final category.

### Done when

The edge case and its design trade-off are explicitly addressed without test-set tuning.

## M11 — Correct dependence claims and make clustering transparent

> The classifier and clustering share an encoder representation, so they are not independent evidence sources.

### Interpretation

Sharing the encoder is not inherently incorrect. It means the bottleneck classifier and clustering signals are dependent, so the four components must not be described as independent evidence.

### Work

Replace “four independent models” with “four model components spanning supervised and unsupervised learning.” Explain that:

- bottleneck classification and HDBSCAN share an encoder and scaler;
- HDBSCAN itself is unsupervised;
- assigning empirical failure fractions to clusters is label-informed;
- cluster failure fractions from a balanced training set are not fleet probabilities.

Report:

- HDBSCAN parameters;
- cluster count and cluster sizes;
- noise count and fraction;
- failure count per cluster;
- outlier/fallback risk;
- use of HDBSCAN `approximate_predict` for new points;
- membership strength if incorporated.

Correct the paper’s “nearest cluster” description.

### Worked example

For one untouched 2026 record, show:

- relevant SMART values and engineered features;
- four component scores;
- each weighted squared contribution;
- final AHI in points;
- cluster assignment and description;
- a non-causal interpretation of the result.

**Note:** Add this worked example in the Clustering section (not Discussion) after the 2026 evaluation, using a real 2026 record.

### Done when

Dependence is acknowledged, clustering is reproducible, and interpretability is demonstrated with a concrete example.

# Part II — Smaller reviewer comments

## S1 — Reconcile the public README and identify the evaluated version

> The public DiskGuard README differs from the paper on splits, thresholds, manufacturer encoding, and encoder setup.

### Work

- Reconcile splits, thresholds, manufacturer encoding, encoder dimensions, and terminology.
- Clarify that the anomaly encoder uses a 12-dimensional bottleneck and the classifier/clustering encoder uses 8 dimensions.
- Clarify that UMAP is visualization-only while HDBSCAN fits the 8-dimensional bottleneck.
- Remove “before they die” and similar future-prediction claims.
- Record the currently inspected baseline commit: `c1a1070dbd65c946b4ec6fb57a0e0cf485d1271d`.
- Record the final evaluated revision hash only if one exists; do not create a commit automatically.

## S2 — Improve reproducibility details

> The healthy-only training size, feature formulas, imputation rules, and vendor/hardware composition would improve reproducibility.

### Work

Report:

- anomaly encoder’s existing 292,000 training and 73,000 validation healthy rows;
- redesigned grouped-split counts;
- all 19 features;
- exact engineered-feature formulas;
- per-feature imputation rules;
- missingness rates;
- model/manufacturer counts;
- HDD/SSD counts;
- capacity distribution;
- exact date ranges.

Do not automatically equate every missing SMART value with a measured zero. Distinguish true zero, unsupported attribute, and missing observation where possible. If full redesign is infeasible, document the limitation and consider missingness indicators.

## S3 — Add RF ROC-AUC and use metric-specific “best” wording

> RF ROC-AUC would be useful if available; RF F1 is slightly above the bottleneck classifier.

### Work

- Calculate RF ROC-AUC and PR-AUC on the common clean evaluation records.
- Include them in the comparison table.
- Use metric-specific statements such as highest ROC-AUC, highest recall, or highest F1.
- Do not call one component universally best based on mismatched evaluations.

## S4 — Remove or test the silent-failure explanation

> The silent-failure explanation is currently speculative.

### Work

Analyze false negatives by:

- missingness;
- model/vendor;
- critical SMART counters;
- component scores;
- cluster assignment.

Use “silent failure” only if the records support it. Otherwise state that the available features did not separate those records and the cause is unknown.

## S5 — Measure or weaken edge-deployment claims

> Strong edge-deployment claims would be more convincing with runtime and memory measurements.

### Work

Measure where feasible:

- artifact sizes;
- total memory use;
- cold-start loading time;
- per-record inference latency;
- optional batch throughput;
- test hardware, OS, Python, TensorFlow, and scikit-learn versions.

If reliable benchmarking is not completed, state only that the pipeline runs locally/offline and that edge suitability was not benchmarked.

# Implementation phases

## Phase 1 — Immediate document and data inventory

1. Add the DOI.
2. Inventory Q1 2026 after extraction.
3. Verify target semantics from Backblaze documentation.
4. Create 2025 and 2026 manifests.
5. Record row, serial-number, date, class, vendor, hardware, and missingness counts.
6. Decide the exact 2026 evaluation cohort before calculating model performance.

## Phase 2 — Leakage-safe training redesign

1. Add serial-grouped splitting utilities and assertions.
2. Create fixed 2025 train/validation/internal-test manifests.
3. Fit preprocessing and models only on allowed partitions.
4. Fit cluster risks only on training labels.
5. Select thresholds and optional weights only on validation.
6. Save new artifacts to a separate revision directory.
7. Request explicit approval before launching full retraining or replacing artifacts.

## Phase 3 — Unified evaluation

1. Evaluate the grouped 2025 internal test.
2. Evaluate untouched Q1 2026.
3. Compare all component and aggregation variants on identical records.
4. Calculate error overlap and score correlation.
5. Evaluate balanced and prevalence-aware settings.
6. Produce clustering diagnostics and one worked example.
7. Benchmark runtime and memory if feasible.
8. Save machine-readable result tables so every paper number is reproducible.

## Phase 4 — Paper revision

1. Reframe the task as current-state assessment.
2. Replace the split description with a precise dataset-flow table.
3. Replace mismatched results with the unified comparison.
4. Present AHI in index points.
5. Add imbalance, RMS, dependence, clustering, and external-validity limitations.
6. Revise abstract and conclusion according to the actual new results.
7. Preserve a modest student-level writing style.
8. Keep the paper within the conference page limit.

## Phase 5 — Cross-project consistency

1. Reconcile `README.md`, `srcML/README.md`, paper, figures, backend, and frontend.
2. Remove unsupported early-warning and probability language.
3. Ensure thresholds, weights, feature counts, and architecture descriptions agree.
4. Identify the exact evaluated code and artifact versions.

## Phase 6 — Verification and submission preparation

1. Run split-overlap and leakage assertions.
2. Test AHI formulas, ablations, and threshold boundaries.
3. Regenerate tables and figures from saved results.
4. Compile with `pdflatex`, `biber`, `pdflatex`, `pdflatex`.
5. Check citations, references, DOI, figure descriptions, warnings, and page count.
6. Perform a final claim-to-evidence audit.
7. Review `git status` and `git diff` without committing or pushing.
8. Leave EasyChair upload to the user unless explicitly requested.

# Deadline-prioritized minimum revision

If time prevents the complete redesign, prioritize:

1. Correct DOI.
2. Current-state terminology and removal of early-warning claims.
3. Exact split and dataset accounting.
4. Q1 2026 same-record comparison of components, RMS AHI, arithmetic mean, and no-anomaly AHI.
5. Error-overlap analysis.
6. AHI as a 0–100 uncalibrated risk index.
7. Balanced-versus-operational-prevalence discussion.
8. Consistent manual weights and thresholds.
9. Clustering dependence and assignment details.
10. README reconciliation.

# Target final claim

A defensible target claim is:

> The system provides a preliminary, component-wise current-state risk assessment from SMART snapshots and is evaluated for unseen-drive and later-period generalization.

Do not claim that the hybrid is superior to the bottleneck classifier unless the common Q1 2026 evaluation supports that conclusion.
