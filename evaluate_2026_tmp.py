
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    confusion_matrix,
)

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from srcML.ahi_final import (
    PROJECT_ROOT,
    _load_sklearn_pipeline,
    _load_tf_clf_artifacts,
    _load_anomaly_artifacts,
    _load_clustering_artifacts,
    _compute_ahi,
    W_SKLEARN, W_TF_CLF, W_ANOMALY, W_CLUSTER,
)
from srcML.nn_preprocessing.preprocessing import prepare_features

DATA_DIR = PROJECT_ROOT / "DiskData2026" / "data_Q1_2026"
OUTPUT_DIR = PROJECT_ROOT / "DiskJson"
GRAPH_DIR = PROJECT_ROOT / "Graphs"
PREPARED_2025_CSV = PROJECT_ROOT / "csv" / "koncniPodatkiZaModel.csv"

# 1. Data loading — extract failure-day + sample never-failed healthy

def load_2026_cohorts(n_healthy: int, random_state: int) -> pd.DataFrame:
    import glob
    csv_files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    print(f"Scanning {len(csv_files)} CSV files...")

    # Pass 1: collect all failure rows + track failed serials
    failure_parts = []
    failed_serials = set()
    for i, f in enumerate(csv_files, 1):
        df = pd.read_csv(f, low_memory=False)
        if "failure" not in df.columns:
            continue
        failed = df[df["failure"] == 1]
        if not failed.empty:
            failure_parts.append(failed)
            failed_serials.update(failed["serial_number"].tolist())
        if i % 30 == 0:
            print(f"  [{i}/{len(csv_files)}] failures so far: {sum(len(x) for x in failure_parts):,}")

    failure_df = pd.concat(failure_parts, ignore_index=True)
    print(f"  Failure-day rows: {len(failure_df)}")

    # Pass 2: sample healthy rows from never-failed serials
    n_target = min(n_healthy, len(failure_df))
    # Sample from a few files to get enough healthy rows
    healthy_parts = []
    needed = n_target
    rng = np.random.default_rng(random_state)
    for i, f in enumerate(csv_files, 1):
        if needed <= 0:
            break
        df = pd.read_csv(f, low_memory=False)
        if "failure" not in df.columns:
            continue
        healthy = df[(df["failure"] == 0) & (~df["serial_number"].isin(failed_serials))]
        if not healthy.empty:
            n_take = min(len(healthy), needed)
            healthy_parts.append(healthy.sample(n=n_take, random_state=random_state + i))
            needed -= n_take
        if i % 30 == 0:
            collected = sum(len(x) for x in healthy_parts)
            print(f"  [{i}/{len(csv_files)}] healthy sampled: {collected:,} / {n_target:,}")

    healthy_df = pd.concat(healthy_parts, ignore_index=True)
    print(f"  Healthy rows sampled: {len(healthy_df)}")

    # Combine
    sample = pd.concat([healthy_df, failure_df], ignore_index=True)
    print(f"  Total evaluation set: {len(sample)} rows ({(sample['failure']==1).sum()} failed, {(sample['failure']==0).sum()} healthy)")
    return sample


def load_2025_serials() -> set:
    """Load serial numbers from the prepared 2025 dataset for overlap flagging."""
    if not PREPARED_2025_CSV.exists():
        return set()
    df = pd.read_csv(PREPARED_2025_CSV, usecols=["serial_number"], low_memory=False)
    return set(df["serial_number"].tolist())

# 2. Batch scoring — all 4 models on the same records

def score_all_models(sample: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Score all 4 models + AHI on the same records. Returns DataFrame with per-record scores."""
    n = len(sample)
    print(f"\nScoring {n} records through all 4 models...")

    # --- Random Forest (sklearn pipeline) ---
    print("  RF...")
    rf_scores = np.zeros(n)
    pipeline = artifacts["sklearn_pipeline"]
    for i in range(n):
        row = sample.iloc[[i]]
        result = pipeline.analyze(row)
        rf_scores[i] = float(result["failure_probability"])

    # --- Bottleneck classifier (batched) ---
    print("  Bottleneck classifier...")
    X = prepare_features(sample)
    clf_scaler = artifacts["clf_scaler"]
    encoder = artifacts["encoder"]
    classifier = artifacts["classifier"]
    X_scaled = clf_scaler.transform(X).astype("float32")
    bottleneck = encoder.predict(X_scaled, batch_size=256, verbose=0)
    clf_scores = classifier.predict(bottleneck, batch_size=256, verbose=0).flatten()

    # --- Anomaly detector (batched) ---
    print("  Anomaly detector...")
    ae_model = artifacts["ae_model"]
    ae_scaler = artifacts["ae_scaler"]
    ae_meta = artifacts["ae_meta"]
    X_ae = prepare_features(sample)
    X_ae_scaled = ae_scaler.transform(X_ae).astype("float32")
    reconstructed = ae_model.predict(X_ae_scaled, batch_size=256, verbose=0)
    errors = np.mean(np.abs(X_ae_scaled - reconstructed), axis=1)
    threshold = ae_meta["threshold"]
    p999 = ae_meta["normalization"]["score_p999"]
    if p999 <= threshold:
        anomaly_scores = np.zeros(n)
    else:
        anomaly_scores = np.clip((errors - threshold) / (p999 - threshold), 0.0, 1.0)

    # --- HDBSCAN clustering (batched via approximate_predict) ---
    print("  HDBSCAN clustering...")
    clusterer = artifacts["clusterer"]
    cluster_meta = artifacts["cluster_meta"]
    cluster_risks = cluster_meta["cluster_risk"]
    fallback_score = float(cluster_risks.get("-1", {}).get("risk_score", 0.5))

    try:
        import hdbscan as hdbscan_lib
        labels, _ = hdbscan_lib.approximate_predict(clusterer, bottleneck)
        cluster_scores = np.array([
            float(cluster_risks.get(str(int(l)), {}).get("risk_score", fallback_score))
            for l in labels
        ])
    except Exception as e:
        print(f"  [WARNING] HDBSCAN failed ({e}), using fallback {fallback_score}")
        cluster_scores = np.full(n, fallback_score)

    # --- AHI ---
    print("  AHI...")
    ahi_scores = np.zeros(n)
    for i in range(n):
        result = _compute_ahi(rf_scores[i], clf_scores[i], anomaly_scores[i], cluster_scores[i])
        ahi_scores[i] = result["ahi_score"]

    # --- Assemble output ---
    labels = sample["failure"].astype(int).values
    result_df = pd.DataFrame({
        "serial_number": sample["serial_number"].values,
        "date": sample["date"].values,
        "model": sample["model"].values,
        "label": labels,
        "rf_score": rf_scores,
        "bottleneck_score": clf_scores,
        "anomaly_score": anomaly_scores,
        "cluster_score": cluster_scores,
        "ahi": ahi_scores,
    })

    return result_df

# 3. Metrics

def compute_metrics(y_true, y_score, name: str) -> dict:
    """Compute metrics for a single model's scores."""
    y_pred = (y_score >= 0.5).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = float("nan")

    recall = recall_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "model": name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def compute_all_metrics(result_df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for all models + AHI."""
    y_true = result_df["label"].values
    metrics = []

    # Individual models (threshold 0.5)
    for col, name in [
        ("rf_score", "Random Forest"),
        ("bottleneck_score", "Bottleneck Classifier"),
        ("anomaly_score", "Anomaly Detector"),
        ("cluster_score", "HDBSCAN Cluster Risk"),
    ]:
        metrics.append(compute_metrics(y_true, result_df[col].values, name))

    # AHI (threshold 45 for WARNING+, 65 for CRITICAL)
    y_pred_ahi = (result_df["ahi"] >= 45).astype(int)
    try:
        roc_auc_ahi = roc_auc_score(y_true, result_df["ahi"].values / 100.0)
    except ValueError:
        roc_auc_ahi = float("nan")
    try:
        pr_auc_ahi = average_precision_score(y_true, result_df["ahi"].values / 100.0)
    except ValueError:
        pr_auc_ahi = float("nan")

    recall_ahi = recall_score(y_true, y_pred_ahi, pos_label=1)
    precision_ahi = precision_score(y_true, y_pred_ahi, pos_label=1, zero_division=0)
    f1_ahi = f1_score(y_true, y_pred_ahi, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_ahi, labels=[0, 1]).ravel()

    metrics.append({
        "model": "AHI (threshold 45)",
        "roc_auc": round(roc_auc_ahi, 4),
        "pr_auc": round(pr_auc_ahi, 4),
        "recall": round(recall_ahi, 4),
        "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "precision": round(precision_ahi, 4),
        "f1": round(f1_ahi, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    })

    return pd.DataFrame(metrics)

# 4. AHI plot

def plot_ahi(result_df: pd.DataFrame, out_path: Path, dataset_label: str = "Q1 2026"):
    BG = "#111111"
    rng = np.random.default_rng(seed=0)

    labels = result_df["label"].astype(int).to_numpy()
    ahi = result_df["ahi"].to_numpy()

    jitter = rng.uniform(-0.18, 0.18, size=len(labels))
    x_pos = labels.astype(float) + jitter

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sc = ax.scatter(x_pos, ahi, c=ahi, cmap="RdYlGn_r", vmin=0, vmax=100,
                    s=22, alpha=0.88, linewidths=0)

    for level, color, lbl in [(45, "#f0c419", "WARNING  45"), (65, "#ff5555", "CRITICAL  65")]:
        ax.axhline(level, color=color, lw=1.0, ls="--", alpha=0.65)
        ax.text(1.44, level, lbl, color=color, va="center", ha="right", fontsize=7.5)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Failure = 0\n(Healthy)", "Failure = 1\n(Failed)"], color="#dddddd", fontsize=11)
    ax.set_ylabel("AHI (index points)", color="#dddddd", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("AHI (index points)", color="#dddddd", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    for lbl, color in [(0, "#55dd88"), (1, "#ff7777")]:
        mean_val = float(result_df[result_df["label"] == lbl]["ahi"].mean())
        ax.hlines(mean_val, lbl - 0.35, lbl + 0.35, colors=color, lw=2.0, alpha=0.9, zorder=5)
        ax.text(lbl + 0.37, mean_val, f"mean {mean_val:.1f}", color=color, va="center", fontsize=8)

    n_h = int((result_df["label"] == 0).sum())
    n_f = int((result_df["label"] == 1).sum())
    ax.set_title(f"AHI vs. actual disk failure (n={n_h+n_f}, {dataset_label})",
                 color="#eeeeee", fontsize=11, pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")

# 5. Main

def main():
    parser = argparse.ArgumentParser(description="2026 AHI evaluation")
    parser.add_argument("--n-healthy", type=int, default=1030,
                        help="Number of healthy rows to sample (default: 1030 = all failures)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print("STEP 1: Loading Q1 2026 data")
    print("=" * 60)
    sample = load_2026_cohorts(args.n_healthy, args.random_state)

    # Load 2025 serials for overlap flagging
    serials_2025 = load_2025_serials()
    sample["seen_in_2025"] = sample["serial_number"].isin(serials_2025)
    n_overlap = sample["seen_in_2025"].sum()
    print(f"  Serials seen in 2025 training: {n_overlap} of {len(sample)}")

    # Load artifacts
    print("\n" + "=" * 60)
    print("STEP 2: Loading trained artifacts")
    print("=" * 60)
    artifacts = {}
    artifacts["sklearn_pipeline"] = _load_sklearn_pipeline(PROJECT_ROOT / "srcML" / "sklearn")
    encoder, classifier, clf_scaler, clf_meta = _load_tf_clf_artifacts(
        PROJECT_ROOT / "srcML" / "tensorflow_classification")
    artifacts["encoder"] = encoder
    artifacts["classifier"] = classifier
    artifacts["clf_scaler"] = clf_scaler
    ae_model, ae_scaler, ae_meta = _load_anomaly_artifacts(
        PROJECT_ROOT / "srcML" / "tensorflow_anomaly")
    artifacts["ae_model"] = ae_model
    artifacts["ae_scaler"] = ae_scaler
    artifacts["ae_meta"] = ae_meta
    clusterer, cluster_meta = _load_clustering_artifacts(
        PROJECT_ROOT / "srcML" / "tensorflow_clustering")
    artifacts["clusterer"] = clusterer
    artifacts["cluster_meta"] = cluster_meta
    print("  All artifacts loaded.")

    # Score
    print("\n" + "=" * 60)
    print("STEP 3: Scoring all models on 2026 records")
    print("=" * 60)
    result_df = score_all_models(sample, artifacts)
    result_df["seen_in_2025"] = sample["seen_in_2025"].values

    # Metrics
    print("\n" + "=" * 60)
    print("STEP 4: Computing metrics")
    print("=" * 60)
    metrics_df = compute_all_metrics(result_df)
    print("\n--- All Q1 2026 (balanced) ---")
    print(metrics_df.to_string(index=False))

    # Overlap subset
    unseen_df = result_df[~result_df["seen_in_2025"]]
    if len(unseen_df) > 0 and unseen_df["label"].nunique() > 1:
        unseen_metrics = compute_all_metrics(unseen_df)
        print("\n--- 2026 serials NOT seen in 2025 (strict unseen) ---")
        print(unseen_metrics.to_string(index=False))
    else:
        print("\n[SKIP] Not enough class diversity in unseen subset for metrics.")

    # Save results
    print("\n" + "=" * 60)
    print("STEP 5: Saving outputs")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    scores_path = OUTPUT_DIR / "ahi_2026_scores.csv"
    result_df.to_csv(scores_path, index=False)
    print(f"  Per-record scores: {scores_path}")

    metrics_path = OUTPUT_DIR / "ahi_2026_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics table: {metrics_path}")

    plot_path = GRAPH_DIR / "ahi_2026.png"
    plot_ahi(result_df, plot_path, "Q1 2026")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_h = int((result_df["label"] == 0).sum())
    n_f = int((result_df["label"] == 1).sum())
    mean_h = result_df[result_df["label"] == 0]["ahi"].mean()
    mean_f = result_df[result_df["label"] == 1]["ahi"].mean()
    std_h = result_df[result_df["label"] == 0]["ahi"].std()
    std_f = result_df[result_df["label"] == 1]["ahi"].std()
    print(f"  Records: {n_h} healthy + {n_f} failed = {n_h + n_f} total")
    print(f"  AHI healthy: mean={mean_h:.1f} (std={std_h:.1f})")
    print(f"  AHI failed:  mean={mean_f:.1f} (std={std_f:.1f})")
    print(f"  Separation: {mean_f - mean_h:.1f} points")
    print(f"  Serials seen in 2025: {n_overlap} / {len(result_df)}")


if __name__ == "__main__":
    main()
