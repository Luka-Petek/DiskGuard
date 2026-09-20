#  python srcML/evaluate_2026.py --data-dir DiskData2026/data_Q1_2026

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from srcML.ahi_final import (
    PROJECT_ROOT,
    _load_sklearn_pipeline,
    _load_tf_clf_artifacts,
    _load_anomaly_artifacts,
    _load_clustering_artifacts,
    _score_sklearn,
    _score_tf_clf,
    _score_anomaly,
    _score_clustering,
    _compute_ahi,
)
from srcML.nn_preprocessing.preprocessing import build_current_state_evaluation_from_csvs


def _read_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return pd.read_csv(csv_path, sep=";", low_memory=False)


def _balanced_sample(df: pd.DataFrame, n_per_class: int, random_state: int) -> pd.DataFrame:
    failed  = df[df["failure"] == 1]
    healthy = df[df["failure"] == 0]
    n_f = min(n_per_class, len(failed))
    n_h = min(n_per_class, len(healthy))
    part_f = failed.sample(n=n_f, random_state=random_state)
    part_h = healthy.sample(n=n_h, random_state=random_state + 1)
    return pd.concat([part_h, part_f], ignore_index=True)


def _compute_ahi_for_row(raw_row: pd.DataFrame,
                          sklearn_pipeline,
                          encoder, classifier, clf_scaler,
                          ae_model, ae_scaler, ae_meta,
                          clusterer, cluster_meta) -> dict:
    s_skl         = _score_sklearn(sklearn_pipeline, raw_row)
    s_clf         = _score_tf_clf(encoder, classifier, clf_scaler, raw_row)
    s_an          = _score_anomaly(ae_model, ae_scaler, ae_meta, raw_row)
    cluster_result = _score_clustering(clusterer, cluster_meta, encoder, clf_scaler, raw_row)
    result = _compute_ahi(s_skl, s_clf, s_an, cluster_result["score"])
    result["cluster_info"] = {
        "cluster_id":  cluster_result["cluster_id"],
        "risk_label":  cluster_result["risk_label"],
        "risk_score":  round(cluster_result["score"], 4),
        "description": cluster_result["description"],
    }
    return result


def _metric_row(name: str, labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "recall": recall_score(labels, predictions),
        "specificity": tn / (tn + fp),
        "false_positive_rate": fp / (fp + tn),
        "precision": precision_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _component_metrics(result_df: pd.DataFrame, classifier_threshold: float) -> pd.DataFrame:
    labels = result_df["label"].to_numpy()
    rows = [
        _metric_row("Random Forest", labels, result_df["sklearn_failure_prob"].to_numpy(), 0.5),
        _metric_row("Bottleneck classifier", labels, result_df["tf_clf_failure_prob"].to_numpy(), classifier_threshold),
        _metric_row("Anomaly detector", labels, result_df["anomaly_score"].to_numpy(), np.nextafter(0.0, 1.0)),
        _metric_row("HDBSCAN cluster risk", labels, result_df["cluster_risk_score"].to_numpy(), 0.5),
        _metric_row("AHI", labels, result_df["ahi"].to_numpy(), 45.0),
    ]
    return pd.DataFrame(rows)


def evaluate_ahi(
    sample: pd.DataFrame,
    sklearn_dir: Path,
    clf_dir: Path,
    anomaly_dir: Path,
    clustering_dir: Path,
    random_state: int,
    output_csv: Optional[Path],
    metrics_out: Optional[Path],
    plot_out: Optional[Path],
    dataset_label: str,
) -> dict:
    print(f"Vzorec: {len(sample)} diskov  ({sample['failure'].sum():.0f} failed, {(sample['failure']==0).sum():.0f} healthy)")

    print("Nalagam artefakte...")
    sklearn_pipeline = _load_sklearn_pipeline(sklearn_dir)
    encoder, classifier, clf_scaler, clf_meta = _load_tf_clf_artifacts(clf_dir)
    ae_model, ae_scaler, ae_meta = _load_anomaly_artifacts(anomaly_dir)
    try:
        clusterer, cluster_meta = _load_clustering_artifacts(clustering_dir)
    except Exception as e:
        print(f"[OPOZORILO] Clustering ni na voljo ({e}), fallback = 0.5")
        clusterer, cluster_meta = None, {"cluster_risk": {"-1": {"risk_score": 0.5}}}

    rows = []
    total = len(sample)
    for i in range(total):
        raw_row = sample.iloc[[i]]
        label = int(raw_row["failure"].iloc[0])
        print(f"  Disk {i+1:>3}/{total}  (failure={label})", end="\r", flush=True)

        fused = _compute_ahi_for_row(
            raw_row,
            sklearn_pipeline,
            encoder, classifier, clf_scaler,
            ae_model, ae_scaler, ae_meta,
            clusterer, cluster_meta,
        )
        serial_number = str(raw_row["serial_number"].iloc[0]) if "serial_number" in raw_row else ""
        rows.append({
            "disk_idx":  i,
            "date": raw_row["date"].iloc[0] if "date" in raw_row else "",
            "serial_number": serial_number,
            "model": raw_row["model"].iloc[0] if "model" in raw_row else "",
            "label":     label,
            "ahi":       fused["ahi_score"],
            "verdict":   fused["verdict"],
            "sklearn_failure_prob": fused["components"]["sklearn_failure_prob"],
            "tf_clf_failure_prob":  fused["components"]["tf_clf_failure_prob"],
            "anomaly_score":        fused["components"]["anomaly_score"],
            "cluster_risk_score":   fused["components"]["cluster_risk_score"],
        })

    print(f"\nDone — {total} diskov ocenjenih.")
    result_df = pd.DataFrame(rows)
    classifier_threshold = float(clf_meta.get("threshold", 0.5))
    metrics_df = _component_metrics(result_df, classifier_threshold)
    print("\nEvalvacija na novih serijskih številkah:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Rezultati shranjeni: {output_csv}")

    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_out, index=False)
        print(f"Metrike shranjene: {metrics_out}")

    if plot_out is not None:
        _plot_color_rock(result_df, plot_out, dataset_label)
        print(f"Graf shranjen: {plot_out}")

    summary = {
        "sample_size": int(len(result_df)),
        "n_failed":  int(result_df["label"].sum()),
        "n_healthy": int((result_df["label"] == 0).sum()),
        "ahi_mean_failed":  round(float(result_df[result_df["label"]==1]["ahi"].mean()), 2),
        "ahi_mean_healthy": round(float(result_df[result_df["label"]==0]["ahi"].mean()), 2),
        "output_csv": str(output_csv) if output_csv else None,
        "plot":       str(plot_out)   if plot_out   else None,
    }
    return summary


def _plot_color_rock(df: pd.DataFrame, out_path: Path, dataset_label: str = "in-sample") -> None:
    BG   = "#111111"
    rng  = np.random.default_rng(seed=0)

    labels = df["label"].astype(int).to_numpy()
    ahi    = df["ahi"].to_numpy()

    jitter = rng.uniform(-0.18, 0.18, size=len(labels))
    x_pos  = labels.astype(float) + jitter

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sc = ax.scatter(
        x_pos, ahi,
        c=ahi, cmap="RdYlGn_r",
        vmin=0, vmax=100,
        s=22, alpha=0.88, linewidths=0,
    )

    for level, color, label in [
        (45, "#f0c419", "WARNING  45"),
        (65, "#ff5555", "CRITICAL  65"),
    ]:
        ax.axhline(level, color=color, lw=1.0, ls="--", alpha=0.65)
        ax.text(1.44, level, label, color=color, va="center", ha="right", fontsize=7.5)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Failure = 0\n(Healthy)", "Failure = 1\n(Failed)"],
                       color="#dddddd", fontsize=11)
    ax.set_ylabel("AHI (index points)", color="#dddddd", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("AHI (index points)", color="#dddddd", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    for lbl, color in [(0, "#55dd88"), (1, "#ff7777")]:
        mean_val = float(df[df["label"] == lbl]["ahi"].mean())
        ax.hlines(mean_val, lbl - 0.35, lbl + 0.35,
                  colors=color, lw=2.0, alpha=0.9, zorder=5)
        ax.text(lbl + 0.37, mean_val, f"mean {mean_val:.1f}",
                color=color, va="center", fontsize=8)

    n_h = int((df["label"] == 0).sum())
    n_f = int((df["label"] == 1).sum())
    ax.set_title(
        f"AHI vs. actual disk failure  (n={n_h+n_f}, {dataset_label})",
        color="#eeeeee", fontsize=11, pad=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Evalvacija vseh 4 modelov + AHI na istih 2026 zapisih.")
    p.add_argument("--data-csv",   type=str, default=None,
                   help="Pot do ene CSV datoteke s stolpcem 'failure'.")
    p.add_argument("--data-dir",   type=str, default=None,
                   help="Mapa z CSV datotekami (npr. DiskData2026/data_Q1_2026).")
    p.add_argument("--n-per-class", type=int, default=None,
                   help="Stevilo zdravih diskov za vzorce. Privzeto enako kot stevilo failure zapisov.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--dataset-label", type=str, default="Q1 2026")
    p.add_argument("--training-csv", type=str,
                   default=str(PROJECT_ROOT / "csv" / "koncniPodatkiZaModel.csv"))
    p.add_argument("--output-csv",  type=str,
                   default=str(PROJECT_ROOT / "DiskJson" / "ahi_2026_scores.csv"))
    p.add_argument("--metrics-out", type=str,
                   default=str(PROJECT_ROOT / "DiskJson" / "ahi_2026_metrics.csv"))
    p.add_argument("--plot-out",    type=str,
                   default=str(PROJECT_ROOT / "Graphs" / "ahi_2026.png"))
    p.add_argument("--sklearn-dir", type=str,
                   default=str(PROJECT_ROOT / "srcML" / "sklearn"))
    p.add_argument("--clf-dir",     type=str,
                   default=str(PROJECT_ROOT / "srcML" / "tensorflow_classification"))
    p.add_argument("--anomaly-dir", type=str,
                   default=str(PROJECT_ROOT / "srcML" / "tensorflow_anomaly"))
    p.add_argument("--clustering-dir", type=str,
                   default=str(PROJECT_ROOT / "srcML" / "tensorflow_clustering"))

    args = p.parse_args()

    if not args.data_csv and not args.data_dir:
        p.error("Podaj --data-csv ali --data-dir")

    training_csv = Path(args.training_csv)
    if not training_csv.exists():
        raise FileNotFoundError(f"Učna množica ne obstaja: {training_csv}")
    training_serials = set(
        pd.read_csv(training_csv, usecols=["serial_number"])["serial_number"].astype(str)
    )

    if args.data_dir:
        data_dir = Path(args.data_dir)
        print(f"Branje CSV datotek iz: {data_dir}")
        healthy_df, failure_df = build_current_state_evaluation_from_csvs(
            data_dir=data_dir,
            n_healthy=args.n_per_class,
            random_state=args.random_state,
            excluded_serials=training_serials,
        )
        sample = pd.concat([healthy_df, failure_df], ignore_index=True)
        dataset_label = args.dataset_label
    else:
        csv_path = Path(args.data_csv)
        df = _read_csv(csv_path)
        if df.empty:
            raise RuntimeError(f"CSV je prazen: {csv_path}")
        if "failure" not in df.columns or "serial_number" not in df.columns:
            raise RuntimeError("CSV mora vsebovati stolpca 'failure' in 'serial_number'.")
        df = df[~df["serial_number"].astype(str).isin(training_serials)]
        sample = _balanced_sample(df, n_per_class=args.n_per_class or 50, random_state=args.random_state)
        dataset_label = args.dataset_label

    if sample["serial_number"].astype(str).isin(training_serials).any():
        raise RuntimeError("Evalvacijska množica vsebuje serijsko številko iz učne množice.")

    summary = evaluate_ahi(
        sample=sample,
        sklearn_dir=Path(args.sklearn_dir),
        clf_dir=Path(args.clf_dir),
        anomaly_dir=Path(args.anomaly_dir),
        clustering_dir=Path(args.clustering_dir),
        random_state=args.random_state,
        output_csv=Path(args.output_csv),
        metrics_out=Path(args.metrics_out),
        plot_out=Path(args.plot_out),
        dataset_label=dataset_label,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
