from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        raise ValueError("Validation set must contain normal and anomalous instances.")
    # Candidate thresholds from score quantiles plus unique values around the anomaly tail.
    qs = np.unique(np.concatenate([
        np.linspace(0.80, 0.999, 300),
        np.linspace(0.50, 0.80, 80),
    ]))
    candidates = np.unique(np.quantile(scores, qs))
    best_t = float(candidates[0])
    best_f1 = -1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    pr_auc = average_precision_score(y_true, scores)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def add_intensity_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def band(x: float) -> str:
        if x <= 1.0 / 3.0:
            return "Baja"
        if x <= 2.0 / 3.0:
            return "Media"
        return "Alta"
    out["intensity_band"] = out["anomaly_intensity"].apply(band)
    return out


def stratified_metrics(pred_df: pd.DataFrame, model_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Only anomaly recall is meaningful within anomaly-only strata; F1 requires normals too.
    d = add_intensity_band(pred_df)
    intensity_rows = []
    for band in ["Baja", "Media", "Alta"]:
        sub = d[(d["is_anomaly"] == 1) & (d["intensity_band"] == band)]
        if len(sub):
            intensity_rows.append({
                "model": model_col,
                "intensity": band,
                "n_anomalies": int(len(sub)),
                "recall": float(sub[f"pred_{model_col}"].mean()),
                "mean_score": float(sub[f"score_{model_col}"].mean()),
            })
    dim_rows = []
    dims = ["amount", "hour", "frequency", "recipient", "device", "location"]
    for dim in dims:
        sub = d[(d["is_anomaly"] == 1) & d["changed_dimensions"].str.contains(dim, regex=False)]
        if len(sub):
            dim_rows.append({
                "model": model_col,
                "dimension": dim,
                "n_anomalies": int(len(sub)),
                "recall": float(sub[f"pred_{model_col}"].mean()),
                "mean_intensity": float(sub["anomaly_intensity"].mean()),
            })
    return pd.DataFrame(intensity_rows), pd.DataFrame(dim_rows)
