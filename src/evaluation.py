from __future__ import annotations

from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def threshold_candidates(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Evaluate candidate thresholds on the external validation partition only."""
    if len(np.unique(y_true)) < 2:
        raise ValueError("Validation set must contain normal and anomalous instances.")
    qs = np.unique(np.concatenate([
        np.linspace(0.80, 0.999, 300),
        np.linspace(0.50, 0.80, 80),
    ]))
    candidates = np.unique(np.quantile(scores, qs))
    rows = []
    for t in candidates:
        pred = (scores >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "alert_rate": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """F1-optimal validation threshold retained as an experimental reference policy."""
    table = threshold_candidates(y_true, scores)
    best = table.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), float(best["f1"])


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


def precision_at_alert_budgets(
    y_true: np.ndarray,
    scores: np.ndarray,
    budgets: Iterable[float] = (0.005, 0.01, 0.02, 0.05),
) -> pd.DataFrame:
    """Operational view: metrics if analysts inspect only the top-k score fraction."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    total_pos = int(y_true.sum())
    rows = []
    for budget in budgets:
        if not (0 < budget <= 1):
            raise ValueError("Alert budgets must be in (0,1].")
        k = max(1, int(np.ceil(len(scores) * float(budget))))
        idx = order[:k]
        tp = int(y_true[idx].sum())
        precision = tp / k
        recall = tp / total_pos if total_pos else 0.0
        cutoff = float(scores[idx[-1]])
        rows.append({
            "alert_budget": float(budget),
            "top_k": int(k),
            "score_cutoff": cutoff,
            "precision_at_k": float(precision),
            "recall_at_k": float(recall),
            "true_positives_at_k": tp,
        })
    return pd.DataFrame(rows)


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


def recall_vs_intensity_curve(
    pred_df: pd.DataFrame,
    model_col: str,
    bin_width: float = 0.10,
) -> pd.DataFrame:
    """Recall as a function of continuous anomaly intensity D.

    Fixed-width bins preserve D as the underlying continuous control variable and
    avoid treating low/medium/high bands as the only analysis.
    """
    if not (0 < bin_width <= 1):
        raise ValueError("bin_width must be in (0,1].")
    anomalies = pred_df[pred_df["is_anomaly"] == 1].copy()
    if anomalies.empty:
        return pd.DataFrame()
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)
    edges[-1] = 1.0000001
    anomalies["D_bin"] = pd.cut(
        anomalies["anomaly_intensity"],
        bins=edges,
        include_lowest=True,
        right=False,
    )
    rows = []
    for interval, sub in anomalies.groupby("D_bin", observed=True):
        if not len(sub):
            continue
        rows.append({
            "model": model_col,
            "D_left": float(interval.left),
            "D_right": min(1.0, float(interval.right)),
            "D_mid": float((interval.left + min(1.0, interval.right)) / 2.0),
            "mean_D": float(sub["anomaly_intensity"].mean()),
            "n_anomalies": int(len(sub)),
            "recall": float(sub[f"pred_{model_col}"].mean()),
            "mean_score": float(sub[f"score_{model_col}"].mean()),
        })
    return pd.DataFrame(rows)


def stratified_metrics(pred_df: pd.DataFrame, model_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Bands remain as a descriptive summary, while recall_vs_intensity_curve
    # provides the requested continuous-D analysis.
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


def paired_bootstrap_pr_auc(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Paired bootstrap CI for ΔPR-AUC = A - B on the same test observations."""
    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("Paired bootstrap arrays must have equal length.")
    point = float(average_precision_score(y_true, scores_a) - average_precision_score(y_true, scores_b))
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y_true)
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(
            float(average_precision_score(yb, scores_a[idx]) - average_precision_score(yb, scores_b[idx]))
        )
    if not diffs:
        return {"delta_pr_auc": point, "ci95_low": float("nan"), "ci95_high": float("nan"), "n_bootstrap_valid": 0}
    lo, hi = np.quantile(np.asarray(diffs), [0.025, 0.975])
    return {
        "delta_pr_auc": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_bootstrap_valid": int(len(diffs)),
    }
