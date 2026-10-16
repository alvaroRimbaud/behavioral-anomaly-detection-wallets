from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.simulator import SimulationConfig, simulate_transactions
from src.features import build_features, RAW_FEATURES, HYBRID_FEATURES
from src.models import (
    set_seed, fit_iforest, iforest_scores, fit_lof, lof_scores,
    AutoencoderDetector, AEConfig,
)
from src.evaluation import choose_threshold, metrics_at_threshold, stratified_metrics
from src.report import build_report


def run(config: SimulationConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    t0 = time.perf_counter()
    transactions, profiles, baselines = simulate_transactions(config)
    t_sim = time.perf_counter() - t0

    features, _ = build_features(transactions, baselines)
    post = features[features["phase"].isin(["train", "validation", "test"])].copy()
    if int(post.loc[post["phase"] == "train", "is_anomaly"].sum()) != 0:
        raise RuntimeError("Design violation: training contains anomalies.")

    # Causal split assertions.
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    train_max = tx.loc[tx["phase"] == "train", "timestamp"].max()
    val_min = tx.loc[tx["phase"] == "validation", "timestamp"].min()
    val_max = tx.loc[tx["phase"] == "validation", "timestamp"].max()
    test_min = tx.loc[tx["phase"] == "test", "timestamp"].min()
    assert train_max < val_min and val_max < test_min, "Temporal split overlap detected."

    results = []
    prediction_frames = []
    all_intensity = []
    all_dimensions = []

    feature_sets = {"raw": RAW_FEATURES, "raw+individualizadas": HYBRID_FEATURES}
    for feature_set, cols in feature_sets.items():
        train = post[post["phase"] == "train"]
        val = post[post["phase"] == "validation"]
        test = post[post["phase"] == "test"]
        X_train = train[cols].to_numpy(float)
        X_val = val[cols].to_numpy(float)
        X_test = test[cols].to_numpy(float)
        y_val = val["is_anomaly"].to_numpy(int)
        y_test = test["is_anomaly"].to_numpy(int)

        # Isolation Forest — thesis core model.
        start = time.perf_counter()
        ifm = fit_iforest(X_train, seed=config.seed)
        fit_s = time.perf_counter() - start
        val_scores = iforest_scores(ifm, X_val)
        test_start = time.perf_counter()
        test_scores = iforest_scores(ifm, X_test)
        infer_s = time.perf_counter() - test_start
        th, _ = choose_threshold(y_val, val_scores)
        met = metrics_at_threshold(y_test, test_scores, th)
        met.update({"model": "Isolation Forest", "feature_set": feature_set, "fit_seconds": fit_s, "inference_seconds": infer_s})
        results.append(met)
        pred = test[["transaction_id", "user_id", "timestamp", "is_anomaly", "anomaly_type", "anomaly_intensity", "anomaly_breadth", "changed_dimensions"]].copy()
        pred["score_if"] = test_scores
        pred["pred_if"] = (test_scores >= th).astype(int)
        if feature_set == "raw+individualizadas":
            ii, dd = stratified_metrics(pred, "if")
            ii["feature_set"] = feature_set
            dd["feature_set"] = feature_set
            all_intensity.append(ii)
            all_dimensions.append(dd)
            prediction_frames.append(pred)

        # Autoencoder — thesis core model.
        start = time.perf_counter()
        ae = AutoencoderDetector(seed=config.seed, config=AEConfig(epochs=15, batch_size=512, patience=4))
        ae.fit(X_train)
        fit_s = time.perf_counter() - start
        val_scores = ae.score_samples(X_val)
        test_start = time.perf_counter()
        test_scores = ae.score_samples(X_test)
        infer_s = time.perf_counter() - test_start
        th, _ = choose_threshold(y_val, val_scores)
        met = metrics_at_threshold(y_test, test_scores, th)
        met.update({"model": "Autoencoder", "feature_set": feature_set, "fit_seconds": fit_s, "inference_seconds": infer_s})
        results.append(met)
        pred_ae = test[["transaction_id", "user_id", "timestamp", "is_anomaly", "anomaly_type", "anomaly_intensity", "anomaly_breadth", "changed_dimensions"]].copy()
        pred_ae["score_ae"] = test_scores
        pred_ae["pred_ae"] = (test_scores >= th).astype(int)
        if feature_set == "raw+individualizadas":
            ii, dd = stratified_metrics(pred_ae, "ae")
            ii["feature_set"] = feature_set
            dd["feature_set"] = feature_set
            all_intensity.append(ii)
            all_dimensions.append(dd)
            prediction_frames.append(pred_ae)

    # LOF is kept as an auxiliary baseline for the partial-exam demo, not as a primary thesis model.
    cols = HYBRID_FEATURES
    train = post[post["phase"] == "train"]
    val = post[post["phase"] == "validation"]
    test = post[post["phase"] == "test"]
    X_train = train[cols].to_numpy(float)
    X_val = val[cols].to_numpy(float)
    X_test = test[cols].to_numpy(float)
    y_val = val["is_anomaly"].to_numpy(int)
    y_test = test["is_anomaly"].to_numpy(int)
    start = time.perf_counter()
    lof, lof_scaler = fit_lof(X_train)
    fit_s = time.perf_counter() - start
    val_scores = lof_scores(lof, lof_scaler, X_val)
    test_start = time.perf_counter()
    test_scores = lof_scores(lof, lof_scaler, X_test)
    infer_s = time.perf_counter() - test_start
    th, _ = choose_threshold(y_val, val_scores)
    met = metrics_at_threshold(y_test, test_scores, th)
    met.update({"model": "LOF (baseline auxiliar)", "feature_set": "raw+individualizadas", "fit_seconds": fit_s, "inference_seconds": infer_s})
    results.append(met)
    pred_lof = test[["transaction_id", "user_id", "timestamp", "is_anomaly", "anomaly_type", "anomaly_intensity", "anomaly_breadth", "changed_dimensions"]].copy()
    pred_lof["score_lof"] = test_scores
    pred_lof["pred_lof"] = (test_scores >= th).astype(int)
    ii, dd = stratified_metrics(pred_lof, "lof")
    ii["feature_set"] = "raw+individualizadas"
    dd["feature_set"] = "raw+individualizadas"
    all_intensity.append(ii)
    all_dimensions.append(dd)
    prediction_frames.append(pred_lof)

    metrics = pd.DataFrame(results)
    intensity = pd.concat(all_intensity, ignore_index=True)
    dimensions = pd.concat(all_dimensions, ignore_index=True)

    # Merge hybrid predictions for an alert panel.
    merged = prediction_frames[0]
    for p in prediction_frames[1:]:
        score_cols = [c for c in p.columns if c.startswith("score_") or c.startswith("pred_")]
        merged = merged.merge(p[["transaction_id"] + score_cols], on="transaction_id", how="left")
    principal = merged.sort_values("score_if", ascending=False)
    alerts = principal[["transaction_id", "user_id", "timestamp", "score_if", "pred_if", "is_anomaly", "anomaly_intensity", "changed_dimensions"]].copy()

    transactions.to_csv(output_dir / "transactions_demo.csv", index=False)
    profiles.to_csv(output_dir / "user_profiles_demo.csv", index=False)
    features.to_csv(output_dir / "features_demo.csv", index=False)
    metrics.to_csv(output_dir / "metrics_global.csv", index=False)
    intensity.to_csv(output_dir / "metrics_by_intensity.csv", index=False)
    dimensions.to_csv(output_dir / "metrics_by_dimension.csv", index=False)
    merged.to_csv(output_dir / "predictions_test.csv", index=False)
    alerts.head(200).to_csv(output_dir / "top_alerts.csv", index=False)

    summary = {
        "seed": config.seed,
        "n_users": config.n_users,
        "n_transactions": int(len(transactions)),
        "warmup_transactions": int((transactions["phase"] == "warmup").sum()),
        "train_transactions": int((transactions["phase"] == "train").sum()),
        "validation_transactions": int((transactions["phase"] == "validation").sum()),
        "test_transactions": int((transactions["phase"] == "test").sum()),
        "train_anomalies": int(transactions.loc[transactions["phase"] == "train", "is_anomaly"].sum()),
        "validation_anomalies": int(transactions.loc[transactions["phase"] == "validation", "is_anomaly"].sum()),
        "test_anomalies": int(transactions.loc[transactions["phase"] == "test", "is_anomaly"].sum()),
        "anomaly_rate_validation": float(transactions.loc[transactions["phase"] == "validation", "is_anomaly"].mean()),
        "anomaly_rate_test": float(transactions.loc[transactions["phase"] == "test", "is_anomaly"].mean()),
        "raw_features": len(RAW_FEATURES),
        "hybrid_features": len(HYBRID_FEATURES),
        "baseline_policy": "Frozen after legitimate warm-up; no ground-truth-based updates during evaluation.",
        "training_policy": "Clean normal-only train by design; anomalies are introduced only in validation and test.",
        "simulation_seconds": t_sim,
        "temporal_assertion": "train_max < validation_min and validation_max < test_min",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = build_report(output_dir, summary, metrics, intensity, dimensions, alerts)

    print("\n=== T12 DEMO COMPLETADA ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nMétricas de test:")
    print(metrics[["model", "feature_set", "precision", "recall", "f1", "pr_auc", "fp", "fn"]].to_string(index=False))
    print(f"\nReporte: {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo reproducible T12 - anomalías conductuales en billeteras digitales")
    p.add_argument("--quick", action="store_true", help="Usa 120 usuarios (12k transacciones) para una demo rápida.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--users", type=int, default=300, help="Usuarios para la corrida base S8 (default: 300).")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        cfg = SimulationConfig(n_users=120, seed=args.seed)
    else:
        cfg = SimulationConfig(n_users=args.users, seed=args.seed)
    run(cfg, args.output)
