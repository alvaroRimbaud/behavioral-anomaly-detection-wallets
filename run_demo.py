from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.simulator import SimulationConfig, simulate_transactions, simulator_assumptions
from src.features import build_features, RAW_FEATURES, HYBRID_FEATURES
from src.models import (
    set_seed,
    fit_iforest,
    iforest_scores,
    fit_lof,
    lof_scores,
    AutoencoderDetector,
    IFConfig,
    AEConfig,
    LOFConfig,
    if_grid,
    ae_grid,
    lof_grid,
)
from src.evaluation import (
    choose_threshold,
    metrics_at_threshold,
    stratified_metrics,
    recall_vs_intensity_curve,
    threshold_candidates,
    precision_at_alert_budgets,
    paired_bootstrap_pr_auc,
    estimate_d50_from_curve,
)
from src.report import build_report


def _temporal_internal_holdout(train: pd.DataFrame, fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normal-only temporal holdout for AE early stopping; labels are never consulted."""
    if not (0.05 <= fraction <= 0.40):
        raise ValueError("early-stop fraction must be between 0.05 and 0.40")
    d = train.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp")
    cut = max(1, int(round(len(d) * (1.0 - fraction))))
    cut = min(cut, len(d) - 1)
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def _jsonable_config(cfg: Any) -> str:
    data = cfg.to_dict()
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def _select_if(X_train, X_val, y_val, seed: int, tuning: str):
    rows = []
    best = None
    for i, cfg in enumerate(if_grid(tuning)):
        start = time.perf_counter()
        model = fit_iforest(X_train, seed=seed, config=cfg)
        fit_s = time.perf_counter() - start
        scores = iforest_scores(model, X_val)
        pr = float(average_precision_score(y_val, scores))
        rows.append({"candidate": i, "model": "Isolation Forest", "config": _jsonable_config(cfg), "validation_pr_auc": pr, "fit_seconds": fit_s})
        if best is None or pr > best[0]:
            best = (pr, model, cfg, scores, fit_s)
    return best, pd.DataFrame(rows)


def _select_ae(X_fit, X_es, X_val, y_val, seed: int, tuning: str):
    rows = []
    best = None
    for i, cfg in enumerate(ae_grid(tuning)):
        start = time.perf_counter()
        model = AutoencoderDetector(seed=seed, config=cfg)
        model.fit(X_fit, X_es)
        fit_s = time.perf_counter() - start
        scores = model.score_samples(X_val)
        pr = float(average_precision_score(y_val, scores))
        rows.append({
            "candidate": i,
            "model": "Autoencoder",
            "config": _jsonable_config(cfg),
            "validation_pr_auc": pr,
            "fit_seconds": fit_s,
            "best_epoch": model.best_epoch,
            "early_stop_loss": model.best_early_stop_loss,
        })
        if best is None or pr > best[0]:
            best = (pr, model, cfg, scores, fit_s)
    return best, pd.DataFrame(rows)


def _select_lof(X_train, X_val, y_val, tuning: str):
    rows = []
    best = None
    for i, cfg in enumerate(lof_grid(tuning)):
        start = time.perf_counter()
        model, scaler = fit_lof(X_train, config=cfg)
        fit_s = time.perf_counter() - start
        scores = lof_scores(model, scaler, X_val)
        pr = float(average_precision_score(y_val, scores))
        rows.append({"candidate": i, "model": "LOF baseline", "config": _jsonable_config(cfg), "validation_pr_auc": pr, "fit_seconds": fit_s})
        if best is None or pr > best[0]:
            best = (pr, model, scaler, cfg, scores, fit_s)
    return best, pd.DataFrame(rows)


def _plot_recall_vs_d(curve: pd.DataFrame, output_dir: Path) -> Path | None:
    if curve.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in curve.groupby("model"):
        sub = sub.sort_values("D_mid")
        ax.plot(sub["D_mid"], sub["recall"], marker="o", label=model)
    ax.set_xlabel("Intensidad de anomalía D")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.0)
    ax.set_title("Recall frente a intensidad continua D")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "recall_vs_D.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(
    config: SimulationConfig,
    output_dir: Path,
    tuning: str = "quick",
    bootstrap_iterations: int = 1000,
    intensity_bin_width: float = 0.10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    t0 = time.perf_counter()
    transactions, profiles, baselines = simulate_transactions(config)
    t_sim = time.perf_counter() - t0

    features, _ = build_features(transactions, baselines)
    post = features[features["phase"].isin(["train", "validation", "test"])].copy()
    if int(post.loc[post["phase"] == "train", "is_anomaly"].sum()) != 0:
        raise RuntimeError("Design violation: training contains anomalies.")

    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    train_max = tx.loc[tx["phase"] == "train", "timestamp"].max()
    val_min = tx.loc[tx["phase"] == "validation", "timestamp"].min()
    val_max = tx.loc[tx["phase"] == "validation", "timestamp"].max()
    test_min = tx.loc[tx["phase"] == "test", "timestamp"].min()
    assert train_max < val_min and val_max < test_min, "Temporal split overlap detected."

    train = post[post["phase"] == "train"].copy()
    val = post[post["phase"] == "validation"].copy()
    test = post[post["phase"] == "test"].copy()
    train_fit, train_es = _temporal_internal_holdout(train, fraction=0.20)
    y_val = val["is_anomaly"].to_numpy(int)
    y_test = test["is_anomaly"].to_numpy(int)

    results = []
    all_intensity = []
    all_dimensions = []
    all_curve = []
    hyper_search = []
    threshold_search = []
    budgets = []
    selected_configs = []
    score_store: dict[tuple[str, str], np.ndarray] = {}
    pred_store: dict[tuple[str, str], np.ndarray] = {}

    feature_sets = {"raw": RAW_FEATURES, "raw+individualizadas": HYBRID_FEATURES}

    for feature_set, cols in feature_sets.items():
        X_train = train[cols].to_numpy(float)
        X_fit = train_fit[cols].to_numpy(float)
        X_es = train_es[cols].to_numpy(float)
        X_val = val[cols].to_numpy(float)
        X_test = test[cols].to_numpy(float)

        # Isolation Forest: hyperparameters selected on validation PR-AUC.
        best_if, search_if = _select_if(X_train, X_val, y_val, config.seed, tuning)
        hyper_search.append(search_if.assign(feature_set=feature_set))
        _, ifm, if_cfg, val_scores, fit_s = best_if
        test_start = time.perf_counter()
        test_scores = iforest_scores(ifm, X_test)
        infer_s = time.perf_counter() - test_start
        th, val_f1 = choose_threshold(y_val, val_scores)
        th_table = threshold_candidates(y_val, val_scores).assign(model="Isolation Forest", feature_set=feature_set)
        threshold_search.append(th_table)
        met = metrics_at_threshold(y_test, test_scores, th)
        met.update({
            "model": "Isolation Forest",
            "feature_set": feature_set,
            "fit_seconds": fit_s,
            "inference_seconds": infer_s,
            "validation_f1_at_selected_threshold": val_f1,
            "selected_config": _jsonable_config(if_cfg),
        })
        results.append(met)
        selected_configs.append({"model": "Isolation Forest", "feature_set": feature_set, "config": _jsonable_config(if_cfg)})
        score_store[("Isolation Forest", feature_set)] = test_scores
        pred_store[("Isolation Forest", feature_set)] = (test_scores >= th).astype(int)
        b = precision_at_alert_budgets(y_test, test_scores).assign(model="Isolation Forest", feature_set=feature_set)
        budgets.append(b)

        # Autoencoder: normal-only internal temporal holdout controls early stopping;
        # external validation selects hyperparameters and threshold.
        best_ae, search_ae = _select_ae(X_fit, X_es, X_val, y_val, config.seed, tuning)
        hyper_search.append(search_ae.assign(feature_set=feature_set))
        _, ae, ae_cfg, val_scores, fit_s = best_ae
        test_start = time.perf_counter()
        test_scores = ae.score_samples(X_test)
        infer_s = time.perf_counter() - test_start
        th, val_f1 = choose_threshold(y_val, val_scores)
        th_table = threshold_candidates(y_val, val_scores).assign(model="Autoencoder", feature_set=feature_set)
        threshold_search.append(th_table)
        met = metrics_at_threshold(y_test, test_scores, th)
        met.update({
            "model": "Autoencoder",
            "feature_set": feature_set,
            "fit_seconds": fit_s,
            "inference_seconds": infer_s,
            "validation_f1_at_selected_threshold": val_f1,
            "selected_config": _jsonable_config(ae_cfg),
            "ae_best_epoch": ae.best_epoch,
            "ae_early_stop_loss": ae.best_early_stop_loss,
        })
        results.append(met)
        selected_configs.append({"model": "Autoencoder", "feature_set": feature_set, "config": _jsonable_config(ae_cfg), "best_epoch": ae.best_epoch})
        score_store[("Autoencoder", feature_set)] = test_scores
        pred_store[("Autoencoder", feature_set)] = (test_scores >= th).astype(int)
        b = precision_at_alert_budgets(y_test, test_scores).assign(model="Autoencoder", feature_set=feature_set)
        budgets.append(b)

        # Observation 1: LOF is now a formal baseline and is evaluated under the
        # same feature sets, split and validation protocol.
        best_lof, search_lof = _select_lof(X_train, X_val, y_val, tuning)
        hyper_search.append(search_lof.assign(feature_set=feature_set))
        _, lof, lof_scaler, lof_cfg, val_scores, fit_s = best_lof
        test_start = time.perf_counter()
        test_scores = lof_scores(lof, lof_scaler, X_test)
        infer_s = time.perf_counter() - test_start
        th, val_f1 = choose_threshold(y_val, val_scores)
        th_table = threshold_candidates(y_val, val_scores).assign(model="LOF baseline", feature_set=feature_set)
        threshold_search.append(th_table)
        met = metrics_at_threshold(y_test, test_scores, th)
        met.update({
            "model": "LOF baseline",
            "feature_set": feature_set,
            "fit_seconds": fit_s,
            "inference_seconds": infer_s,
            "validation_f1_at_selected_threshold": val_f1,
            "selected_config": _jsonable_config(lof_cfg),
        })
        results.append(met)
        selected_configs.append({"model": "LOF baseline", "feature_set": feature_set, "config": _jsonable_config(lof_cfg)})
        score_store[("LOF baseline", feature_set)] = test_scores
        pred_store[("LOF baseline", feature_set)] = (test_scores >= th).astype(int)
        b = precision_at_alert_budgets(y_test, test_scores).assign(model="LOF baseline", feature_set=feature_set)
        budgets.append(b)

    # Stratified analysis and continuous Recall-vs-D use the hybrid representation.
    base_cols = [
        "transaction_id", "user_id", "timestamp", "is_anomaly", "anomaly_type",
        "target_anomaly_intensity", "anomaly_intensity", "anomaly_breadth", "changed_dimensions",
    ]
    model_keys = {"if": "Isolation Forest", "ae": "Autoencoder", "lof": "LOF baseline"}
    hybrid_predictions = test[base_cols].copy()
    for short, model_name in model_keys.items():
        scores = score_store[(model_name, "raw+individualizadas")]
        preds = pred_store[(model_name, "raw+individualizadas")]
        hybrid_predictions[f"score_{short}"] = scores
        hybrid_predictions[f"pred_{short}"] = preds
        temp = test[base_cols].copy()
        temp[f"score_{short}"] = scores
        temp[f"pred_{short}"] = preds
        ii, dd = stratified_metrics(temp, short)
        ii["feature_set"] = "raw+individualizadas"
        dd["feature_set"] = "raw+individualizadas"
        all_intensity.append(ii)
        all_dimensions.append(dd)
        curve = recall_vs_intensity_curve(temp, short, bin_width=intensity_bin_width)
        curve["feature_set"] = "raw+individualizadas"
        all_curve.append(curve)

    metrics = pd.DataFrame(results)
    intensity = pd.concat(all_intensity, ignore_index=True)
    dimensions = pd.concat(all_dimensions, ignore_index=True)
    curve = pd.concat(all_curve, ignore_index=True)
    d50_rows = []
    for short, model_name in model_keys.items():
        sub = curve[curve["model"] == short].copy()
        d50 = estimate_d50_from_curve(sub)
        d50_rows.append({
            "model": model_name,
            "feature_set": "raw+individualizadas",
            **d50,
        })
    d50_df = pd.DataFrame(d50_rows)
    hyper_search_df = pd.concat(hyper_search, ignore_index=True)
    threshold_search_df = pd.concat(threshold_search, ignore_index=True)
    budgets_df = pd.concat(budgets, ignore_index=True)
    selected_configs_df = pd.DataFrame(selected_configs)

    # Paired bootstrap: formal model comparisons and raw-vs-hybrid ablation.
    comparisons = [
        ("AE_hybrid_vs_IF_hybrid", ("Autoencoder", "raw+individualizadas"), ("Isolation Forest", "raw+individualizadas")),
        ("AE_hybrid_vs_LOF_hybrid", ("Autoencoder", "raw+individualizadas"), ("LOF baseline", "raw+individualizadas")),
        ("IF_hybrid_vs_LOF_hybrid", ("Isolation Forest", "raw+individualizadas"), ("LOF baseline", "raw+individualizadas")),
        ("IF_hybrid_vs_raw", ("Isolation Forest", "raw+individualizadas"), ("Isolation Forest", "raw")),
        ("AE_hybrid_vs_raw", ("Autoencoder", "raw+individualizadas"), ("Autoencoder", "raw")),
        ("LOF_hybrid_vs_raw", ("LOF baseline", "raw+individualizadas"), ("LOF baseline", "raw")),
    ]
    bootstrap_rows = []
    test_user_ids = test["user_id"].to_numpy()
    for j, (label, a, b) in enumerate(comparisons):
        res = paired_bootstrap_pr_auc(
            y_test,
            score_store[a],
            score_store[b],
            cluster_ids=test_user_ids,
            n_bootstrap=bootstrap_iterations,
            seed=config.seed * 100 + j,
        )
        res.update({"comparison": label, "A": f"{a[0]} | {a[1]}", "B": f"{b[0]} | {b[1]}"})
        bootstrap_rows.append(res)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # Neutral auxiliary alert ranking: average percentile rank of the three
    # hybrid detector scores. It is not a third decision model.
    rank_cols = []
    for short in model_keys:
        rc = f"rank_{short}"
        hybrid_predictions[rc] = hybrid_predictions[f"score_{short}"].rank(pct=True)
        rank_cols.append(rc)
    hybrid_predictions["alert_rank_score"] = hybrid_predictions[rank_cols].mean(axis=1)
    alerts = hybrid_predictions.sort_values("alert_rank_score", ascending=False)[
        ["transaction_id", "user_id", "timestamp", "alert_rank_score", "score_if", "score_ae", "score_lof",
         "pred_if", "pred_ae", "pred_lof", "is_anomaly", "anomaly_intensity", "anomaly_breadth", "changed_dimensions"]
    ].copy()

    transactions.to_csv(output_dir / "transactions_demo.csv", index=False)
    profiles.to_csv(output_dir / "user_profiles_demo.csv", index=False)
    features.to_csv(output_dir / "features_demo.csv", index=False)
    metrics.to_csv(output_dir / "metrics_global.csv", index=False)
    intensity.to_csv(output_dir / "metrics_by_intensity.csv", index=False)
    dimensions.to_csv(output_dir / "metrics_by_dimension.csv", index=False)
    curve.to_csv(output_dir / "recall_vs_D.csv", index=False)
    d50_df.to_csv(output_dir / "d50_estimates.csv", index=False)
    hyper_search_df.to_csv(output_dir / "hyperparameter_search.csv", index=False)
    threshold_search_df.to_csv(output_dir / "threshold_candidates_validation.csv", index=False)
    budgets_df.to_csv(output_dir / "precision_at_k.csv", index=False)
    selected_configs_df.to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    bootstrap_df.to_csv(output_dir / "paired_bootstrap_pr_auc.csv", index=False)
    hybrid_predictions.to_csv(output_dir / "predictions_test.csv", index=False)
    alerts.head(200).to_csv(output_dir / "top_alerts.csv", index=False)
    simulator_assumptions(config).to_csv(output_dir / "simulator_assumptions.csv", index=False)
    _plot_recall_vs_d(curve, output_dir)

    summary = {
        "seed": config.seed,
        "n_users": config.n_users,
        "n_transactions": int(len(transactions)),
        "warmup_transactions": int((transactions["phase"] == "warmup").sum()),
        "train_transactions": int((transactions["phase"] == "train").sum()),
        "train_fit_transactions": int(len(train_fit)),
        "train_early_stop_transactions": int(len(train_es)),
        "validation_transactions": int((transactions["phase"] == "validation").sum()),
        "test_transactions": int((transactions["phase"] == "test").sum()),
        "train_anomalies": int(transactions.loc[transactions["phase"] == "train", "is_anomaly"].sum()),
        "validation_anomalies": int(transactions.loc[transactions["phase"] == "validation", "is_anomaly"].sum()),
        "test_anomalies": int(transactions.loc[transactions["phase"] == "test", "is_anomaly"].sum()),
        "anomaly_rate_validation": float(transactions.loc[transactions["phase"] == "validation", "is_anomaly"].mean()),
        "anomaly_rate_test": float(transactions.loc[transactions["phase"] == "test", "is_anomaly"].mean()),
        "raw_features": len(RAW_FEATURES),
        "hybrid_features": len(HYBRID_FEATURES),
        "baseline_policy": "Frozen after legitimate warm-up; no post-warm-up observation updates it.",
        "training_policy": "Clean normal-only train by design; anomalies only in validation/test.",
        "early_stopping_policy": "Normal-only temporal holdout carved from train; external validation is not used for AE stopping.",
        "threshold_policy": "F1-optimal threshold chosen on external validation as reference; top-k alert budgets are also reported.",
        "model_policy": "Isolation Forest and Autoencoder are thesis core models; LOF is a formal comparison baseline.",
        "tuning_profile": tuning,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_unit": "user",
        "d50_definition": "First fixed-D recall curve crossing of 0.50, linearly interpolated between adjacent bin midpoints.",
        "simulation_seconds": t_sim,
        "temporal_assertion": "train_max < validation_min and validation_max < test_min",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = build_report(
        output_dir,
        summary,
        metrics,
        intensity,
        dimensions,
        alerts,
        curve=curve,
        budgets=budgets_df,
        bootstrap=bootstrap_df,
        selected_configs=selected_configs_df,
    )

    print("\n=== T12 EXPERIMENTO COMPLETADO ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nMétricas de test:")
    print(metrics[["model", "feature_set", "precision", "recall", "f1", "pr_auc", "fp", "fn"]].to_string(index=False))
    print("\nBootstrap pareado (ΔPR-AUC = A - B):")
    print(bootstrap_df[["comparison", "delta_pr_auc", "ci95_low", "ci95_high"]].to_string(index=False))
    print(f"\nReporte: {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T12 - experimento reproducible de anomalías conductuales")
    p.add_argument("--quick", action="store_true", help="Usa 120 usuarios para una demostración rápida.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--users", type=int, default=300, help="Usuarios de la corrida (300 S8; 1000 experimento final).")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--tuning", choices=["off", "quick", "full"], default="quick", help="Perfil de búsqueda de hiperparámetros.")
    p.add_argument("--bootstrap", type=int, default=1000, help="Remuestreos de bootstrap pareado.")
    p.add_argument("--d-bin-width", type=float, default=0.10, help="Ancho de bin para curva Recall vs D.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        cfg = SimulationConfig(n_users=120, seed=args.seed)
    else:
        cfg = SimulationConfig(n_users=args.users, seed=args.seed)
    run(
        cfg,
        args.output,
        tuning=args.tuning,
        bootstrap_iterations=args.bootstrap,
        intensity_bin_width=args.d_bin_width,
    )
