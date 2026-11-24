from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from run_demo import run
from src.simulator import SimulationConfig


REQUIRED_SEED_FILES = [
    "run_summary.json", "metrics_global.csv", "precision_at_k.csv",
    "paired_bootstrap_pr_auc.csv", "selected_hyperparameters.csv",
    "d50_estimates.csv", "scores_test_all_models.csv", "hyperparameter_search.csv",
    "simulator_assumptions.csv", "reporte_demo.html",
]


def completed_seed(out: Path, *, seed: int, users: int, tuning: str, bootstrap: int) -> bool:
    """Guard --resume against incomplete outputs or a changed experimental plan."""
    if any(not (out / filename).is_file() or (out / filename).stat().st_size == 0
           for filename in REQUIRED_SEED_FILES):
        return False
    try:
        meta = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
        return (meta.get("seed") == seed and meta.get("n_users") == users
                and meta.get("tuning_profile") == tuning
                and meta.get("bootstrap_iterations") == bootstrap
                and meta.get("bootstrap_unit") == "user"
                and meta.get("train_anomalies") == 0
                and meta.get("raw_features") == 6 and meta.get("hybrid_features") == 12)
    except (ValueError, KeyError, OSError):
        return False


def aggregate(root: Path, seeds: list[int]) -> None:
    metrics_all = []
    budgets_all = []
    bootstrap_all = []
    selected_all = []
    d50_all = []
    for seed in seeds:
        out = root / f"seed_{seed}"
        m = pd.read_csv(out / "metrics_global.csv").assign(seed=seed)
        b = pd.read_csv(out / "precision_at_k.csv").assign(seed=seed)
        bp = pd.read_csv(out / "paired_bootstrap_pr_auc.csv").assign(seed=seed)
        sc = pd.read_csv(out / "selected_hyperparameters.csv").assign(seed=seed)
        d50 = pd.read_csv(out / "d50_estimates.csv").assign(seed=seed)
        metrics_all.append(m)
        budgets_all.append(b)
        bootstrap_all.append(bp)
        selected_all.append(sc)
        d50_all.append(d50)

    metrics = pd.concat(metrics_all, ignore_index=True)
    budgets = pd.concat(budgets_all, ignore_index=True)
    boot = pd.concat(bootstrap_all, ignore_index=True)
    selected = pd.concat(selected_all, ignore_index=True)
    d50 = pd.concat(d50_all, ignore_index=True)

    metrics.to_csv(root / "metrics_all_seeds.csv", index=False)
    budgets.to_csv(root / "precision_at_k_all_seeds.csv", index=False)
    boot.to_csv(root / "bootstrap_all_seeds.csv", index=False)
    selected.to_csv(root / "selected_hyperparameters_all_seeds.csv", index=False)
    d50.to_csv(root / "d50_all_seeds.csv", index=False)

    metric_summary = metrics.groupby(["model", "feature_set"], as_index=False).agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        threshold_mean=("threshold", "mean"),
        threshold_std=("threshold", "std"),
        threshold_min=("threshold", "min"),
        threshold_max=("threshold", "max"),
        n_seeds=("seed", "nunique"),
    )
    metric_summary.to_csv(root / "metrics_and_threshold_stability.csv", index=False)

    budget_summary = budgets.groupby(["model", "feature_set", "alert_budget"], as_index=False).agg(
        precision_at_k_mean=("precision_at_k", "mean"),
        precision_at_k_std=("precision_at_k", "std"),
        recall_at_k_mean=("recall_at_k", "mean"),
        recall_at_k_std=("recall_at_k", "std"),
        n_seeds=("seed", "nunique"),
    )
    budget_summary.to_csv(root / "precision_at_k_summary.csv", index=False)

    # WARNING: averaging within-seed CI endpoints is NOT a valid aggregate CI.
    # Aggregate point effects across seeds here; analyze_pi14.py performs the
    # seed-aware inferential analysis (and records small-n limitations).
    bootstrap_summary = boot.groupby("comparison", as_index=False).agg(
        delta_pr_auc_mean=("delta_pr_auc", "mean"),
        delta_pr_auc_std=("delta_pr_auc", "std"),
        n_seeds=("seed", "nunique"),
    )
    bootstrap_summary["note"] = "Descriptivo; no promediar límites IC por semilla"
    bootstrap_summary.to_csv(root / "bootstrap_summary.csv", index=False)

    d50_summary = d50.groupby(["model", "feature_set"], as_index=False).agg(
        d50_mean=("d50", "mean"),
        d50_std=("d50", "std"),
        d50_min=("d50", "min"),
        d50_max=("d50", "max"),
        max_recall_mean=("max_recall", "mean"),
        n_seeds=("seed", "nunique"),
    )
    d50_summary.to_csv(root / "d50_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T12 - ejecución y consolidación multi-semilla")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--users", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("outputs_final_5seeds"))
    p.add_argument("--tuning", choices=["off", "quick", "full"], default="full")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--resume", action="store_true", help="Reutiliza semillas completas ya generadas.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        cfg = SimulationConfig(n_users=args.users, seed=seed)
        out = args.output / f"seed_{seed}"
        if args.resume and completed_seed(out, seed=seed, users=args.users,
                                          tuning=args.tuning, bootstrap=args.bootstrap):
            print(f"Seed {seed}: evidencia COMPLETA y compatible; se reutiliza.", flush=True)
            continue
        if args.resume and out.exists():
            print(f"Seed {seed}: resultado incompleto o parámetros distintos; se vuelve a ejecutar.",flush=True)
            (out / "run_summary.json").unlink(missing_ok=True)
        run(
            cfg,
            args.output / f"seed_{seed}",
            tuning=args.tuning,
            bootstrap_iterations=args.bootstrap,
        )
    aggregate(args.output, args.seeds)
    print(f"\nConsolidado multi-semilla: {args.output}")
