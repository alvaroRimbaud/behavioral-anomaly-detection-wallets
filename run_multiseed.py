from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from run_demo import run
from src.simulator import SimulationConfig


def aggregate(root: Path, seeds: list[int]) -> None:
    metrics_all = []
    budgets_all = []
    bootstrap_all = []
    selected_all = []
    for seed in seeds:
        out = root / f"seed_{seed}"
        m = pd.read_csv(out / "metrics_global.csv").assign(seed=seed)
        b = pd.read_csv(out / "precision_at_k.csv").assign(seed=seed)
        bp = pd.read_csv(out / "paired_bootstrap_pr_auc.csv").assign(seed=seed)
        sc = pd.read_csv(out / "selected_hyperparameters.csv").assign(seed=seed)
        metrics_all.append(m)
        budgets_all.append(b)
        bootstrap_all.append(bp)
        selected_all.append(sc)

    metrics = pd.concat(metrics_all, ignore_index=True)
    budgets = pd.concat(budgets_all, ignore_index=True)
    boot = pd.concat(bootstrap_all, ignore_index=True)
    selected = pd.concat(selected_all, ignore_index=True)

    metrics.to_csv(root / "metrics_all_seeds.csv", index=False)
    budgets.to_csv(root / "precision_at_k_all_seeds.csv", index=False)
    boot.to_csv(root / "bootstrap_all_seeds.csv", index=False)
    selected.to_csv(root / "selected_hyperparameters_all_seeds.csv", index=False)

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

    bootstrap_summary = boot.groupby("comparison", as_index=False).agg(
        delta_pr_auc_mean=("delta_pr_auc", "mean"),
        delta_pr_auc_std=("delta_pr_auc", "std"),
        ci95_low_mean=("ci95_low", "mean"),
        ci95_high_mean=("ci95_high", "mean"),
        n_seeds=("seed", "nunique"),
    )
    bootstrap_summary.to_csv(root / "bootstrap_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T12 - ejecución y consolidación multi-semilla")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--users", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("outputs_final_5seeds"))
    p.add_argument("--tuning", choices=["off", "quick", "full"], default="full")
    p.add_argument("--bootstrap", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        cfg = SimulationConfig(n_users=args.users, seed=seed)
        run(
            cfg,
            args.output / f"seed_{seed}",
            tuning=args.tuning,
            bootstrap_iterations=args.bootstrap,
        )
    aggregate(args.output, args.seeds)
    print(f"\nConsolidado multi-semilla: {args.output}")
