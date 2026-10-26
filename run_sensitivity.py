from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from run_demo import run
from src.simulator import SimulationConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T12 - sensibilidad de supuestos del simulador")
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs_sensitivity"))
    p.add_argument("--tuning", choices=["off", "quick"], default="off")
    p.add_argument("--bootstrap", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # The baseline length is an explicit design assumption. Keep 100 tx/user by
    # compensating train length so validation/test remain unchanged.
    scenarios = [
        ("warmup20", 20, 59, 0.02),
        ("warmup30", 30, 49, 0.02),
        ("warmup40", 40, 39, 0.02),
        ("prevalence01", 30, 49, 0.01),
        ("prevalence05", 30, 49, 0.05),
    ]
    rows = []
    for name, warmup, train, rate in scenarios:
        cfg = SimulationConfig(
            n_users=args.users,
            transactions_per_user=100,
            warmup_per_user=warmup,
            train_per_user=train,
            val_per_user=10,
            test_per_user=11,
            anomaly_rate_val=rate,
            anomaly_rate_test=rate,
            seed=args.seed,
        )
        out = args.output / name
        run(cfg, out, tuning=args.tuning, bootstrap_iterations=args.bootstrap)
        m = pd.read_csv(out / "metrics_global.csv")
        for _, r in m.iterrows():
            rows.append({
                "scenario": name,
                "warmup": warmup,
                "train": train,
                "anomaly_rate": rate,
                "model": r["model"],
                "feature_set": r["feature_set"],
                "pr_auc": r["pr_auc"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
            })
    pd.DataFrame(rows).to_csv(args.output / "sensitivity_summary.csv", index=False)
    print(f"Sensibilidad guardada en {args.output / 'sensitivity_summary.csv'}")
