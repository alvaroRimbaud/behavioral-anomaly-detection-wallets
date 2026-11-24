"""Independent-generator challenge: evaluate on ORIGINAL MoMTSim/PaySim labels.

Nothing here invokes the T12 simulator or injects new anomalies. This is an
external *synthetic-generator* challenge, NOT validation on real wallet data.
Only fields present in BOTH public datasets are used, so the hybrid set has
10 (not 12) features: 6 raw/contextual + 4 individual/history features.

No ground-truth label is used for warm-up eligibility, baseline fitting,
model training, or feature engineering. Original external labels are used only
for hyperparameter / threshold selection on validation and reporting on test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from run_demo import _select_if, _select_ae, _select_lof
from src.models import set_seed, iforest_scores, lof_scores
from src.evaluation import (
    metrics_at_threshold, choose_threshold, precision_at_alert_budgets,
    paired_cluster_bootstrap_pr_auc,
)

RAW = ["amount_scaled", "hour_sin", "hour_cos", "time_since_last_tx",
       "tx_count_last_1h", "tx_count_last_24h"]
INDIVIDUAL = ["amount_deviation_user", "hour_novelty_user",
              "frequency_deviation_user", "recipient_novelty_user"]
HYBRID = RAW + INDIVIDUAL
SCHEMAS = {
    "momtsim": {"step": "step", "amount": "amount", "sender": "initiator",
                "recipient": "recipient", "label": "isFraud", "type": "transactionType"},
    "paysim": {"step": "step", "amount": "amount", "sender": "nameOrig",
               "recipient": "nameDest", "label": "isFraud", "type": "type"},
}


def load_native(path: Path, schema: str, max_rows: int = 0,
                user_fraction: float = 1.0) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f"External CSV not found: {path}. Download the ORIGINAL dataset first; no fallback to T12 data.")
    cols = SCHEMAS[schema]
    if not (0 < user_fraction <= 1):
        raise ValueError("user_fraction must be within (0,1]")
    # Chunked hash; the file hash is vital for a reproducible external challenge.
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            sha.update(data)
    # Predefined label-blind sampling of SENDER IDs. Unlike sampling individual
    # rows, this preserves each retained person's temporal history across the
    # entire original simulation. Deterministic SHA256 makes this portable.
    # The fraction is fixed BEFORE seeing original labels or test scores.
    selection = {}
    selected = []
    for chunk in pd.read_csv(path, usecols=list(cols.values()), nrows=max_rows or None,
                             chunksize=150_000):
        if user_fraction < 1:
            sender_col = cols["sender"]
            new_ids = chunk[sender_col].astype(str).unique()
            for uid in new_ids:
                if uid not in selection:
                    sample_int = int.from_bytes(hashlib.sha256(("T12_O2_v1|"+uid).encode()).digest()[:8], "big")
                    selection[uid] = sample_int / (2**64) < user_fraction
            chunk = chunk[chunk[sender_col].astype(str).map(selection)].copy()
        if not chunk.empty:
            selected.append(chunk)
    if not selected:
        raise ValueError("No external users selected at fixed user_fraction; change only before inspecting results.")
    frame = pd.concat(selected, ignore_index=True)
    frame = frame.rename(columns={native: canonical for canonical, native in cols.items()})
    # Require complete original ground truth, instead of selecting users/rows
    # conditionally on label presence (which could bias the baseline).
    if frame["label"].isna().any():
        raise ValueError("Missing original fraud labels: cannot evaluate without label-based row selection.")
    frame = frame.dropna(subset=["step", "amount", "sender", "recipient"]).copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["step", "amount"])
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("External original label column must be binary {0,1}.")
    frame["label"] = frame["label"].astype(int)
    if len(frame) < 200:
        raise ValueError("Not enough external rows: at least 200 needed to test the pipeline.")
    # Sorting must not consult the fraud label. Original row number breaks same-step ties.
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["step", "_row_order"], kind="stable").reset_index(drop=True)
    return frame, sha.hexdigest()


def assign_temporal_phases(frame: pd.DataFrame) -> pd.DataFrame:
    unique_steps = np.sort(frame["step"].unique())
    if len(unique_steps) < 20:
        raise ValueError("At least 20 distinct time steps required for nonoverlapping temporal phases.")
    # Hold out complete hourly steps, avoiding same-timestamp train/test crossover.
    cut = np.maximum(1, np.round(np.array([.20, .70, .85]) * len(unique_steps)).astype(int))
    if len(set(cut)) != 3 or cut[-1] >= len(unique_steps):
        raise ValueError("Too few distinct time steps after temporal partition.")
    steps = frame["step"].to_numpy(float)
    phase = np.select(
        [steps < unique_steps[cut[0]], steps < unique_steps[cut[1]],
         steps < unique_steps[cut[2]]],
        ["warmup", "train", "validation"], default="test"
    )
    out = frame.copy()
    out["phase"] = phase
    return out


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return max(float(np.median(np.abs(values - median))), 1e-4)


def _novelty(c: Counter, key: str) -> float:
    return 1.0 if key not in c else float(1.0 / (1.0 + c[key]))


def build_external_features(frame: pd.DataFrame, min_warmup: int = 5) -> tuple[pd.DataFrame, dict]:
    if min_warmup < 1:
        raise ValueError("min_warmup must be at least 1")
    warm = frame[frame["phase"] == "warmup"]
    counts = warm.groupby("sender").size()
    eligible = counts[counts >= min_warmup].index
    d = frame[frame["sender"].isin(eligible)].copy()
    # No labels were used to determine eligibility or build profiles.
    profiles = {}
    for uid, g in d.loc[d["phase"] == "warmup"].groupby("sender", sort=False):
        g = g.sort_values(["step", "_row_order"])
        amounts = g["amount"].to_numpy(float)
        steps = g["step"].to_numpy(float)
        hours = (steps.astype(int) % 24)
        prior = np.searchsorted(steps, steps, side="left")
        gaps = steps[prior > 0] - steps[prior[prior > 0] - 1]
        profiles[uid] = {
            "amount_median": float(np.median(amounts)),
            "amount_mad": _mad(amounts),
            "modal_hour": int(Counter(hours).most_common(1)[0][0]),
            "gap_median": max(float(np.median(gaps)) if len(gaps) else 1.0, 1e-2),
            "recipient_counts": Counter(g["recipient"].astype(str)),
        }
    rows = []
    # Sequential scan per sender: past history is visible, future is not.
    for uid, g in d.groupby("sender", sort=False):
        p = profiles[uid]
        ordered = g.sort_values(["step", "_row_order"])
        steps = ordered["step"].to_numpy(float)
        # Half-open windows [t-w, t) exclude every simultaneous transaction.
        ends = np.searchsorted(steps, steps, side="left")
        counts_1h = ends - np.searchsorted(steps, steps - 1)
        counts_24h = ends - np.searchsorted(steps, steps - 24)
        for pos, (idx, row) in enumerate(ordered.iterrows()):
            step = steps[pos]
            end = ends[pos]
            gap = step - steps[end - 1] if end else p["gap_median"]
            hour = int(step) % 24
            diff = min(abs(hour - p["modal_hour"]), 24 - abs(hour - p["modal_hour"]))
            amount_dev = min(abs(float(row["amount"]) - p["amount_median"]) /
                             (8.0 * p["amount_mad"]), 1.0)
            gap_ratio = max(gap, 1e-3) / p["gap_median"]
            frequency_dev = min(abs(float(np.log(gap_ratio))) / float(np.log(20)), 1.0)
            rows.append({
                "source_index": idx, "sender": uid, "phase": row["phase"],
                "step": step, "label": int(row["label"]), "amount_raw": float(row["amount"]),
                "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
                "time_since_last_tx": float(np.log1p(gap)),
                "tx_count_last_1h": float(counts_1h[pos]),
                "tx_count_last_24h": float(counts_24h[pos]),
                "amount_deviation_user": float(amount_dev),
                "hour_novelty_user": float(diff / 12),
                "frequency_deviation_user": float(frequency_dev),
                "recipient_novelty_user": _novelty(p["recipient_counts"], str(row["recipient"])),
            })
    feat = pd.DataFrame(rows).sort_values("source_index").reset_index(drop=True)
    if feat.empty:
        raise ValueError("No users have enough early history. Full original MoMTSim is recommended.")
    scaler = StandardScaler().fit(feat.loc[feat["phase"] == "train", ["amount_raw"]])
    feat["amount_scaled"] = scaler.transform(feat[["amount_raw"]]).ravel()
    summary = {
        "original_rows": len(frame), "retained_rows": len(feat),
        "n_users_with_warmup": len(profiles), "n_user_coverage": len(profiles) / frame["sender"].nunique(),
        "fraud_in_train_audit_only": int(feat.loc[feat["phase"] == "train", "label"].sum()),
        "fraud_in_validation": int(feat.loc[feat["phase"] == "validation", "label"].sum()),
        "fraud_in_test": int(feat.loc[feat["phase"] == "test", "label"].sum()),
        "n_features_raw": len(RAW), "n_features_hybrid": len(HYBRID),
        "missing_from_original_t12": ["device_novelty_user", "location_novelty_user"],
    }
    return feat, summary


def external_holdout(train: pd.DataFrame):
    steps = np.sort(train["step"].unique())
    if len(steps) < 2:
        raise ValueError("AE holdout requires two distinct training steps")
    cut = min(len(steps) - 1, max(1, round(0.8 * len(steps))))
    earlier = train["step"] < steps[cut]
    return train.loc[earlier].copy(), train.loc[~earlier].copy()


def run_external(data: Path, dataset: str, output: Path, seed: int = 42,
                 min_warmup: int = 5, tuning: str = "quick", bootstrap: int = 300,
                 max_rows: int = 0, source_verified: bool = False,
                 user_fraction: float = 0.05) -> dict:
    if tuning not in {"off", "quick", "full"}:
        raise ValueError("Invalid tuning profile")
    set_seed(seed)
    df, digest = load_native(data, dataset, max_rows, user_fraction)
    df = assign_temporal_phases(df)
    feat, data_summary = build_external_features(df, min_warmup)
    train = feat[feat["phase"] == "train"].copy()
    val = feat[feat["phase"] == "validation"].copy()
    test = feat[feat["phase"] == "test"].copy()
    if len(train) < 100 or len(val) < 30 or len(test) < 30:
        raise ValueError("Insufficient retained sample for train/validation/test.")
    if val["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise ValueError("External validation/test lacks positive or negative original labels. Do not inject replacements.")
    fit, es = external_holdout(train)
    y_val = val["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)
    metrics, budgets, searches, scores_test = [], [], [], {}
    for label, cols in {"raw": RAW, "raw+individualizadas_observables": HYBRID}.items():
        x_tr, x_fit, x_es = train[cols].to_numpy(float), fit[cols].to_numpy(float), es[cols].to_numpy(float)
        x_v, x_te = val[cols].to_numpy(float), test[cols].to_numpy(float)
        for name in ("Isolation Forest", "Autoencoder", "LOF baseline"):
            if name == "Isolation Forest":
                best, log = _select_if(x_tr, x_v, y_val, seed, tuning)
                val_score, tes_score = best[3], iforest_scores(best[1], x_te)
            elif name == "Autoencoder":
                best, log = _select_ae(x_fit, x_es, x_v, y_val, seed, tuning)
                val_score, tes_score = best[3], best[1].score_samples(x_te)
            else:
                best, log = _select_lof(x_tr, x_v, y_val, tuning)
                val_score, tes_score = best[4], lof_scores(best[1], best[2], x_te)
            t, _ = choose_threshold(y_val, val_score)
            m = metrics_at_threshold(y_test, tes_score, t)
            m.update({"model": name, "feature_set": label,
                      "n_test": len(test), "n_test_fraud": int(y_test.sum()),
                      "prevalence_test": float(np.mean(y_test))})
            metrics.append(m)
            searches.append(log.assign(feature_set=label))
            budgets.append(precision_at_alert_budgets(y_test, tes_score).assign(model=name,feature_set=label))
            scores_test[(name,label)] = tes_score
    boot = []
    for i,name in enumerate(("Isolation Forest", "Autoencoder", "LOF baseline")):
        r = paired_cluster_bootstrap_pr_auc(y_test, scores_test[(name,"raw+individualizadas_observables")],
                        scores_test[(name,"raw")], test["sender"].to_numpy(), n_bootstrap=bootstrap, seed=seed*100+i)
        r.update({"comparison": f"{name}: externally_observable_hybrid_vs_raw"})
        boot.append(r)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output / "external_metrics.csv",index=False)
    pd.concat(budgets,ignore_index=True).to_csv(output / "external_precision_at_k.csv",index=False)
    pd.concat(searches,ignore_index=True).to_csv(output / "external_hyperparameter_search.csv",index=False)
    pd.DataFrame(boot).to_csv(output / "external_cluster_bootstrap.csv",index=False)
    status={"status":("EXTERNAL_VALIDATION_EXECUTED_SOURCE_DECLARED_USER_SAMPLE" if source_verified and not max_rows
                      else "SCHEMA_SMOKE_OR_SUBSAMPLE_NOT_EXTERNAL_VALIDATION"), "dataset":dataset,
            "source_file_sha256":digest,"source_path":str(data),
            "original_labels_used_for_test":True,"seed":seed,"tuning":tuning,
            "bootstrap_iterations":bootstrap,"min_warmup":min_warmup,
            "max_rows":max_rows,
            "label_blind_sender_sample_fraction": user_fraction,
            "sender_sampling_method":"SHA256(T12_O2_v1|sender_id) < fraction, deterministic across all time steps",
            "early_stopping_policy":"unlabelled internal train holdout; possibly contaminated",
            "training_policy":"unlabelled external train may contain original fraud; no label-based filtering",
            "original_data_source_attested_by_executor":bool(source_verified),
            "external_validation_accepted":bool(source_verified and not max_rows),
            "not_real_peruvian_wallet_data":True, **data_summary}
    (output/"external_run_summary.json").write_text(json.dumps(status,indent=2,ensure_ascii=False))
    print(pd.DataFrame(metrics)[["model","feature_set","pr_auc","precision","recall","f1"]].to_string(index=False))
    return status


if __name__ == "__main__":
    p=argparse.ArgumentParser(description="T12 external generator challenge; original labels only")
    p.add_argument("--data",type=Path,required=True,help="Original MoMTSim or PaySim CSV (not a T12-generated file)")
    p.add_argument("--dataset",choices=list(SCHEMAS),default="momtsim")
    p.add_argument("--output",type=Path,default=Path("outputs_external_momtsim"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--min-warmup",type=int,default=5)
    p.add_argument("--user-fraction",type=float,default=0.05,
         help="Predefined label-blind fraction of original sender IDs; default 5%% for feasible local runs")
    p.add_argument("--max-rows",type=int,default=0,help="Exploratory prefix only; 0 uses complete CSV")
    p.add_argument("--tuning",choices=["off","quick","full"],default="quick")
    p.add_argument("--bootstrap",type=int,default=300)
    p.add_argument("--confirm-original-source",action="store_true",
         help="Manually attest that CSV is an original dataset release, not a T12 file or rebalanced sample")
    args=p.parse_args()
    print(json.dumps(run_external(args.data,args.dataset,args.output,args.seed,
                      args.min_warmup,args.tuning,args.bootstrap,args.max_rows,
                      args.confirm_original_source,args.user_fraction),indent=2))
