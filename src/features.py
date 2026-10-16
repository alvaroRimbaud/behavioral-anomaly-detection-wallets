from __future__ import annotations

from collections import deque
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FEATURES = [
    "amount_scaled",
    "hour_sin",
    "hour_cos",
    "time_since_last_tx",
    "tx_count_last_1h",
    "tx_count_last_24h",
]
BEHAVIORAL_FEATURES = [
    "amount_deviation_user",
    "hour_novelty_user",
    "frequency_deviation_user",
    "recipient_novelty_user",
    "device_novelty_user",
    "location_novelty_user",
]
HYBRID_FEATURES = RAW_FEATURES + BEHAVIORAL_FEATURES


def _novelty(counts: dict, value: str | int) -> float:
    """Smoothed historical novelty in [0,1].

    Unseen values map to 1. Seen values decrease with historical support:
    count=1 -> 0.5, count=4 -> 0.2, count=9 -> 0.1.
    This preserves rarity while keeping a clear separation for truly new values.
    """
    key = value
    if key not in counts:
        return 1.0
    return float(1.0 / (1.0 + float(counts[key])))


def build_features(transactions: pd.DataFrame, baselines: Dict[str, dict]) -> Tuple[pd.DataFrame, StandardScaler]:
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    rows = []
    for uid, g in df.groupby("user_id", sort=False):
        b = baselines[uid]
        recent_1h: deque[pd.Timestamp] = deque()
        recent_24h: deque[pd.Timestamp] = deque()
        prev_ts = None
        for _, r in g.iterrows():
            ts = pd.Timestamp(r["timestamp"])
            while recent_1h and (ts - recent_1h[0]).total_seconds() > 3600:
                recent_1h.popleft()
            while recent_24h and (ts - recent_24h[0]).total_seconds() > 86400:
                recent_24h.popleft()

            gap_h = float((ts - prev_ts).total_seconds() / 3600.0) if prev_ts is not None else float(b["gap_median_hours"])
            gap_h = float(np.clip(gap_h, 0.0, 24 * 30))
            count_1h = len(recent_1h)
            count_24h = len(recent_24h)

            robust_z = abs(float(r["amount"]) - b["amount_median"]) / (b["amount_mad"] + 1e-9)
            amount_dev = float(np.clip(robust_z / 8.0, 0.0, 1.0))
            hour = int(ts.hour)
            modal_hour = int(max(b["hour_counts"], key=b["hour_counts"].get)) if b["hour_counts"] else 12
            hour_dist = abs(hour - modal_hour) % 24
            hour_dist = min(hour_dist, 24 - hour_dist)
            hour_novelty = float(np.clip(hour_dist / 12.0, 0.0, 1.0))
            # Frequency deviation combines inter-arrival surprise with recent daily activity.
            gap_ratio = max(gap_h, 1e-3) / (b["gap_median_hours"] + 1e-9)
            gap_dev = min(abs(np.log(gap_ratio)) / np.log(20.0), 1.0)
            count_ratio = abs(count_24h - b["daily_count_median"]) / (b["daily_count_median"] + 1.0)
            count_dev = min(count_ratio / 3.0, 1.0)
            freq_dev = float(max(gap_dev, count_dev))

            rows.append({
                "transaction_id": r["transaction_id"],
                "user_id": uid,
                "timestamp": ts,
                "phase": r["phase"],
                "amount_raw": float(r["amount"]),
                "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
                "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
                "time_since_last_tx": float(np.log1p(gap_h)),
                "tx_count_last_1h": float(count_1h),
                "tx_count_last_24h": float(count_24h),
                "amount_deviation_user": amount_dev,
                "hour_novelty_user": hour_novelty,
                "frequency_deviation_user": freq_dev,
                "recipient_novelty_user": _novelty(b["recipient_counts"], str(r["recipient_id"])),
                "device_novelty_user": _novelty(b["device_counts"], str(r["device_id"])),
                "location_novelty_user": _novelty(b["location_counts"], str(r["location_zone"])),
                "is_anomaly": int(r["is_anomaly"]),
                "anomaly_type": str(r["anomaly_type"]),
                "anomaly_intensity": float(r["anomaly_intensity"]),
                "anomaly_breadth": int(r["anomaly_breadth"]),
                "changed_dimensions": str(r["changed_dimensions"]),
            })
            recent_1h.append(ts)
            recent_24h.append(ts)
            prev_ts = ts

    feat = pd.DataFrame(rows)
    # Frozen-baseline rows are valid, but warm-up itself is not used as model input.
    train_mask = feat["phase"] == "train"
    amount_scaler = StandardScaler()
    amount_scaler.fit(feat.loc[train_mask, ["amount_raw"]])
    feat["amount_scaled"] = amount_scaler.transform(feat[["amount_raw"]]).ravel()
    return feat, amount_scaler
