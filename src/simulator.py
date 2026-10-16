from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_users: int = 1000
    transactions_per_user: int = 100
    warmup_per_user: int = 30
    train_per_user: int = 49
    val_per_user: int = 10
    test_per_user: int = 11
    anomaly_rate_val: float = 0.02
    anomaly_rate_test: float = 0.02
    seed: int = 42

    def validate(self) -> None:
        expected = self.warmup_per_user + self.train_per_user + self.val_per_user + self.test_per_user
        if expected != self.transactions_per_user:
            raise ValueError(f"Phase counts ({expected}) must equal transactions_per_user ({self.transactions_per_user}).")
        if self.n_users < 10:
            raise ValueError("n_users should be at least 10 for a meaningful demo.")


ZONES = [
    "San Borja", "Miraflores", "Surco", "San Isidro", "Jesus Maria",
    "Magdalena", "Pueblo Libre", "San Miguel", "Lince", "La Molina",
    "Surquillo", "Barranco", "Breña", "Cercado", "Los Olivos",
    "Comas", "Ate", "Chorrillos", "Rimac", "Independencia",
]


def _circular_hour_distance(a: float, b: float) -> float:
    d = abs(a - b) % 24.0
    return min(d, 24.0 - d)


def _phase_bounds() -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        "warmup": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-30 23:59:59")),
        "train": (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-20 23:59:59")),
        "validation": (pd.Timestamp("2026-03-21"), pd.Timestamp("2026-03-30 23:59:59")),
        "test": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-11 23:59:59")),
    }


def _sample_timestamps(rng: np.random.Generator, n: int, start: pd.Timestamp, end: pd.Timestamp,
                       hour_center: float, hour_spread: float) -> List[pd.Timestamp]:
    total_days = max(1, (end.normalize() - start.normalize()).days + 1)
    # Sampling with replacement intentionally allows multiple operations on active days.
    day_offsets = rng.integers(0, total_days, size=n)
    # A wrapped Gaussian is sufficient for a controlled synthetic baseline.
    hours = (rng.normal(hour_center, hour_spread, size=n) % 24.0)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    stamps = []
    for d, h, m, s in zip(day_offsets, hours, minutes, seconds):
        base = start.normalize() + pd.Timedelta(days=int(d))
        ts = base + pd.Timedelta(hours=float(h), minutes=int(m), seconds=int(s))
        if ts < start:
            ts = start + pd.Timedelta(minutes=1)
        if ts > end:
            ts = end - pd.Timedelta(minutes=1)
        stamps.append(ts)
    return sorted(stamps)


def _profile_for_user(rng: np.random.Generator, user_idx: int, recipient_universe: np.ndarray) -> dict:
    # Four broad archetypes only control simulator heterogeneity; they are not model inputs.
    archetype = rng.choice(["low", "regular", "active", "night"], p=[0.20, 0.45, 0.25, 0.10])
    if archetype == "low":
        amount_med = rng.lognormal(mean=3.15, sigma=0.40)
        hour_center = rng.normal(18, 2)
        hour_spread = rng.uniform(1.8, 3.0)
    elif archetype == "regular":
        amount_med = rng.lognormal(mean=3.55, sigma=0.45)
        hour_center = rng.normal(14, 3)
        hour_spread = rng.uniform(2.0, 4.0)
    elif archetype == "active":
        amount_med = rng.lognormal(mean=3.85, sigma=0.55)
        hour_center = rng.normal(13, 3)
        hour_spread = rng.uniform(3.0, 5.0)
    else:
        amount_med = rng.lognormal(mean=3.45, sigma=0.50)
        hour_center = rng.normal(22, 1.5)
        hour_spread = rng.uniform(1.5, 3.0)

    amount_med = float(np.clip(amount_med, 8.0, 450.0))
    amount_sigma = float(rng.uniform(0.28, 0.62))

    recipients = rng.choice(recipient_universe, size=5, replace=False)
    recipient_weights = rng.dirichlet(np.array([5.0, 3.0, 1.8, 1.0, 0.7]))

    primary_device = f"D{user_idx:04d}_A"
    secondary_device = f"D{user_idx:04d}_B"
    devices = np.array([primary_device, secondary_device])
    device_weights = np.array([0.91, 0.09])

    locs = rng.choice(np.array(ZONES), size=3, replace=False)
    loc_weights = np.array([0.72, 0.20, 0.08])

    return {
        "user_id": f"U{user_idx:04d}",
        "archetype": archetype,
        "amount_median_gen": amount_med,
        "amount_sigma_gen": amount_sigma,
        "hour_center_gen": float(hour_center % 24),
        "hour_spread_gen": hour_spread,
        "recipients": recipients,
        "recipient_weights": recipient_weights,
        "devices": devices,
        "device_weights": device_weights,
        "locations": locs,
        "location_weights": loc_weights,
    }


def _generate_normal_rows(rng: np.random.Generator, profile: dict, phase: str, n: int,
                          start: pd.Timestamp, end: pd.Timestamp) -> List[dict]:
    timestamps = _sample_timestamps(
        rng, n, start, end, profile["hour_center_gen"], profile["hour_spread_gen"]
    )
    rows = []
    # Parameterize lognormal so its median is close to amount_median_gen.
    mu = math.log(max(profile["amount_median_gen"], 0.01))
    sigma = profile["amount_sigma_gen"]
    amounts = np.clip(rng.lognormal(mean=mu, sigma=sigma, size=n), 1.0, 2500.0)
    recipients_draw = [str(rng.choice(profile["recipients"], p=profile["recipient_weights"])) for _ in range(n)]
    devices_draw = [str(rng.choice(profile["devices"], p=profile["device_weights"])) for _ in range(n)]
    locations_draw = [str(rng.choice(profile["locations"], p=profile["location_weights"])) for _ in range(n)]
    # The warm-up is deliberately a baseline-establishment phase: ensure the habitual
    # support is observed at least once so novelty means "new relative to baseline".
    if phase == "warmup":
        for j, val in enumerate(profile["recipients"]):
            if j < n:
                recipients_draw[j] = str(val)
        for j, val in enumerate(profile["devices"]):
            if j < n:
                devices_draw[-(j + 1)] = str(val)
        for j, val in enumerate(profile["locations"]):
            if j < n:
                locations_draw[-(j + 3)] = str(val)
    for i, ts in enumerate(timestamps):
        rows.append({
            "user_id": profile["user_id"],
            "archetype": profile["archetype"],
            "phase": phase,
            "timestamp": ts,
            "amount": float(round(amounts[i], 2)),
            "transaction_type": "P2P_TRANSFER",
            "recipient_id": recipients_draw[i],
            "device_id": devices_draw[i],
            "location_zone": locations_draw[i],
            "is_anomaly": 0,
            "anomaly_type": "normal",
            "anomaly_intensity": 0.0,
            "anomaly_breadth": 0,
            "changed_dimensions": "",
        })
    return rows


def _baseline_from_warmup(user_warmup: pd.DataFrame) -> dict:
    amounts = user_warmup["amount"].to_numpy(float)
    med = float(np.median(amounts))
    mad = float(np.median(np.abs(amounts - med)))
    if mad < 1e-6:
        mad = max(1.0, 0.05 * med)
    hour_counts = user_warmup["timestamp"].dt.hour.value_counts().to_dict()
    rec_counts = user_warmup["recipient_id"].value_counts().to_dict()
    dev_counts = user_warmup["device_id"].value_counts().to_dict()
    loc_counts = user_warmup["location_zone"].value_counts().to_dict()
    daily_counts = user_warmup.set_index("timestamp").resample("1D").size()
    daily_nonzero = daily_counts[daily_counts > 0]
    daily_median = float(np.median(daily_nonzero)) if len(daily_nonzero) else 1.0
    sorted_ts = user_warmup["timestamp"].sort_values()
    gaps = sorted_ts.diff().dropna().dt.total_seconds() / 3600.0
    gap_median = float(np.median(gaps)) if len(gaps) else 24.0
    return {
        "amount_median": med,
        "amount_mad": mad,
        "hour_counts": hour_counts,
        "recipient_counts": rec_counts,
        "device_counts": dev_counts,
        "location_counts": loc_counts,
        "daily_count_median": max(daily_median, 1.0),
        "gap_median_hours": max(gap_median, 0.25),
    }


def _novelty(counts: Dict[str, int], value: str) -> float:
    if value not in counts:
        return 1.0
    return float(1.0 / (1.0 + float(counts[value])))


def _closest_known_by_novelty(counts: Dict[str, int], target: float, fallback: str) -> str:
    if not counts:
        return fallback
    options = []
    for val in counts:
        options.append((abs(_novelty(counts, val) - target), val))
    options.sort(key=lambda x: x[0])
    return options[0][1]


def _mutate_anomaly(rng: np.random.Generator, row: pd.Series, baseline: dict,
                    all_recipient_universe: np.ndarray, user_idx: int,
                    target_intensity: float, dimensions: List[str], previous_ts: pd.Timestamp | None) -> Tuple[pd.Series, List[float]]:
    out = row.copy()
    achieved = []
    for dim in dimensions:
        d = float(np.clip(rng.normal(target_intensity, 0.06), 0.05, 1.0))
        if dim == "amount":
            # Magnitude maps to a multiplicative factor from x1.1 to x8.0 around the warm-up median.
            factor = math.exp(d * math.log(12.0))
            if rng.random() < 0.85:
                out["amount"] = round(float(baseline["amount_median"] * factor), 2)
            else:
                out["amount"] = round(float(max(1.0, baseline["amount_median"] / factor)), 2)
            robust_z = abs(out["amount"] - baseline["amount_median"]) / (baseline["amount_mad"] + 1e-9)
            achieved.append(float(np.clip(robust_z / 8.0, 0.0, 1.0)))
        elif dim == "hour":
            center = int(max(baseline["hour_counts"], key=baseline["hour_counts"].get)) if baseline["hour_counts"] else 12
            shift = max(1, int(round(d * 12)))
            new_hour = (center + (shift if rng.random() < 0.5 else -shift)) % 24
            ts = pd.Timestamp(out["timestamp"])
            out["timestamp"] = ts.normalize() + pd.Timedelta(hours=new_hour, minutes=int(rng.integers(0, 60)))
            achieved.append(_circular_hour_distance(new_hour, center) / 12.0)
        elif dim == "recipient":
            if d >= 0.67:
                candidate = f"RX{user_idx:04d}_{int(rng.integers(1000, 9999))}"
            else:
                candidate = _closest_known_by_novelty(baseline["recipient_counts"], d, str(rng.choice(all_recipient_universe)))
            out["recipient_id"] = candidate
            achieved.append(_novelty(baseline["recipient_counts"], candidate))
        elif dim == "device":
            if d >= 0.67:
                candidate = f"DX{user_idx:04d}_{int(rng.integers(10, 99))}"
            else:
                candidate = _closest_known_by_novelty(baseline["device_counts"], d, f"DX{user_idx:04d}")
            out["device_id"] = candidate
            achieved.append(_novelty(baseline["device_counts"], candidate))
        elif dim == "location":
            if d >= 0.67:
                unseen = [z for z in ZONES if z not in baseline["location_counts"]]
                candidate = str(rng.choice(unseen if unseen else ZONES))
            else:
                candidate = _closest_known_by_novelty(baseline["location_counts"], d, str(rng.choice(ZONES)))
            out["location_zone"] = candidate
            achieved.append(_novelty(baseline["location_counts"], candidate))
        elif dim == "frequency":
            if previous_ts is not None:
                gap_h = max(0.02, baseline["gap_median_hours"] * (1.0 - 0.92 * d))
                out["timestamp"] = pd.Timestamp(previous_ts) + pd.Timedelta(hours=gap_h)
                achieved.append(float(np.clip(1.0 - gap_h / (baseline["gap_median_hours"] + 1e-9), 0.0, 1.0)))
            else:
                achieved.append(d)
        else:
            raise ValueError(f"Unsupported anomaly dimension: {dim}")
    return out, achieved


def _inject_anomalies(df: pd.DataFrame, baselines: Dict[str, dict], rng: np.random.Generator,
                      rate_val: float, rate_test: float, recipient_universe: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    dimensions = np.array(["amount", "hour", "frequency", "recipient", "device", "location"])
    user_index_map = {uid: int(uid[1:]) for uid in df["user_id"].unique()}

    selected_indices: List[int] = []
    for phase, rate in [("validation", rate_val), ("test", rate_test)]:
        candidates = df.index[df["phase"] == phase].to_numpy()
        n = max(1, int(round(len(candidates) * rate)))
        picked = rng.choice(candidates, size=n, replace=False)
        selected_indices.extend(int(i) for i in picked)

    # Pre-register intensity strata in equal widths for descriptive analysis; D remains continuous.
    strata = [(0.18, 0.33), (0.34, 0.66), (0.67, 0.96)]
    # Shuffle labels so each phase gets a mix of difficulty levels.
    rng.shuffle(selected_indices)

    # Work user by user so frequency mutations can reference the previous observed timestamp.
    selected_set = set(selected_indices)
    for uid, group in df.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        prev_ts: pd.Timestamp | None = None
        for idx, row in group.iterrows():
            if idx in selected_set:
                stratum = strata[int(rng.integers(0, len(strata)))]
                target = float(rng.uniform(*stratum))
                k = int(rng.choice([1, 2, 3], p=[0.48, 0.34, 0.18]))
                dims = list(rng.choice(dimensions, size=k, replace=False))
                mutated, achieved = _mutate_anomaly(
                    rng, row, baselines[uid], recipient_universe, user_index_map[uid], target, dims, prev_ts
                )
                # Preserve the pre-registered temporal split even after timestamp perturbations.
                phase_start, phase_end = _phase_bounds()[str(row["phase"])]
                mts = pd.Timestamp(mutated["timestamp"])
                if mts < phase_start:
                    mts = phase_start + pd.Timedelta(minutes=int(rng.integers(1, 60)))
                if mts > phase_end:
                    mts = phase_end - pd.Timedelta(minutes=int(rng.integers(1, 60)))
                mutated["timestamp"] = mts
                D = float(np.mean(achieved)) if achieved else target
                df.loc[idx, ["timestamp", "amount", "recipient_id", "device_id", "location_zone"]] = [
                    mutated["timestamp"], mutated["amount"], mutated["recipient_id"], mutated["device_id"], mutated["location_zone"]
                ]
                df.loc[idx, "is_anomaly"] = 1
                df.loc[idx, "anomaly_type"] = "ATO"
                df.loc[idx, "anomaly_intensity"] = D
                df.loc[idx, "anomaly_breadth"] = k
                df.loc[idx, "changed_dimensions"] = "|".join(sorted(dims))
                prev_ts = pd.Timestamp(mutated["timestamp"])
            else:
                prev_ts = pd.Timestamp(row["timestamp"])

    return df.sort_values(["timestamp", "user_id"]).reset_index(drop=True)


def simulate_transactions(config: SimulationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    config.validate()
    rng = np.random.default_rng(config.seed)
    recipient_universe = np.array([f"R{i:05d}" for i in range(max(5000, config.n_users * 5))])
    bounds = _phase_bounds()

    all_rows: List[dict] = []
    profile_rows: List[dict] = []
    for u in range(config.n_users):
        profile = _profile_for_user(rng, u, recipient_universe)
        profile_rows.append({
            "user_id": profile["user_id"],
            "archetype": profile["archetype"],
            "amount_median_gen": profile["amount_median_gen"],
            "hour_center_gen": profile["hour_center_gen"],
            "hour_spread_gen": profile["hour_spread_gen"],
        })
        counts = {
            "warmup": config.warmup_per_user,
            "train": config.train_per_user,
            "validation": config.val_per_user,
            "test": config.test_per_user,
        }
        for phase, n in counts.items():
            start, end = bounds[phase]
            all_rows.extend(_generate_normal_rows(rng, profile, phase, n, start, end))

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])

    baselines: Dict[str, dict] = {}
    for uid, g in df[df["phase"] == "warmup"].groupby("user_id"):
        baselines[uid] = _baseline_from_warmup(g.sort_values("timestamp"))

    df = _inject_anomalies(
        df, baselines, rng, config.anomaly_rate_val, config.anomaly_rate_test, recipient_universe
    )
    profiles = pd.DataFrame(profile_rows)
    return df, profiles, baselines
