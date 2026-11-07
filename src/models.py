from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple
import random

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Fix pseudo-random generators used by numpy/PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Avoid excessive CPU thread oversubscription on shared and laptop hardware.
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class AutoencoderNet(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        h1 = 8 if d >= 8 else max(4, d)
        latent = 4 if d >= 6 else max(2, d // 2)
        self.net = nn.Sequential(
            nn.Linear(d, h1), nn.ReLU(),
            nn.Linear(h1, latent), nn.ReLU(),
            nn.Linear(latent, h1), nn.ReLU(),
            nn.Linear(h1, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class IFConfig:
    n_estimators: int = 200
    max_samples: int | str = 256
    max_features: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LOFConfig:
    n_neighbors: int = 35

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AEConfig:
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 25
    weight_decay: float = 1e-5
    patience: int = 5
    min_delta: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoencoderDetector:
    """Compact global Autoencoder used as an unsupervised novelty detector.

    Training labels are never consumed. Early stopping can observe a *normal-only*
    internal temporal holdout supplied through ``X_early_stop``. This separates
    optimization from the external validation set used to select hyperparameters
    and the anomaly threshold.
    """

    def __init__(self, seed: int = 42, config: AEConfig | None = None):
        self.seed = seed
        self.config = config or AEConfig()
        self.scaler = StandardScaler()
        self.model: AutoencoderNet | None = None
        self.train_history: list[float] = []
        self.early_stop_history: list[float] = []
        self.best_epoch: int | None = None
        self.best_early_stop_loss: float | None = None

    def fit(self, X_train: np.ndarray, X_early_stop: np.ndarray | None = None) -> "AutoencoderDetector":
        set_seed(self.seed)
        Xs = self.scaler.fit_transform(X_train).astype(np.float32)
        tensor = torch.from_numpy(Xs)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )

        early_tensor: torch.Tensor | None = None
        if X_early_stop is not None and len(X_early_stop):
            early_scaled = self.scaler.transform(X_early_stop).astype(np.float32)
            early_tensor = torch.from_numpy(early_scaled)

        self.model = AutoencoderNet(Xs.shape[1])
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        best = float("inf")
        stale = 0
        best_state = None
        self.train_history = []
        self.early_stop_history = []
        self.best_epoch = None

        for epoch in range(self.config.epochs):
            self.model.train()
            losses = []
            for (xb,) in loader:
                optimizer.zero_grad()
                recon = self.model(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(losses)) if losses else float("nan")
            self.train_history.append(train_loss)

            # Observation 6: early stopping is no longer based on the same data
            # whose weights are being optimized. If a normal-only holdout is
            # provided, it is the stopping criterion.
            if early_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    early_loss = float(loss_fn(self.model(early_tensor), early_tensor).cpu())
            else:
                early_loss = train_loss
            self.early_stop_history.append(early_loss)

            if early_loss < best - self.config.min_delta:
                best = early_loss
                stale = 0
                self.best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_early_stop_loss = best if np.isfinite(best) else None
        return self

    def score_samples(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("AutoencoderDetector is not fitted.")
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(Xs), batch_size):
                xb = torch.from_numpy(Xs[start:start + batch_size])
                recon = self.model(xb)
                err = torch.mean((recon - xb) ** 2, dim=1)
                scores.append(err.cpu().numpy())
        return np.concatenate(scores)


def fit_iforest(X_train: np.ndarray, seed: int = 42, config: IFConfig | None = None) -> IsolationForest:
    cfg = config or IFConfig()
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        max_features=cfg.max_features,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train)
    return model


def iforest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # Common experiment convention: larger score = more anomalous.
    return -model.score_samples(X)


def fit_lof(X_train: np.ndarray, config: LOFConfig | None = None) -> Tuple[LocalOutlierFactor, StandardScaler]:
    cfg = config or LOFConfig()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = LocalOutlierFactor(
        n_neighbors=cfg.n_neighbors,
        novelty=True,
        contamination="auto",
        n_jobs=1,
    )
    model.fit(Xs)
    return model, scaler


def lof_scores(model: LocalOutlierFactor, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(scaler.transform(X))


def if_grid(profile: str = "full") -> list[IFConfig]:
    if profile == "off":
        return [IFConfig()]
    if profile == "quick":
        return [
            IFConfig(n_estimators=100, max_samples=256, max_features=1.0),
            IFConfig(n_estimators=200, max_samples="auto", max_features=1.0),
        ]
    return [
        IFConfig(n_estimators=n, max_samples=ms, max_features=mf)
        for n in (100, 200, 300)
        for ms in (256, 512, "auto")
        for mf in (0.75, 1.0)
    ]


def ae_grid(profile: str = "full") -> list[AEConfig]:
    if profile == "off":
        return [AEConfig()]
    if profile == "quick":
        return [
            AEConfig(learning_rate=1e-3, batch_size=128, weight_decay=1e-5, epochs=15, patience=4),
            AEConfig(learning_rate=1e-4, batch_size=128, weight_decay=1e-5, epochs=15, patience=4),
        ]
    # Thesis search space: learning rate {1e-4,1e-3}, batch {32,64,128};
    # L2 is represented by weight_decay. Epochs are an upper bound because
    # normal-only early stopping can terminate earlier.
    return [
        AEConfig(
            learning_rate=lr,
            batch_size=bs,
            weight_decay=wd,
            epochs=25,
            patience=5,
        )
        for lr in (1e-4, 1e-3)
        for bs in (32, 64, 128)
        for wd in (0.0, 1e-5)
    ]


def lof_grid(profile: str = "full") -> list[LOFConfig]:
    if profile == "off":
        return [LOFConfig()]
    if profile == "quick":
        return [LOFConfig(n_neighbors=20), LOFConfig(n_neighbors=35)]
    return [LOFConfig(n_neighbors=n) for n in (20, 35, 50)]
