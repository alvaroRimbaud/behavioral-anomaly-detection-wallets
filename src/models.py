from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


@dataclass
class AEConfig:
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 25
    weight_decay: float = 1e-5
    patience: int = 5


class AutoencoderDetector:
    def __init__(self, seed: int = 42, config: AEConfig | None = None):
        self.seed = seed
        self.config = config or AEConfig()
        self.scaler = StandardScaler()
        self.model: AutoencoderNet | None = None
        self.history: list[float] = []

    def fit(self, X_train: np.ndarray) -> "AutoencoderDetector":
        set_seed(self.seed)
        Xs = self.scaler.fit_transform(X_train).astype(np.float32)
        tensor = torch.from_numpy(Xs)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.config.batch_size, shuffle=True)
        self.model = AutoencoderNet(Xs.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        loss_fn = nn.MSELoss()
        best = float("inf")
        stale = 0
        best_state = None
        for _ in range(self.config.epochs):
            self.model.train()
            losses = []
            for (xb,) in loader:
                optimizer.zero_grad()
                recon = self.model(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(losses)) if losses else float("nan")
            self.history.append(epoch_loss)
            if epoch_loss < best - 1e-6:
                best = epoch_loss
                stale = 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
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


def fit_iforest(X_train: np.ndarray, seed: int = 42) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        max_features=1.0,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def iforest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # Larger score = more anomalous.
    return -model.score_samples(X)


def fit_lof(X_train: np.ndarray) -> Tuple[LocalOutlierFactor, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1)
    model.fit(Xs)
    return model, scaler


def lof_scores(model: LocalOutlierFactor, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(scaler.transform(X))
