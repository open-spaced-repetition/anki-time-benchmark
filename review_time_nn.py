from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def featurize_dsrg(
    d: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """
    Raw -> model features:
      [log1p(D), log1p(S), logit(R), grade_z]
    """
    d = np.log1p(np.clip(d, 0.0, None))
    s = np.log1p(np.clip(s, 0.0, None))
    r = _logit(r.astype(np.float32))
    # grade in {1,2,3,4}, standardized to roughly zero mean / unit-ish variance
    g = (g.astype(np.float32) - 2.5) / 1.11803398875
    return np.stack([d, s, r, g], axis=1).astype(np.float32)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return Normalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


class ReviewTimeNN(nn.Module):
    """
    4 linear layers total:
      4 -> 32 -> 16 -> 8 -> 1
    Final layer params = 8*1 + 1 = 9 (within requested 8-12).
    """
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
        )
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h).squeeze(-1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True


def train_regressor(
    model: ReviewTimeNN,
    x: np.ndarray,
    y_log_sec: np.ndarray,
    device: torch.device,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 8,
    batch_size: int = 1024,
    train_head_only: bool = False,
) -> None:
    if len(x) == 0:
        return

    model.to(device)
    if train_head_only:
        model.freeze_backbone()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y_log_sec.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def predict_seconds(
    model: ReviewTimeNN,
    x: np.ndarray,
    device: torch.device,
    *,
    min_sec: float = 0.05,
    max_sec: float = 1800.0,
) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    model.to(device)
    xb = torch.from_numpy(x).to(device)
    y_log = model(xb).cpu().numpy()
    y_sec = np.exp(y_log)
    y_sec = np.clip(y_sec, min_sec, max_sec)
    return y_sec.astype(np.float32)