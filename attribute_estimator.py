"""Training-split calibration of the class-attribute prior Q."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from physical_attributes import (
    PHYSICAL_ATTRIBUTES,
    PhysicalAttributeResponseExtractor,
    unpack_csi_batch,
)
from prior import save_attribute_prior


def resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


class LightweightAttributeEstimator(nn.Module):
    """Small CSI-to-attribute network used only to estimate Q."""

    def __init__(
        self,
        input_dim: int,
        num_attrs: int = 8,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_attrs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x.float().transpose(1, 2)))


@torch.no_grad()
def collect_raw_attribute_responses(
    train_loader: Iterable,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Collect raw weak targets and labels from the training split."""
    responses: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    input_dim: int | None = None
    target_device = resolve_device(device)

    for batch in train_loader:
        x, y = unpack_csi_batch(batch)
        if input_dim is None:
            input_dim = int(x.shape[-1])
        responses.append(
            PhysicalAttributeResponseExtractor.raw_responses(
                x.to(target_device).float()
            ).cpu()
        )
        labels.append(y.detach().cpu().long().view(-1))

    if not responses or input_dim is None:
        raise ValueError("train_loader is empty; cannot estimate Q.")
    return torch.cat(responses), torch.cat(labels), input_dim


def aggregate_class_prior(
    responses: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    aggregation: str = "mean",
) -> torch.Tensor:
    """Aggregate sample attribute predictions into one Q row per class."""
    supported = {"mean", "median", "trimmed_mean"}
    if aggregation not in supported:
        raise ValueError(
            f"aggregation must be one of {sorted(supported)}, got {aggregation!r}."
        )

    rows = []
    for class_id in range(len(class_names)):
        class_responses = responses[labels == class_id]
        if class_responses.numel() == 0:
            row = torch.full((len(PHYSICAL_ATTRIBUTES),), 0.5)
        elif aggregation == "median":
            row = class_responses.median(dim=0).values
        elif aggregation == "trimmed_mean":
            sorted_responses = class_responses.sort(dim=0).values
            count = sorted_responses.size(0)
            low = int(0.1 * count)
            high = max(low + 1, int(0.9 * count))
            row = sorted_responses[low:high].mean(dim=0)
        else:
            row = class_responses.mean(dim=0)
        rows.append(row.clamp(0.0, 1.0))
    return torch.stack(rows)


def pretrain_attribute_estimator_and_build_prior(
    train_loader: Iterable,
    class_names: list[str],
    device: str | torch.device = "cuda",
    save_path: str | None = None,
    estimator_ckpt_path: str | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    low_percentile: float = 0.05,
    high_percentile: float = 0.95,
    aggregation: str = "mean",
    hidden: int = 128,
    dropout: float = 0.1,
) -> tuple[torch.Tensor, LightweightAttributeEstimator, dict[str, Any]]:
    """Pretrain the estimator and build Q using training samples only."""
    if not 0.0 <= low_percentile < high_percentile <= 1.0:
        raise ValueError("Quantiles must satisfy 0 <= low < high <= 1.")

    target_device = resolve_device(device)
    raw, _, input_dim = collect_raw_attribute_responses(
        train_loader, device=target_device
    )
    q_low = torch.quantile(raw, low_percentile, dim=0).to(target_device)
    q_high = torch.quantile(raw, high_percentile, dim=0).to(target_device)

    estimator = LightweightAttributeEstimator(
        input_dim=input_dim,
        num_attrs=len(PHYSICAL_ATTRIBUTES),
        hidden=hidden,
        dropout=dropout,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        estimator.parameters(), lr=lr, weight_decay=weight_decay
    )

    loss_history = []
    estimator.train()
    for _ in range(epochs):
        total_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            x, _ = unpack_csi_batch(batch)
            x = x.to(target_device).float()
            with torch.no_grad():
                raw_target = PhysicalAttributeResponseExtractor.raw_responses(x)
                target = PhysicalAttributeResponseExtractor.robust_normalize(
                    raw_target, q_low, q_high
                )

            prediction = torch.sigmoid(estimator(x))
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * x.size(0)
            sample_count += x.size(0)
        loss_history.append(total_loss / max(1, sample_count))

    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    estimator.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, y = unpack_csi_batch(batch)
            predictions.append(
                torch.sigmoid(estimator(x.to(target_device).float())).cpu()
            )
            labels.append(y.detach().cpu().long().view(-1))

    prior = aggregate_class_prior(
        torch.cat(predictions),
        torch.cat(labels),
        class_names,
        aggregation=aggregation,
    )
    metadata: dict[str, Any] = {
        "construction": "lightweight CSI-to-attribute pretraining on training split",
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "aggregation": aggregation,
        "loss": "mse_on_sigmoid_outputs",
        "loss_history": loss_history,
        "normalization_low": q_low.detach().cpu().tolist(),
        "normalization_high": q_high.detach().cpu().tolist(),
        "note": "Q uses only the training split and is fixed before model training.",
    }

    if save_path:
        save_attribute_prior(save_path, class_names, prior, metadata)
    if estimator_ckpt_path:
        import os

        os.makedirs(
            os.path.dirname(os.path.abspath(estimator_ckpt_path)), exist_ok=True
        )
        torch.save(
            {
                "model_state_dict": estimator.state_dict(),
                "input_dim": input_dim,
                "attribute_names": list(PHYSICAL_ATTRIBUTES),
                "metadata": metadata,
            },
            estimator_ckpt_path,
        )
    return prior, estimator, metadata
