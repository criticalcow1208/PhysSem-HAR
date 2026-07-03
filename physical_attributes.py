"""Physical-semantic attribute definitions and CSI response extraction."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch


PHYSICAL_ATTRIBUTES = {
    "periodic_motion": (
        "repeated temporal oscillations in CSI amplitude caused by cyclic body movement."
    ),
    "impulsive_change": (
        "a sudden transient peak in CSI amplitude followed by rapid stabilization."
    ),
    "stable_channel": "low temporal variation with nearly stationary CSI amplitude.",
    "high_frequency_variation": (
        "rapid short-term fluctuations in CSI amplitude over time."
    ),
    "strong_amplitude_change": "large magnitude perturbation across CSI subcarriers.",
    "smooth_transition": "gradual rise or fall of the CSI amplitude envelope.",
    "directional_sweep": (
        "asymmetric monotonic change indicating directional body movement."
    ),
    "broadband_disturbance": (
        "simultaneous disturbance distributed over many CSI subcarriers."
    ),
}


def normalize_name(name: str) -> str:
    """Normalize class names for robust matching across metadata files."""
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")


def unpack_csi_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(CSI, label)`` from common tuple/list/dictionary batch formats."""
    if isinstance(batch, Mapping):
        x_keys = ("x_enc", "x", "csi", "input", "inputs", "data")
        y_keys = ("labels", "label", "y", "target", "targets")
        x = next((batch[key] for key in x_keys if key in batch), None)
        y = next((batch[key] for key in y_keys if key in batch), None)
        if x is None or y is None:
            raise ValueError(
                "Cannot unpack batch dictionary. Expected a CSI key in "
                f"{x_keys} and a label key in {y_keys}; got {list(batch.keys())}."
            )
        return x, y

    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if len(batch) >= 2:
            return batch[0], batch[1]

    raise ValueError(
        "Unsupported batch format. Use (x, y), (x, y, ...), or a dictionary "
        "containing CSI and label fields."
    )


class PhysicalAttributeResponseExtractor:
    """Extract eight weak physical-attribute targets from CSI amplitude.

    Input must have shape ``[batch, time, subcarrier]``. Output has shape
    ``[batch, 8]`` in the order defined by :data:`PHYSICAL_ATTRIBUTES`.
    """

    attribute_names = list(PHYSICAL_ATTRIBUTES)

    @staticmethod
    def raw_responses(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected CSI shape [B, T, C], got {tuple(x.shape)}.")

        x = x.float()
        batch_size, time_steps, _ = x.shape
        eps = 1e-6
        x_abs = x.abs()
        x_centered = x - x.mean(dim=1, keepdim=True)

        if time_steps > 1:
            dx = x[:, 1:, :] - x[:, :-1, :]
        else:
            dx = torch.zeros_like(x[:, :1, :])
        abs_dx = dx.abs()

        if dx.size(1) > 1:
            abs_ddx = (dx[:, 1:, :] - dx[:, :-1, :]).abs()
        else:
            abs_ddx = torch.zeros_like(dx[:, :1, :])

        temporal_var = x.var(dim=1, unbiased=False).mean(dim=1)
        mean_abs_diff = abs_dx.mean(dim=(1, 2))
        max_abs_diff = abs_dx.amax(dim=(1, 2))
        second_order = abs_ddx.mean(dim=(1, 2))

        peak = x_abs.amax(dim=(1, 2))
        mean_abs = x_abs.mean(dim=(1, 2))
        peak_to_mean = peak / (mean_abs + eps)

        fft = torch.fft.rfft(x_centered, dim=1).abs()
        energy = fft.square().mean(dim=2)
        non_dc_energy = energy[:, 1:] if energy.size(1) > 1 else energy
        peak_freq_ratio = non_dc_energy.max(dim=1).values / (
            non_dc_energy.sum(dim=1) + eps
        )

        half = max(1, energy.size(1) // 2)
        if half < energy.size(1):
            high_energy = energy[:, half:].mean(dim=1)
        else:
            high_energy = energy[:, -1]
        high_freq_ratio = high_energy / (energy.mean(dim=1) + eps)

        stable_raw = 1.0 / (1.0 + temporal_var + mean_abs_diff)
        smooth_raw = (
            mean_abs_diff / (mean_abs_diff + second_order + eps)
        ) * (1.0 - stable_raw)

        if time_steps > 1:
            time_axis = torch.linspace(
                -1.0, 1.0, time_steps, device=x.device, dtype=x.dtype
            ).view(1, time_steps, 1)
            slope = (x_centered * time_axis).mean(dim=1) / (
                time_axis.square().mean() + eps
            )
            trend_strength = slope.abs().mean(dim=1)
            temporal_std = (
                x_centered.std(dim=1, unbiased=False).mean(dim=1) + eps
            )
            directional_raw = trend_strength / temporal_std
        else:
            directional_raw = torch.zeros(
                batch_size, device=x.device, dtype=x.dtype
            )

        subcarrier_var = x.var(dim=2, unbiased=False).mean(dim=1)
        per_subcarrier_disturbance = (
            abs_dx.mean(dim=1) if time_steps > 1 else x_abs.mean(dim=1)
        )
        threshold = per_subcarrier_disturbance.mean(dim=1, keepdim=True)
        threshold = threshold + per_subcarrier_disturbance.std(
            dim=1, keepdim=True, unbiased=False
        )
        disturbed_ratio = (
            per_subcarrier_disturbance > threshold
        ).float().mean(dim=1)

        raw = torch.stack(
            [
                peak_freq_ratio,
                max_abs_diff / (mean_abs_diff + eps),
                stable_raw,
                high_freq_ratio,
                peak_to_mean + torch.log1p(temporal_var.clamp_min(0.0)),
                smooth_raw,
                directional_raw,
                subcarrier_var + disturbed_ratio,
            ],
            dim=-1,
        )
        return torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0)

    @staticmethod
    def robust_normalize(
        raw: torch.Tensor,
        q_low: torch.Tensor,
        q_high: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Scale responses to ``[0, 1]`` using robust quantile bounds."""
        return ((raw - q_low) / (q_high - q_low + eps)).clamp(0.0, 1.0)
