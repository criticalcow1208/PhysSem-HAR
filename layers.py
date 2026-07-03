"""Reusable neural-network layers for PhysSem-HAR."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert a CSI sequence into overlapping patch embeddings."""

    def __init__(
        self,
        hidden_size: int,
        patch_len: int,
        stride: int,
        dropout: float,
        input_dim: int,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive.")
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len * input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if x.ndim != 3:
            raise ValueError(f"Expected CSI shape [B, T, C], got {tuple(x.shape)}.")
        batch_size, time_steps, channels = x.shape
        if time_steps < self.patch_len:
            x = F.pad(x, (0, 0, 0, self.patch_len - time_steps))
        else:
            remainder = (time_steps - self.patch_len) % self.stride
            if remainder:
                x = F.pad(x, (0, 0, 0, self.stride - remainder))

        patches = x.permute(0, 2, 1).unfold(
            2, self.patch_len, self.stride
        )
        patches = (
            patches.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, -1, self.patch_len * channels)
        )
        patches = self.dropout(self.norm(self.value_embedding(patches)))
        return patches, patches.size(1)


class SignalDescriptor(nn.Module):
    """Build a differentiable physical descriptor token from CSI amplitude."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_abs = x.abs()
        temporal_var = x.var(dim=1, unbiased=False).mean(dim=1, keepdim=True)

        if x.size(1) > 1:
            dx = x[:, 1:, :] - x[:, :-1, :]
            mean_abs_diff = dx.abs().mean(dim=(1, 2)).unsqueeze(-1)
        else:
            dx = torch.zeros_like(x[:, :1, :])
            mean_abs_diff = torch.zeros(
                batch_size, 1, device=x.device, dtype=x.dtype
            )

        peak = x_abs.amax(dim=(1, 2)).unsqueeze(-1)
        mean = x_abs.mean(dim=(1, 2)).unsqueeze(-1)
        peak_to_mean = peak / (mean + 1e-6)

        energy = torch.fft.rfft(x, dim=1).abs().square()
        total_energy = energy.mean(dim=(1, 2)).unsqueeze(-1)
        high_start = max(1, energy.size(1) // 2)
        if high_start < energy.size(1):
            high_energy = energy[:, high_start:, :].mean(dim=(1, 2)).unsqueeze(-1)
        else:
            high_energy = energy[:, -1:, :].mean(dim=(1, 2)).unsqueeze(-1)
        high_frequency_ratio = high_energy / (total_energy + 1e-6)

        subcarrier_var = x.var(dim=2, unbiased=False).mean(dim=1, keepdim=True)
        if dx.size(1) > 1:
            smoothness = (
                (dx[:, 1:, :] - dx[:, :-1, :])
                .abs()
                .mean(dim=(1, 2))
                .unsqueeze(-1)
            )
        else:
            smoothness = torch.zeros(
                batch_size, 1, device=x.device, dtype=x.dtype
            )

        descriptor = torch.cat(
            [
                temporal_var,
                mean_abs_diff,
                peak_to_mean,
                high_frequency_ratio,
                subcarrier_var,
                smoothness,
            ],
            dim=-1,
        )
        return self.projection(
            torch.log1p(descriptor.clamp_min(0.0))
        ).unsqueeze(1)


class AttributeConditionedFusion(nn.Module):
    """Fuse physical-attribute tokens with CSI patch tokens bidirectionally."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attr_to_signal = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.signal_to_attr = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attr_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.patch_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.attr_norm = nn.LayerNorm(hidden_size)
        self.patch_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        attribute_embeddings: torch.Tensor,
        patch_embeddings: torch.Tensor,
        descriptor_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal_memory = torch.cat([descriptor_token, patch_embeddings], dim=1)
        attr_context, _ = self.attr_to_signal(
            attribute_embeddings, signal_memory, signal_memory
        )
        patch_context, _ = self.signal_to_attr(
            patch_embeddings, attribute_embeddings, attribute_embeddings
        )

        attr_gate = torch.sigmoid(
            self.attr_gate(
                torch.cat([attribute_embeddings, attr_context], dim=-1)
            )
        )
        enhanced_attributes = self.attr_norm(
            attr_gate * self.dropout(attr_context)
            + (1.0 - attr_gate) * attribute_embeddings
        )

        descriptor = descriptor_token.expand(-1, patch_embeddings.size(1), -1)
        patch_gate = torch.sigmoid(
            self.patch_gate(
                torch.cat(
                    [patch_embeddings, patch_context, descriptor], dim=-1
                )
            )
        )
        enhanced_patches = self.patch_norm(
            patch_gate * self.dropout(patch_context)
            + (1.0 - patch_gate) * patch_embeddings
        )
        return enhanced_attributes, enhanced_patches


class AttributeGroundedClassifier(nn.Module):
    """Fuse signal logits with class-prior semantic logits."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_attrs: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.signal_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.attr_predictor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_attrs),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size + num_attrs, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.attr_logit_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        patch_features: torch.Tensor,
        attribute_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal_logits = self.signal_classifier(patch_features)
        attribute_logits = self.attr_predictor(patch_features)
        attribute_probabilities = torch.sigmoid(attribute_logits)
        semantic_logits = (
            self.attr_logit_scale.exp().clamp(max=50.0)
            * F.normalize(attribute_probabilities, dim=-1)
            @ F.normalize(attribute_prior, dim=-1).T
        )
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat([patch_features, attribute_probabilities], dim=-1)
            )
        )
        return (
            gate * signal_logits + (1.0 - gate) * semantic_logits,
            attribute_logits,
        )
