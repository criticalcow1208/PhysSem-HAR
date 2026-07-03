"""Configuration schema for PhysSem-HAR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysSemHARConfig:
    """Minimal configuration accepted by :class:`PhysSemHAR`."""

    num_classes: int
    patch_len: int
    stride: int
    enc_in: int
    llm_path: str
    label2id_path: str
    attr_prior_path: str
    d_model: int = 768
    dropout: float = 0.1
    n_heads: int = 8
    ablation_mode: str = "full"
