"""Public entry point for PhysSem-HAR.

Import the model with ``from main import PhysSemHAR`` or instantiate it from a
JSON configuration with ``python main.py --config config.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attribute_estimator import (
    LightweightAttributeEstimator,
    pretrain_attribute_estimator_and_build_prior,
)
from config import PhysSemHARConfig
from model import PhysSemHAR
from physical_attributes import (
    PHYSICAL_ATTRIBUTES,
    PhysicalAttributeResponseExtractor,
)
from prior import load_attribute_prior, save_attribute_prior

__all__ = [
    "PHYSICAL_ATTRIBUTES",
    "LightweightAttributeEstimator",
    "PhysicalAttributeResponseExtractor",
    "PhysSemHAR",
    "PhysSemHARConfig",
    "load_attribute_prior",
    "pretrain_attribute_estimator_and_build_prior",
    "save_attribute_prior",
]


def load_config(path: str | Path) -> PhysSemHARConfig:
    """Load a :class:`PhysSemHARConfig` from JSON."""
    with open(path, "r", encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)
    return PhysSemHARConfig(**payload)


def build_model(config: PhysSemHARConfig | str | Path) -> PhysSemHAR:
    """Build PhysSem-HAR from a config object or JSON path."""
    if isinstance(config, (str, Path)):
        config = load_config(config)
    return PhysSemHAR(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize PhysSem-HAR.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a PhysSem-HAR JSON configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.config)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {total:,} total, {trainable:,} trainable")


if __name__ == "__main__":
    main()
