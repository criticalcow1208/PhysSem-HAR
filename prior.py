"""Class-attribute prior (Q) serialization and validation."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from physical_attributes import PHYSICAL_ATTRIBUTES, normalize_name


def save_attribute_prior(
    save_path: str,
    class_names: list[str],
    prior_matrix: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a calibrated class-attribute prior as a portable JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    payload = {
        "attribute_names": list(PHYSICAL_ATTRIBUTES),
        "class_names": list(class_names),
        "matrix": prior_matrix.detach().cpu().tolist(),
        "metadata": metadata or {},
    }
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_attribute_prior(
    path: str,
    class_names: list[str],
) -> torch.Tensor:
    """Load Q, validate its schema, and reorder rows to ``class_names``."""
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    required_keys = {"attribute_names", "class_names", "matrix", "metadata"}
    missing = required_keys.difference(payload) if isinstance(payload, dict) else required_keys
    if missing:
        raise ValueError(
            f"Invalid Q file {path!r}; missing keys: {sorted(missing)}."
        )

    expected_attributes = list(PHYSICAL_ATTRIBUTES)
    if list(payload["attribute_names"]) != expected_attributes:
        raise ValueError(
            "Attribute order mismatch. Expected "
            f"{expected_attributes}, got {payload['attribute_names']}."
        )

    matrix = torch.tensor(payload["matrix"], dtype=torch.float32)
    file_classes = list(payload["class_names"])
    if matrix.shape != (len(file_classes), len(expected_attributes)):
        raise ValueError(
            f"Invalid Q shape {tuple(matrix.shape)}; expected "
            f"({len(file_classes)}, {len(expected_attributes)})."
        )

    class_to_row = {
        normalize_name(class_name): index
        for index, class_name in enumerate(file_classes)
    }
    rows = []
    for class_name in class_names:
        key = normalize_name(class_name)
        if key not in class_to_row:
            raise KeyError(f"Class {class_name!r} is missing from {path!r}.")
        rows.append(matrix[class_to_row[key]])

    return torch.stack(rows, dim=0).clamp(0.0, 1.0)


# Compatibility name used by the original research script.
save_attr_prior_json = save_attribute_prior
