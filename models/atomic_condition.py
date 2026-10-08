"""Stable compositional vocabularies for atomic-operation conditioning."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

CONDITION_FIELDS = ("skill", "participants", "style")

# Dataset annotations use contiguous model-facing skill IDs.
VOCABS = {
    "skill": {
        "unknown": 0,
        "idle": 0,
        "grasp": 1,
        "place": 2,
        "click": 3,
        "bimanual_same": 4,
        "bimanual_two": 5,
        "handover": 6,
        "shake": 7,
        "pour": 8,
        "hang": 9,
        "pull": 10,
        "open_lid": 11,
        "switch_on": 12,
        "beat": 13,
        "scan": 14,
        "rotate": 15,
        "homing": 16,
        "postplace": 17,
        "lift": 18,
        "retract": 19,
        "pre_handover": 20,
        "prepare_skillet": 21,
        "remove": 22,
    },
    "participants": {"unknown": 0, "none": 0, "left": 1, "right": 2, "both": 3},
    "style": {
        "unknown": 0,
        "none": 0,
        "handover_L2R": 1,
        "handover_R2L": 2,
        "place_left_of_reference": 3,
        "place_right_of_reference": 4,
    },
}
VOCAB_SIZES = {field: max(values.values()) + 1 for field, values in VOCABS.items()}


def _id_for(field: str, value: Any) -> int:
    if isinstance(value, str):
        return int(VOCABS[field].get(value, 0))
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return value if 0 <= value < VOCAB_SIZES[field] else 0


def encode_condition_ids(
    values: Mapping[str, Any] | None,
    *,
    batch_size: int,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    values = values or {}
    result = {}
    for field in CONDITION_FIELDS:
        value = values.get(field, 0)
        if isinstance(value, Sequence) and not isinstance(value, str):
            raise TypeError(f"{field} batch values must be a Tensor, not a Python sequence")
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=torch.long).reshape(-1)
            if tensor.numel() == 1 and batch_size != 1:
                tensor = tensor.expand(batch_size)
            if tensor.numel() != batch_size:
                raise ValueError(f"{field} must contain batch_size={batch_size} values")
            invalid = (tensor < 0) | (tensor >= VOCAB_SIZES[field])
            tensor = tensor.masked_fill(invalid, 0)
        else:
            tensor = torch.full((batch_size,), _id_for(field, value), dtype=torch.long, device=device)
        result[field] = tensor
    return result
