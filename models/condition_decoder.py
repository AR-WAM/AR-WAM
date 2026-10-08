"""Decode non-geometric atomic metadata from final mask-query states."""
from __future__ import annotations

import torch
import torch.nn as nn

from .atomic_condition import VOCAB_SIZES


CONDITION_DECODE_SIZES = {
    "skill": VOCAB_SIZES["skill"],
    "participants": VOCAB_SIZES["participants"],
    "style": VOCAB_SIZES["style"],
    "valid_l": 2,
    "valid_r": 2,
    "kind_l": 3,
    "kind_r": 3,
}


class MaskQueryConditionDecoder(nn.Module):
    """Classify all non-bbox condition fields from four mask-query states."""

    def __init__(self, hidden_dim: int, num_queries: int = 4):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * self.num_queries),
            nn.Linear(self.hidden_dim * self.num_queries, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.heads = nn.ModuleDict({
            field: nn.Linear(self.hidden_dim, classes)
            for field, classes in CONDITION_DECODE_SIZES.items()
        })
        self.run_endpoint_state_head = nn.Linear(self.hidden_dim, 16)

    def _fuse(self, mask_prompts: torch.Tensor) -> torch.Tensor:
        if mask_prompts.ndim != 3 or mask_prompts.shape[1:] != (
            self.num_queries, self.hidden_dim
        ):
            raise ValueError(
                f"condition decoder expects (B,{self.num_queries},{self.hidden_dim})"
            )
        return self.fusion(mask_prompts.flatten(1))

    def decode_all(
        self, mask_prompts: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Decode current labels and the absolute normalized run-end EE state."""
        fused = self._fuse(mask_prompts)
        logits = {field: head(fused) for field, head in self.heads.items()}
        return logits, self.run_endpoint_state_head(fused)

    def forward(self, mask_prompts: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, _ = self.decode_all(mask_prompts)
        return logits


__all__ = ["CONDITION_DECODE_SIZES", "MaskQueryConditionDecoder"]
