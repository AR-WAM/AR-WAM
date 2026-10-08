"""Four-prompt decoder for current-frame RoboTwin masks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoPromptMaskDecoder(nn.Module):
    """Decode dense last-layer DINO tokens conditioned by four DiT prompts."""

    def __init__(
        self,
        dino_dim: int = 1024,
        prompt_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_register_tokens: int = 4,
        patch_grid: tuple[int, int] = (15, 20),
        num_mask_channels: int = 6,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("mask decoder hidden_dim must be divisible by num_heads")
        self.dino_dim = int(dino_dim)
        self.prompt_dim = int(prompt_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.patch_grid = tuple(int(value) for value in patch_grid)
        self.num_mask_channels = int(num_mask_channels)
        if self.num_mask_channels <= 0:
            raise ValueError("num_mask_channels must be positive")
        if len(self.patch_grid) != 2 or any(value <= 0 for value in self.patch_grid):
            raise ValueError("patch_grid must contain two positive dimensions")
        def input_mlp(input_dim):
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.cls_mlp = input_mlp(self.dino_dim)
        self.register_mlp = input_mlp(self.dino_dim)
        self.patch_mlp = input_mlp(self.dino_dim)
        self.prompt_mlp = input_mlp(self.prompt_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.patch_head = nn.Linear(self.hidden_dim, self.num_mask_channels)

    def forward(
        self,
        dense_dino: torch.Tensor,
        mask_prompts: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Return current-frame semantic mask logits with shape ``(B,C,H,W)``."""
        if dense_dino.ndim != 3 or mask_prompts.ndim != 3:
            raise ValueError("mask decoder expects DINO (B,N,D) and prompts (B,4,C)")
        batch, token_count, channels = dense_dino.shape
        if mask_prompts.shape[:2] != (batch, 4):
            raise ValueError("mask decoder requires exactly four mask prompts")
        if channels != self.dino_dim or mask_prompts.shape[-1] != self.prompt_dim:
            raise ValueError("mask decoder input width does not match its projections")
        patch_count = self.patch_grid[0] * self.patch_grid[1]
        expected_tokens = 1 + self.num_register_tokens + patch_count
        if token_count != expected_tokens:
            raise ValueError(
                f"mask decoder expects {expected_tokens} DINO tokens, got {token_count}"
            )

        decoder_dtype = self.cls_mlp[0].weight.dtype
        dense_dino = dense_dino.to(dtype=decoder_dtype)
        cls_token = self.cls_mlp(dense_dino[:, :1])
        register_tokens = self.register_mlp(
            dense_dino[:, 1:1 + self.num_register_tokens]
        )
        patch_start = 1 + self.num_register_tokens
        patch_tokens = self.patch_mlp(dense_dino[:, patch_start:])
        prompt_tokens = self.prompt_mlp(mask_prompts.to(dtype=decoder_dtype))
        tokens = self.transformer(torch.cat([
            cls_token,
            register_tokens,
            patch_tokens,
            prompt_tokens,
        ], dim=1))
        patch_tokens = tokens[:, patch_start:patch_start + patch_count]
        patch_logits = self.patch_head(self.output_norm(patch_tokens)).transpose(1, 2)
        patch_logits = patch_logits.reshape(
            batch, self.num_mask_channels, *self.patch_grid
        )
        if output_size is not None:
            if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
                raise ValueError("output_size must be two positive dimensions")
            patch_logits = F.interpolate(
                patch_logits,
                size=tuple(int(value) for value in output_size),
                mode="bilinear",
                align_corners=False,
            )
        return patch_logits


__all__ = ["DinoPromptMaskDecoder"]
