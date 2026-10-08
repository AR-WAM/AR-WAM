from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_block import AttentionBlock


class SimpleTokenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_tokens: int = 16,
        zero_init: bool = False,
        token_channels: Optional[int] = None,
        norm_output_tokens: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.zero_init = zero_init
        self.norm_output_tokens = norm_output_tokens

        if num_tokens > 0: # for internal tokens
            self.tokens = nn.Parameter(torch.randn(num_tokens, model_channels) * 0.2)
        else:
            self.tokens = None
        if token_channels is not None and token_channels != model_channels: # for input tokens
            self.token_proj = nn.Linear(token_channels, model_channels)
        else:
            self.token_proj = None
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    channels=model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_init)
        if self.zero_init:
            nn.init.zeros_(self.out_layer.weight)
            nn.init.zeros_(self.out_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, Lp, C)
        h = self.input_layer(x)
       
        if tokens is not None:
            if self.token_proj is not None:
                tokens = self.token_proj(tokens)
            if self.tokens is not None:
                tokens = tokens + self.tokens.unsqueeze(0)
        else:
            if self.num_tokens > 0:
                tokens = self.tokens.unsqueeze(0).repeat(h.shape[0], 1, 1)


        if tokens is not None:
            num_tokens = tokens.shape[1]       
            h = torch.cat([tokens, h], dim=1)
        else:
            num_tokens = 0
        for block in self.blocks:
            h = block(h)

        out = self.out_layer(self.norm_out(h))
        tokens, visuals = out[:, :num_tokens], out[:, num_tokens:]
        if self.norm_output_tokens:
            tokens = F.normalize(tokens, dim=-1)
        return tokens, visuals

