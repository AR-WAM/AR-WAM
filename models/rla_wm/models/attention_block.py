import torch
import torch.nn as nn



class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        cond_channels: int = -1,
        use_condition: bool = False,
    ):
        super().__init__()
        self.channels = channels

        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        if use_condition:
            self.cond_proj = nn.Linear(cond_channels, channels)
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels)
            )
        else:
            self.cond_proj = None
            self.modulation = None
        self.use_condition = use_condition

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None=None,
    ) -> torch.Tensor:
        if self.use_condition:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation(self.cond_proj(cond)).chunk(6, dim=-1)
            )
            if scale_msa.ndim == 2:
                scale_msa = scale_msa[:,  None, :]
                shift_msa = shift_msa[:,  None, :]
                gate_msa = gate_msa[:,  None, :]
                scale_mlp = scale_mlp[:,  None, :]
                shift_mlp = shift_mlp[:,  None, :]
                gate_mlp = gate_mlp[:,  None, :]

            h = self.norm1(x)
            h = h * (1.0 + scale_msa) + shift_msa
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h * gate_msa
            h2 = self.norm2(x)
            h2 = h2 * (1.0 + scale_mlp) + shift_mlp
            h2 = self.mlp(h2)
            x = x + h2 * gate_mlp
        else:
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h 
            h2 = self.norm2(x)
            h2 = self.mlp(h2)
            x = x + h2 
        return x

