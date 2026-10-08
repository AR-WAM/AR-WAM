import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .atomic_condition import encode_condition_ids
from .condition_decoder import MaskQueryConditionDecoder
from .mask_decoder import DinoPromptMaskDecoder


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Standard Transformer SinCos positional encoding.
    Returns: (1, length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)


# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return (
        x.float() * (1 + scale.float().unsqueeze(1))
        + shift.float().unsqueeze(1)
    ).to(dtype=x.dtype)


def _stable_layer_norm(module: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        value.float(),
        module.normalized_shape,
        module.weight.float() if module.weight is not None else None,
        module.bias.float() if module.bias is not None else None,
        module.eps,
    ).to(dtype=value.dtype)


def _autocast_disabled(device):
    """Disable CUDA autocast for an explicitly FP32 numerical-safety island."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _require_finite(name, value):
    finite = torch.isfinite(value).all()
    if value.is_cuda:
        # Queue the diagnostic on the CUDA stream without synchronizing every
        # block back to Python.  A failed assertion still reports its tensor
        # boundary name before the optimizer is allowed to advance.
        torch._assert_async(finite, f"non-finite {name}")
        return
    if not finite:
        maximum = torch.nan_to_num(value.detach().float()).abs().max().item()
        raise FloatingPointError(f"non-finite {name}; finite_abs_max={maximum:.6g}")


# t=1 in flow matching is the clean / fully-denoised endpoint (x_t = x1).
PREFIX_CLEAN_T = 1.0


class MultiLayerConcatFusion(nn.Module):
    """
    Multi-layer DINO feature fusion (concat mode):
      - Each layer is normalized by its own LayerNorm first (feature norms differ
        a lot across DINOv3 layers; shallow layers would be drowned out otherwise)
      - Concatenate token-wise along the feature dim -> (B, N, L*feat_dim)
      - Linear projection down to -> (B, N, out_dim)
    The output retains dense CLS / register / patch tokens for the DiT.
    """
    def __init__(self, feat_dim, num_layers, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        in_dim = feat_dim * num_layers
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, feats_list):
        """feats_list: List[(B, N, feat_dim)] (all layers must share the same N)"""
        assert len(feats_list) == self.num_layers, \
            f"MultiLayerConcatFusion expects {self.num_layers} layers, got {len(feats_list)}"
        target_dtype = self.proj.weight.dtype
        normalized = []
        for norm, feature in zip(self.layer_norms, feats_list):
            weight = norm.weight
            normalized.append(F.layer_norm(
                feature.float(),
                norm.normalized_shape,
                weight.float() if weight is not None else None,
                norm.bias.float() if norm.bias is not None else None,
                norm.eps,
            ).to(dtype=target_dtype))
        feats_list = normalized
        x = torch.cat(feats_list, dim=-1)   # (B, N, L*feat_dim)
        return self.proj(x)                 # (B, N, out_dim)


def _validate_future_frame_mask(mask, batch, horizon, device):
    if mask is None:
        raise ValueError("future frame mask is required for masked future loss")
    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != (batch, horizon):
        raise ValueError(f"future frame mask must be {(batch, horizon)}, got {tuple(mask.shape)}")
    return mask


class DiTBlock(nn.Module):
    """DiT block with explicit QKV and a clean-prefix teacher prefill."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = dropout
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.time_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.task_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.xavier_uniform_(self.task_adaLN_modulation[-1].weight, gain=0.01)
        nn.init.zeros_(self.task_adaLN_modulation[-1].bias)
        nn.init.constant_(self.time_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.time_adaLN_modulation[-1].bias, 0)

    def _ada(self, time_emb, task_emb):
        values = (
            self.time_adaLN_modulation(time_emb)
            + self.task_adaLN_modulation(task_emb)
        )
        chunks = list(values.chunk(6, dim=1))
        for index in (1, 2, 4, 5):
            chunks[index] = chunks[index].tanh()
        return tuple(chunks)

    def _shape_qkv(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, S, D
        return qkv[0], qkv[1], qkv[2]

    def _attend(self, q, k, v, split_prefix_len=None):
        # q,k,v: (B, H, Sq/Sk, D).  The joint training path supplies the
        # structural prefix length directly, avoiding both the masked-off
        # prefix→suffix score block and any GPU→CPU mask inspection.
        B, H, Sq, D = q.shape
        dropout_p = self.attn_drop if self.training and self.attn_drop > 0 else 0.0
        if split_prefix_len is not None:
            prefix_len = int(split_prefix_len)
            if not 0 < prefix_len < Sq or k.shape[2] != Sq:
                raise ValueError(
                    f"invalid split_prefix_len={prefix_len} for q={Sq}, k={k.shape[2]}"
                )
            prefix_out = F.scaled_dot_product_attention(
                q[:, :, :prefix_len],
                k[:, :, :prefix_len],
                v[:, :, :prefix_len],
                dropout_p=dropout_p,
                is_causal=False,
            )
            suffix_out = F.scaled_dot_product_attention(
                q[:, :, prefix_len:],
                k,
                v,
                dropout_p=dropout_p,
                is_causal=False,
            )
            out = torch.cat([prefix_out, suffix_out], dim=2)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p, is_causal=False,
            )
        out = out.transpose(1, 2).reshape(B, Sq, H * D)
        return self.attn_out(out)


    def prefill(self, x, time_emb, task_emb):
        """Run the clean prefix block for the future teacher."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada(
            time_emb, task_emb
        )
        x_norm = modulate(_stable_layer_norm(self.norm1, x), shift_msa, scale_msa)
        q, k, v = self._shape_qkv(x_norm)
        x = x + gate_msa.unsqueeze(1) * self._attend(q, k, v)
        x_norm = modulate(_stable_layer_norm(self.norm2, x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

class VLAModel(nn.Module):
    """Dense DINO + EE-state AdaLN model for joint action/RLA/mask training."""
    def __init__(self,
                 action_dim=14,
                 proprio_dim=16,
                 hidden_dim=768,
                 num_heads=8,
                 depth=12,
                 action_len=32,
                 proprio_len=1,
                 dino_feat_dims=(1024,),
                 concat_out_dim=None,
                 num_dino_registers=4,
                 rla_tokens_per_frame=32,
                 rla_token_dim=64,
                 mask_decoder_dim=256,
                 mask_decoder_layers=4,
                 mask_decoder_heads=8,
                 dino_patch_grid=(15, 20),
                 num_mask_channels=6,
                 future_mask_bottleneck_dim=64,
                 condition_embedding_init_std=2.0,
                 state_dropout_prob=0.0,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_len = action_len
        self.proprio_len = proprio_len
        self.state_dropout_prob = float(state_dropout_prob)
        if not 0.0 <= self.state_dropout_prob <= 1.0:
            raise ValueError("state_dropout_prob must be in [0, 1]")
        self.num_dino_layers = len(dino_feat_dims)
        self.rla_tokens_per_frame = int(rla_tokens_per_frame)
        self.rla_token_dim = int(rla_token_dim)
        self.num_mask_queries = 4
        self.dino_feature_dim = int(dino_feat_dims[0])
        self.num_mask_channels = int(num_mask_channels)
        self.future_mask_bottleneck_dim = int(future_mask_bottleneck_dim)
        self.latent_prediction_type = "z"
        if self.future_mask_bottleneck_dim <= 0:
            raise ValueError("future_mask_bottleneck_dim must be positive")
        self.num_dino_registers = int(num_dino_registers)
        self.dino_patch_grid = tuple(int(value) for value in dino_patch_grid)
        self.mask_queries = nn.Parameter(
            torch.randn(1, self.num_mask_queries, hidden_dim) * 0.02
        )
        from .bbox_condition import BBoxCondition
        self.bbox_cond = BBoxCondition(
            hidden_dim=hidden_dim,
            embedding_init_std=condition_embedding_init_std,
        )

        # --- 1. Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # --- 2. Projections & PosEmb ---
        def input_mlp(input_dim):
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.proprio_mlp = nn.Linear(proprio_dim, hidden_dim)
        self.state_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.state_mask_token, std=0.02)
        self.mask_query_mlp = input_mlp(hidden_dim)
        self.rla_mlp = input_mlp(self.rla_token_dim)
        # The teacher encodes first-frame query states into the bottleneck; the
        # student decodes denoised bottleneck states back to DiT width. Separate
        # weights keep the target and prediction spaces independently learnable.
        bottleneck = self.future_mask_bottleneck_dim
        self.future_mask_teacher_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, bottleneck),
        )
        self.future_mask_student_proj = nn.Sequential(
            nn.LayerNorm(bottleneck), nn.Linear(bottleneck, hidden_dim),
        )
        self.future_mask_mlp = input_mlp(hidden_dim)

        self.register_buffer('action_pos_emb', get_1d_sincos_pos_embed(hidden_dim, action_len))
        self.register_buffer('proprio_pos_emb', get_1d_sincos_pos_embed(hidden_dim, proprio_len))
        self.register_buffer('mask_query_pos_emb', get_1d_sincos_pos_embed(hidden_dim, self.num_mask_queries))
        self.register_buffer(
            'rla_pos_emb',
            get_1d_sincos_pos_embed(hidden_dim, self.rla_tokens_per_frame),
        )
        self.register_buffer(
            'future_mask_pos_emb',
            get_1d_sincos_pos_embed(hidden_dim, self.num_mask_queries),
        )

        self.type_emb_action = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_rla = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_future_mask = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.type_emb_action, std=0.02)
        nn.init.normal_(self.type_emb_rla, std=0.02)
        nn.init.normal_(self.type_emb_future_mask, std=0.02)

        # --- 4. Dense DINO fusion ---
        fusion_out_dim = concat_out_dim if concat_out_dim is not None else dino_feat_dims[0]
        self.concat_fusion = MultiLayerConcatFusion(
            feat_dim=dino_feat_dims[0],
            num_layers=self.num_dino_layers,
            out_dim=fusion_out_dim,
        )
        # CLS / DINO-registers / patches each get their own MLP into DiT width.
        self.patch_mlp = nn.Sequential(
            nn.LayerNorm(fusion_out_dim),
            nn.Linear(fusion_out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls_mlp = nn.Sequential(
            nn.LayerNorm(fusion_out_dim),
            nn.Linear(fusion_out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dino_register_mlp = nn.Sequential(
            nn.LayerNorm(fusion_out_dim),
            nn.Linear(fusion_out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- 5. Core Transformer Blocks ---
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])

        # --- 6. Output Head ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        self.rla_output_proj = nn.Linear(hidden_dim, self.rla_token_dim)
        self.future_mask_output_proj = nn.Linear(hidden_dim, bottleneck)

        # --- 7. Current-frame mask decoder ---
        self.mask_decoder = DinoPromptMaskDecoder(
            dino_dim=self.dino_feature_dim,
            prompt_dim=hidden_dim,
            hidden_dim=int(mask_decoder_dim),
            num_layers=int(mask_decoder_layers),
            num_heads=int(mask_decoder_heads),
            num_register_tokens=self.num_dino_registers,
            patch_grid=tuple(dino_patch_grid),
            num_mask_channels=self.num_mask_channels,
        )
        self.condition_decoder = MaskQueryConditionDecoder(
            hidden_dim=hidden_dim,
            num_queries=self.num_mask_queries,
        )

    def _condition_embeddings(self, t, valid_l, valid_r, B, device,
                              condition_ids=None, bbox_l=None, bbox_r=None,
                              kind_l=None, kind_r=None):
        t = t.to(dtype=self.time_mlp[1].weight.dtype)
        time_emb = self.time_mlp(t)
        if condition_ids is None:
            raise ValueError("atomic condition IDs are required")
        condition_ids = encode_condition_ids(
            condition_ids, batch_size=B, device=device
        )
        # The discrete task condition (embedding tables, bbox slots, fusion)
        # stays in FP32 even under a CUDA autocast(bfloat16) training step.
        # Cast back to the model parameter dtype for downstream modulation.
        with _autocast_disabled(device):
            task_emb = self.bbox_cond.non_time_condition(
                valid_l=valid_l,
                valid_r=valid_r,
                condition_ids=condition_ids,
                bbox_l=bbox_l,
                bbox_r=bbox_r,
                kind_l=kind_l,
                kind_r=kind_r,
            ).to(dtype=self.time_mlp[1].weight.dtype)
        return time_emb, task_emb

    def _encode_visual(self, dino_features_list):
        assert len(dino_features_list) == self.num_dino_layers, \
            f"Expected {self.num_dino_layers} DINO feature layers, got {len(dino_features_list)}"
        fused = self.concat_fusion(dino_features_list)  # (B, 1+R+P, Cf)
        r = self.num_dino_registers
        cls_t = self.cls_mlp(fused[:, :1])
        reg_t = self.dino_register_mlp(fused[:, 1:1 + r])
        pat_t = self.patch_mlp(fused[:, 1 + r:])
        return torch.cat([cls_t, reg_t, pat_t], dim=1)

    def _encode_proprio(self, qpos_history, *, apply_dropout: bool):
        """Project EE state and optionally replace the whole token per sample."""
        state = self.proprio_mlp(
            qpos_history.to(dtype=self.proprio_mlp.weight.dtype)
        )
        batch = state.shape[0]
        dropout_mask = torch.zeros(batch, device=state.device, dtype=torch.bool)
        if apply_dropout and self.training and self.state_dropout_prob > 0.0:
            dropout_mask = torch.rand(batch, device=state.device) < self.state_dropout_prob
            mask_token = self.state_mask_token.expand(batch, state.shape[1], -1)
            state = torch.where(dropout_mask[:, None, None], mask_token, state)
        state = state + self.proprio_pos_emb[:, :state.shape[1]].to(
            device=state.device, dtype=state.dtype
        )
        return state, dropout_mask

    def _build_prefix(self, B, device, qpos_history, visual_tokens,
                      apply_state_dropout=False):
        if qpos_history is None:
            raise ValueError("the training prefix requires measured proprioception")
        if qpos_history.dim() == 2:
            qpos_history = qpos_history.unsqueeze(1)
        if qpos_history.shape[1] != 1:
            raise ValueError("the training prefix requires exactly one proprio token")
        vis = visual_tokens
        if vis is None or vis.shape[1] != 1 + self.num_dino_registers + self.dino_patch_grid[0] * self.dino_patch_grid[1]:
            raise ValueError("the training prefix requires dense CLS/register/patch DINO tokens")
        state, state_dropout_mask = self._encode_proprio(
            qpos_history, apply_dropout=apply_state_dropout
        )
        prompts = self.mask_query_mlp(
            self.mask_queries.expand(B, -1, -1).to(dtype=self.mask_query_mlp[0].weight.dtype)
        )
        prompts = prompts + self.mask_query_pos_emb.to(device=device, dtype=prompts.dtype)
        return torch.cat([state, vis, prompts], dim=1), state_dropout_mask

    def _build_suffix(self, noisy_rla, noisy_future_mask, noisy_actions):
        if noisy_rla is None or noisy_future_mask is None:
            raise ValueError("joint suffix requires noisy RLA and future-mask tokens")
        if noisy_rla.shape[1:] != (self.rla_tokens_per_frame, self.rla_token_dim):
            raise ValueError("noisy_rla must have shape (B,32,64)")
        if noisy_future_mask.shape[1:] != (
            self.num_mask_queries, self.future_mask_bottleneck_dim
        ):
            raise ValueError(
                "noisy_future_mask must have shape (B,4,bottleneck)"
            )
        rla = self.rla_mlp(noisy_rla.to(dtype=self.rla_mlp[0].weight.dtype))
        rla = rla + self.rla_pos_emb.to(device=rla.device, dtype=rla.dtype) + self.type_emb_rla
        projected = self.future_mask_student_proj(
            noisy_future_mask.to(
                dtype=self.future_mask_student_proj[0].weight.dtype
            )
        )
        future_mask = self.future_mask_mlp(projected)
        future_mask = (
            future_mask
            + self.future_mask_pos_emb.to(
                device=future_mask.device, dtype=future_mask.dtype
            )
            + self.type_emb_future_mask
        )
        noisy_actions = noisy_actions.to(dtype=self.action_proj.weight.dtype)
        action = self.action_proj(noisy_actions) + \
                 self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                 self.type_emb_action
        return torch.cat([rla, future_mask, action], dim=1)

    def prefill_prefix(
        self,
        dino_features_list,
        qpos_history,
        condition_ids,
        bbox_l, bbox_r,
        valid_l, valid_r,
        kind_l, kind_r,
    ):
        """Run the future teacher prefix at clean t=1 for its mask bottleneck."""
        vis = self._encode_visual(dino_features_list)
        B, device, dtype = vis.shape[0], vis.device, vis.dtype
        prefix, _ = self._build_prefix(
            B, device, qpos_history, vis, apply_state_dropout=False,
        )
        t_clean = torch.full((B,), PREFIX_CLEAN_T, device=device, dtype=dtype)
        time_emb, task_emb = self._condition_embeddings(
            t_clean, valid_l, valid_r, B, device, condition_ids=condition_ids,
            bbox_l=bbox_l, bbox_r=bbox_r, kind_l=kind_l, kind_r=kind_r,
        )
        x = prefix
        for block in self.blocks:
            x = block.prefill(x, time_emb, task_emb)
        x_norm = self.final_norm(x)
        mask_prompt_start = 1 + vis.shape[1]
        mask_prompt_hidden = x_norm[:, mask_prompt_start:mask_prompt_start + self.num_mask_queries]
        future_mask_bottleneck = self.future_mask_teacher_proj(
            mask_prompt_hidden.to(
                dtype=self.future_mask_teacher_proj[0].weight.dtype
            )
        )
        return {"future_mask_bottleneck": future_mask_bottleneck}

    def forward(self,
                t,
                noisy_actions,
                noisy_rla=None,
                noisy_future_mask=None,
                qpos_history=None,
                dino_features_list=None,
                condition_ids=None,
                bbox_l=None, bbox_r=None,
                valid_l=None, valid_r=None,
                kind_l=None, kind_r=None,
                mask_decoder_dino=None,
                mask_output_size=None,
                ):
        """Joint training forward: prefix (t=1 adaLN) + suffix (step t), prefix-suffix mask.

        Action head is LiLa-WAM: Linear on the action-token slice → velocity field.
        """
        B = noisy_actions.shape[0]
        device = noisy_actions.device
        visual_tokens = self._encode_visual(dino_features_list)
        prefix, state_dropout_mask = self._build_prefix(
            B, device, qpos_history, visual_tokens,
            apply_state_dropout=True,
        )
        suffix = self._build_suffix(noisy_rla, noisy_future_mask, noisy_actions)
        P = prefix.shape[1]
        R = self.rla_tokens_per_frame
        M = self.num_mask_queries
        A = self.action_len
        suffix_len = R + M + A
        visual_len = visual_tokens.shape[1]

        t_clean = torch.full((B,), PREFIX_CLEAN_T, device=device, dtype=t.dtype)
        time_emb_prefix, task_emb = self._condition_embeddings(
            t_clean, valid_l, valid_r, B, device, condition_ids=condition_ids,
            bbox_l=bbox_l, bbox_r=bbox_r, kind_l=kind_l, kind_r=kind_r,
        )
        time_emb_suffix = self.time_mlp(
            t.to(dtype=self.time_mlp[1].weight.dtype)
        )

        _require_finite("visual_tokens", visual_tokens)
        _require_finite("prefix_input", prefix)
        _require_finite("suffix_input", suffix)
        _require_finite("time_prefix", time_emb_prefix)
        _require_finite("time_suffix", time_emb_suffix)
        _require_finite("task_embedding", task_emb)
        x = torch.cat([prefix, suffix], dim=1)
        for block_index, block in enumerate(self.blocks):
            ada_p = torch.cat(block._ada(time_emb_prefix, task_emb), dim=1)
            ada_s = torch.cat(block._ada(time_emb_suffix, task_emb), dim=1)
            _require_finite(f"block_{block_index}_ada_prefix", ada_p)
            _require_finite(f"block_{block_index}_ada_suffix", ada_s)
            x = _block_two_times(block, x, P, ada_p, ada_s)
            _require_finite(f"block_{block_index}_output", x)

        x = self.final_norm(x)
        suffix_start = P
        pred_rla = self.rla_output_proj(x[:, suffix_start:suffix_start + R])
        pred_future_mask = self.future_mask_output_proj(
            x[:, suffix_start + R:suffix_start + R + M]
        )
        final_pred = self.output_proj(
            x[:, suffix_start + R + M:suffix_start + R + M + A]
        )
        mask_prompt_start = 1 + visual_len
        mask_prompt_hidden = x[:, mask_prompt_start:mask_prompt_start + self.num_mask_queries]
        condition_logits, pred_run_endpoint_state = (
            self.condition_decoder.decode_all(mask_prompt_hidden)
        )
        result = {
            "pred_rla": pred_rla,
            "pred_future_mask": pred_future_mask,
            "final_pred": final_pred,
            "condition_logits": condition_logits,
            "pred_run_endpoint_state": pred_run_endpoint_state,
        }
        if mask_decoder_dino is not None:
            result["pred_mask"] = self.mask_decoder(
                mask_decoder_dino,
                mask_prompt_hidden,
                output_size=mask_output_size,
            )
        return result



def _block_two_times(block: DiTBlock, x, prefix_len, ada_prefix, ada_suffix):
    """One DiT block with different adaLN for prefix vs suffix tokens."""
    def _modulate_split(xn, ada):
        s_msa, c_msa, g_msa, s_mlp, c_mlp, g_mlp = ada.chunk(6, dim=1)
        return s_msa, c_msa, g_msa, s_mlp, c_mlp, g_mlp

    sp = _modulate_split(x, ada_prefix)
    ss = _modulate_split(x, ada_suffix)
    # attention
    xn = _stable_layer_norm(block.norm1, x)
    xn_p = modulate(xn[:, :prefix_len], sp[0], sp[1])
    xn_s = modulate(xn[:, prefix_len:], ss[0], ss[1])
    xn = torch.cat([xn_p, xn_s], dim=1)
    q, k, v = block._shape_qkv(xn)
    attn = block._attend(q, k, v, split_prefix_len=prefix_len)
    gate = torch.cat([
        sp[2].unsqueeze(1).expand(-1, prefix_len, -1),
        ss[2].unsqueeze(1).expand(-1, x.shape[1] - prefix_len, -1),
    ], dim=1)
    x = x + gate * attn
    xn = _stable_layer_norm(block.norm2, x)
    xn_p = modulate(xn[:, :prefix_len], sp[3], sp[4])
    xn_s = modulate(xn[:, prefix_len:], ss[3], ss[4])
    xn = torch.cat([xn_p, xn_s], dim=1)
    gate = torch.cat([
        sp[5].unsqueeze(1).expand(-1, prefix_len, -1),
        ss[5].unsqueeze(1).expand(-1, x.shape[1] - prefix_len, -1),
    ], dim=1)
    x = x + gate * block.mlp(xn)
    return x


def calc_flow_matching_loss(
    model,
    x1,
    dino_features_list,
    qpos_history,
    condition_ids=None,
    action_mask=None,
    mask_decoder_dino=None,
    visual_mask=None,
    visual_mask_valid=None,
    future_mask=None,
    future_rla_target=None,
    future_mask_target=None,
    lambda_action=5.0,
    lambda_rla=0.0,
    lambda_mask=0.0,
    lambda_mask_token=0.0,
    lambda_condition_decode=0.0,
    lambda_run_endpoint_state=0.0,
    run_endpoint_state_target=None,
    condition_skill=None,
    condition_participants=None,
    condition_style=None,
    condition_valid_l=None,
    condition_valid_r=None,
    condition_kind_l=None,
    condition_kind_r=None,
    time_sampler="uniform",
    time_mu=0.0,
    time_sigma=1.0,
    bbox_l=None, bbox_r=None,
    valid_l=None, valid_r=None,
    kind_l=None, kind_r=None,
):
    """
    Joint action flow, RLA/mask-bottleneck, and current-condition supervision.
    """
    device = x1.device
    bs = x1.shape[0]

    # 1. Sample noise x0
    x0 = torch.randn_like(x1)

    # 2. Sample timestep t
    if time_sampler == "uniform":
        t = torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        normal_samples = torch.randn(bs, device=device)
        normal_samples = normal_samples * time_sigma + time_mu
        t = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")

    # 3. Interpolate
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1

    # 4. Target vector fields
    target_v = x1 - x0
    if future_rla_target is None or future_mask_target is None:
        raise ValueError("joint flow requires RLA and future-mask targets")
    rla_x1 = torch.as_tensor(future_rla_target, device=device, dtype=x1.dtype)
    mask_x1 = torch.as_tensor(
        future_mask_target, device=device, dtype=x1.dtype
    ).detach()
    if rla_x1.ndim == 4 and rla_x1.shape[1] == 1:
        rla_x1 = rla_x1[:, 0]
    if mask_x1.ndim == 4 and mask_x1.shape[1] == 1:
        mask_x1 = mask_x1[:, 0]
    model_contract = getattr(model, "module", model)
    expected_rla = (
        bs, model_contract.rla_tokens_per_frame, model_contract.rla_token_dim
    )
    expected_mask = (
        bs, model_contract.num_mask_queries,
        model_contract.future_mask_bottleneck_dim,
    )
    if tuple(rla_x1.shape) != expected_rla:
        raise ValueError(f"future_rla_target must have shape {expected_rla}")
    if tuple(mask_x1.shape) != expected_mask:
        raise ValueError(f"future_mask_target must have shape {expected_mask}")
    rla_x0 = torch.randn_like(rla_x1)
    mask_x0 = torch.randn_like(mask_x1)
    rla_x_t = (1 - t_expand) * rla_x0 + t_expand * rla_x1
    mask_x_t = (1 - t_expand) * mask_x0 + t_expand * mask_x1
    target_rla = rla_x1
    target_mask = mask_x1

    # 5. Forward
    preds = model(t,
                  noisy_actions=x_t,
                  noisy_rla=rla_x_t,
                  noisy_future_mask=mask_x_t,
                  dino_features_list=dino_features_list,
                  condition_ids=condition_ids,
                  qpos_history=qpos_history,
                  bbox_l=bbox_l, bbox_r=bbox_r,
                  valid_l=valid_l, valid_r=valid_r,
                  kind_l=kind_l, kind_r=kind_r,
                  mask_decoder_dino=mask_decoder_dino,
                  mask_output_size=tuple(visual_mask.shape[-2:]) if visual_mask is not None else None)

    pred_v_final = preds["final_pred"]

    # ==================== Final Loss ====================
    loss_final_unreduced = F.mse_loss(
        pred_v_final.float(), target_v.float(), reduction='none'
    )
    if action_mask is not None:
        mask = torch.as_tensor(action_mask, device=device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != x1.shape[:2]:
            raise ValueError(f"action_mask shape {tuple(mask.shape)} does not match actions {tuple(x1.shape[:2])}")
        valid = mask.unsqueeze(-1).expand_as(loss_final_unreduced)
        denom = valid.sum(dim=(1, 2)).clamp_min(1)
        loss_final_per_sample = (loss_final_unreduced * valid).sum(dim=(1, 2)) / denom
    else:
        loss_final_per_sample = torch.mean(loss_final_unreduced, dim=(1, 2))
    loss_mse = torch.mean(loss_final_per_sample)

    valid_future = _validate_future_frame_mask(future_mask, bs, 1, device)[:, 0]
    valid_future_f = valid_future.float()
    valid_future_count = valid_future_f.sum().clamp_min(1.0)
    rla_per_sample = F.mse_loss(
        preds["pred_rla"].float(), target_rla.float(), reduction="none"
    ).mean(dim=(1, 2))
    mask_token_per_sample = F.mse_loss(
        preds["pred_future_mask"].float(), target_mask.float(), reduction="none"
    ).mean(dim=(1, 2))
    loss_rla = (rla_per_sample * valid_future_f).sum() / valid_future_count
    loss_mask_token = (
        mask_token_per_sample * valid_future_f
    ).sum() / valid_future_count

    loss_condition_decode = loss_mse * 0.0
    if lambda_condition_decode and lambda_condition_decode != 0:
        logits = preds.get("condition_logits")
        if logits is None:
            raise ValueError("condition decode loss requires condition_logits")
        targets = {
            "skill": condition_skill,
            "participants": condition_participants,
            "style": condition_style,
            "valid_l": condition_valid_l,
            "valid_r": condition_valid_r,
            "kind_l": condition_kind_l,
            "kind_r": condition_kind_r,
        }
        losses = []
        for field, target in targets.items():
            if target is None:
                raise ValueError(f"condition decode loss requires {field} target")
            target = torch.as_tensor(target, device=device, dtype=torch.long).reshape(bs)
            losses.append(F.cross_entropy(logits[field].float(), target))
        loss_condition_decode = torch.stack(losses).mean()

    loss_run_endpoint_state = loss_mse * 0.0
    if run_endpoint_state_target is not None:
        prediction = preds.get("pred_run_endpoint_state")
        if prediction is None:
            raise ValueError(
                "run endpoint state loss requires pred_run_endpoint_state"
            )
        target = torch.as_tensor(
            run_endpoint_state_target, device=device, dtype=torch.float32
        ).detach()
        if tuple(target.shape) != (bs, 16):
            raise ValueError(
                f"run_endpoint_state_target must have shape {(bs, 16)}, "
                f"got {tuple(target.shape)}"
            )
        if tuple(prediction.shape) != (bs, 16):
            raise ValueError(
                f"pred_run_endpoint_state must have shape {(bs, 16)}, "
                f"got {tuple(prediction.shape)}"
            )
        error = (prediction.float() - target).square()
        loss_run_endpoint_state = error.mean()
    elif lambda_run_endpoint_state and lambda_run_endpoint_state != 0:
        raise ValueError(
            "run endpoint state loss requires run_endpoint_state_target"
        )

    loss_mask = loss_mse * 0.0
    if lambda_mask and lambda_mask != 0:
        if visual_mask is None or visual_mask_valid is None:
            raise ValueError("mask loss requires current visual_mask and visual_mask_valid")
        if "pred_mask" not in preds:
            raise ValueError("mask loss requires current pred_mask output")
        target_mask = (torch.as_tensor(visual_mask, device=device).float() > 0).float()
        if target_mask.ndim != 4:
            raise ValueError("visual_mask must have shape (B,C,Hpx,Wpx)")
        if preds["pred_mask"].shape != target_mask.shape:
            raise ValueError(
                f"pred_mask shape {tuple(preds['pred_mask'].shape)} does not match "
                f"visual_mask shape {tuple(target_mask.shape)}"
            )
        valid_mask = torch.as_tensor(visual_mask_valid, device=device, dtype=torch.bool)
        if valid_mask.shape != (bs,):
            raise ValueError("visual_mask_valid must have shape (B,)")
        per_sample = F.binary_cross_entropy_with_logits(
            preds["pred_mask"].float(),
            target_mask,
            reduction="none",
        ).mean(dim=(1, 2, 3))
        valid_mask_f = valid_mask.float()
        loss_mask = (
            per_sample * valid_mask_f
        ).sum() / valid_mask_f.sum().clamp_min(1.0)

    loss = (
        lambda_action * loss_mse
        + lambda_mask * loss_mask
        + lambda_rla * loss_rla
        + lambda_mask_token * loss_mask_token
        + lambda_condition_decode * loss_condition_decode
        + lambda_run_endpoint_state * loss_run_endpoint_state
    )
    _require_finite("joint_loss", loss)

    return loss, {
        "loss_action": loss_mse.detach(),
        "loss_rla": loss_rla.detach(),
        "loss_mask_token": loss_mask_token.detach(),
        "loss_mask": loss_mask.detach(),
        "loss_condition_decode": loss_condition_decode.detach(),
        "loss_run_endpoint_state": loss_run_endpoint_state.detach(),
        "loss": loss.detach(),
    }
