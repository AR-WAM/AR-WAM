"""Encode bbox geometry and atomic conditions for AdaLN modulation.

Left/right bbox geometry is encoded independently and merged into the task
condition. Time is kept separate so each DiT block can use independent time and
task AdaLN branches.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .atomic_condition import CONDITION_FIELDS, VOCAB_SIZES

KIND_NULL = 0
KIND_GRASP = 1
KIND_PLACE = 2

def _orthogonal_rows_(parameter: torch.Tensor, std: float) -> None:
    rows, width = parameter.shape
    if rows > width:
        values = torch.empty(rows, width, device=parameter.device, dtype=parameter.dtype)
        nn.init.normal_(values)
        values = torch.nn.functional.normalize(values, dim=-1)
    else:
        values = torch.empty(width, rows, device=parameter.device, dtype=parameter.dtype)
        nn.init.orthogonal_(values)
        values = values.T
    values = values * (float(std) * math.sqrt(width))
    with torch.no_grad():
        parameter.copy_(values)


class BBoxCondition(nn.Module):
    def __init__(self, hidden_dim: int, box_hidden: int = 256,
                 embedding_init_std: float = 2.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embedding_init_std = float(embedding_init_std)
        if not math.isfinite(self.embedding_init_std) or self.embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be finite and positive")
        self.enc_grasp = nn.Sequential(
            nn.Linear(4, box_hidden),
            nn.GELU(),
            nn.Linear(box_hidden, hidden_dim),
        )
        self.enc_place = nn.Sequential(
            nn.Linear(4, box_hidden),
            nn.GELU(),
            nn.Linear(box_hidden, hidden_dim),
        )
        self.null_l = nn.Parameter(torch.empty(1, hidden_dim))
        self.null_r = nn.Parameter(torch.empty(1, hidden_dim))
        self.arm_emb_l = nn.Parameter(torch.empty(1, hidden_dim))
        self.arm_emb_r = nn.Parameter(torch.empty(1, hidden_dim))

        self.condition_emb = nn.ModuleDict({
            field: nn.Embedding(VOCAB_SIZES[field], hidden_dim)
            for field in CONDITION_FIELDS
        })
        self.valid_l_emb = nn.Embedding(2, hidden_dim)
        self.valid_r_emb = nn.Embedding(2, hidden_dim)
        # skill/participants/style + valid L/R + bbox slot L/R = seven fields.
        self.non_time_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 7),
            nn.Linear(hidden_dim * 7, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self._reset_condition_parameters()

    def _reset_condition_parameters(self) -> None:
        for table in (*self.condition_emb.values(), self.valid_l_emb, self.valid_r_emb):
            _orthogonal_rows_(table.weight, self.embedding_init_std)
        identities = torch.empty(4, self.hidden_dim)
        _orthogonal_rows_(identities, self.embedding_init_std)
        with torch.no_grad():
            self.arm_emb_l.copy_(identities[0:1])
            self.arm_emb_r.copy_(identities[1:2])
            self.null_l.copy_(identities[2:3])
            self.null_r.copy_(identities[3:4])

    def _encode_one(self, box, valid, kind, which: str):
        batch = box.shape[0]
        arm = self.arm_emb_l if which == "L" else self.arm_emb_r
        null = self.null_l if which == "L" else self.null_r
        target_dtype = self.enc_grasp[0].weight.dtype
        box = box.to(dtype=target_dtype)
        arm = arm.to(dtype=target_dtype)
        grasp = self.enc_grasp(box) + arm
        place = self.enc_place(box) + arm
        null_value = null.expand(batch, -1).to(dtype=box.dtype)
        kind = kind.long()
        valid = valid > 0.5
        use_grasp = valid & (kind == KIND_GRASP)
        use_place = valid & (kind == KIND_PLACE)
        token = null_value.clone()
        token = torch.where(use_grasp.unsqueeze(-1), grasp, token)
        token = torch.where(use_place.unsqueeze(-1), place, token)
        return token

    def encode_slots(self, bbox_l, bbox_r, valid_l, valid_r, kind_l, kind_r):
        return (
            self._encode_one(bbox_l, valid_l, kind_l, "L"),
            self._encode_one(bbox_r, valid_r, kind_r, "R"),
        )

    def non_time_condition(self, condition_ids, valid_l, valid_r,
                           bbox_l, bbox_r, kind_l, kind_r):
        first = torch.as_tensor(condition_ids[CONDITION_FIELDS[0]])
        batch = first.reshape(-1).shape[0]
        device = first.device
        bbox_l = torch.as_tensor(bbox_l, device=device)
        bbox_r = torch.as_tensor(bbox_r, device=device)
        valid_l = torch.as_tensor(valid_l, device=device)
        valid_r = torch.as_tensor(valid_r, device=device)
        kind_l = torch.as_tensor(kind_l, device=device, dtype=torch.long)
        kind_r = torch.as_tensor(kind_r, device=device, dtype=torch.long)
        slot_l, slot_r = self.encode_slots(
            bbox_l, bbox_r, valid_l, valid_r, kind_l, kind_r,
        )
        parts = []
        for field in CONDITION_FIELDS:
            ids = torch.as_tensor(condition_ids[field], device=device).long()
            if ids.ndim != 1 or ids.shape[0] != batch:
                raise ValueError(f"{field} condition IDs must have shape ({batch},)")
            in_range = ((ids >= 0) & (ids < VOCAB_SIZES[field])).all()
            message = (
                f"{field} condition ID is outside vocabulary range "
                f"[0, {VOCAB_SIZES[field] - 1}]"
            )
            if ids.is_cuda:
                torch._assert_async(in_range, message)
            elif not in_range:
                raise ValueError(message)
            parts.append(self.condition_emb[field](ids))
        parts.extend([
            self.valid_l_emb((valid_l > 0.5).long()),
            self.valid_r_emb((valid_r > 0.5).long()),
            slot_l,
            slot_r,
        ])
        dtype = self.non_time_fusion[0].weight.dtype
        fused = self.non_time_fusion(
            torch.cat([part.to(dtype=dtype) for part in parts], dim=-1)
        )
        return torch.nn.functional.normalize(fused.float(), dim=-1).to(dtype=dtype)
