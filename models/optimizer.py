"""FP32 AdamW for bfloat16 mixed-precision policy training."""
from __future__ import annotations

from collections.abc import Iterable

import torch


class FP32AdamW:
    """Training-loop adapter around ordinary FP32 ``torch.optim.AdamW``."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        device: torch.device | None = None,
        **adamw_kwargs,
    ):
        del device  # kept for call-site compatibility; dtype is model-defined
        pairs = [(name, value) for name, value in named_parameters if value.requires_grad]
        if not pairs:
            raise ValueError("FP32AdamW requires trainable parameters")
        non_fp32 = [name for name, value in pairs if value.dtype != torch.float32]
        if non_fp32:
            raise ValueError(
                "FP32AdamW requires float32 trainable parameters; "
                f"got non-float32 parameters: {non_fp32[:4]}"
            )
        self.parameter_pairs = pairs
        self.names = [name for name, _ in pairs]
        self.model_parameters = [value for _, value in pairs]
        self.optimizer = torch.optim.AdamW(
            self.model_parameters, lr=float(lr), **adamw_kwargs
        )
        self._prepared = False
        self._all_finite = True

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self._prepared = False
        self._all_finite = True

    def backward(self, loss: torch.Tensor) -> None:
        loss.float().backward()

    def prepare_grads(self, max_norm: float) -> tuple[bool, float]:
        if self._prepared:
            raise RuntimeError("FP32 gradients were already prepared")
        gradients = [parameter for parameter in self.model_parameters if parameter.grad is not None]
        if gradients:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                gradients, float(max_norm)
            )
        else:
            grad_norm = torch.zeros((), device=self.model_parameters[0].device)
        finite = torch.isfinite(grad_norm)
        if grad_norm.is_cuda:
            # A single global norm detects every NaN/Inf gradient without one
            # Python bool synchronization per parameter.
            torch._assert_async(finite, "non-finite global gradient norm")
            all_finite = True
        else:
            all_finite = bool(finite)
        self._prepared = True
        self._all_finite = all_finite
        return all_finite, grad_norm

    def step(self) -> bool:
        if not self._prepared:
            raise RuntimeError("call prepare_grads before step")
        if self._all_finite:
            self.optimizer.step()
        stepped = self._all_finite
        self._prepared = False
        return stepped

    def state_dict(self) -> dict:
        return {
            "format": "fp32_adamw_v1",
            "parameter_names": list(self.names),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("format") != "fp32_adamw_v1":
            raise ValueError("checkpoint does not contain FP32 AdamW state")
        if list(state.get("parameter_names", ())) != self.names:
            raise ValueError("FP32 optimizer parameter names do not match")
        self.optimizer.load_state_dict(state["optimizer"])


__all__ = ["FP32AdamW"]
