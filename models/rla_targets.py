"""Frozen RLA encoder targets from DINO patch residuals."""
from pathlib import Path

import torch
import yaml

from .rla_wm.models.simple_token_transformer import SimpleTokenTransformer


class FrozenRLAAdapter:
    def __init__(self, work_dir, *, device="cpu",
                 checkpoint_step=150000):
        if work_dir is None or checkpoint_step is None:
            raise ValueError("work_dir and checkpoint_step are required for the frozen encoder")
        self.work_dir = Path(work_dir)
        self.device = torch.device(device)
        self.checkpoint_step = int(checkpoint_step)
        self.encoder = None
        self.expected_input_dim = 1024
        self.expected_tokens = 32
        self.expected_token_dim = 64

    def ensure_loaded(self):
        if self.encoder is not None:
            return
        with (self.work_dir / "config.yaml").open() as handle:
            config = yaml.safe_load(handle)
        encoder_config = config["models"]["encoder"]
        if encoder_config["name"] != "SimpleTokenTransformer":
            raise ValueError("RLA encoder must be SimpleTokenTransformer")
        args = dict(encoder_config["args"])
        args.pop("use_fp16", None)
        expected = {"in_channels": 1024, "out_channels": 64, "num_tokens": 32}
        if any(int(args.get(key, -1)) != value for key, value in expected.items()):
            raise ValueError("RLA encoder must map 1024-D patches to 32x64 latents")
        encoder = SimpleTokenTransformer(**args).to(self.device, dtype=torch.float32)
        checkpoint = self.work_dir / "ckpts" / f"encoder_step{self.checkpoint_step:07d}.pt"
        encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        encoder.eval().requires_grad_(False)
        self.encoder = encoder

    def encode_targets(self, current_patch_tokens, future_patch_tokens,
                       valid_frames=None):
        """Encode future-current residuals as `(B,H,32,64)` latent targets."""
        self.ensure_loaded()
        if current_patch_tokens.ndim != 3 or future_patch_tokens.ndim != 4:
            raise ValueError("RLA inputs must be current (B,P,C) and future (B,H,P,C)")
        batch, horizon, patches, channels = future_patch_tokens.shape
        if current_patch_tokens.shape != (batch, patches, channels) or channels != 1024:
            raise ValueError("Current/future DINO patch shapes must match at width1024")
        if valid_frames is not None and tuple(valid_frames.shape) != (batch, horizon):
            raise ValueError("valid_frames must have shape (B,H)")
        delta = (future_patch_tokens.to(device=self.device, dtype=torch.float32) -
                 current_patch_tokens.to(device=self.device, dtype=torch.float32).unsqueeze(1))
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=False,
        ):
            latent, _ = self.encoder(delta.reshape(batch * horizon, patches, channels), tokens=None)
        if tuple(latent.shape) != (batch * horizon, 32, 64):
            raise ValueError("RLA encoder must return (B*H,32,64) latents")
        return latent.reshape(batch, horizon, 32, 64)
