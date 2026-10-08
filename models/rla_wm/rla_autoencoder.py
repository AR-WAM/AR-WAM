"""Token-only residual-latent autoencoder adapted from RLA-WM."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models.simple_token_transformer import SimpleTokenTransformer


class FrozenDINOFeatures(nn.Module):
    """Final normalized DINO patches from pairs of camera frames in [0,1]."""

    def __init__(self, vision):
        super().__init__()
        self.vision = vision.eval().requires_grad_(False)
        self.patch_size = vision.config.patch_size
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images):
        batch, frames, cameras, channels, height, width = images.shape
        images = images.reshape(batch * frames * cameras, channels, height, width).float()
        images = (images - self.mean) / self.std
        pad_h, pad_w = -height % self.patch_size, -width % self.patch_size
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.autocast(device_type=images.device.type, enabled=False):
            output = self.vision(pixel_values=images, return_dict=True)
        patches = (images.shape[-2] // self.patch_size) * (images.shape[-1] // self.patch_size)
        tokens = output.last_hidden_state[:, -patches:].contiguous()
        tokens = tokens.reshape(batch, frames, cameras * patches, tokens.shape[-1])
        return tokens[:, 0], tokens[:, 1]


class RlaAutoencoder(nn.Module):
    """Encode future-current patches; reconstruct future given current+latent."""

    def __init__(self, models):
        super().__init__()
        if any(models[role]['name'] != 'SimpleTokenTransformer' for role in ('encoder', 'decoder')):
            raise ValueError('foresight-gist pretraining supports only SimpleTokenTransformer')
        self.encoder = SimpleTokenTransformer(**models['encoder']['args'])
        self.decoder = SimpleTokenTransformer(**models['decoder']['args'])

    def forward(self, current, future):
        latent, _ = self.encoder(future - current)
        _, prediction = self.decoder(current, tokens=latent)
        return prediction

    def training_losses(self, current, future, *, lambda_l1=1., lambda_mse=1.):
        prediction = self(current, future)
        return reconstruction_loss(prediction, future, lambda_l1, lambda_mse)


def reconstruction_loss(prediction, future, lambda_l1=1., lambda_mse=1.):
    """Original token-only L1+MSE objective, without RGB/VQ objectives."""
    l1 = F.l1_loss(prediction, future)
    mse = F.mse_loss(prediction, future.clone())
    return lambda_l1 * l1 + lambda_mse * mse, {'loss_l1': l1.detach(), 'loss_mse': mse.detach()}
