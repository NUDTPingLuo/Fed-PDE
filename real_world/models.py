"""Compact field-to-field models for public PDE benchmarks."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        scale = 1.0 / np.sqrt(max(in_channels * out_channels, 1))
        shape = (in_channels, out_channels, self.modes_x, self.modes_y)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _multiply(values, weights):
        return torch.einsum('bixy,ioxy->boxy', values, weights)

    def forward(self, values):
        batch, _, height, width = values.shape
        transformed = torch.fft.rfft2(values, norm='ortho')
        output = torch.zeros(
            batch, self.weight_pos.shape[1], height, width // 2 + 1,
            dtype=torch.cfloat, device=values.device)
        modes_x = min(self.modes_x, max(height // 2, 1))
        modes_y = min(self.modes_y, width // 2 + 1)
        output[:, :, :modes_x, :modes_y] = self._multiply(
            transformed[:, :, :modes_x, :modes_y],
            self.weight_pos[:, :, :modes_x, :modes_y])
        output[:, :, -modes_x:, :modes_y] = self._multiply(
            transformed[:, :, -modes_x:, :modes_y],
            self.weight_neg[:, :, :modes_x, :modes_y])
        return torch.fft.irfft2(output, s=(height, width), norm='ortho')


class FNO2d(nn.Module):
    """Fourier neural operator for regular-grid PDE fields."""
    def __init__(self, in_channels, out_channels, width=32, modes=12, layers=4):
        super().__init__()
        width = int(width)
        self.lift = nn.Conv2d(int(in_channels) + 2, width, 1)
        self.spectral = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes)
            for _ in range(int(layers))
        ])
        self.local = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(int(layers))
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(),
            nn.Conv2d(width * 2, int(out_channels), 1))

    @staticmethod
    def _grid(batch, height, width, device, dtype):
        x = torch.linspace(0, 1, height, device=device, dtype=dtype)
        y = torch.linspace(0, 1, width, device=device, dtype=dtype)
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([gx, gy])[None].expand(batch, -1, -1, -1)

    def forward(self, inputs):
        grid = self._grid(
            len(inputs), inputs.shape[-2], inputs.shape[-1],
            inputs.device, inputs.dtype)
        values = self.lift(torch.cat([inputs, grid], dim=1))
        for spectral, local in zip(self.spectral, self.local):
            values = F.gelu(spectral(values) + local(values))
        return self.project(values)


class InversionNetLite(nn.Module):
    """Memory-conscious encoder-decoder for OpenFWI shot gathers.

    This preserves InversionNet's waveform-encoder/image-decoder structure but
    is intentionally smaller than the official reproduction model so that ten
    federated client replicas fit on a workstation GPU.
    """
    def __init__(self, in_channels, out_channels, output_shape, width=32):
        super().__init__()
        width = int(width)
        self.output_shape = tuple(int(value) for value in output_shape)
        channels = [int(in_channels), width, width * 2, width * 4, width * 4]
        encoder = []
        for source, target in zip(channels[:-1], channels[1:]):
            encoder.extend([
                nn.Conv2d(source, target, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, target), target), nn.GELU()])
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels[-1], width * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, int(out_channels), 1))

    def forward(self, inputs):
        encoded = F.adaptive_avg_pool2d(self.encoder(inputs), (18, 18))
        decoded = self.decoder(encoded)
        return F.interpolate(
            decoded, size=self.output_shape, mode='bilinear', align_corners=False)


def make_model(model_cfg, input_shape, target_shape):
    name = str(model_cfg['type']).lower()
    if len(input_shape) != 3 or len(target_shape) != 3:
        raise ValueError("public PDE models require [C,H,W] input and target shapes")
    if name == 'fno2d':
        if tuple(input_shape[-2:]) != tuple(target_shape[-2:]):
            raise ValueError("FNO2D requires input and target on the same spatial grid")
        return FNO2d(
            input_shape[0], target_shape[0],
            width=model_cfg.get('width', 32),
            modes=model_cfg.get('modes', 12),
            layers=model_cfg.get('layers', 4))
    if name == 'inversionnet_lite':
        return InversionNetLite(
            input_shape[0], target_shape[0], target_shape[-2:],
            width=model_cfg.get('width', 32))
    raise ValueError("model.type must be fno2d or inversionnet_lite")
