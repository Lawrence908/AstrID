"""PyTorch U-Net architecture for astronomical anomaly detection.

This module provides a reusable UNet implementation compatible with the
training pipeline used in notebooks and services.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net architecture with correct decoder channel dimensions.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale)
        out_channels: Number of output channels (e.g., 1 for binary mask)
        initial_filters: Base number of filters for the first encoder block
        depth: Number of encoder/decoder levels
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        initial_filters: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.initial_filters = initial_filters

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for level_index in range(depth):
            out_ch = initial_filters * (2**level_index)
            self.encoders.append(self._conv_block(in_ch, out_ch))
            if level_index < depth - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = self._conv_block(
            initial_filters * (2 ** (depth - 1)),
            initial_filters * (2**depth),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        prev_ch = initial_filters * (2**depth)
        for level_index in range(depth - 1, 0, -1):
            out_ch = initial_filters * (2 ** (level_index - 1))
            self.upsamples.append(
                nn.ConvTranspose2d(prev_ch, out_ch, kernel_size=2, stride=2)
            )
            # After upsample, concat with encoder skip (out_ch) -> 2*out_ch in
            self.decoders.append(self._conv_block(out_ch * 2, out_ch))
            prev_ch = out_ch

        # Final 1x1 conv to map to out_channels
        self.final = nn.Conv2d(initial_filters, out_channels, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        encoder_outputs: list[torch.Tensor] = []
        for block_index, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outputs.append(x)
            if block_index < len(self.pools):
                x = self.pools[block_index](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for step_index, (upsample, decoder) in enumerate(
            zip(self.upsamples, self.decoders, strict=False)
        ):
            x = upsample(x)
            skip_idx = len(encoder_outputs) - 2 - step_index
            x = torch.cat([x, encoder_outputs[skip_idx]], dim=1)
            x = decoder(x)

        # Final projection
        x = self.final(x)
        return x


__all__ = ["UNet"]
