"""Braai-style CNN for real/bogus classification of image triplets.

Input: (batch, 3, H, W) — science, reference, difference channels.
Output: (batch,) logits or probabilities; 1 = real transient, 0 = bogus.

Reference: SUPERNOVA_DETECTION_PIPELINE_GUIDE.md Phase 5; ZTF Braai classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RealBogusCNN(nn.Module):
    """Small CNN for classifying (science, reference, difference) triplets as real or bogus.

    Architecture (Braai-style):
        Conv2D(32, 3x3) -> ReLU -> MaxPool
        Conv2D(64, 3x3) -> ReLU -> MaxPool
        Conv2D(128, 3x3) -> ReLU -> MaxPool
        Flatten -> Dense(256) -> ReLU -> Dropout(0.5) -> Dense(1)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: tuple[int, ...] = (32, 64, 128),
        dense_size: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters

        layers: list[nn.Module] = []
        ch = in_channels
        for nf in num_filters:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, nf, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            ch = nf
        self.conv = nn.Sequential(*layers)

        # For 63x63 input: 63 -> 31 -> 15 -> 7
        self._feature_size = num_filters[-1] * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size, dense_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (batch, 1)."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities in [0, 1]; 1 = real."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
