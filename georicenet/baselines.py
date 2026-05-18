from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def column(kernel: int) -> nn.Sequential:
            padding = kernel // 2
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel, padding=padding),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel, padding=padding),
                nn.ReLU(inplace=True),
            )

        self.columns = nn.ModuleList([column(9), column(7), column(5)])
        self.head = nn.Sequential(nn.Conv2d(48, 32, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = torch.cat([column(image) for column in self.columns], dim=1)
        return F.softplus(self.head(features), beta=1.0)


class CSRNetLite(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        density = self.backend(self.frontend(image))
        density = F.interpolate(density, scale_factor=4, mode="bilinear", align_corners=False)
        return F.softplus(density, beta=1.0)


def build_baseline(name: str) -> nn.Module:
    name = name.lower()
    if name == "mcnn":
        return MCNN()
    if name in {"csrnet", "csrnet_lite"}:
        return CSRNetLite()
    raise ValueError(f"Unknown baseline: {name}")
