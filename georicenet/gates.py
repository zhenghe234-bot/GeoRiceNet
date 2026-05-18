from __future__ import annotations

import torch
from torch import nn


def normalize_density_map(density: torch.Tensor) -> torch.Tensor:
    flat = density.detach().flatten(1)
    low = flat.amin(dim=1).view(-1, 1, 1, 1)
    high = flat.amax(dim=1).view(-1, 1, 1, 1)
    return torch.clamp((density.detach() - low) / (high - low + 1e-6), 0.0, 1.0)


class ReflectionGeometryFeatureGate(nn.Module):
    """Feature-level GeoGate for reflection-aware density feature calibration."""

    def __init__(
        self,
        in_channels: int = 9,
        feature_channels: int = 32,
        hidden_channels: int = 32,
        max_delta: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_delta = max_delta
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.feature_head = nn.Conv2d(hidden_channels, feature_channels, 1)
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        nn.init.zeros_(self.feature_head.weight)
        nn.init.zeros_(self.feature_head.bias)

    def forward(self, gate_input: torch.Tensor, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = torch.tanh(self.feature_head(self.encoder(gate_input)))
        scale = 1.0 + self.max_delta * residual
        return feature * scale, scale, residual


class LearnedSpatialReliabilityGate(nn.Module):
    """Transferable density-level gate for fusing RGB and geometry-guided responses."""

    def __init__(self, in_channels: int = 11, hidden_channels: int = 24, residual_logit_scale: float = 1.5) -> None:
        super().__init__()
        self.residual_logit_scale = residual_logit_scale
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(4, hidden_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.local_head = nn.Conv2d(hidden_channels // 2, 1, 1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )
        self.reset_to_prior()

    def reset_to_prior(self) -> None:
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)

    def forward(
        self,
        rgb_prior: torch.Tensor,
        geometry_prior: torch.Tensor,
        rgb_density: torch.Tensor,
        geo_density: torch.Tensor,
        prior_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        density_rgb = normalize_density_map(rgb_density)
        density_geo = normalize_density_map(geo_density)
        density_diff = normalize_density_map(torch.abs(geo_density - rgb_density))
        features = torch.cat(
            [
                rgb_prior,
                geometry_prior,
                density_rgb.detach(),
                density_geo.detach(),
                density_diff.detach(),
                prior_weight.detach(),
            ],
            dim=1,
        )
        hidden = self.encoder(features)
        residual_logits = self.local_head(hidden) + self.global_head(hidden)
        residual_logits = self.residual_logit_scale * torch.tanh(residual_logits)
        prior = torch.clamp(prior_weight.detach(), 1e-4, 1.0 - 1e-4)
        weight = torch.sigmoid(torch.logit(prior) + residual_logits)
        fused_density = (1.0 - weight) * rgb_density.detach() + weight * geo_density.detach()
        return fused_density, weight, residual_logits


def prior_reliability_weight(
    geometry_prior: torch.Tensor,
    texture_prior: torch.Tensor | None = None,
    tau: float = 0.35,
    sharpness: float = 18.0,
) -> torch.Tensor:
    glare = torch.clamp(geometry_prior[:, 0:1], 0.0, 1.0)
    flat_water = torch.clamp(geometry_prior[:, 1:2], 0.0, 1.0)
    depth_gradient = torch.clamp(geometry_prior[:, 2:3], 0.0, 1.0)
    saliency = torch.clamp(geometry_prior[:, 3:4], 0.0, 1.0)
    if texture_prior is None:
        low_texture = torch.ones_like(glare)
    else:
        low_texture = torch.sigmoid((tau - texture_prior) * sharpness)
    weight = low_texture * (0.78 + 0.14 * flat_water + 0.08 * glare)
    weight = torch.maximum(weight, 0.18 * flat_water)
    weight = torch.maximum(weight, 0.12 * glare)
    weight = torch.maximum(weight, 0.10 * saliency * low_texture)
    weight = weight * (1.0 - 0.18 * depth_gradient)
    return torch.clamp(weight, 0.0, 1.0)
