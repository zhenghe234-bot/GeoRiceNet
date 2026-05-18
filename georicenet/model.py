from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .gates import ReflectionGeometryFeatureGate, normalize_density_map
from .ricenet import RiceNet


class GeoRiceNet(nn.Module):
    """RiceNet with reflection-aware geometry feature calibration."""

    def __init__(
        self,
        gate_in_channels: int = 9,
        gate_max_delta: float = 0.2,
        freeze_vgg: bool = True,
        freeze_backend: bool = True,
        freeze_output: bool = False,
    ) -> None:
        super().__init__()
        self.ricenet = RiceNet()
        self.freeze_vgg = freeze_vgg
        self.freeze_backend = freeze_backend
        self.freeze_output = freeze_output
        self.feature_gate = ReflectionGeometryFeatureGate(
            in_channels=gate_in_channels,
            feature_channels=32,
            max_delta=gate_max_delta,
        )
        self.apply_freeze_policy()

    def load_ricenet_checkpoint(self, checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("model", checkpoint)
        self.ricenet.load_state_dict(state_dict, strict=False)

    def apply_freeze_policy(self) -> None:
        for parameter in self.ricenet.vgg.parameters():
            parameter.requires_grad_(not self.freeze_vgg)

        for module in (self.ricenet.amp, self.ricenet.dmp, self.ricenet.conv_att):
            for parameter in module.parameters():
                parameter.requires_grad_(not self.freeze_backend)

        for parameter in self.ricenet.conv_out.parameters():
            parameter.requires_grad_(not self.freeze_output)

    def set_freeze_policy(
        self,
        freeze_vgg: bool | None = None,
        freeze_backend: bool | None = None,
        freeze_output: bool | None = None,
    ) -> None:
        if freeze_vgg is not None:
            self.freeze_vgg = freeze_vgg
        if freeze_backend is not None:
            self.freeze_backend = freeze_backend
        if freeze_output is not None:
            self.freeze_output = freeze_output
        self.apply_freeze_policy()

    def train(self, mode: bool = True) -> "GeoRiceNet":
        super().train(mode)
        if mode:
            if self.freeze_vgg:
                self.ricenet.vgg.eval()
            if self.freeze_backend:
                self.ricenet.amp.eval()
                self.ricenet.dmp.eval()
                self.ricenet.conv_att.eval()
            if self.freeze_output:
                self.ricenet.conv_out.eval()
        return self

    def _extract_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freeze_vgg:
            with torch.no_grad():
                return tuple(feature.detach() for feature in self.ricenet.vgg(image))
        return self.ricenet.vgg(image)

    def _fuse_density_features(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.freeze_backend:
            with torch.no_grad():
                attention_feature = self.ricenet.amp(*features)
                density_feature = self.ricenet.dmp(*features)
                attention = self.ricenet.conv_att(attention_feature)
                fused_feature = attention * density_feature
            return fused_feature.detach(), attention.detach()

        attention_feature = self.ricenet.amp(*features)
        density_feature = self.ricenet.dmp(*features)
        attention = self.ricenet.conv_att(attention_feature)
        return attention * density_feature, attention

    def build_gate_input(
        self,
        rgb_prior: torch.Tensor,
        geometry_prior: torch.Tensor,
        baseline_density: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        density_norm = normalize_density_map(baseline_density)
        attention = torch.clamp(attention.detach(), 0.0, 1.0)
        return torch.cat([rgb_prior, density_norm, attention, geometry_prior], dim=1)

    def forward(self, image: torch.Tensor, rgb_prior: torch.Tensor, geometry_prior: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._extract_features(image)
        fused_feature, attention = self._fuse_density_features(features)
        baseline_density = self.ricenet.conv_out(fused_feature)
        gate_input = self.build_gate_input(rgb_prior, geometry_prior, baseline_density, attention)
        enhanced_feature, scale, residual = self.feature_gate(gate_input, fused_feature)
        density = self.ricenet.conv_out(enhanced_feature)
        return {
            "density": density,
            "baseline_density": baseline_density.detach(),
            "attention": attention,
            "scale": scale,
            "residual": residual,
        }
