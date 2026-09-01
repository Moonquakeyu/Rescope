from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from relmeasure3d.model import (
    DirectOutput,
    SegmentationOutput,
    SurfaceCriticalAttentionOutput,
    _surface_field_loss,
)


def _groups(channels: int) -> int:
    return next(group for group in range(min(8, channels), 0, -1) if channels % group == 0)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class PublicationEncoder3D(nn.Module):
    """Four-scale residual encoder shared by publication baselines and RelMeasure3D."""

    def __init__(self, base: int = 24, blocks_per_stage: int = 2):
        super().__init__()
        widths = (base, base * 2, base * 4, base * 8)
        self.widths = widths
        self.stem = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.SiLU(inplace=True),
        )
        stages: list[nn.Module] = []
        in_channels = base
        for index, width in enumerate(widths):
            stage: list[nn.Module] = [ResidualBlock3D(in_channels, width, stride=1 if index == 0 else 2)]
            stage.extend(ResidualBlock3D(width, width) for _ in range(blocks_per_stage - 1))
            stages.append(nn.Sequential(*stage))
            in_channels = width
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class PublicationUNet3D(nn.Module):
    def __init__(self, base: int = 24, blocks_per_stage: int = 2):
        super().__init__()
        self.encoder = PublicationEncoder3D(base=base, blocks_per_stage=blocks_per_stage)
        w0, w1, w2, w3 = self.encoder.widths
        self.up2 = nn.ConvTranspose3d(w3, w2, 2, stride=2)
        self.dec2 = nn.Sequential(ResidualBlock3D(w2 * 2, w2), ResidualBlock3D(w2, w2))
        self.up1 = nn.ConvTranspose3d(w2, w1, 2, stride=2)
        self.dec1 = nn.Sequential(ResidualBlock3D(w1 * 2, w1), ResidualBlock3D(w1, w1))
        self.up0 = nn.ConvTranspose3d(w1, w0, 2, stride=2)
        self.dec0 = nn.Sequential(ResidualBlock3D(w0 * 2, w0), ResidualBlock3D(w0, w0))

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != reference.shape[2:]:
            x = F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e0, e1, e2, e3 = self.encoder(x)
        d2 = self.dec2(torch.cat([self._resize_like(self.up2(e3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._resize_like(self.up1(d2), e1), e1], dim=1))
        d0 = self.dec0(torch.cat([self._resize_like(self.up0(d1), e0), e0], dim=1))
        return d0, e3


class PublicationDirectRegression(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.encoder = PublicationEncoder3D(base=base)
        self.head = nn.Sequential(
            nn.Linear(base * 8, base * 8),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(base * 8, 2),
        )

    def forward(self, image: torch.Tensor) -> DirectOutput:
        feature = self.encoder(image)[-1].mean((2, 3, 4))
        head = self.head(feature)
        return DirectOutput(F.softplus(head[:, 0]), F.softplus(head[:, 1]) + 1e-3)


class PublicationSegmentationBaseline(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.backbone = PublicationUNet3D(base=base)
        self.segmentation_head = nn.Conv3d(base, 2, 1)

    def forward(self, image: torch.Tensor) -> SegmentationOutput:
        feature, _ = self.backbone(image)
        logits = self.segmentation_head(feature)
        batch = image.shape[0]
        return SegmentationOutput(
            segmentation_logits=logits,
            distance_mm=torch.zeros(batch, device=image.device, dtype=image.dtype),
            sigma_mm=torch.ones(batch, device=image.device, dtype=image.dtype),
        )


@dataclass
class SurfaceGlobalOutput:
    distance_mm: torch.Tensor
    sigma_mm: torch.Tensor
    sdf: torch.Tensor


class PublicationSurfaceGlobalRegression(nn.Module):
    """Capacity-matched multi-task baseline without measurement-critical search."""

    def __init__(self, base: int = 24, truncate_mm: float = 10.0):
        super().__init__()
        self.backbone = PublicationUNet3D(base=base)
        self.sdf_head = nn.Conv3d(base, 2, 1)
        self.truncate_mm = truncate_mm
        self.head = nn.Sequential(
            nn.Linear(base * 8, base * 8),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(base * 8, 2),
        )

    def forward(self, image: torch.Tensor) -> SurfaceGlobalOutput:
        feature, bottleneck = self.backbone(image)
        sdf = torch.tanh(self.sdf_head(feature)) * self.truncate_mm
        head = self.head(bottleneck.mean((2, 3, 4)))
        return SurfaceGlobalOutput(
            distance_mm=F.softplus(head[:, 0]),
            sigma_mm=F.softplus(head[:, 1]) + 1e-3,
            sdf=sdf,
        )


class PublicationRelMeasure3D(nn.Module):
    """RelScope model used in the reported experiments.

    ``PublicationRelMeasure3D`` is retained as the development class name so
    released checkpoints and experiment commands remain compatible.
    """

    def __init__(self, base: int = 24, truncate_mm: float = 10.0):
        super().__init__()
        self.backbone = PublicationUNet3D(base=base)
        self.sdf_head = nn.Conv3d(base, 2, 1)
        self.critical_head = nn.Sequential(
            ResidualBlock3D(base + 2, base),
            nn.Conv3d(base, 2, 1),
        )
        self.truncate_mm = truncate_mm
        relation_dim = base * 3 + base * 8 + 9
        self.head = nn.Sequential(
            nn.Linear(relation_dim, base * 12),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(base * 12, base * 6),
            nn.SiLU(),
            nn.Linear(base * 6, 2),
        )

    def forward(self, image: torch.Tensor) -> SurfaceCriticalAttentionOutput:
        feature, bottleneck = self.backbone(image)
        sdf = torch.tanh(self.sdf_head(feature)) * self.truncate_mm
        critical_logits = self.critical_head(torch.cat([feature, sdf / self.truncate_mm], dim=1))
        attention = torch.softmax(critical_logits.flatten(2), dim=-1).reshape_as(critical_logits)
        attention_a, attention_b = attention[:, 0:1], attention[:, 1:2]
        za = (attention_a * feature).sum((2, 3, 4))
        zb = (attention_b * feature).sum((2, 3, 4))
        global_feature = bottleneck.mean((2, 3, 4))
        _, _, depth, height, width = critical_logits.shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, depth, device=image.device),
            torch.linspace(-1.0, 1.0, height, device=image.device),
            torch.linspace(-1.0, 1.0, width, device=image.device),
            indexing="ij",
        )
        coordinates = torch.stack([zz, yy, xx], dim=0)[None]
        center_a = (attention_a * coordinates).sum((2, 3, 4))
        center_b = (attention_b * coordinates).sum((2, 3, 4))
        geometry = torch.linalg.vector_norm(center_a - center_b, dim=1)
        entropy_a = -(attention_a.flatten(1) * attention_a.flatten(1).clamp_min(1e-12).log()).sum(1)
        entropy_b = -(attention_b.flatten(1) * attention_b.flatten(1).clamp_min(1e-12).log()).sum(1)
        relation = torch.cat(
            [
                global_feature,
                za,
                zb,
                (za - zb).abs(),
                center_a,
                center_b,
                geometry[:, None],
                entropy_a[:, None],
                entropy_b[:, None],
            ],
            dim=1,
        )
        head = self.head(relation)
        return SurfaceCriticalAttentionOutput(
            distance_mm=F.softplus(head[:, 0]),
            sigma_mm=F.softplus(head[:, 1]) + 1e-3,
            critical_logits=critical_logits,
            attention_a=attention_a,
            attention_b=attention_b,
            geometry_distance_vox=geometry,
            sdf=sdf,
        )


RelScope = PublicationRelMeasure3D


def _analytic_surface_attention(
    sdf: torch.Tensor,
    surface_temperature_mm: float,
    distance_temperature_mm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locate the closest surface regions using only predicted SDF geometry."""
    distance_to_a = sdf[:, 0:1].abs()
    distance_to_b = sdf[:, 1:2].abs()
    logits_a = -distance_to_a / surface_temperature_mm - distance_to_b / distance_temperature_mm
    logits_b = -distance_to_b / surface_temperature_mm - distance_to_a / distance_temperature_mm
    attention_a = torch.softmax(logits_a.flatten(2), dim=-1).reshape_as(logits_a)
    attention_b = torch.softmax(logits_b.flatten(2), dim=-1).reshape_as(logits_b)
    return logits_a, logits_b, attention_a, attention_b


class PublicationAnalyticSoftmin(nn.Module):
    """Predicted SDF followed by an analytic differentiable closest-surface estimate."""

    def __init__(
        self,
        base: int = 24,
        truncate_mm: float = 10.0,
        surface_temperature_mm: float = 0.5,
        distance_temperature_mm: float = 0.75,
    ):
        super().__init__()
        self.backbone = PublicationUNet3D(base=base)
        self.sdf_head = nn.Conv3d(base, 2, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(base * 8, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, 1),
        )
        self.truncate_mm = truncate_mm
        self.surface_temperature_mm = surface_temperature_mm
        self.distance_temperature_mm = distance_temperature_mm

    def forward(self, image: torch.Tensor) -> SurfaceGlobalOutput:
        feature, bottleneck = self.backbone(image)
        sdf = torch.tanh(self.sdf_head(feature)) * self.truncate_mm
        _, _, attention_a, attention_b = _analytic_surface_attention(
            sdf, self.surface_temperature_mm, self.distance_temperature_mm
        )
        distance_a = (attention_a * sdf[:, 1:2].abs()).sum((2, 3, 4))
        distance_b = (attention_b * sdf[:, 0:1].abs()).sum((2, 3, 4))
        distance_mm = 0.5 * (distance_a + distance_b).squeeze(1)
        sigma_mm = F.softplus(self.sigma_head(bottleneck.mean((2, 3, 4))).squeeze(1)) + 1e-3
        return SurfaceGlobalOutput(distance_mm=distance_mm, sigma_mm=sigma_mm, sdf=sdf)


class PublicationAnalyticCriticalPooling(nn.Module):
    """Capacity-matched relation head with critical regions fixed by analytic SDF proximity."""

    def __init__(
        self,
        base: int = 24,
        truncate_mm: float = 10.0,
        surface_temperature_mm: float = 0.5,
        distance_temperature_mm: float = 0.75,
    ):
        super().__init__()
        self.backbone = PublicationUNet3D(base=base)
        self.sdf_head = nn.Conv3d(base, 2, 1)
        self.truncate_mm = truncate_mm
        self.surface_temperature_mm = surface_temperature_mm
        self.distance_temperature_mm = distance_temperature_mm
        relation_dim = base * 3 + base * 8 + 9
        self.head = nn.Sequential(
            nn.Linear(relation_dim, base * 12),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(base * 12, base * 6),
            nn.SiLU(),
            nn.Linear(base * 6, 2),
        )

    def forward(self, image: torch.Tensor) -> SurfaceCriticalAttentionOutput:
        feature, bottleneck = self.backbone(image)
        sdf = torch.tanh(self.sdf_head(feature)) * self.truncate_mm
        logits_a, logits_b, attention_a, attention_b = _analytic_surface_attention(
            sdf, self.surface_temperature_mm, self.distance_temperature_mm
        )
        za = (attention_a * feature).sum((2, 3, 4))
        zb = (attention_b * feature).sum((2, 3, 4))
        global_feature = bottleneck.mean((2, 3, 4))
        _, _, depth, height, width = sdf.shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, depth, device=image.device),
            torch.linspace(-1.0, 1.0, height, device=image.device),
            torch.linspace(-1.0, 1.0, width, device=image.device),
            indexing="ij",
        )
        coordinates = torch.stack([zz, yy, xx], dim=0)[None]
        center_a = (attention_a * coordinates).sum((2, 3, 4))
        center_b = (attention_b * coordinates).sum((2, 3, 4))
        geometry = torch.linalg.vector_norm(center_a - center_b, dim=1)
        entropy_a = -(attention_a.flatten(1) * attention_a.flatten(1).clamp_min(1e-12).log()).sum(1)
        entropy_b = -(attention_b.flatten(1) * attention_b.flatten(1).clamp_min(1e-12).log()).sum(1)
        relation = torch.cat(
            [
                global_feature,
                za,
                zb,
                (za - zb).abs(),
                center_a,
                center_b,
                geometry[:, None],
                entropy_a[:, None],
                entropy_b[:, None],
            ],
            dim=1,
        )
        head = self.head(relation)
        return SurfaceCriticalAttentionOutput(
            distance_mm=F.softplus(head[:, 0]),
            sigma_mm=F.softplus(head[:, 1]) + 1e-3,
            critical_logits=torch.cat([logits_a, logits_b], dim=1),
            attention_a=attention_a,
            attention_b=attention_b,
            geometry_distance_vox=geometry,
            sdf=sdf,
        )


def surface_global_loss(
    output: SurfaceGlobalOutput,
    batch: dict[str, torch.Tensor | list[str]],
    surface_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sdf = batch["sdf"]
    target_mask = batch["mask"]
    target_distance = batch["distance_mm"]
    assert isinstance(target_sdf, torch.Tensor)
    assert isinstance(target_mask, torch.Tensor)
    assert isinstance(target_distance, torch.Tensor)
    surface = _surface_field_loss(output.sdf, target_sdf, target_mask)
    error = target_distance - output.distance_mm
    distance = F.smooth_l1_loss(output.distance_mm, target_distance)
    nll = (0.5 * (error / output.sigma_mm).square() + output.sigma_mm.log()).mean()
    total = distance + 0.02 * nll + surface_weight * surface
    return total, {
        "loss": float(total.detach()),
        "field_loss": float(surface.detach()),
        "distance_loss": float(distance.detach()),
        "nll": float(nll.detach()),
        "critical_loss": 0.0,
        "critical_alignment": 0.0,
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": 0.0,
    }


def surface_global_warmup_loss(
    output: SurfaceGlobalOutput,
    batch: dict[str, torch.Tensor | list[str]],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sdf = batch["sdf"]
    target_mask = batch["mask"]
    target_distance = batch["distance_mm"]
    assert isinstance(target_sdf, torch.Tensor)
    assert isinstance(target_mask, torch.Tensor)
    assert isinstance(target_distance, torch.Tensor)
    surface = _surface_field_loss(output.sdf, target_sdf, target_mask)
    error = target_distance - output.distance_mm
    return surface, {
        "loss": float(surface.detach()),
        "field_loss": float(surface.detach()),
        "distance_loss": 0.0,
        "nll": 0.0,
        "critical_loss": 0.0,
        "critical_alignment": 0.0,
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": 0.0,
    }
