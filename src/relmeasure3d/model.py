from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    groups = next(group for group in range(min(8, cout), 0, -1) if cout % group == 0)
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.SiLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.SiLU(inplace=True),
    )


class TinyFieldBackbone(nn.Module):
    def __init__(self, base: int = 12):
        super().__init__()
        self.e0 = _block(1, base)
        self.e1 = _block(base, base * 2, stride=2)
        self.e2 = _block(base * 2, base * 4, stride=2)
        self.b = _block(base * 4, base * 6, stride=2)
        self.d2 = _block(base * 6 + base * 4, base * 4)
        self.d1 = _block(base * 4 + base * 2, base * 2)
        self.d0 = _block(base * 2 + base, base)
        self.field = nn.Conv3d(base, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        b = self.b(e2)
        d2 = self.d2(torch.cat([F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False), e1], 1))
        d0 = self.d0(torch.cat([F.interpolate(d1, size=e0.shape[2:], mode="trilinear", align_corners=False), e0], 1))
        return self.field(d0), d0


class TinyDirectBackbone(nn.Module):
    """Encoder-only baseline with no explicit surface geometry."""

    def __init__(self, base: int = 12):
        super().__init__()
        self.e0 = _block(1, base)
        self.e1 = _block(base, base * 2, stride=2)
        self.e2 = _block(base * 2, base * 4, stride=2)
        self.b = _block(base * 4, base * 6, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.e0(x)
        x = self.e1(x)
        x = self.e2(x)
        return self.b(x)


def _field_gradients(field: torch.Tensor) -> torch.Tensor:
    """Central differences; returns B,C,3,D,H,W."""
    dx = F.pad((field[:, :, 2:] - field[:, :, :-2]) * 0.5, (0, 0, 0, 0, 1, 1))
    dy = F.pad((field[:, :, :, 2:] - field[:, :, :, :-2]) * 0.5, (0, 0, 1, 1, 0, 0))
    dz = F.pad((field[:, :, :, :, 2:] - field[:, :, :, :, :-2]) * 0.5, (1, 1, 0, 0, 0, 0))
    return torch.stack([dx, dy, dz], dim=2)


def _gather_flat(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather B,C,N at B,K -> B,K,C."""
    x = x.flatten(2)
    gathered = torch.gather(x, 2, index[:, None].expand(-1, x.shape[1], -1))
    return gathered.transpose(1, 2)


def _surface_field_loss(
    predicted_sdf: torch.Tensor,
    target_sdf: torch.Tensor,
    target_mask: torch.Tensor,
    surface_scale_mm: float = 2.0,
) -> torch.Tensor:
    """Balance global SDF fit with zero-level-set and foreground accuracy."""
    pointwise = F.smooth_l1_loss(predicted_sdf / 10.0, target_sdf / 10.0, reduction="none")
    surface_weight = 1.0 + 8.0 * torch.exp(-target_sdf.abs() / surface_scale_mm)
    regression = (pointwise * surface_weight).sum() / surface_weight.sum().clamp_min(1.0)

    foreground = target_mask.float()
    probability = torch.sigmoid(-predicted_sdf / 0.75)
    reduce_dims = (2, 3, 4)
    intersection = (probability * foreground).sum(reduce_dims)
    denominator = probability.sum(reduce_dims) + foreground.sum(reduce_dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    sign_pointwise = F.binary_cross_entropy_with_logits(-predicted_sdf / 0.75, foreground, reduction="none")
    sign_weight = torch.exp(-target_sdf.abs() / surface_scale_mm)
    sign = (sign_pointwise * sign_weight).sum() / sign_weight.sum().clamp_min(1.0)
    return regression + 0.5 * dice + 0.1 * sign


@dataclass
class ModelOutput:
    sdf: torch.Tensor
    distance_mm: torch.Tensor
    sigma_mm: torch.Tensor
    pair_distance_mm: torch.Tensor
    marginal_a: torch.Tensor
    marginal_b: torch.Tensor
    index_a: torch.Tensor
    index_b: torch.Tensor
    coupling_entropy: torch.Tensor
    geometry_distance_mm: torch.Tensor
    residual_mm: torch.Tensor


@dataclass
class DirectOutput:
    distance_mm: torch.Tensor
    sigma_mm: torch.Tensor


@dataclass
class CriticalAttentionOutput:
    distance_mm: torch.Tensor
    sigma_mm: torch.Tensor
    critical_logits: torch.Tensor
    attention_a: torch.Tensor
    attention_b: torch.Tensor
    geometry_distance_vox: torch.Tensor


@dataclass
class SegmentationOutput:
    segmentation_logits: torch.Tensor
    distance_mm: torch.Tensor
    sigma_mm: torch.Tensor


@dataclass
class SurfaceCriticalAttentionOutput:
    distance_mm: torch.Tensor
    sigma_mm: torch.Tensor
    critical_logits: torch.Tensor
    attention_a: torch.Tensor
    attention_b: torch.Tensor
    geometry_distance_vox: torch.Tensor
    sdf: torch.Tensor


class DirectRegressionSmoke(nn.Module):
    """Capacity-matched image-to-scalar pilot baseline."""

    def __init__(self, base: int = 12):
        super().__init__()
        self.backbone = TinyDirectBackbone(base=base)
        self.head = nn.Sequential(
            nn.Linear(base * 6, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, image: torch.Tensor) -> DirectOutput:
        feat = self.backbone(image).mean(dim=(2, 3, 4))
        head = self.head(feat)
        return DirectOutput(
            distance_mm=F.softplus(head[:, 0]),
            sigma_mm=F.softplus(head[:, 1]) + 1e-3,
        )


class SegmentationDistanceBaseline(nn.Module):
    """Two-structure 3D U-Net baseline; distance is derived from its predicted masks."""

    def __init__(self, base: int = 12):
        super().__init__()
        self.backbone = TinyFieldBackbone(base=base)

    def forward(self, image: torch.Tensor) -> SegmentationOutput:
        logits, _ = self.backbone(image)
        batch = image.shape[0]
        return SegmentationOutput(
            segmentation_logits=logits,
            distance_mm=torch.zeros(batch, device=image.device, dtype=image.dtype),
            sigma_mm=torch.ones(batch, device=image.device, dtype=image.dtype),
        )


class CriticalAttentionRegression(nn.Module):
    """Image-to-distance model with training-only critical-boundary supervision."""

    def __init__(self, base: int = 12):
        super().__init__()
        self.backbone = TinyFieldBackbone(base=base)
        relation_dim = base * 4 + 7
        self.head = nn.Sequential(
            nn.Linear(relation_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, image: torch.Tensor) -> CriticalAttentionOutput:
        critical_logits, feature = self.backbone(image)
        _, _, d, h, w = critical_logits.shape
        attention = torch.softmax(critical_logits.flatten(2), dim=-1).reshape_as(critical_logits)
        attention_a = attention[:, 0:1]
        attention_b = attention[:, 1:2]
        za = (attention_a * feature).sum((2, 3, 4))
        zb = (attention_b * feature).sum((2, 3, 4))
        global_feature = feature.mean((2, 3, 4))

        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, d, device=image.device),
            torch.linspace(-1.0, 1.0, h, device=image.device),
            torch.linspace(-1.0, 1.0, w, device=image.device),
            indexing="ij",
        )
        coords = torch.stack([zz, yy, xx], 0)[None]
        center_a = (attention_a * coords).sum((2, 3, 4))
        center_b = (attention_b * coords).sum((2, 3, 4))
        geometry = torch.linalg.vector_norm(center_a - center_b, dim=1)
        relation = torch.cat(
            [global_feature, za, zb, (za - zb).abs(), center_a, center_b, geometry[:, None]],
            dim=1,
        )
        head = self.head(relation)
        return CriticalAttentionOutput(
            distance_mm=F.softplus(head[:, 0]),
            sigma_mm=F.softplus(head[:, 1]) + 1e-3,
            critical_logits=critical_logits,
            attention_a=attention_a,
            attention_b=attention_b,
            geometry_distance_vox=geometry,
        )


class SurfaceCriticalAttentionRegression(nn.Module):
    """Search-then-measure model with surface geometry and critical-region supervision."""

    def __init__(self, base: int = 12, truncate_mm: float = 10.0):
        super().__init__()
        self.backbone = TinyFieldBackbone(base=base)
        self.truncate_mm = truncate_mm
        self.critical_head = nn.Sequential(
            nn.Conv3d(base + 2, base, 3, padding=1, bias=False),
            nn.GroupNorm(next(group for group in range(min(8, base), 0, -1) if base % group == 0), base),
            nn.SiLU(inplace=True),
            nn.Conv3d(base, 2, 1),
        )
        relation_dim = base * 4 + 9
        self.head = nn.Sequential(
            nn.Linear(relation_dim, 192),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 2),
        )

    def forward(self, image: torch.Tensor) -> SurfaceCriticalAttentionOutput:
        surface_raw, feature = self.backbone(image)
        sdf = torch.tanh(surface_raw) * self.truncate_mm
        critical_logits = self.critical_head(torch.cat([feature, sdf / self.truncate_mm], dim=1))
        _, _, d, h, w = critical_logits.shape
        attention = torch.softmax(critical_logits.flatten(2), dim=-1).reshape_as(critical_logits)
        attention_a = attention[:, 0:1]
        attention_b = attention[:, 1:2]
        za = (attention_a * feature).sum((2, 3, 4))
        zb = (attention_b * feature).sum((2, 3, 4))
        global_feature = feature.mean((2, 3, 4))
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, d, device=image.device),
            torch.linspace(-1.0, 1.0, h, device=image.device),
            torch.linspace(-1.0, 1.0, w, device=image.device),
            indexing="ij",
        )
        coords = torch.stack([zz, yy, xx], 0)[None]
        center_a = (attention_a * coords).sum((2, 3, 4))
        center_b = (attention_b * coords).sum((2, 3, 4))
        geometry = torch.linalg.vector_norm(center_a - center_b, dim=1)
        entropy_a = -(attention_a.flatten(1) * attention_a.flatten(1).clamp_min(1e-12).log()).sum(1)
        entropy_b = -(attention_b.flatten(1) * attention_b.flatten(1).clamp_min(1e-12).log()).sum(1)
        relation = torch.cat(
            [
                global_feature, za, zb, (za - zb).abs(), center_a, center_b,
                geometry[:, None], entropy_a[:, None], entropy_b[:, None],
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


class CriticalRefinementRegression(nn.Module):
    """Frozen direct anchor with a bounded critical-attention correction."""

    def __init__(self, base: int = 12, max_correction_mm: float = 1.0):
        super().__init__()
        self.direct = DirectRegressionSmoke(base=base)
        for parameter in self.direct.parameters():
            parameter.requires_grad = False
        self.attention_backbone = TinyFieldBackbone(base=base)
        self.max_correction_mm = max_correction_mm
        relation_dim = base * 4 + 9
        self.correction = nn.Sequential(
            nn.Linear(relation_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )
        final_layer = self.correction[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, image: torch.Tensor) -> CriticalAttentionOutput:
        self.direct.eval()
        with torch.no_grad():
            direct = self.direct(image)
        critical_logits, feature = self.attention_backbone(image)
        _, _, d, h, w = critical_logits.shape
        attention = torch.softmax(critical_logits.flatten(2), dim=-1).reshape_as(critical_logits)
        attention_a = attention[:, 0:1]
        attention_b = attention[:, 1:2]
        za = (attention_a * feature).sum((2, 3, 4))
        zb = (attention_b * feature).sum((2, 3, 4))
        global_feature = feature.mean((2, 3, 4))
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, d, device=image.device),
            torch.linspace(-1.0, 1.0, h, device=image.device),
            torch.linspace(-1.0, 1.0, w, device=image.device),
            indexing="ij",
        )
        coords = torch.stack([zz, yy, xx], 0)[None]
        center_a = (attention_a * coords).sum((2, 3, 4))
        center_b = (attention_b * coords).sum((2, 3, 4))
        geometry = torch.linalg.vector_norm(center_a - center_b, dim=1)
        relation = torch.cat(
            [
                global_feature,
                za,
                zb,
                (za - zb).abs(),
                center_a,
                center_b,
                geometry[:, None],
                direct.distance_mm[:, None],
                direct.sigma_mm[:, None],
            ],
            dim=1,
        )
        correction = self.correction(relation)
        distance = (direct.distance_mm + self.max_correction_mm * torch.tanh(correction[:, 0])).clamp_min(0.0)
        sigma = F.softplus(correction[:, 1]) + 1e-3
        return CriticalAttentionOutput(
            distance_mm=distance,
            sigma_mm=sigma,
            critical_logits=critical_logits,
            attention_a=attention_a,
            attention_b=attention_b,
            geometry_distance_vox=geometry,
        )


class RelMeasure3DSmoke(nn.Module):
    """Compact implementation of the proposal's pairwise-coupling critical path."""

    def __init__(
        self,
        base: int = 12,
        token_dim: int = 48,
        token_count: int = 64,
        truncate_mm: float = 10.0,
        spacing_mm: float = 1.0,
        coupling_mode: str = "learned",
        analytic_temperature_mm: float = 0.5,
        max_residual_mm: float = 0.5,
    ):
        super().__init__()
        if coupling_mode not in {"learned", "analytic", "anchored"}:
            raise ValueError(f"unsupported coupling mode: {coupling_mode}")
        self.backbone = TinyFieldBackbone(base=base)
        self.token_count = token_count
        self.truncate_mm = truncate_mm
        self.spacing_mm = spacing_mm
        self.coupling_mode = coupling_mode
        self.analytic_temperature_mm = analytic_temperature_mm
        self.max_residual_mm = max_residual_mm
        raw_dim = base + 3 + 3 + 1
        self.token_a = nn.Sequential(nn.Linear(raw_dim, token_dim), nn.SiLU(), nn.LayerNorm(token_dim))
        self.token_b = nn.Sequential(nn.Linear(raw_dim, token_dim), nn.SiLU(), nn.LayerNorm(token_dim))
        self.q = nn.Linear(token_dim, token_dim, bias=False)
        self.k = nn.Linear(token_dim, token_dim, bias=False)
        self.pair_bias = nn.Sequential(nn.Linear(5, token_dim // 2), nn.SiLU(), nn.Linear(token_dim // 2, 1))
        self.geometry_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.coupling_gate_raw = nn.Parameter(torch.tensor(-3.0))
        relation_dim = token_dim * 4 + 3
        self.measurement = nn.Sequential(
            nn.Linear(relation_dim, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 2)
        )
        if self.coupling_mode == "anchored":
            final_layer = self.measurement[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(self, image: torch.Tensor) -> ModelOutput:
        sdf_raw, feat = self.backbone(image)
        sdf = torch.tanh(sdf_raw) * self.truncate_mm
        b, _, d, h, w = sdf.shape
        nvox = d * h * w
        k = min(self.token_count, nvox)
        index_a = torch.topk(-sdf[:, 0].abs().flatten(1), k=k, dim=1).indices
        index_b = torch.topk(-sdf[:, 1].abs().flatten(1), k=k, dim=1).indices

        zz, yy, xx = torch.meshgrid(
            torch.arange(d, device=image.device),
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij",
        )
        coords = torch.stack([zz, yy, xx], 0).float().reshape(3, -1) * self.spacing_mm
        coords = coords[None].expand(b, -1, -1)
        pa = _gather_flat(coords, index_a)
        pb = _gather_flat(coords, index_b)
        center = torch.tensor([(d - 1), (h - 1), (w - 1)], device=image.device, dtype=pa.dtype) * (0.5 * self.spacing_mm)
        scale = torch.tensor([d, h, w], device=image.device, dtype=pa.dtype) * (0.5 * self.spacing_mm)
        pa_norm = (pa - center) / scale
        pb_norm = (pb - center) / scale

        grad = _field_gradients(sdf)
        ga = _gather_flat(grad[:, 0].reshape(b, 3, d, h, w), index_a)
        gb = _gather_flat(grad[:, 1].reshape(b, 3, d, h, w), index_b)
        ga = F.normalize(ga, dim=-1, eps=1e-6)
        gb = F.normalize(gb, dim=-1, eps=1e-6)
        fa = _gather_flat(feat, index_a)
        fb = _gather_flat(feat, index_b)
        cross_a = _gather_flat(sdf[:, 1:2], index_a)
        cross_b = _gather_flat(sdf[:, 0:1], index_b)
        ta = self.token_a(torch.cat([fa, pa_norm, ga, cross_a / self.truncate_mm], -1))
        tb = self.token_b(torch.cat([fb, pb_norm, gb, cross_b / self.truncate_mm], -1))

        delta = pa[:, :, None, :] - pb[:, None, :, :]
        pair_dist = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-5)
        normal_dot = (ga[:, :, None, :] * gb[:, None, :, :]).sum(-1, keepdim=True)
        pair_raw = torch.cat([delta / (max(d, h, w) * self.spacing_mm), pair_dist[..., None] / 64.0, normal_dot], -1)
        learned_logits = torch.einsum("bkd,bjd->bkj", self.q(ta), self.k(tb)) / math.sqrt(ta.shape[-1])
        learned_logits = learned_logits + self.pair_bias(pair_raw).squeeze(-1)
        if self.coupling_mode == "analytic":
            logits = -pair_dist / self.analytic_temperature_mm
        elif self.coupling_mode == "anchored":
            coupling_strength = torch.sigmoid(self.coupling_gate_raw)
            logits = -pair_dist / self.analytic_temperature_mm + coupling_strength * torch.tanh(learned_logits)
        else:
            logits = learned_logits - F.softplus(self.geometry_scale_raw) * pair_dist
        coupling = torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)
        marginal_a = coupling.sum(2)
        marginal_b = coupling.sum(1)
        d_pair = (coupling * pair_dist).sum((1, 2))

        za = torch.einsum("bk,bkd->bd", marginal_a, ta)
        zb = torch.einsum("bk,bkd->bd", marginal_b, tb)
        entropy = -(coupling.clamp_min(1e-9) * coupling.clamp_min(1e-9).log()).sum((1, 2))
        rel = torch.cat([za, zb, za - zb, za * zb, d_pair[:, None], entropy[:, None], pair_dist.amin((1, 2))[:, None]], 1)
        head = self.measurement(rel)
        if self.coupling_mode == "analytic":
            residual = torch.zeros_like(d_pair)
            distance = d_pair
            sigma = torch.ones_like(d_pair)
        elif self.coupling_mode == "anchored":
            residual = self.max_residual_mm * torch.tanh(head[:, 0])
            distance = (d_pair + residual).clamp_min(0.0)
            sigma = F.softplus(head[:, 1]) + 1e-3
        else:
            residual = head[:, 0]
            distance = (d_pair + residual).clamp_min(0.0)
            sigma = F.softplus(head[:, 1]) + 1e-3
        return ModelOutput(
            sdf=sdf,
            distance_mm=distance,
            sigma_mm=sigma,
            pair_distance_mm=d_pair,
            marginal_a=marginal_a,
            marginal_b=marginal_b,
            index_a=index_a,
            index_b=index_b,
            coupling_entropy=entropy,
            geometry_distance_mm=d_pair,
            residual_mm=residual,
        )


def smoke_loss(
    output: ModelOutput,
    batch: dict[str, torch.Tensor | list[str]],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sdf = batch["sdf"]
    target_critical = batch["critical"]
    target_distance = batch["distance_mm"]
    assert isinstance(target_sdf, torch.Tensor)
    assert isinstance(target_critical, torch.Tensor)
    assert isinstance(target_distance, torch.Tensor)
    field = F.smooth_l1_loss(output.sdf / 10.0, target_sdf / 10.0)
    distance = F.smooth_l1_loss(output.distance_mm, target_distance)
    error = target_distance - output.distance_mm
    nll = (0.5 * (error / output.sigma_mm).square() + output.sigma_mm.log()).mean()

    crit_a = _gather_flat(target_critical[:, 0:1], output.index_a).squeeze(-1) + 1e-4
    crit_b = _gather_flat(target_critical[:, 1:2], output.index_b).squeeze(-1) + 1e-4
    crit_a = crit_a / crit_a.sum(1, keepdim=True)
    crit_b = crit_b / crit_b.sum(1, keepdim=True)
    critical = F.kl_div(output.marginal_a.clamp_min(1e-8).log(), crit_a, reduction="batchmean")
    critical = critical + F.kl_div(output.marginal_b.clamp_min(1e-8).log(), crit_b, reduction="batchmean")
    total = field + distance + 0.05 * nll + 0.1 * critical
    metrics = {
        "loss": float(total.detach()),
        "field_loss": float(field.detach()),
        "distance_loss": float(distance.detach()),
        "nll": float(nll.detach()),
        "critical_loss": float(critical.detach()),
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": float(output.coupling_entropy.mean().detach()),
    }
    return total, metrics


def direct_smoke_loss(
    output: DirectOutput,
    batch: dict[str, torch.Tensor | list[str]],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_distance = batch["distance_mm"]
    assert isinstance(target_distance, torch.Tensor)
    error = target_distance - output.distance_mm
    distance = F.smooth_l1_loss(output.distance_mm, target_distance)
    nll = (0.5 * (error / output.sigma_mm).square() + output.sigma_mm.log()).mean()
    total = distance + 0.05 * nll
    return total, {
        "loss": float(total.detach()),
        "field_loss": 0.0,
        "distance_loss": float(distance.detach()),
        "nll": float(nll.detach()),
        "critical_loss": 0.0,
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": 0.0,
    }


def segmentation_loss(
    output: SegmentationOutput,
    batch: dict[str, torch.Tensor | list[str]],
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["mask"]
    assert isinstance(target, torch.Tensor)
    logits = output.segmentation_logits
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    reduce_dims = (2, 3, 4)
    intersection = (probability * target).sum(reduce_dims)
    denominator = probability.sum(reduce_dims) + target.sum(reduce_dims)
    dice_per_channel = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - dice_per_channel.mean()
    total = bce + dice_loss
    return total, {
        "loss": float(total.detach()),
        "field_loss": float(dice_loss.detach()),
        "distance_loss": 0.0,
        "nll": 0.0,
        "critical_loss": 0.0,
        "mae_mm": float(dice_loss.detach()),
        "sigma_mm": 0.0,
        "coupling_entropy": 0.0,
        "dice": float(dice_per_channel.mean().detach()),
        "bce": float(bce.detach()),
    }


def critical_attention_loss(
    output: CriticalAttentionOutput,
    batch: dict[str, torch.Tensor | list[str]],
    critical_temperature_mm: float = 1.0,
    critical_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    critical, alignment = _critical_localization_terms(
        output, batch, critical_temperature_mm=critical_temperature_mm
    )
    target_distance = batch["distance_mm"]
    assert isinstance(target_distance, torch.Tensor)

    error = target_distance - output.distance_mm
    distance = F.smooth_l1_loss(output.distance_mm, target_distance)
    nll = (0.5 * (error / output.sigma_mm).square() + output.sigma_mm.log()).mean()
    total = distance + 0.02 * nll + critical_weight * critical
    return total, {
        "loss": float(total.detach()),
        "field_loss": 0.0,
        "distance_loss": float(distance.detach()),
        "nll": float(nll.detach()),
        "critical_loss": float(critical.detach()),
        "critical_alignment": float(alignment.detach()),
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": 0.0,
        "geometry_distance_vox": float(output.geometry_distance_vox.mean().detach()),
    }


def _critical_localization_terms(
    output: CriticalAttentionOutput,
    batch: dict[str, torch.Tensor | list[str]],
    critical_temperature_mm: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_sdf = batch["sdf"]
    assert isinstance(target_sdf, torch.Tensor)
    own_a = target_sdf[:, 0].abs()
    own_b = target_sdf[:, 1].abs()
    cross_a = target_sdf[:, 1].clamp_min(0.0)
    cross_b = target_sdf[:, 0].clamp_min(0.0)
    target_a = torch.softmax((-(own_a + cross_a) / critical_temperature_mm).flatten(1), dim=1)
    target_b = torch.softmax((-(own_b + cross_b) / critical_temperature_mm).flatten(1), dim=1)
    log_a = F.log_softmax(output.critical_logits[:, 0].flatten(1), dim=1)
    log_b = F.log_softmax(output.critical_logits[:, 1].flatten(1), dim=1)
    critical = F.kl_div(log_a, target_a, reduction="batchmean")
    critical = critical + F.kl_div(log_b, target_b, reduction="batchmean")
    predicted_a = log_a.exp()
    predicted_b = log_b.exp()
    alignment_a = F.cosine_similarity(predicted_a, target_a, dim=1).mean()
    alignment_b = F.cosine_similarity(predicted_b, target_b, dim=1).mean()
    return critical, 0.5 * (alignment_a + alignment_b)


def critical_localization_loss(
    output: CriticalAttentionOutput,
    batch: dict[str, torch.Tensor | list[str]],
    critical_temperature_mm: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    critical, alignment = _critical_localization_terms(
        output, batch, critical_temperature_mm=critical_temperature_mm
    )
    target_distance = batch["distance_mm"]
    assert isinstance(target_distance, torch.Tensor)
    error = target_distance - output.distance_mm
    return critical, {
        "loss": float(critical.detach()),
        "field_loss": 0.0,
        "distance_loss": 0.0,
        "nll": 0.0,
        "critical_loss": float(critical.detach()),
        "critical_alignment": float(alignment.detach()),
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": 0.0,
        "geometry_distance_vox": float(output.geometry_distance_vox.mean().detach()),
    }


def surface_critical_attention_loss(
    output: SurfaceCriticalAttentionOutput,
    batch: dict[str, torch.Tensor | list[str]],
    critical_weight: float = 0.1,
    surface_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    critical, alignment = _critical_localization_terms(output, batch)
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
    total = distance + 0.02 * nll + critical_weight * critical + surface_weight * surface
    return total, {
        "loss": float(total.detach()),
        "field_loss": float(surface.detach()),
        "distance_loss": float(distance.detach()),
        "nll": float(nll.detach()),
        "critical_loss": float(critical.detach()),
        "critical_alignment": float(alignment.detach()),
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": 0.0,
        "geometry_distance_vox": float(output.geometry_distance_vox.mean().detach()),
    }


def surface_critical_localization_loss(
    output: SurfaceCriticalAttentionOutput,
    batch: dict[str, torch.Tensor | list[str]],
    critical_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    critical, alignment = _critical_localization_terms(output, batch)
    target_sdf = batch["sdf"]
    target_mask = batch["mask"]
    target_distance = batch["distance_mm"]
    assert isinstance(target_sdf, torch.Tensor)
    assert isinstance(target_mask, torch.Tensor)
    assert isinstance(target_distance, torch.Tensor)
    surface = _surface_field_loss(output.sdf, target_sdf, target_mask)
    total = surface + critical_weight * critical
    error = target_distance - output.distance_mm
    return total, {
        "loss": float(total.detach()),
        "field_loss": float(surface.detach()),
        "distance_loss": 0.0,
        "nll": 0.0,
        "critical_loss": float(critical.detach()),
        "critical_alignment": float(alignment.detach()),
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": 0.0,
        "geometry_distance_vox": float(output.geometry_distance_vox.mean().detach()),
    }


def field_only_smoke_loss(
    output: ModelOutput,
    batch: dict[str, torch.Tensor | list[str]],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sdf = batch["sdf"]
    target_mask = batch["mask"]
    target_distance = batch["distance_mm"]
    assert isinstance(target_sdf, torch.Tensor)
    assert isinstance(target_mask, torch.Tensor)
    assert isinstance(target_distance, torch.Tensor)
    field = _surface_field_loss(output.sdf, target_sdf, target_mask)
    error = target_distance - output.distance_mm
    return field, {
        "loss": float(field.detach()),
        "field_loss": float(field.detach()),
        "distance_loss": 0.0,
        "nll": 0.0,
        "critical_loss": 0.0,
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": float(output.coupling_entropy.mean().detach()),
        "geometry_mae_mm": float((target_distance - output.geometry_distance_mm).abs().mean().detach()),
        "residual_abs_mm": float(output.residual_mm.abs().mean().detach()),
    }


def analytic_smoke_loss(
    output: ModelOutput,
    batch: dict[str, torch.Tensor | list[str]],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sdf = batch["sdf"]
    target_mask = batch["mask"]
    target_distance = batch["distance_mm"]
    assert isinstance(target_sdf, torch.Tensor)
    assert isinstance(target_mask, torch.Tensor)
    assert isinstance(target_distance, torch.Tensor)
    field = _surface_field_loss(output.sdf, target_sdf, target_mask)
    distance = F.smooth_l1_loss(output.distance_mm, target_distance)
    error = target_distance - output.distance_mm
    total = field + distance
    return total, {
        "loss": float(total.detach()),
        "field_loss": float(field.detach()),
        "distance_loss": float(distance.detach()),
        "nll": 0.0,
        "critical_loss": 0.0,
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": 1.0,
        "coupling_entropy": float(output.coupling_entropy.mean().detach()),
        "geometry_mae_mm": float(error.abs().mean().detach()),
        "residual_abs_mm": 0.0,
    }


def anchored_smoke_loss(
    output: ModelOutput,
    batch: dict[str, torch.Tensor | list[str]],
    critical_temperature_mm: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sdf = batch["sdf"]
    target_mask = batch["mask"]
    target_distance = batch["distance_mm"]
    assert isinstance(target_sdf, torch.Tensor)
    assert isinstance(target_mask, torch.Tensor)
    assert isinstance(target_distance, torch.Tensor)
    field = _surface_field_loss(output.sdf, target_sdf, target_mask)
    distance = F.smooth_l1_loss(output.distance_mm, target_distance)
    error = target_distance - output.distance_mm
    nll = (0.5 * (error / output.sigma_mm).square() + output.sigma_mm.log()).mean()

    own_a = _gather_flat(target_sdf[:, 0:1], output.index_a).squeeze(-1).abs()
    own_b = _gather_flat(target_sdf[:, 1:2], output.index_b).squeeze(-1).abs()
    cross_a = _gather_flat(target_sdf[:, 1:2], output.index_a).squeeze(-1).clamp_min(0.0)
    cross_b = _gather_flat(target_sdf[:, 0:1], output.index_b).squeeze(-1).clamp_min(0.0)
    target_a = torch.softmax(-(own_a + cross_a) / critical_temperature_mm, dim=1)
    target_b = torch.softmax(-(own_b + cross_b) / critical_temperature_mm, dim=1)
    critical = F.kl_div(output.marginal_a.clamp_min(1e-8).log(), target_a, reduction="batchmean")
    critical = critical + F.kl_div(output.marginal_b.clamp_min(1e-8).log(), target_b, reduction="batchmean")
    total = field + distance + 0.02 * nll + 0.05 * critical
    return total, {
        "loss": float(total.detach()),
        "field_loss": float(field.detach()),
        "distance_loss": float(distance.detach()),
        "nll": float(nll.detach()),
        "critical_loss": float(critical.detach()),
        "mae_mm": float(error.abs().mean().detach()),
        "sigma_mm": float(output.sigma_mm.mean().detach()),
        "coupling_entropy": float(output.coupling_entropy.mean().detach()),
        "geometry_mae_mm": float((target_distance - output.geometry_distance_mm).abs().mean().detach()),
        "residual_abs_mm": float(output.residual_mm.abs().mean().detach()),
    }
