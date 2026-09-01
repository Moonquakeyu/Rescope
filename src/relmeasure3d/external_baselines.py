from __future__ import annotations

import torch
from torch import nn

from relmeasure3d.model import SegmentationOutput


class MonaiSegResNetBaseline(nn.Module):
    """MONAI SegResNet pair-segmentation baseline for segmentation-to-distance evaluation."""

    def __init__(self, base: int = 24):
        super().__init__()
        from monai.networks.nets import SegResNet

        self.network = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=base,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.1,
        )

    def forward(self, image: torch.Tensor) -> SegmentationOutput:
        logits = self.network(image)
        batch = image.shape[0]
        return SegmentationOutput(
            segmentation_logits=logits,
            distance_mm=torch.zeros(batch, device=image.device, dtype=image.dtype),
            sigma_mm=torch.ones(batch, device=image.device, dtype=image.dtype),
        )


class MonaiDynUNetBaseline(nn.Module):
    """MONAI DynUNet configured as a strong nnU-Net-style 3D segmentation baseline."""

    def __init__(self, base: int = 24):
        super().__init__()
        from monai.networks.nets import DynUNet

        filters = (base, base * 2, base * 4, base * 8, min(base * 12, 320))
        self.network = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=((3, 3, 3),) * 5,
            strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            upsample_kernel_size=((2, 2, 2),) * 4,
            filters=filters,
            norm_name=("INSTANCE", {"affine": True}),
            deep_supervision=False,
            res_block=True,
        )

    def forward(self, image: torch.Tensor) -> SegmentationOutput:
        logits = self.network(image)
        batch = image.shape[0]
        return SegmentationOutput(
            segmentation_logits=logits,
            distance_mm=torch.zeros(batch, device=image.device, dtype=image.dtype),
            sigma_mm=torch.ones(batch, device=image.device, dtype=image.dtype),
        )


class MonaiMedNeXtBaseline(nn.Module):
    """MONAI MedNeXt baseline using the official 3D architecture implementation."""

    def __init__(self, base: int = 24):
        super().__init__()
        from monai.networks.nets import MedNeXt

        self.network = MedNeXt(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=base,
            kernel_size=5,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            deep_supervision=False,
            use_residual_connection=True,
        )

    def forward(self, image: torch.Tensor) -> SegmentationOutput:
        logits = self.network(image)
        batch = image.shape[0]
        return SegmentationOutput(
            segmentation_logits=logits,
            distance_mm=torch.zeros(batch, device=image.device, dtype=image.dtype),
            sigma_mm=torch.ones(batch, device=image.device, dtype=image.dtype),
        )


class NNUNetResEncBaseline(nn.Module):
    """nnU-Net v2 residual-encoder architecture under the matched 80^3 protocol."""

    def __init__(self, base: int = 32):
        super().__init__()
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

        widths = (base, base * 2, base * 4, base * 8, min(base * 10, 320))
        self.network = ResidualEncoderUNet(
            input_channels=1,
            n_stages=5,
            features_per_stage=widths,
            conv_op=nn.Conv3d,
            kernel_sizes=(3, 3, 3, 3, 3),
            strides=(1, 2, 2, 2, 2),
            n_blocks_per_stage=(1, 3, 4, 6, 6),
            num_classes=2,
            n_conv_per_stage_decoder=(1, 1, 1, 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )

    def forward(self, image: torch.Tensor) -> SegmentationOutput:
        logits = self.network(image)
        batch = image.shape[0]
        return SegmentationOutput(
            segmentation_logits=logits,
            distance_mm=torch.zeros(batch, device=image.device, dtype=image.dtype),
            sigma_mm=torch.ones(batch, device=image.device, dtype=image.dtype),
        )
