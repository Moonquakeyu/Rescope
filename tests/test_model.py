from __future__ import annotations

import pytest
import torch

from relmeasure3d.publication_model import PublicationRelMeasure3D, RelScope


def test_relscope_cpu_forward() -> None:
    model = RelScope(base=4).eval()
    image = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        output = model(image)
    assert output.distance_mm.shape == (1,)
    assert output.sigma_mm.shape == (1,)
    assert output.sdf.shape == (1, 2, 16, 16, 16)
    assert output.critical_logits.shape == (1, 2, 16, 16, 16)
    assert torch.allclose(output.attention_a.sum((2, 3, 4)), torch.ones(1, 1), atol=1e-5)
    assert torch.allclose(output.attention_b.sum((2, 3, 4)), torch.ones(1, 1), atol=1e-5)


def test_development_class_name_is_checkpoint_compatible_alias() -> None:
    assert RelScope is PublicationRelMeasure3D


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_relscope_cuda_forward_backward() -> None:
    model = RelScope(base=4).cuda()
    image = torch.randn(1, 1, 16, 16, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(image)
        loss = output.sdf.square().mean() + output.distance_mm.mean() + output.sigma_mm.mean()
    loss.backward()
    assert torch.isfinite(loss)
