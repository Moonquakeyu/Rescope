from __future__ import annotations

import torch

from relmeasure3d.model import direct_smoke_loss
from relmeasure3d.publication_model import PublicationDirectRegression


def test_direct_regression_forward_and_loss() -> None:
    model = PublicationDirectRegression(base=4)
    image = torch.randn(2, 1, 16, 16, 16)
    output = model(image)
    assert output.distance_mm.shape == (2,)
    assert output.sigma_mm.shape == (2,)
    loss, metrics = direct_smoke_loss(output, {"distance_mm": torch.tensor([1.0, 2.0])})
    assert torch.isfinite(loss)
    assert metrics["field_loss"] == 0.0
