#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from relmeasure3d.data import save_json
from relmeasure3d.publication_model import (
    PublicationDirectRegression,
    PublicationRelMeasure3D,
    PublicationSegmentationBaseline,
    PublicationSurfaceGlobalRegression,
)


MODELS = {
    "publication_direct": PublicationDirectRegression,
    "publication_multitask": PublicationSurfaceGlobalRegression,
    "publication_relmeasure": PublicationRelMeasure3D,
    "publication_segmentation": PublicationSegmentationBaseline,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda")
    model = MODELS[args.model](base=args.base).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image = torch.zeros(1, 1, args.size, args.size, args.size, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup):
            model(image)
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(image)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    result = {
        "status": "complete",
        "model": args.model,
        "base": args.base,
        "input_size": [1, 1, args.size, args.size, args.size],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "latency_ms": {
            "mean": float(np.mean(timings)),
            "std": float(np.std(timings, ddof=1)),
            "p95": float(np.quantile(timings, 0.95)),
        },
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    save_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
