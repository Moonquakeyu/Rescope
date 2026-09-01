#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from relmeasure3d.data import NPZRelationDataset, save_json
from relmeasure3d.model import (
    CriticalAttentionRegression,
    CriticalRefinementRegression,
    DirectRegressionSmoke,
    RelMeasure3DSmoke,
    SegmentationDistanceBaseline,
    SurfaceCriticalAttentionRegression,
    analytic_smoke_loss,
    anchored_smoke_loss,
    critical_attention_loss,
    critical_localization_loss,
    direct_smoke_loss,
    field_only_smoke_loss,
    smoke_loss,
    segmentation_loss,
    surface_critical_attention_loss,
    surface_critical_localization_loss,
)
from relmeasure3d.publication_model import (
    PublicationAnalyticCriticalPooling,
    PublicationAnalyticSoftmin,
    PublicationDirectRegression,
    PublicationRelMeasure3D,
    PublicationSegmentationBaseline,
    PublicationSurfaceGlobalRegression,
    surface_global_loss,
    surface_global_warmup_loss,
)


def parse_deadline(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_predictions: bool = False,
) -> tuple[float, list[dict[str, float | str]]]:
    model.eval()
    errors = []
    predictions: list[dict[str, float | str]] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            image = batch["image"].to(device)
            pred = model(image)
            gt = float(batch["distance_mm"].item())
            if hasattr(pred, "segmentation_logits"):
                target_mask = batch["mask"].to(device)
                probability = torch.sigmoid(pred.segmentation_logits)
                intersection = (probability * target_mask).sum((2, 3, 4))
                denominator = probability.sum((2, 3, 4)) + target_mask.sum((2, 3, 4))
                value = float(((2.0 * intersection + 1.0) / (denominator + 1.0)).mean().item())
                errors.append(1.0 - value)
            else:
                value = float(pred.distance_mm.item())
                errors.append(abs(value - gt))
            if include_predictions:
                prediction: dict[str, float | str] = {
                    "sample_id": batch["sample_id"][0],
                    "gt_mm": gt,
                    "pred_mm": value,
                    "sigma_mm": float(pred.sigma_mm.item()),
                }
                if hasattr(pred, "geometry_distance_mm"):
                    prediction["geometry_mm"] = float(pred.geometry_distance_mm.item())
                    prediction["residual_mm"] = float(pred.residual_mm.item())
                    prediction["coupling_entropy"] = float(pred.coupling_entropy.item())
                predictions.append(prediction)
    model.train()
    return float(np.mean(errors)), predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--token-count", type=int, default=64)
    parser.add_argument("--base", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--deadline", default=None)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--model",
        choices=(
            "relmeasure", "relmeasure_v2", "analytic", "direct", "critical_attention",
            "critical_refinement", "segmentation", "search_measure", "publication_direct",
            "publication_segmentation", "publication_multitask", "publication_relmeasure",
            "publication_analytic_softmin", "publication_analytic_pool",
            "monai_segresnet", "monai_dynunet", "monai_mednext", "nnunet_resenc",
        ),
        default="publication_relmeasure",
    )
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--field-warmup-steps", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--encoder-init", default=None)
    parser.add_argument("--direct-init", default=None)
    parser.add_argument("--critical-weight", type=float, default=0.05)
    parser.add_argument("--surface-weight", type=float, default=0.1)
    parser.add_argument("--critical-warmup-steps", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--crop-jitter-voxels", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    validation_metrics_path = out / "validation_metrics.jsonl"
    dataset = NPZRelationDataset(args.data)
    train_dataset = NPZRelationDataset(
        args.data,
        crop_size=args.crop_size,
        random_crop=args.crop_jitter_voxels > 0,
        max_jitter_voxels=args.crop_jitter_voxels,
    )
    val_dataset = NPZRelationDataset(args.data, crop_size=args.crop_size)
    if args.split_json:
        split_payload = json.loads(Path(args.split_json).read_text())
        train_ids = set(split_payload["samples"]["train"])
        val_ids = set(split_payload["samples"]["validation"])
        train_indices = [i for i, path in enumerate(dataset.files) if path.stem in train_ids]
        val_indices = [i for i, path in enumerate(dataset.files) if path.stem in val_ids]
        if not train_indices or not val_indices:
            raise SystemExit("patient split did not match cached sample IDs")
        train_set = Subset(train_dataset, train_indices)
        val_set = Subset(val_dataset, val_indices)
        train_count = len(train_indices)
        val_count = len(val_indices)
    else:
        val_count = max(1, len(dataset) // 5)
        train_count = len(dataset) - val_count
        permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_indices = permutation[:train_count]
        val_indices = permutation[train_count:]
        train_set = Subset(train_dataset, train_indices)
        val_set = Subset(val_dataset, val_indices)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    if args.model == "direct":
        model = DirectRegressionSmoke(base=args.base).to(device)
        loss_fn = direct_smoke_loss
    elif args.model == "publication_direct":
        model = PublicationDirectRegression(base=args.base).to(device)
        loss_fn = direct_smoke_loss
    elif args.model == "segmentation":
        model = SegmentationDistanceBaseline(base=args.base).to(device)
        loss_fn = segmentation_loss
    elif args.model == "publication_segmentation":
        model = PublicationSegmentationBaseline(base=args.base).to(device)
        loss_fn = segmentation_loss
    elif args.model in ("monai_segresnet", "monai_dynunet", "monai_mednext", "nnunet_resenc"):
        from relmeasure3d.external_baselines import (
            MonaiDynUNetBaseline,
            MonaiMedNeXtBaseline,
            MonaiSegResNetBaseline,
            NNUNetResEncBaseline,
        )

        model_class = {
            "monai_segresnet": MonaiSegResNetBaseline,
            "monai_dynunet": MonaiDynUNetBaseline,
            "monai_mednext": MonaiMedNeXtBaseline,
            "nnunet_resenc": NNUNetResEncBaseline,
        }[args.model]
        model = model_class(base=args.base).to(device)
        loss_fn = segmentation_loss
    elif args.model == "publication_multitask":
        model = PublicationSurfaceGlobalRegression(base=args.base).to(device)
        loss_fn = partial(surface_global_loss, surface_weight=args.surface_weight)
    elif args.model == "publication_relmeasure":
        model = PublicationRelMeasure3D(base=args.base).to(device)
        loss_fn = partial(
            surface_critical_attention_loss,
            critical_weight=args.critical_weight,
            surface_weight=args.surface_weight,
        )
    elif args.model == "publication_analytic_softmin":
        model = PublicationAnalyticSoftmin(base=args.base).to(device)
        loss_fn = partial(surface_global_loss, surface_weight=args.surface_weight)
    elif args.model == "publication_analytic_pool":
        model = PublicationAnalyticCriticalPooling(base=args.base).to(device)
        loss_fn = partial(
            surface_critical_attention_loss,
            critical_weight=0.0,
            surface_weight=args.surface_weight,
        )
    elif args.model == "critical_attention":
        model = CriticalAttentionRegression(base=args.base).to(device)
        loss_fn = partial(critical_attention_loss, critical_weight=args.critical_weight)
    elif args.model == "search_measure":
        model = SurfaceCriticalAttentionRegression(base=args.base).to(device)
        loss_fn = partial(
            surface_critical_attention_loss,
            critical_weight=args.critical_weight,
            surface_weight=args.surface_weight,
        )
    elif args.model == "critical_refinement":
        model = CriticalRefinementRegression(base=args.base).to(device)
        loss_fn = partial(critical_attention_loss, critical_weight=args.critical_weight)
    elif args.model == "analytic":
        model = RelMeasure3DSmoke(base=args.base, token_count=args.token_count, coupling_mode="analytic").to(device)
        loss_fn = analytic_smoke_loss
    elif args.model == "relmeasure_v2":
        model = RelMeasure3DSmoke(base=args.base, token_count=args.token_count, coupling_mode="anchored").to(device)
        loss_fn = anchored_smoke_loss
    else:
        model = RelMeasure3DSmoke(base=args.base, token_count=args.token_count).to(device)
        loss_fn = smoke_loss
    if args.encoder_init:
        source_checkpoint = torch.load(args.encoder_init, map_location="cpu", weights_only=False)
        source_state = source_checkpoint["model"]
        target_state = model.state_dict()
        encoder_prefixes = ("backbone.e0.", "backbone.e1.", "backbone.e2.", "backbone.b.")
        matched = {
            key: value
            for key, value in source_state.items()
            if key.startswith(encoder_prefixes) and key in target_state and value.shape == target_state[key].shape
        }
        if not matched:
            raise RuntimeError(f"no compatible encoder parameters found in {args.encoder_init}")
        target_state.update(matched)
        model.load_state_dict(target_state)
        print(json.dumps({"encoder_init": args.encoder_init, "matched_parameters": len(matched)}), flush=True)
    if args.direct_init:
        if not hasattr(model, "direct"):
            raise RuntimeError("--direct-init requires a model with a direct anchor")
        source_checkpoint = torch.load(args.direct_init, map_location="cpu", weights_only=False)
        model.direct.load_state_dict(source_checkpoint["model"])
        print(json.dumps({"direct_init": args.direct_init}), flush=True)
    if args.freeze_backbone:
        if not hasattr(model, "backbone"):
            raise SystemExit("selected model has no backbone to freeze")
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)
    start_step = 0
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = {"coupling_gate_raw"}
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"incompatible resume checkpoint: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    deadline = parse_deadline(args.deadline)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    iterator = iter(loader)
    started = time.time()
    last_metrics: dict[str, float] = {}
    best_val_mae = float("inf")
    best_step = 0
    validations_without_improvement = 0
    stopped_early = False
    for step in range(start_step + 1, args.steps + 1):
        if deadline and dt.datetime.now().astimezone() >= deadline:
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
            if (
                args.model in ("search_measure", "publication_relmeasure")
                and args.critical_weight > 0
                and step <= args.critical_warmup_steps
            ):
                loss, train_metrics = surface_critical_localization_loss(
                    output, batch, critical_weight=args.critical_weight
                )
            elif (
                args.model in ("critical_attention", "critical_refinement")
                and args.critical_weight > 0
                and step <= args.critical_warmup_steps
            ):
                loss, train_metrics = critical_localization_loss(output, batch)
            elif args.model in ("publication_multitask", "publication_analytic_softmin", "publication_analytic_pool") and step <= args.field_warmup_steps:
                loss, train_metrics = surface_global_warmup_loss(output, batch)
            elif args.model in ("relmeasure", "relmeasure_v2", "analytic") and step <= args.field_warmup_steps:
                loss, train_metrics = field_only_smoke_loss(output, batch)
            else:
                loss, train_metrics = loss_fn(output, batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_metrics = dict(train_metrics)
        last_metrics.update(
            {
                "step": step,
                "grad_norm": float(grad_norm),
                "elapsed_s": time.time() - started,
                "gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                "timestamp": dt.datetime.now().astimezone().isoformat(),
            }
        )
        if step == 1 or step % 10 == 0:
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(last_metrics) + "\n")
            print(json.dumps(last_metrics), flush=True)
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "args": vars(args)}, out / f"checkpoint_{step:07d}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "args": vars(args)}, out / "latest.pt")
        should_validate = (
            args.validate_every > 0
            and step > max(args.field_warmup_steps, args.critical_warmup_steps)
            and step % args.validate_every == 0
        )
        if should_validate:
            val_mae, _ = evaluate(model, val_loader, device)
            record = {"step": step, "val_mae_mm": val_mae, "timestamp": dt.datetime.now().astimezone().isoformat()}
            with validation_metrics_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_step = step
                validations_without_improvement = 0
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "args": vars(args)}, out / "best.pt")
            else:
                validations_without_improvement += 1
            if args.early_stop_patience > 0 and validations_without_improvement >= args.early_stop_patience:
                stopped_early = True
                break

    selected_step = int(last_metrics.get("step", start_step))
    selected_optimizer = optimizer.state_dict()
    if (out / "best.pt").exists():
        best_checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        selected_optimizer = best_checkpoint["optimizer"]
        selected_step = int(best_checkpoint["step"])
    val_mae, predictions = evaluate(model, val_loader, device, include_predictions=True)
    final_step = int(last_metrics.get("step", start_step))
    checkpoint = {"model": model.state_dict(), "optimizer": selected_optimizer, "step": selected_step, "args": vars(args)}
    torch.save(checkpoint, out / "latest.pt")
    summary = {
        "status": "complete",
        "model": args.model,
        "seed": args.seed,
        "split_json": args.split_json,
        "final_step": final_step,
        "best_step": best_step if best_step else final_step,
        "stopped_early": stopped_early,
        "train_samples": train_count,
        "val_samples": val_count,
        "val_mae_mm": val_mae,
        "validation_metric": "one_minus_soft_dice" if hasattr(model, "network") or "segmentation" in args.model else "mae_mm",
        "max_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "last_train_metrics": last_metrics,
        "predictions": predictions,
    }
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not np.isfinite(summary["val_mae_mm"]):
        raise SystemExit("non-finite validation metric")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    main()
