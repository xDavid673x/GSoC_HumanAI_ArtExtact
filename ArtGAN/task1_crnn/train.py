from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import TASKS, WikiArtMultiTaskDataset
from .metrics import compute_task_metrics, mean_macro_f1
from .model import ConvRecurrentWikiArtClassifier


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
    weights = counts.float().clamp_min(1).pow(-0.5)
    weights = weights / weights.mean()
    return weights


def build_loss_functions(dataset: WikiArtMultiTaskDataset, device: torch.device, use_class_weights: bool) -> dict[str, nn.Module]:
    losses: dict[str, nn.Module] = {}
    for task in TASKS:
        weight = None
        if use_class_weights:
            weight = class_weights_from_counts(dataset.class_counts(task)).to(device)
        losses[task] = nn.CrossEntropyLoss(weight=weight)
    return losses


def compute_losses(
    model_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    losses: dict[str, nn.Module],
    task_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss = None
    task_loss_values: dict[str, float] = {}

    for task in TASKS:
        mask = batch[f"mask_{task}"]
        if mask.sum().item() == 0:
            task_loss_values[task] = 0.0
            continue
        logits = model_outputs[task][mask]
        labels = batch[task][mask]
        loss = losses[task](logits, labels)
        weighted = loss * task_weights[task]
        total_loss = weighted if total_loss is None else total_loss + weighted
        task_loss_values[task] = float(loss.detach().item())

    if total_loss is None:
        total_loss = torch.tensor(0.0, device=next(iter(model_outputs.values())).device)
    return total_loss, task_loss_values


def build_grad_scaler(device: torch.device, use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def save_checkpoint(
    output_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: Any,
    num_classes: dict[str, int],
    class_names: dict[str, list[str]],
    args: argparse.Namespace,
    epoch: int,
    best_score: float,
    best_validation: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "num_classes": num_classes,
        "class_names": class_names,
        "args": vars(args),
        "best_score": best_score,
        "best_validation": best_validation,
        "history": history,
    }
    torch.save(checkpoint, output_path)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
    show_progress: bool = False,
    epoch: int | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    task_labels: dict[str, list[torch.Tensor]] = {task: [] for task in TASKS}
    task_probabilities: dict[str, list[torch.Tensor]] = {task: [] for task in TASKS}
    total_batches = 0

    progress_total = len(loader) if limit_batches is None else min(len(loader), limit_batches)
    batch_iterator = loader
    if show_progress:
        desc = "Validation" if epoch is None else f"Epoch {epoch} val"
        batch_iterator = tqdm(loader, total=progress_total, desc=desc, leave=False)

    for batch in batch_iterator:
        total_batches += 1
        images = batch["image"].to(device)
        with autocast_context(device, use_amp):
            outputs = model(images)
        for task in TASKS:
            mask = batch[f"mask_{task}"]
            if mask.sum().item() == 0:
                continue
            logits = outputs[task][mask.to(device)]
            probabilities = torch.softmax(logits, dim=-1).cpu()
            labels = batch[task][mask].cpu()
            task_probabilities[task].append(probabilities)
            task_labels[task].append(labels)

        if limit_batches is not None and total_batches >= limit_batches:
            break

    if show_progress and hasattr(batch_iterator, "close"):
        batch_iterator.close()

    metrics: dict[str, Any] = {}
    for task in TASKS:
        if task_probabilities[task]:
            probabilities = torch.cat(task_probabilities[task]).numpy()
            labels = torch.cat(task_labels[task]).numpy()
        else:
            probabilities = torch.empty(0, 0).numpy()
            labels = torch.empty(0, dtype=torch.long).numpy()
        metrics[task] = compute_task_metrics(labels, probabilities, loader.dataset.class_names[task])
    metrics["mean_macro_f1"] = mean_macro_f1({task: metrics[task] for task in TASKS})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multitask convolutional-recurrent WikiArt classifier.")
    parser.add_argument("--dataset-dir", default="WikiArt Dataset")
    parser.add_argument("--archive-path", default="wikiart.zip")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="outputs/task1_crnn")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--style-loss-weight", type=float, default=1.0)
    parser.add_argument("--genre-loss-weight", type=float, default=1.0)
    parser.add_argument("--artist-loss-weight", type=float, default=1.2)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = WikiArtMultiTaskDataset(
        dataset_dir=args.dataset_dir,
        archive_path=args.archive_path,
        image_root=args.image_root,
        split="train",
        image_size=args.image_size,
        crop_size=args.crop_size,
    )
    val_dataset = WikiArtMultiTaskDataset(
        dataset_dir=args.dataset_dir,
        archive_path=args.archive_path,
        image_root=args.image_root,
        split="val",
        image_size=args.image_size,
        crop_size=args.crop_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    num_classes = {task: len(train_dataset.class_names[task]) for task in TASKS}
    model = ConvRecurrentWikiArtClassifier(
        num_classes=num_classes,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = build_grad_scaler(device, use_amp)
    losses = build_loss_functions(train_dataset, device, not args.disable_class_weights)
    task_weights = {
        "style": args.style_loss_weight,
        "genre": args.genre_loss_weight,
        "artist": args.artist_loss_weight,
    }

    best_score = float("-inf")
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_score = float(checkpoint.get("best_score", checkpoint.get("best_validation", {}).get("mean_macro_f1", float("-inf"))))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(json.dumps({"resumed_from": args.resume_from, "start_epoch": start_epoch, "best_score": best_score}, indent=2))

    def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for task in TASKS:
            task_metrics = metrics[task]
            compact[task] = {key: value for key, value in task_metrics.items() if key != "confusion_matrix"}
        compact["mean_macro_f1"] = metrics["mean_macro_f1"]
        return compact

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_steps = 0
        progress_total = len(train_loader) if args.limit_train_batches is None else min(len(train_loader), args.limit_train_batches)
        train_iterator = train_loader
        if not args.disable_progress:
            train_iterator = tqdm(train_loader, total=progress_total, desc=f"Epoch {epoch} train", leave=True)

        for step, batch in enumerate(train_iterator, start=1):
            images = batch["image"].to(device)
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                outputs = model(images)
                loss, task_loss_values = compute_losses(outputs, tensor_batch, losses, task_weights)
            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().item())
            running_steps += 1

            if not args.disable_progress:
                train_iterator.set_postfix(
                    avg_loss=f"{running_loss / running_steps:.4f}",
                    style=f"{task_loss_values['style']:.3f}",
                    genre=f"{task_loss_values['genre']:.3f}",
                    artist=f"{task_loss_values['artist']:.3f}",
                )

            if args.limit_train_batches is not None and step >= args.limit_train_batches:
                break

        if not args.disable_progress and hasattr(train_iterator, "close"):
            train_iterator.close()

        scheduler.step()
        validation_metrics = run_validation(
            model,
            val_loader,
            device,
            limit_batches=args.limit_val_batches,
            show_progress=not args.disable_progress,
            epoch=epoch,
            use_amp=use_amp,
        )
        train_loss = running_loss / max(running_steps, 1)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_summary)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "validation": compact_metrics(validation_metrics),
                },
                indent=2,
            )
        )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            num_classes=num_classes,
            class_names=train_dataset.class_names,
            args=args,
            epoch=epoch,
            best_score=best_score,
            best_validation=validation_metrics,
            history=history,
        )

        if validation_metrics["mean_macro_f1"] > best_score:
            best_score = validation_metrics["mean_macro_f1"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                num_classes=num_classes,
                class_names=train_dataset.class_names,
                args=args,
                epoch=epoch,
                best_score=best_score,
                best_validation=validation_metrics,
                history=history,
            )

    summary = {
        "best_mean_macro_f1": best_score,
        "history": history,
        "train_dataset_summary": train_dataset.dataset_summary(),
        "val_dataset_summary": val_dataset.dataset_summary(),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
