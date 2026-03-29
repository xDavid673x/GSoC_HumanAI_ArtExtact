from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TASKS, WikiArtMultiTaskDataset
from .metrics import compute_task_metrics, mean_macro_f1, save_confusion_csv
from .model import ConvRecurrentWikiArtClassifier
from .train import choose_device


@torch.no_grad()
def collect_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, dict[str, Any]]:
    model.eval()
    collected: dict[str, dict[str, list[Any]]] = {
        task: {"labels": [], "probabilities": [], "embeddings": [], "paths": [], "predictions": []}
        for task in TASKS
    }

    total_batches = 0
    for batch in loader:
        total_batches += 1
        outputs = model(batch["image"].to(device))
        embeddings = torch.nn.functional.normalize(outputs["embedding"], dim=-1).cpu()

        for task in TASKS:
            mask = batch[f"mask_{task}"]
            if mask.sum().item() == 0:
                continue
            task_embeddings = embeddings[mask]
            task_probabilities = torch.softmax(outputs[task][mask.to(device)], dim=-1).cpu()
            task_labels = batch[task][mask].cpu()
            task_predictions = task_probabilities.argmax(dim=-1)
            task_paths = [path for path, keep in zip(batch["path"], mask.tolist()) if keep]

            collected[task]["labels"].append(task_labels)
            collected[task]["probabilities"].append(task_probabilities)
            collected[task]["embeddings"].append(task_embeddings)
            collected[task]["predictions"].append(task_predictions)
            collected[task]["paths"].extend(task_paths)

        if limit_batches is not None and total_batches >= limit_batches:
            break

    output: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        if collected[task]["probabilities"]:
            output[task] = {
                "labels": torch.cat(collected[task]["labels"]).numpy(),
                "probabilities": torch.cat(collected[task]["probabilities"]).numpy(),
                "embeddings": torch.cat(collected[task]["embeddings"]).numpy(),
                "predictions": torch.cat(collected[task]["predictions"]).numpy(),
                "paths": collected[task]["paths"],
            }
        else:
            output[task] = {
                "labels": np.empty((0,), dtype=np.int64),
                "probabilities": np.empty((0, 0), dtype=np.float32),
                "embeddings": np.empty((0, 0), dtype=np.float32),
                "predictions": np.empty((0,), dtype=np.int64),
                "paths": [],
            }
    return output


def build_centroids(train_outputs: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    centroids: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in TASKS}
    for task in TASKS:
        labels = train_outputs[task]["labels"]
        embeddings = train_outputs[task]["embeddings"]
        if len(labels) == 0:
            continue
        for label in np.unique(labels):
            label_embeddings = embeddings[labels == label]
            centroid = label_embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            distances = 1.0 - label_embeddings @ centroid
            centroids[task][int(label)] = {
                "centroid": centroid,
                "distance_mean": float(distances.mean()),
                "distance_std": float(max(distances.std(), 1e-3)),
            }
    return centroids


def compute_outliers(
    eval_outputs: dict[str, dict[str, Any]],
    centroids: dict[str, dict[int, dict[str, Any]]],
    class_names: dict[str, list[str]],
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    per_task_outliers: dict[str, list[dict[str, Any]]] = {}

    for task in TASKS:
        labels = eval_outputs[task]["labels"]
        probabilities = eval_outputs[task]["probabilities"]
        embeddings = eval_outputs[task]["embeddings"]
        predictions = eval_outputs[task]["predictions"]
        paths = eval_outputs[task]["paths"]
        rows: list[dict[str, Any]] = []

        for index, (label, prediction, path) in enumerate(zip(labels, predictions, paths)):
            centroid_stats = centroids[task].get(int(label))
            if centroid_stats is None:
                continue
            embedding = embeddings[index]
            true_probability = float(probabilities[index, label])
            predicted_probability = float(probabilities[index, prediction])
            centroid = centroid_stats["centroid"]
            cosine_distance = float(1.0 - embedding @ centroid)
            standardized_distance = (cosine_distance - centroid_stats["distance_mean"]) / centroid_stats["distance_std"]
            outlier_score = float((-np.log(max(true_probability, 1e-8))) + standardized_distance)
            rows.append(
                {
                    "path": path,
                    "true_label": class_names[task][int(label)],
                    "predicted_label": class_names[task][int(prediction)],
                    "true_probability": true_probability,
                    "predicted_probability": predicted_probability,
                    "cosine_distance": cosine_distance,
                    "standardized_distance": standardized_distance,
                    "outlier_score": outlier_score,
                    "misclassified": bool(prediction != label),
                }
            )

        rows.sort(key=lambda item: item["outlier_score"], reverse=True)
        per_task_outliers[task] = rows[:top_n]
    return per_task_outliers


def write_outliers_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the multitask CRNN and extract outliers.")
    parser.add_argument("--checkpoint", default="outputs/task1_crnn/best.pt")
    parser.add_argument("--dataset-dir", default="WikiArt Dataset")
    parser.add_argument("--archive-path", default="wikiart.zip")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="outputs/task1_crnn/eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--top-outliers", type=int, default=25)
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = WikiArtMultiTaskDataset(
        dataset_dir=args.dataset_dir,
        archive_path=args.archive_path,
        image_root=args.image_root,
        split="train",
        image_size=checkpoint["args"]["image_size"],
        crop_size=checkpoint["args"]["crop_size"],
        augment=False,
    )
    val_dataset = WikiArtMultiTaskDataset(
        dataset_dir=args.dataset_dir,
        archive_path=args.archive_path,
        image_root=args.image_root,
        split="val",
        image_size=checkpoint["args"]["image_size"],
        crop_size=checkpoint["args"]["crop_size"],
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConvRecurrentWikiArtClassifier(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_outputs = collect_outputs(model, train_loader, device, limit_batches=args.limit_train_batches)
    val_outputs = collect_outputs(model, val_loader, device, limit_batches=args.limit_val_batches)

    metrics: Dict[str, Any] = {}
    for task in TASKS:
        task_metrics = compute_task_metrics(val_outputs[task]["labels"], val_outputs[task]["probabilities"], val_dataset.class_names[task])
        metrics[task] = task_metrics
        save_confusion_csv(
            task_metrics["confusion_matrix"],
            val_dataset.class_names[task],
            output_dir / f"{task}_confusion.csv",
        )
    metrics["mean_macro_f1"] = mean_macro_f1({task: metrics[task] for task in TASKS})

    centroids = build_centroids(train_outputs)
    outliers = compute_outliers(val_outputs, centroids, val_dataset.class_names, args.top_outliers)
    for task, rows in outliers.items():
        write_outliers_csv(rows, output_dir / f"{task}_outliers.csv")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "outliers.json").write_text(json.dumps(outliers, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "outliers": outliers}, indent=2))


if __name__ == "__main__":
    main()
