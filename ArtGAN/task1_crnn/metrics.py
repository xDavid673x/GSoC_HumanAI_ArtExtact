from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    if len(y_true) == 0:
        return 0.0
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * in_bin.mean()
    return float(ece)


def top_k_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    topk = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    hits = np.array([label in row for label, row in zip(y_true, topk)], dtype=np.float32)
    return float(hits.mean())


def compute_task_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_names: Iterable[str],
    topk: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    label_names = list(label_names)
    label_indices = list(range(len(label_names)))
    if len(y_true) == 0:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_accuracy": 0.0,
            "ece": 0.0,
            "confusion_matrix": [],
            "topk_accuracy": {},
        }

    predictions = probabilities.argmax(axis=1)
    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "ece": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=label_indices).tolist(),
        "topk_accuracy": {},
    }
    for k in topk:
        if k <= probabilities.shape[1]:
            metrics["topk_accuracy"][str(k)] = top_k_accuracy(y_true, probabilities, k)
    return metrics


def mean_macro_f1(task_metrics: Dict[str, dict[str, Any]]) -> float:
    values = [payload["macro_f1"] for payload in task_metrics.values() if payload["num_samples"] > 0]
    return float(np.mean(values)) if values else 0.0


def save_confusion_csv(matrix: list[list[int]], label_names: list[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + label_names)
        for label_name, row in zip(label_names, matrix):
            writer.writerow([label_name] + row)
