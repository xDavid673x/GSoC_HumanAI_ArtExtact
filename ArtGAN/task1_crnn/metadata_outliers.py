from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import WikiArtRecord, load_class_names, load_records


def write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_conditional_counts(
    records: list[WikiArtRecord],
    condition_field: str,
    attribute_field: str,
) -> tuple[Counter[int], Counter[tuple[int, int]], Counter[int]]:
    condition_totals: Counter[int] = Counter()
    pair_totals: Counter[tuple[int, int]] = Counter()
    attribute_totals: Counter[int] = Counter()

    for record in records:
        condition_value = getattr(record, condition_field)
        attribute_value = getattr(record, attribute_field)
        if condition_value is None or attribute_value is None:
            continue
        condition = int(condition_value)
        attribute = int(attribute_value)
        condition_totals[condition] += 1
        pair_totals[(condition, attribute)] += 1
        attribute_totals[attribute] += 1

    return condition_totals, pair_totals, attribute_totals


def compute_surprise(
    *,
    condition: int,
    attribute: int,
    condition_totals: Counter[int],
    pair_totals: Counter[tuple[int, int]],
    attribute_totals: Counter[int],
    attribute_vocab_size: int,
    alpha: float,
) -> dict[str, float]:
    condition_total = int(condition_totals[condition])
    pair_total = int(pair_totals[(condition, attribute)])
    global_total = int(sum(attribute_totals.values()))
    global_attribute_total = int(attribute_totals[attribute])

    conditional_probability = (pair_total + alpha) / (condition_total + alpha * attribute_vocab_size)
    global_probability = (global_attribute_total + alpha) / (global_total + alpha * attribute_vocab_size)
    outlier_score = math.log(global_probability / conditional_probability)

    return {
        "condition_total": condition_total,
        "pair_total": pair_total,
        "global_attribute_total": global_attribute_total,
        "conditional_probability": conditional_probability,
        "global_probability": global_probability,
        "outlier_score": outlier_score,
    }


def rank_conditional_outliers(
    *,
    eval_records: list[WikiArtRecord],
    condition_field: str,
    attribute_field: str,
    condition_names: list[str],
    attribute_names: list[str],
    condition_totals: Counter[int],
    pair_totals: Counter[tuple[int, int]],
    attribute_totals: Counter[int],
    alpha: float,
    min_condition_count: int,
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in eval_records:
        condition_value = getattr(record, condition_field)
        attribute_value = getattr(record, attribute_field)
        if condition_value is None or attribute_value is None:
            continue

        condition = int(condition_value)
        attribute = int(attribute_value)
        if condition_totals[condition] < min_condition_count:
            continue

        stats = compute_surprise(
            condition=condition,
            attribute=attribute,
            condition_totals=condition_totals,
            pair_totals=pair_totals,
            attribute_totals=attribute_totals,
            attribute_vocab_size=len(attribute_names),
            alpha=alpha,
        )
        rows.append(
            {
                "path": record.path,
                "split": record.global_split,
                "condition_label": condition_names[condition],
                "attribute_label": attribute_names[attribute],
                "conditional_count": stats["pair_total"],
                "condition_total": stats["condition_total"],
                "global_attribute_count": stats["global_attribute_total"],
                "conditional_probability": stats["conditional_probability"],
                "global_probability": stats["global_probability"],
                "outlier_score": stats["outlier_score"],
                "source_splits": "|".join(record.source_splits),
                "conflicting_split": record.conflicting_split,
            }
        )

    rows.sort(
        key=lambda row: (
            row["outlier_score"],
            -row["conditional_count"],
            -row["conditional_probability"],
        ),
        reverse=True,
    )
    return rows[:top_n]


def rank_artist_profile_outliers(
    *,
    eval_records: list[WikiArtRecord],
    class_names: dict[str, list[str]],
    artist_totals: Counter[int],
    artist_style_totals: Counter[tuple[int, int]],
    style_totals: Counter[int],
    artist_genre_totals: Counter[tuple[int, int]],
    genre_totals: Counter[int],
    alpha: float,
    min_condition_count: int,
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in eval_records:
        if record.artist is None:
            continue
        artist = int(record.artist)
        if artist_totals[artist] < min_condition_count:
            continue

        style_stats = compute_surprise(
            condition=artist,
            attribute=int(record.style),
            condition_totals=artist_totals,
            pair_totals=artist_style_totals,
            attribute_totals=style_totals,
            attribute_vocab_size=len(class_names["style"]),
            alpha=alpha,
        )
        genre_stats = None
        if record.genre is not None:
            genre_stats = compute_surprise(
                condition=artist,
                attribute=int(record.genre),
                condition_totals=artist_totals,
                pair_totals=artist_genre_totals,
                attribute_totals=genre_totals,
                attribute_vocab_size=len(class_names["genre"]),
                alpha=alpha,
            )

        total_score = style_stats["outlier_score"]
        if genre_stats is not None:
            total_score += genre_stats["outlier_score"]

        rows.append(
            {
                "path": record.path,
                "split": record.global_split,
                "artist": class_names["artist"][artist],
                "style": class_names["style"][int(record.style)],
                "genre": "" if record.genre is None else class_names["genre"][int(record.genre)],
                "artist_total": artist_totals[artist],
                "artist_style_count": style_stats["pair_total"],
                "artist_style_probability": style_stats["conditional_probability"],
                "global_style_probability": style_stats["global_probability"],
                "artist_style_score": style_stats["outlier_score"],
                "artist_genre_count": None if genre_stats is None else genre_stats["pair_total"],
                "artist_genre_probability": None if genre_stats is None else genre_stats["conditional_probability"],
                "global_genre_probability": None if genre_stats is None else genre_stats["global_probability"],
                "artist_genre_score": None if genre_stats is None else genre_stats["outlier_score"],
                "profile_outlier_score": total_score,
                "source_splits": "|".join(record.source_splits),
                "conflicting_split": record.conflicting_split,
            }
        )

    rows.sort(
        key=lambda row: (
            row["profile_outlier_score"],
            -(row["artist_style_count"] if row["artist_style_count"] is not None else 0),
        ),
        reverse=True,
    )
    return rows[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine metadata-only WikiArt outlier candidates from rare label combinations.")
    parser.add_argument("--dataset-dir", default="WikiArt Dataset")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-dir", default="outputs/task1_crnn/metadata_outliers")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-condition-count", type=int, default=25)
    args = parser.parse_args()

    class_names = load_class_names(args.dataset_dir)
    records = load_records(args.dataset_dir)
    train_records = [record for record in records if record.global_split == "train"]
    eval_records = [record for record in records if record.global_split == args.split]

    artist_totals, artist_style_totals, style_totals = build_conditional_counts(train_records, "artist", "style")
    _, artist_genre_totals, genre_totals = build_conditional_counts(train_records, "artist", "genre")
    genre_condition_totals, genre_style_totals, _ = build_conditional_counts(train_records, "genre", "style")

    outputs = {
        "artist_style_outliers": rank_conditional_outliers(
            eval_records=eval_records,
            condition_field="artist",
            attribute_field="style",
            condition_names=class_names["artist"],
            attribute_names=class_names["style"],
            condition_totals=artist_totals,
            pair_totals=artist_style_totals,
            attribute_totals=style_totals,
            alpha=args.alpha,
            min_condition_count=args.min_condition_count,
            top_n=args.top_n,
        ),
        "artist_genre_outliers": rank_conditional_outliers(
            eval_records=eval_records,
            condition_field="artist",
            attribute_field="genre",
            condition_names=class_names["artist"],
            attribute_names=class_names["genre"],
            condition_totals=artist_totals,
            pair_totals=artist_genre_totals,
            attribute_totals=genre_totals,
            alpha=args.alpha,
            min_condition_count=args.min_condition_count,
            top_n=args.top_n,
        ),
        "genre_style_outliers": rank_conditional_outliers(
            eval_records=eval_records,
            condition_field="genre",
            attribute_field="style",
            condition_names=class_names["genre"],
            attribute_names=class_names["style"],
            condition_totals=genre_condition_totals,
            pair_totals=genre_style_totals,
            attribute_totals=style_totals,
            alpha=args.alpha,
            min_condition_count=args.min_condition_count,
            top_n=args.top_n,
        ),
        "artist_profile_outliers": rank_artist_profile_outliers(
            eval_records=eval_records,
            class_names=class_names,
            artist_totals=artist_totals,
            artist_style_totals=artist_style_totals,
            style_totals=style_totals,
            artist_genre_totals=artist_genre_totals,
            genre_totals=genre_totals,
            alpha=args.alpha,
            min_condition_count=args.min_condition_count,
            top_n=args.top_n,
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_csv(rows, output_dir / f"{name}.csv")

    summary = {
        "split": args.split,
        "alpha": args.alpha,
        "min_condition_count": args.min_condition_count,
        "top_n": args.top_n,
        "counts": {
            "train_records": len(train_records),
            "eval_records": len(eval_records),
        },
        "top_hits": {name: rows[:5] for name, rows in outputs.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
