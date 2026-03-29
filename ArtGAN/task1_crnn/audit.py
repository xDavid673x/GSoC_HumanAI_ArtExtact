from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_class_names, load_records, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the leak-free multitask WikiArt split.")
    parser.add_argument(
        "--dataset-dir",
        default="WikiArt Dataset",
        help="Directory containing the Style, Genre, and Artist split files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the audit summary as JSON.",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.dataset_dir)
    records = load_records(args.dataset_dir)
    summary = summarize_records(records, class_names)
    train_records = [record for record in records if record.global_split == "train"]
    val_records = [record for record in records if record.global_split == "val"]
    split_summary = {
        "all": summary,
        "train": summarize_records(train_records, class_names),
        "val": summarize_records(val_records, class_names),
    }

    print(json.dumps(split_summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
