from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import warnings
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageFile, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

TASKS = ("style", "genre", "artist")
MISSING_LABEL = -1


@dataclass(frozen=True)
class WikiArtRecord:
    path: str
    global_split: str
    style: int
    genre: Optional[int]
    artist: Optional[int]
    source_splits: tuple[str, ...]
    conflicting_split: bool


def _read_style_or_genre_split(path: Path) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_path, label = line.rsplit(",", 1)
            items.append((image_path, int(label)))
    return items


def _read_artist_split(path: Path) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("(Path to image)"):
                continue
            image_path, label = line.rsplit(",,", 1)
            items.append((image_path.strip(), int(label.strip())))
    return items


def _read_class_map(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            index, name = line.split(maxsplit=1)
            mapping[int(index)] = name
    return mapping


def load_class_names(dataset_dir: str | Path) -> dict[str, list[str]]:
    dataset_path = Path(dataset_dir)
    style_map = _read_class_map(dataset_path / "Style" / "style_class.txt")
    genre_map = _read_class_map(dataset_path / "Genre" / "genre_class")
    artist_map = _read_class_map(dataset_path / "Artist" / "artist_class")
    return {
        "style": [style_map[index] for index in sorted(style_map)],
        "genre": [genre_map[index] for index in sorted(genre_map)],
        "artist": [artist_map[index] for index in sorted(artist_map)],
    }


def load_records(dataset_dir: str | Path) -> list[WikiArtRecord]:
    dataset_path = Path(dataset_dir)
    merged: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "style": None,
            "genre": None,
            "artist": None,
            "source_splits": set(),
        }
    )

    for split_name in ("train", "val"):
        for image_path, label in _read_style_or_genre_split(dataset_path / "Style" / f"style_{split_name}.csv"):
            entry = merged[image_path]
            entry["style"] = label
            entry["source_splits"].add(f"style_{split_name}")

        for image_path, label in _read_style_or_genre_split(dataset_path / "Genre" / f"genre_{split_name}.csv"):
            entry = merged[image_path]
            entry["genre"] = label
            entry["source_splits"].add(f"genre_{split_name}")

        for image_path, label in _read_artist_split(dataset_path / "Artist" / f"artist_{split_name}"):
            entry = merged[image_path]
            entry["artist"] = label
            entry["source_splits"].add(f"artist_{split_name}")

    records: list[WikiArtRecord] = []
    for image_path, entry in merged.items():
        has_train = any(name.endswith("train") for name in entry["source_splits"])
        has_val = any(name.endswith("val") for name in entry["source_splits"])
        global_split = "val" if has_val else "train"
        if entry["style"] is None:
            raise ValueError(f"Style label missing for {image_path}")
        records.append(
            WikiArtRecord(
                path=image_path,
                global_split=global_split,
                style=int(entry["style"]),
                genre=None if entry["genre"] is None else int(entry["genre"]),
                artist=None if entry["artist"] is None else int(entry["artist"]),
                source_splits=tuple(sorted(entry["source_splits"])),
                conflicting_split=has_train and has_val,
            )
        )

    records.sort(key=lambda item: item.path)
    return records


def summarize_records(records: Iterable[WikiArtRecord], class_names: Optional[dict[str, list[str]]] = None) -> dict[str, Any]:
    items = list(records)
    summary: dict[str, Any] = {
        "num_records": len(items),
        "conflicting_split_records": sum(record.conflicting_split for record in items),
        "split_counts": dict(Counter(record.global_split for record in items)),
        "label_coverage": {},
    }

    for task in TASKS:
        counts = Counter(getattr(record, task) for record in items if getattr(record, task) is not None)
        summary["label_coverage"][task] = {
            "num_labeled_examples": sum(counts.values()),
            "num_classes_observed": len(counts),
            "class_counts": {int(label): int(count) for label, count in sorted(counts.items())},
        }
        if class_names is not None:
            summary["label_coverage"][task]["class_names"] = class_names[task]
    return summary


class WikiArtMultiTaskDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        archive_path: str | Path,
        split: str,
        image_root: Optional[str | Path] = None,
        image_size: int = 256,
        crop_size: int = 224,
        augment: Optional[bool] = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.dataset_dir = Path(dataset_dir)
        self.archive_path = Path(archive_path)
        self.image_root = None if image_root is None else Path(image_root)
        self.split = split
        self.class_names = load_class_names(self.dataset_dir)
        self.records = [record for record in load_records(self.dataset_dir) if record.global_split == split]
        self._zip_file: Optional[ZipFile] = None
        self._archive_names: Optional[set[str]] = None
        self._bad_paths: set[str] = set()
        self._warned_bad_paths: set[str] = set()
        self.augment = split == "train" if augment is None else augment

        resize_size = max(image_size, crop_size)
        if self.augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resize_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomResizedCrop(
                        crop_size,
                        scale=(0.7, 1.0),
                        ratio=(0.85, 1.15),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self) -> int:
        return len(self.records)

    def __del__(self) -> None:
        self._reset_zip_file()

    def dataset_summary(self) -> dict[str, Any]:
        return summarize_records(self.records, self.class_names)

    def class_counts(self, task: str) -> torch.Tensor:
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        counts = torch.zeros(len(self.class_names[task]), dtype=torch.long)
        for record in self.records:
            label = getattr(record, task)
            if label is None:
                continue
            counts[label] += 1
        return counts

    def _get_zip_file(self) -> ZipFile:
        if self._zip_file is None:
            self._zip_file = ZipFile(self.archive_path)
        return self._zip_file

    def _get_archive_names(self) -> set[str]:
        if self._archive_names is None:
            self._archive_names = set(self._get_zip_file().namelist())
        return self._archive_names

    def _reset_zip_file(self) -> None:
        if self._zip_file is not None:
            self._zip_file.close()
        self._zip_file = None
        self._archive_names = None

    def _resolve_archive_name(self, relative_path: str) -> str:
        candidates = (
            relative_path,
            f"wikiart/{relative_path}",
            relative_path.lstrip("/"),
            f"wikiart/{relative_path.lstrip('/')}",
        )
        archive_names = self._get_archive_names()
        for candidate in candidates:
            if candidate in archive_names:
                return candidate
        raise FileNotFoundError(f"{relative_path} not found in {self.archive_path}")

    def _load_image(self, relative_path: str) -> Image.Image:
        if self.image_root is not None:
            image_path = self.image_root / relative_path
            image = Image.open(image_path)
            return image.convert("RGB")

        archive_name = self._resolve_archive_name(relative_path)
        with self._get_zip_file().open(archive_name) as handle:
            image_bytes = handle.read()
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")

    def _warning_once(self, path: str, error: Exception) -> None:
        if path in self._warned_bad_paths:
            return
        self._warned_bad_paths.add(path)
        warnings.warn(
            f"Skipping unreadable image '{path}' due to {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _build_sample(self, record: WikiArtRecord) -> dict[str, Any]:
        image = self._load_image(record.path)
        image_tensor = self.transform(image)

        genre_label = MISSING_LABEL if record.genre is None else int(record.genre)
        artist_label = MISSING_LABEL if record.artist is None else int(record.artist)

        return {
            "image": image_tensor,
            "path": record.path,
            "style": torch.tensor(int(record.style), dtype=torch.long),
            "genre": torch.tensor(genre_label, dtype=torch.long),
            "artist": torch.tensor(artist_label, dtype=torch.long),
            "mask_style": torch.tensor(1, dtype=torch.bool),
            "mask_genre": torch.tensor(record.genre is not None, dtype=torch.bool),
            "mask_artist": torch.tensor(record.artist is not None, dtype=torch.bool),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        current_index = index
        max_attempts = min(32, len(self.records))

        for _ in range(max_attempts):
            record = self.records[current_index]
            if record.path in self._bad_paths:
                current_index = (current_index + 1) % len(self.records)
                continue

            try:
                return self._build_sample(record)
            except (BadZipFile, FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as error:
                self._bad_paths.add(record.path)
                self._warning_once(record.path, error)
                self._reset_zip_file()
                current_index = (current_index + 1) % len(self.records)

        raise RuntimeError(
            f"Unable to load a valid image after {max_attempts} attempts starting from index {index}. "
            "The archive may be heavily corrupted."
        )


def records_to_dicts(records: Iterable[WikiArtRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
