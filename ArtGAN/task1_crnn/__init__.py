"""Multi-task convolutional-recurrent classifier for WikiArt."""

from .dataset import TASKS, WikiArtMultiTaskDataset, load_class_names, load_records
from .model import ConvRecurrentWikiArtClassifier

__all__ = [
    "TASKS",
    "ConvRecurrentWikiArtClassifier",
    "WikiArtMultiTaskDataset",
    "load_class_names",
    "load_records",
]
