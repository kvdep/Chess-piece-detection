"""Data generation, annotation conversion, and PyTorch dataset modules."""

from src.data.synthetic_annotator import convert_synthetic_targets_to_yolo
from src.data.video_collector import capture_turntable_frames
from src.data.torch_dataset import (
    ChessDetectionDataset,
    collate_fn,
    get_faster_rcnn_transforms,
    get_ssd_transforms,
)

__all__ = [
    "convert_synthetic_targets_to_yolo",
    "capture_turntable_frames",
    "ChessDetectionDataset",
    "collate_fn",
    "get_ssd_transforms",
    "get_faster_rcnn_transforms",
]
