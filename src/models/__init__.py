"""Model definitions and wrappers for YOLOv11, SSD300-VGG16, and Faster R-CNN."""

from src.models.yolo_detector import YOLOChessDetector
from src.models.ssd_detector import build_ssd_model, predict_ssd, train_ssd
from src.models.faster_rcnn_detector import (
    build_faster_rcnn_model,
    predict_faster_rcnn,
    train_faster_rcnn,
)

__all__ = [
    "YOLOChessDetector",
    "build_ssd_model",
    "predict_ssd",
    "train_ssd",
    "build_faster_rcnn_model",
    "predict_faster_rcnn",
    "train_faster_rcnn",
]
