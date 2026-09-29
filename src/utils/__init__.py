"""Utility functions for geometry, visualization, and metric evaluation."""

from src.utils.box_conversions import (
    calculate_iou,
    voc_to_yolo,
    yolo_to_voc,
    board_cell_to_yolo_bbox,
    pixel_to_chess_square,
)
from src.utils.visualization import (
    draw_bounding_boxes,
    imread_unicode,
    imwrite_unicode,
    plot_image_with_boxes,
)
from src.utils.metrics import (
    compute_ap,
    compute_confusion_matrix,
    evaluate_detections,
)

__all__ = [
    "calculate_iou",
    "voc_to_yolo",
    "yolo_to_voc",
    "board_cell_to_yolo_bbox",
    "pixel_to_chess_square",
    "draw_bounding_boxes",
    "imread_unicode",
    "imwrite_unicode",
    "plot_image_with_boxes",
    "compute_ap",
    "compute_confusion_matrix",
    "evaluate_detections",
]
