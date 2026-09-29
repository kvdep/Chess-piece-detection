"""Bounding box conversion and geometric utility functions."""

from typing import Tuple, Union
import numpy as np
import torch

from src.constants import (
    CHESS_FILES,
    CHESS_RANKS,
    SYNTHETIC_CELL_SIZE_PX,
    SYNTHETIC_IMG_SIZE,
    SYNTHETIC_OFFSET_PX,
    SYNTHETIC_BB_WIDTH_PX,
    SYNTHETIC_BB_HEIGHT_PX,
)


def yolo_to_voc(
    x_c: float, y_c: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert normalized YOLO format [x_c, y_c, w, h] to pixel VOC format [x1, y1, x2, y2].

    Args:
        x_c: Center X coordinate normalized in [0, 1].
        y_c: Center Y coordinate normalized in [0, 1].
        w: Bounding box width normalized in [0, 1].
        h: Bounding box height normalized in [0, 1].
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Tuple of (x1, y1, x2, y2) in absolute pixel coordinates.
    """
    x1 = (x_c - w / 2.0) * img_w
    y1 = (y_c - h / 2.0) * img_h
    x2 = (x_c + w / 2.0) * img_w
    y2 = (y_c + h / 2.0) * img_h
    return max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)


def voc_to_yolo(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert pixel VOC format [x1, y1, x2, y2] to normalized YOLO format [x_c, y_c, w, h].

    Args:
        x1: Left bounding box coordinate in pixels.
        y1: Top bounding box coordinate in pixels.
        x2: Right bounding box coordinate in pixels.
        y2: Bottom bounding box coordinate in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Tuple of (x_c, y_c, w, h) normalized in [0, 1].
    """
    box_w = x2 - x1
    box_h = y2 - y1
    x_c = (x1 + box_w / 2.0) / img_w
    y_c = (y1 + box_h / 2.0) / img_h
    w = box_w / img_w
    h = box_h / img_h
    return x_c, y_c, w, h


def calculate_iou(
    box_a: Union[np.ndarray, torch.Tensor, list],
    box_b: Union[np.ndarray, torch.Tensor, list],
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        IoU value in [0.0, 1.0].
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def board_cell_to_yolo_bbox(
    cell_x: int,
    cell_y: int,
    img_size: int = SYNTHETIC_IMG_SIZE,
    offset: int = SYNTHETIC_OFFSET_PX,
    cell_size: int = SYNTHETIC_CELL_SIZE_PX,
    bb_w: int = SYNTHETIC_BB_WIDTH_PX,
    bb_h: int = SYNTHETIC_BB_HEIGHT_PX,
) -> Tuple[float, float, float, float]:
    """Calculate normalized YOLO bounding box from discrete 8x8 chessboard cell indices.

    Formulas:
        cx_px = offset + (cell_x + 0.5) * cell_size
        cy_px = img_size - (offset + (cell_y + 0.5) * cell_size)
        yolo_x = cx_px / img_size
        yolo_y = cy_px / img_size
        yolo_w = bb_w / img_size
        yolo_h = bb_h / img_size
    """
    cx_px = offset + (cell_x + 0.5) * cell_size
    cy_px = img_size - (offset + (cell_y + 0.5) * cell_size)

    yolo_x = cx_px / float(img_size)
    yolo_y = cy_px / float(img_size)
    yolo_w = bb_w / float(img_size)
    yolo_h = bb_h / float(img_size)
    return yolo_x, yolo_y, yolo_w, yolo_h


def pixel_to_chess_square(
    px: float,
    py: float,
    img_size: int = SYNTHETIC_IMG_SIZE,
    offset: int = SYNTHETIC_OFFSET_PX,
    cell_size: int = SYNTHETIC_CELL_SIZE_PX,
) -> Tuple[int, int, str]:
    """Map image pixel coordinate (px, py) to chessboard grid indices and algebraic notation.

    Returns:
        Tuple of (file_idx, rank_idx, square_name e.g. 'e4'). If outside, indices will be clamped or -1.
    """
    col = int((px - offset) // cell_size)
    row = int((img_size - py - offset) // cell_size)

    if 0 <= col < 8 and 0 <= row < 8:
        square = f"{CHESS_FILES[col]}{CHESS_RANKS[row]}"
        return col, row, square
    return -1, -1, "outside"
