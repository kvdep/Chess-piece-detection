"""Full chessboard piece detection and algebraic board square localization."""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.constants import (
    CLASS_NAMES,
    ID_TO_LABEL,
    SYNTHETIC_CELL_SIZE_PX,
    SYNTHETIC_IMG_SIZE,
    SYNTHETIC_OFFSET_PX,
)
from src.models.yolo_detector import YOLOChessDetector
from src.utils.box_conversions import pixel_to_chess_square
from src.utils.visualization import draw_bounding_boxes, imread_unicode, imwrite_unicode


class BoardStatePredictor:
    """Detects chess pieces on a board and projects them into algebraic grid coordinates."""

    def __init__(
        self,
        weights_path: str = "weights/best.pt",
        model_type: str = "yolo",
        conf_threshold: float = 0.25,
    ):
        self.model_type = model_type
        self.conf_threshold = conf_threshold

        if model_type == "yolo":
            self.detector = YOLOChessDetector(model_path=weights_path)
        else:
            raise NotImplementedError(f"Model type {model_type} not configured in BoardStatePredictor")

    def predict(
        self,
        image_path: str,
        img_size: int = SYNTHETIC_IMG_SIZE,
        offset: int = SYNTHETIC_OFFSET_PX,
        cell_size: int = SYNTHETIC_CELL_SIZE_PX,
    ) -> Dict[str, Any]:
        """Detect pieces and map each detected piece to its closest chessboard square.

        Args:
            image_path: Path to the input image.
            img_size: Resized image dimension.
            offset: Offset from border to first board square in pixels.
            cell_size: Dimension of single board square in pixels.

        Returns:
            Dictionary containing 'detections', 'board_state' (square -> piece), and 'annotated_image'.
        """
        img_bgr = imread_unicode(image_path)
        h, w = img_bgr.shape[:2]

        results = self.detector.predict(source=image_path, conf=self.conf_threshold)
        detections = results[0]

        boxes = detections["boxes"]
        scores = detections["scores"]
        labels = detections["labels"]

        board_state: Dict[str, Dict[str, Any]] = {}
        annotated_labels = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # Scale center coordinate to synthetic reference dimension if needed
            norm_cx = (center_x / float(w)) * img_size
            norm_cy = (center_y / float(h)) * img_size

            col, row, square = pixel_to_chess_square(
                px=norm_cx,
                py=norm_cy,
                img_size=img_size,
                offset=offset,
                cell_size=cell_size,
            )

            piece_name = ID_TO_LABEL.get(int(labels[idx]), str(labels[idx]))
            conf = float(scores[idx])

            if square != "outside":
                board_state[square] = {
                    "piece": piece_name,
                    "confidence": conf,
                    "box": box.tolist(),
                }
                annotated_labels.append(f"{piece_name} [{square}]")
            else:
                annotated_labels.append(piece_name)

        # Draw boxes and labels
        annotated_img = draw_bounding_boxes(
            image=img_bgr,
            boxes=boxes,
            labels=labels,
            scores=scores,
            id_to_label={idx: annotated_labels[idx] for idx in range(len(annotated_labels))},
            box_format="voc",
            color=(0, 220, 0),
            thickness=2,
        )

        return {
            "image_path": image_path,
            "detections_count": len(boxes),
            "board_state": board_state,
            "annotated_image": annotated_img,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict chess pieces and board position")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output_prediction.png", help="Path to save annotated image")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    predictor = BoardStatePredictor(weights_path=cli_args.weights, conf_threshold=cli_args.conf)
    result = predictor.predict(image_path=cli_args.image)

    print("=" * 60)
    print(f"Detected {result['detections_count']} pieces:")
    for square, info in sorted(result["board_state"].items()):
        print(f"  {square:>3}: {info['piece']} (conf: {info['confidence']:.3f})")
    print("=" * 60)

    imwrite_unicode(cli_args.output, result["annotated_image"])
    print(f"Saved annotated image to {cli_args.output}")
