"""Visualization and image I/O utilities supporting Unicode paths on Windows."""

from typing import Dict, List, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.constants import ID_TO_LABEL


def imread_unicode(path: str) -> np.ndarray:
    """Read image safely from paths containing Cyrillic or Unicode characters.

    Standard cv2.imread fails silently on non-ASCII Windows paths.
    Using np.fromfile with cv2.imdecode avoids this limitation.
    """
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image from: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Write image safely to paths containing Cyrillic or Unicode characters."""
    ext = "." + path.rsplit(".", 1)[-1]
    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        return False
    encoded_img.tofile(path)
    return True


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Union[List[List[float]], np.ndarray],
    labels: Union[List[int], np.ndarray],
    scores: Optional[Union[List[float], np.ndarray]] = None,
    id_to_label: Optional[Dict[int, str]] = None,
    box_format: str = "voc",
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and class labels onto an image copy.

    Args:
        image: BGR or RGB image numpy array.
        boxes: List of bounding boxes.
        labels: List of integer class labels.
        scores: Optional detection confidence scores in [0, 1].
        id_to_label: Dictionary mapping class integer ID to name.
        box_format: 'voc' ([x1, y1, x2, y2]) or 'yolo' ([xc, yc, w, h]).
        color: Bounding box BGR color tuple.
        thickness: Line thickness.

    Returns:
        Annotated image numpy array.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    mapping = id_to_label or ID_TO_LABEL

    for idx, box in enumerate(boxes):
        if box_format == "yolo":
            xc, yc, bw, bh = box
            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)
        else:
            x1, y1, x2, y2 = map(int, box)

        label_id = int(labels[idx])
        label_text = mapping.get(label_id, str(label_id))

        if scores is not None and len(scores) > idx:
            label_text = f"{label_text} {float(scores[idx]):.2f}"

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label background box
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(
            annotated,
            (x1, bg_y1),
            (x1 + tw + 4, bg_y1 + th + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label_text,
            (x1 + 2, bg_y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def plot_image_with_boxes(
    image: np.ndarray,
    boxes: Union[List[List[float]], np.ndarray],
    labels: Union[List[int], np.ndarray],
    scores: Optional[Union[List[float], np.ndarray]] = None,
    title: str = "Predictions",
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
) -> None:
    """Render image with bounding boxes via matplotlib."""
    drawn = draw_bounding_boxes(image, boxes, labels, scores=scores)
    # Convert BGR to RGB for matplotlib display if needed
    if len(drawn.shape) == 3 and drawn.shape[2] == 3:
        drawn_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
    else:
        drawn_rgb = drawn

    plt.figure(figsize=figsize)
    plt.imshow(drawn_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
