"""Object detection evaluation metrics: IoU, Precision, Recall, mAP50, and Confusion Matrix."""

from typing import Dict, List, Tuple
import numpy as np
from src.utils.box_conversions import calculate_iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision (AP) using standard 11-point or continuous interpolation (COCO/VOC 2012).

    Formula:
        AP = sum_{k=1}^n (R_k - R_{k-1}) * max_{r >= R_k} P(r)
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Integrate area under curve
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def evaluate_detections(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    num_classes: int = 12,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate detection metrics across a dataset.

    Args:
        predictions: List of dicts, each with 'boxes' [N, 4], 'scores' [N], 'labels' [N].
        ground_truths: List of dicts, each with 'boxes' [M, 4], 'labels' [M].
        num_classes: Total number of classes.
        iou_threshold: Minimum IoU to consider a detection a true positive.

    Returns:
        Dictionary containing per-class AP, overall mean Precision, mean Recall, and mAP50.
    """
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for cls_idx in range(num_classes):
        tp_list = []
        fp_list = []
        scores_list = []
        n_gt_class = 0

        for pred_dict, gt_dict in zip(predictions, ground_truths):
            gt_mask = gt_dict["labels"] == cls_idx
            gt_boxes = gt_dict["boxes"][gt_mask]
            n_gt_class += len(gt_boxes)

            pred_mask = pred_dict["labels"] == cls_idx
            pred_boxes = pred_dict["boxes"][pred_mask]
            pred_scores = pred_dict["scores"][pred_mask]

            # Sort predictions by descending confidence score
            if len(pred_scores) > 0:
                sort_indices = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[sort_indices]
                pred_scores = pred_scores[sort_indices]

            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            for p_box, p_score in zip(pred_boxes, pred_scores):
                scores_list.append(p_score)
                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, g_box in enumerate(gt_boxes):
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                    tp_list.append(1)
                    fp_list.append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        total_gt += n_gt_class
        total_tp += sum(tp_list)
        total_fp += sum(fp_list)

        if n_gt_class == 0:
            continue

        if len(tp_list) == 0:
            aps.append(0.0)
            continue

        # Sort by score descending across all images
        sort_order = np.argsort(-np.array(scores_list))
        tp_arr = np.array(tp_list)[sort_order]
        fp_arr = np.array(fp_list)[sort_order]

        cumsum_tp = np.cumsum(tp_arr)
        cumsum_fp = np.cumsum(fp_arr)

        recalls = cumsum_tp / float(n_gt_class)
        precisions = cumsum_tp / np.maximum(cumsum_tp + cumsum_fp, np.finfo(float).eps)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    map50 = float(np.mean(aps)) if len(aps) > 0 else 0.0
    overall_p = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    overall_r = float(total_tp / total_gt) if total_gt > 0 else 0.0

    return {
        "precision": overall_p,
        "recall": overall_r,
        "map50": map50,
        "per_class_ap": {cls_idx: ap for cls_idx, ap in enumerate(aps)},
    }


def compute_confusion_matrix(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    num_classes: int = 12,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> np.ndarray:
    """Calculate confusion matrix of size [num_classes + 1, num_classes + 1].

    Last row/column (index num_classes) represents background / unassigned detections or missed GTs.
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for pred_dict, gt_dict in zip(predictions, ground_truths):
        gt_boxes = gt_dict["boxes"]
        gt_labels = gt_dict["labels"]

        valid_pred_mask = pred_dict["scores"] >= score_threshold
        pred_boxes = pred_dict["boxes"][valid_pred_mask]
        pred_labels = pred_dict["labels"][valid_pred_mask]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for p_box, p_lbl in zip(pred_boxes, pred_labels):
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g_box in enumerate(gt_boxes):
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                true_lbl = gt_labels[best_gt_idx]
                matrix[true_lbl, p_lbl] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive: predicted class, but no ground truth (background)
                matrix[num_classes, p_lbl] += 1

        # False negatives: unassigned ground truth pieces (missed detection)
        for g_idx, matched in enumerate(gt_matched):
            if not matched:
                matrix[gt_labels[g_idx], num_classes] += 1

    return matrix
