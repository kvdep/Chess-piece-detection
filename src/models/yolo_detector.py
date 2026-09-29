"""YOLOv11 Detector wrapper for training, validation, and inference."""

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ultralytics import YOLO

from src.constants import CLASS_NAMES, ID_TO_LABEL


class YOLOChessDetector:
    """Wrapper for Ultralytics YOLOv11 detector for chess piece recognition."""

    def __init__(self, model_path: str = "yolo11n.pt"):
        """Initialize YOLO detector with pretrained weights or fine-tuned checkpoint."""
        self.model_path = model_path
        self.model = YOLO(model_path)

    def train(
        self,
        data_yaml: str,
        hyp_yaml: Optional[str] = None,
        epochs: int = 100,
        batch_size: int = 8,
        imgsz: int = 500,
        project: str = "runs/detect",
        name: str = "yolov11n_chess",
        device: Union[int, str] = 0,
        **kwargs: Any,
    ) -> Any:
        """Train YOLO model with custom loss weights and augmentations.

        Args:
            data_yaml: Path to data.yaml dataset definition.
            hyp_yaml: Path to custom hyperparameter YAML file.
            epochs: Total training epochs.
            batch_size: Batch size.
            imgsz: Input image dimension.
            project: Run directory root.
            name: Experiment name.
            device: CUDA device index or 'cpu'.
            **kwargs: Additional YOLO training arguments.
        """
        train_args = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "project": project,
            "name": name,
            "device": device,
            **kwargs,
        }

        # Override hyperparameters if provided
        if hyp_yaml and os.path.exists(hyp_yaml):
            import yaml

            with open(hyp_yaml, "r", encoding="utf-8") as f:
                hyp_dict = yaml.safe_load(f)
                train_args.update(hyp_dict)

        results = self.model.train(**train_args)
        return results

    def validate(
        self,
        data_yaml: str,
        split: str = "val",
        batch_size: int = 8,
        imgsz: int = 500,
    ) -> Dict[str, float]:
        """Evaluate model on dataset split."""
        metrics = self.model.val(
            data=data_yaml,
            split=split,
            batch=batch_size,
            imgsz=imgsz,
        )
        return {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }

    def predict(
        self,
        source: Union[str, np.ndarray, List[str]],
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """Run inference on an image path, directory, or numpy image.

        Returns:
            List of dictionaries with 'boxes' [N, 4] in VOC pixel format, 'scores' [N], and 'labels' [N].
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=False,
            device=device,
            verbose=False,
        )

        outputs = []
        for r in results:
            boxes_np = r.boxes.xyxy.cpu().numpy()
            scores_np = r.boxes.conf.cpu().numpy()
            labels_np = r.boxes.cls.cpu().numpy().astype(int)

            outputs.append(
                {
                    "boxes": boxes_np,
                    "scores": scores_np,
                    "labels": labels_np,
                    "orig_img": r.orig_img,
                }
            )
        return outputs
