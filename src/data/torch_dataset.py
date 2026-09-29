"""PyTorch Dataset implementation for SSD and Faster R-CNN detectors."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights, VGG16_Weights

from src.utils.box_conversions import yolo_to_voc


def get_ssd_transforms(img_size: int = 300) -> A.Compose:
    """Standard Albumentations pipeline for SSD300-VGG16."""
    w = VGG16_Weights.DEFAULT
    mean, std = w.transforms().mean, w.transforms().std

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 360),
                sat_shift_limit=int(0.01 * 255),
                val_shift_limit=int(0.01 * 255),
                p=0.5,
            ),
            A.Affine(
                rotate=(-3.0, 3.0),
                translate_percent=(-0.05, 0.05),
                scale=(0.95, 1.05),
                shear=(-2.0, 2.0),
                p=0.5,
            ),
            A.HorizontalFlip(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.1,
            check_each_transform=False,
        ),
    )


def get_faster_rcnn_transforms(img_size: int = 500) -> A.Compose:
    """Standard Albumentations pipeline for Faster R-CNN ResNet50-FPN."""
    w = ResNet50_Weights.DEFAULT
    mean, std = w.transforms().mean, w.transforms().std

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.1,
            check_each_transform=False,
        ),
    )


class ChessDetectionDataset(Dataset):
    """PyTorch Dataset loading images and YOLO format bounding box annotations for Torchvision models.

    Converts YOLO normalized [xc, yc, w, h] to absolute pixel VOC [x1, y1, x2, y2].
    Shifts class labels by +1 because torchvision models reserve class 0 for background.
    """

    def __init__(
        self,
        img_dir: str,
        lbl_dir: str,
        transform: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        rebuild_cache: bool = False,
    ):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.data: List[Tuple[str, Dict[str, torch.Tensor]]] = []

        if cache_path and os.path.exists(cache_path) and not rebuild_cache:
            print(f"Loading cached dataset from: {cache_path}")
            self.data = torch.load(cache_path, weights_only=False)
        else:
            self._load_and_parse_annotations()
            if cache_path:
                print(f"Saving dataset cache to: {cache_path}")
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                torch.save(self.data, cache_path)

    def _load_and_parse_annotations(self) -> None:
        """Scan directory and parse images and labels."""
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")

        filenames = [
            f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for fn in filenames:
            base_name = fn.rsplit(".", 1)[0]
            txt_path = os.path.join(self.lbl_dir, f"{base_name}.txt")
            if not os.path.exists(txt_path):
                continue

            full_img_path = os.path.join(self.img_dir, fn)
            try:
                with Image.open(full_img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                continue

            boxes: List[List[float]] = []
            labels: List[int] = []

            with open(txt_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])

                    x1, y1, x2, y2 = yolo_to_voc(xc, yc, bw, bh, img_w, img_h)
                    # Enforce valid box geometry
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        # Shift class by 1 for torchvision models (class 0 is background)
                        labels.append(cls_id + 1)

            if len(boxes) > 0:
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                }
                self.data.append((fn, target))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fn, target = self.data[idx]
        img_path = os.path.join(self.img_dir, fn)
        img = np.array(Image.open(img_path).convert("RGB"))

        boxes_list = target["boxes"].tolist()
        labels_list = target["labels"].tolist()

        if self.transform is not None:
            augmented = self.transform(image=img, bboxes=boxes_list, labels=labels_list)
            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_labels = augmented["labels"]

            # Fallback if augmentations eliminated all bounding boxes
            if len(aug_boxes) == 0:
                h, w = img.shape[:2]
                default_transform = A.Compose(
                    [A.Resize(h, w), ToTensorV2()],
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
                )
                res = default_transform(image=img, bboxes=boxes_list, labels=labels_list)
                aug_img = res["image"]
                aug_boxes = res["bboxes"]
                aug_labels = res["labels"]

            boxes_tensor = torch.as_tensor(aug_boxes, dtype=torch.float32).reshape(-1, 4)
            labels_tensor = torch.as_tensor(aug_labels, dtype=torch.int64)
            return aug_img, {"boxes": boxes_tensor, "labels": labels_tensor}

        # No transform: return standard normalized float tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor, target


def collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
    """Collate function for PyTorch DataLoader handling variable bounding box counts per image."""
    return tuple(zip(*batch))
