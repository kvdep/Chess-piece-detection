"""Dataset verification script: checks bounding box alignment on random samples."""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import CLASS_NAMES, ID_TO_LABEL
from src.utils.visualization import imread_unicode, plot_image_with_boxes


def verify_dataset(
    dataset_dir: str,
    split: str = "train",
    num_samples: int = 3,
    save_output_dir: str = "docs/assets/verification_samples",
) -> None:
    """Load random images from split, parse bounding boxes, and save visualized verification images."""
    img_dir = os.path.join(dataset_dir, "images", split)
    lbl_dir = os.path.join(dataset_dir, "labels", split)

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Directory not found: {img_dir} or {lbl_dir}")
        return

    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(images) == 0:
        print(f"No images found in {img_dir}")
        return

    os.makedirs(save_output_dir, exist_ok=True)
    sample_files = random.sample(images, min(num_samples, len(images)))

    for fn in sample_files:
        base = fn.rsplit(".", 1)[0]
        lbl_file = os.path.join(lbl_dir, f"{base}.txt")
        img_file = os.path.join(img_dir, fn)

        if not os.path.exists(lbl_file):
            print(f"Warning: Label file missing for {fn}")
            continue

        img = imread_unicode(img_file)
        h, w = img.shape[:2]

        boxes = []
        labels = []
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        out_path = os.path.join(save_output_dir, f"verified_{base}.png")
        plot_image_with_boxes(
            image=img,
            boxes=boxes,
            labels=labels,
            title=f"Sample: {fn} ({len(boxes)} objects)",
            save_path=out_path,
        )
        print(f"Verification sample saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset labels and bounding box alignment")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    verify_dataset(dataset_dir=args.dataset_dir, split=args.split, num_samples=args.samples)


if __name__ == "__main__":
    main()
