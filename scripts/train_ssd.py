"""CLI script to train SSD300-VGG16 on the chess piece dataset."""

import argparse
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ssd_detector import train_ssd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SSD300-VGG16 on Chess Piece Dataset")
    parser.add_argument("--config", type=str, default="configs/ssd_config.yaml", help="Path to ssd_config.yaml")
    parser.add_argument("--train_images", type=str, default="datasets/images/train")
    parser.add_argument("--train_labels", type=str, default="datasets/labels/train")
    parser.add_argument("--output_weights", type=str, default="weights/ssd_chess.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_ssd(
        train_img_dir=args.train_images,
        train_lbl_dir=args.train_labels,
        output_weights=args.output_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
