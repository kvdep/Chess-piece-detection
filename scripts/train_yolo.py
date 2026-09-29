"""CLI script to train YOLOv11 on the chess piece dataset."""

import argparse
import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.yolo_detector import YOLOChessDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv11 on Chess Piece Dataset")
    parser.add_argument("--data", type=str, default="configs/yolo_chess.yaml", help="Path to data.yaml")
    parser.add_argument("--hyp", type=str, default="configs/yolo_hyp.yaml", help="Path to custom hyp.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=500, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project folder")
    parser.add_argument("--name", type=str, default="yolov11n_chess", help="Experiment run name")
    args = parser.parse_args()

    detector = YOLOChessDetector(model_path=args.model)
    detector.train(
        data_yaml=args.data,
        hyp_yaml=args.hyp,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
