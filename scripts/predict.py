"""CLI script to run inference on an image or directory of images."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.predict_board import BoardStatePredictor
from src.utils.visualization import imwrite_unicode


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess Piece Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output_prediction.png", help="Output visualized image path")
    args = parser.parse_args()

    predictor = BoardStatePredictor(weights_path=args.weights, conf_threshold=args.conf)
    result = predictor.predict(image_path=args.image)

    print("=" * 60)
    print(f"Results for: {args.image}")
    print(f"Detected pieces count: {result['detections_count']}")
    print("-" * 60)
    for square, data in sorted(result["board_state"].items()):
        print(f"Square [{square:>2}]: {data['piece']:<14} | Confidence: {data['confidence']:.3f}")
    print("=" * 60)

    imwrite_unicode(args.output, result["annotated_image"])
    print(f"Annotated result successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
