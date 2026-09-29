"""Automated physical dataset acquisition rig video capture and real-time labeling script."""

import argparse
import os
from typing import Optional
import cv2

from src.constants import (
    CLASS_NAMES,
    LABEL_TO_ID,
    MAX_RECORDING_FRAMES,
    REAL_BB_HEIGHT_PX,
    REAL_BB_WIDTH_PX,
    REAL_FRAME_HEIGHT,
    REAL_FRAME_WIDTH,
)


def capture_turntable_frames(
    piece_name: str,
    color: str,
    checkerboard: str,
    angle: str,
    camera_index: int = 0,
    output_root: str = "datasets/real_captured",
    max_frames: int = MAX_RECORDING_FRAMES,
    bb_size: int = REAL_BB_WIDTH_PX,
    target_dim: int = REAL_FRAME_WIDTH,
) -> int:
    """Capture frames from rotating motorized rig, crop center square, and save YOLO labels.

    Args:
        piece_name: e.g. 'pawn', 'bishop', 'knight', 'rook', 'queen', 'king'
        color: 'black' or 'white'
        checkerboard: 'black' or 'white' (background square)
        angle: elevation tilt angle string, e.g. '10', '20', '30'
        camera_index: OpenCV VideoCapture device index
        output_root: Root folder for captured images
        max_frames: Total frames per full rotation (default 720 for 24s at 30 fps)
        bb_size: Bounding box size in pixels (70 px)
        target_dim: Target square image dimension (500 px)

    Returns:
        Number of frames recorded.
    """
    full_class_name = f"{piece_name}_{color}"
    if full_class_name not in LABEL_TO_ID:
        raise ValueError(
            f"Unknown class: {full_class_name}. Must be one of {CLASS_NAMES}"
        )

    class_id = LABEL_TO_ID[full_class_name]

    # Destination directories
    img_dir = os.path.join(output_root, full_class_name, "images")
    lbl_dir = os.path.join(output_root, full_class_name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video camera index {camera_index}")

    hw = bb_size // 2
    yolo_cx = 0.5
    yolo_cy = 0.5
    yolo_w = bb_size / float(target_dim)
    yolo_h = bb_size / float(target_dim)

    frame_counter = 0
    recording_active = False

    print("=" * 60)
    print(f"Target: {full_class_name} | Background: {checkerboard} | Angle: {angle}")
    print("Controls: Press 's' to START recording, 'q' to QUIT")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stream interrupted.")
                break

            h, w = frame.shape[:2]
            size = min(h, w)
            cx, cy = w // 2, h // 2

            # Central square crop
            x1 = cx - size // 2
            y1 = cy - size // 2
            x2 = cx + size // 2
            y2 = cy + size // 2
            cropped = frame[y1:y2, x1:x2]

            # Resize to target 500x500
            resized = cv2.resize(cropped, (target_dim, target_dim))

            # Display preview with alignment box
            preview = resized.copy()
            r1 = target_dim // 2 - hw
            r2 = target_dim // 2 + hw
            box_color = (0, 255, 0) if recording_active else (0, 0, 255)
            cv2.rectangle(preview, (r1, r1), (r2, r2), box_color, 2)

            status_text = f"Frames: {frame_counter}/{max_frames}" if recording_active else "READY (Press 's')"
            cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            cv2.imshow("Turntable Capture Rig", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and not recording_active:
                recording_active = True
                print("Recording started...")
            elif key == ord("q"):
                print("Cancelled by user.")
                break

            if recording_active and frame_counter < max_frames:
                filename = f"{full_class_name}_{frame_counter:04d}_{checkerboard}_angle_{angle}"
                out_img_path = os.path.join(img_dir, f"{filename}.png")
                out_lbl_path = os.path.join(lbl_dir, f"{filename}.txt")

                cv2.imwrite(out_img_path, resized)
                with open(out_lbl_path, "w", encoding="utf-8") as lf:
                    lf.write(f"{class_id} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")

                frame_counter += 1
                if frame_counter >= max_frames:
                    print(f"Completed rotation: {frame_counter} frames recorded.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return frame_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire real-world chess piece video frames")
    parser.add_argument("--piece", type=str, required=True, choices=["pawn", "bishop", "knight", "rook", "queen", "king"])
    parser.add_argument("--color", type=str, required=True, choices=["black", "white"])
    parser.add_argument("--checkerboard", type=str, default="white", choices=["black", "white"])
    parser.add_argument("--angle", type=str, default="0", help="Elevation tilt angle (e.g. 10, 20, 30)")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera device index")
    parser.add_argument("--output_root", type=str, default="datasets/real_captured")
    parser.add_argument("--max_frames", type=int, default=MAX_RECORDING_FRAMES)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    captured = capture_turntable_frames(
        piece_name=cli_args.piece,
        color=cli_args.color,
        checkerboard=cli_args.checkerboard,
        angle=cli_args.angle,
        camera_index=cli_args.camera_index,
        output_root=cli_args.output_root,
        max_frames=cli_args.max_frames,
    )
    print(f"Captured {captured} frames for {cli_args.piece}_{cli_args.color}.")
