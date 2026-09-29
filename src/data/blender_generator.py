"""Procedural 3D Synthetic Chessboard and Piece Generator for Blender.

Usage:
    blender -b -P src/data/blender_generator.py -- \
        --model_dir "path/to/3d_models" \
        --output_dir "path/to/output" \
        --num_samples 10000 \
        --device CUDA
"""

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Set, Tuple

# Verify if executed inside Blender's python runtime
try:
    import bpy
    import numpy as np
except ImportError:
    bpy = None
    np = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments passed after '--' in Blender CLI command."""
    parser = argparse.ArgumentParser(description="Blender Synthetic Chess Generator")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing .obj files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for images and targets.csv")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of scene renders to generate")
    parser.add_argument("--samples_per_pixel", type=int, default=32, help="Cycles render samples")
    parser.add_argument("--device", type=str, default="CUDA", choices=["CUDA", "OPTIX", "HIP", "METAL", "CPU"])
    parser.add_argument("--resolution", type=int, default=500, help="Render image square resolution")

    # Extract arguments passed after '--'
    if "--" in sys.argv:
        args_start = sys.argv.index("--") + 1
        return parser.parse_args(sys.argv[args_start:])
    return parser.parse_args()


def setup_scene(
    model_dir: str,
    resolution: int = 500,
    samples_per_pixel: int = 32,
    device: str = "CUDA",
) -> Tuple[object, Dict[str, object]]:
    """Clear scene, configure Cycles GPU renderer, camera, lights, and import board/pieces."""
    if bpy is None:
        raise RuntimeError("bpy is not available. This script must run inside Blender.")

    # 1. Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # 2. Import chessboard
    board_file = os.path.join(model_dir, "chess_board.obj")
    if not os.path.exists(board_file):
        raise FileNotFoundError(f"Board file not found: {board_file}")
    bpy.ops.wm.obj_import(filepath=board_file)
    board = bpy.context.selected_objects[0]
    board.name = "Chessboard"

    # 3. Import pieces and park them outside the visible view
    piece_files = {
        "bishop_black": "bishop_black.obj",
        "bishop_white": "bishop_white.obj",
        "king_black": "king_black.obj",
        "king_white": "king_white.obj",
        "knight_black": "knight_black.obj",
        "knight_white": "knight_white.obj",
        "pawn_black": "pawn_black.obj",
        "pawn_white": "pawn_white.obj",
        "queen_black": "queen_black.obj",
        "queen_white": "queen_white.obj",
        "rook_black": "rook_black.obj",
        "rook_white": "rook_white.obj",
    }
    pieces = {}
    for piece_name, file_name in piece_files.items():
        obj_path = os.path.join(model_dir, file_name)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Piece file not found: {obj_path}")
        bpy.ops.wm.obj_import(filepath=obj_path)
        piece_obj = bpy.context.selected_objects[0]
        piece_obj.name = piece_name
        piece_obj.location = (100.0, 100.0, 0.0)
        pieces[piece_name] = piece_obj

    # 4. Camera setup: top-down perspective view
    bpy.ops.object.camera_add(location=(0.0, 0.0, 35.0))
    cam = bpy.context.object
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.type = "PERSP"
    bpy.context.scene.camera = cam

    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100

    # 5. Lighting: 1 Sun + 4 Area lights at corners to prevent harsh cast shadows
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 8.0))
    sun = bpy.context.object
    sun.rotation_euler = (0.0, 0.0, 0.0)

    area_positions = [
        (0.0, 0.0, 10.0),
        (-10.0, -10.0, 10.0),
        (10.0, -10.0, 10.0),
        (10.0, 10.0, 10.0),
        (-10.0, 10.0, 10.0),
    ]
    for pos in area_positions:
        bpy.ops.object.light_add(type="AREA", location=pos)
        bpy.context.object.data.energy = 1000.0

    # 6. Cycles GPU settings
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.image_settings.file_format = "PNG"

    if device != "CPU":
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = device
        bpy.context.scene.cycles.device = "GPU"
    else:
        bpy.context.scene.cycles.device = "CPU"

    bpy.context.scene.cycles.samples = samples_per_pixel
    return board, pieces


def generate_synthetic_dataset(
    model_dir: str,
    output_dir: str,
    num_samples: int = 1000,
    samples_per_pixel: int = 32,
    device: str = "CUDA",
    resolution: int = 500,
) -> None:
    """Generate synthetic images with randomized piece placements and orientations."""
    os.makedirs(output_dir, exist_ok=True)
    target_file = os.path.join(output_dir, "targets.csv")

    board, pieces = setup_scene(
        model_dir=model_dir,
        resolution=resolution,
        samples_per_pixel=samples_per_pixel,
        device=device,
    )

    # Resume from last index if file exists
    start_id = 0
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                start_id = int(last_line.split(",")[0]) + 1

    # World coordinate calibration constants for board grid
    x_origin = -8.96654
    y_origin = -8.91536
    cell_spacing = 2.5474

    with open(target_file, "a" if start_id > 0 else "w", encoding="utf-8") as f:
        if start_id == 0:
            f.write("image_id,piece,x,y,rot_z\n")

        for sample_idx in range(start_id, start_id + num_samples):
            placed_records = []
            occupied_cells: Set[Tuple[int, int]] = set()

            for piece_name, piece_obj in pieces.items():
                while True:
                    place_x = random.randint(0, 7)
                    place_y = random.randint(0, 7)
                    if (place_x, place_y) not in occupied_cells:
                        occupied_cells.add((place_x, place_y))
                        break

                world_x = x_origin + place_x * cell_spacing
                world_y = y_origin + place_y * cell_spacing
                rot_z = random.uniform(0.0, 2.0 * math.pi)

                piece_obj.location = (world_x, world_y, 1.0)
                piece_obj.rotation_euler = (math.pi / 2.0, 0.0, rot_z)

                placed_records.append((piece_name, place_x, place_y, rot_z))

            # Render frame
            image_path = os.path.join(output_dir, f"{sample_idx}.png")
            bpy.context.scene.render.filepath = image_path
            bpy.ops.render.render(write_still=True)

            # Record annotations
            for record in placed_records:
                f.write(f"{sample_idx},{record[0]},{record[1]},{record[2]},{record[3]:.6f}\n")
            f.flush()

            # Return pieces to off-screen positions for next iteration
            for piece_obj in pieces.values():
                piece_obj.location = (100.0, 100.0, 0.0)


if __name__ == "__main__":
    cli_args = parse_args()
    generate_synthetic_dataset(
        model_dir=cli_args.model_dir,
        output_dir=cli_args.output_dir,
        num_samples=cli_args.num_samples,
        samples_per_pixel=cli_args.samples_per_pixel,
        device=cli_args.device,
        resolution=cli_args.resolution,
    )
