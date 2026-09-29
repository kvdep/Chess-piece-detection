"""Constants and configuration mappings for chess piece detection."""

from typing import Dict, List

# 12 Classes for YOLO (0-indexed without background)
CLASS_NAMES: List[str] = [
    "bishop_black",
    "bishop_white",
    "king_black",
    "king_white",
    "knight_black",
    "knight_white",
    "pawn_black",
    "pawn_white",
    "queen_black",
    "queen_white",
    "rook_black",
    "rook_white",
]

# 13 Classes for Torchvision models (SSD, Faster R-CNN) with Background as class 0
CLASS_NAMES_TORCHVISION: List[str] = ["background"] + CLASS_NAMES

# Direct mappings
LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

TORCH_LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES_TORCHVISION)}
TORCH_ID_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES_TORCHVISION)}

# Synthetic dataset geometry parameters (Blender camera top-down at z=35)
SYNTHETIC_IMG_SIZE: int = 500
SYNTHETIC_OFFSET_PX: int = 40
SYNTHETIC_CELL_SIZE_PX: int = 52
SYNTHETIC_BB_WIDTH_PX: int = 70
SYNTHETIC_BB_HEIGHT_PX: int = 70
BOARD_GRID_SIZE: int = 8

# Real dataset acquisition parameters
REAL_FRAME_WIDTH: int = 500
REAL_FRAME_HEIGHT: int = 500
REAL_BB_WIDTH_PX: int = 70
REAL_BB_HEIGHT_PX: int = 70
TURNTABLE_ROTATION_SECONDS: int = 24
CAMERA_FPS: int = 30
MAX_RECORDING_FRAMES: int = 720  # 30 fps * 24 seconds = 720 frames per rotation

# Chess algebraic notation files and ranks
CHESS_FILES: List[str] = ["a", "b", "c", "d", "e", "f", "g", "h"]
CHESS_RANKS: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8"]
