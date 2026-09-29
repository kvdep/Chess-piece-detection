"""Unit tests for box conversions, IoU, and geometry mapping."""

import unittest
from src.utils.box_conversions import (
    board_cell_to_yolo_bbox,
    calculate_iou,
    pixel_to_chess_square,
    voc_to_yolo,
    yolo_to_voc,
)


class TestBoxConversions(unittest.TestCase):
    def test_yolo_to_voc_and_back(self):
        img_w, img_h = 500, 500
        orig_xc, orig_yc, orig_w, orig_h = 0.5, 0.5, 0.2, 0.2

        x1, y1, x2, y2 = yolo_to_voc(orig_xc, orig_yc, orig_w, orig_h, img_w, img_h)
        self.assertAlmostEqual(x1, 200.0)
        self.assertAlmostEqual(y1, 200.0)
        self.assertAlmostEqual(x2, 300.0)
        self.assertAlmostEqual(y2, 300.0)

        rec_xc, rec_yc, rec_w, rec_h = voc_to_yolo(x1, y1, x2, y2, img_w, img_h)
        self.assertAlmostEqual(orig_xc, rec_xc)
        self.assertAlmostEqual(orig_yc, rec_yc)
        self.assertAlmostEqual(orig_w, rec_w)
        self.assertAlmostEqual(orig_h, rec_h)

    def test_calculate_iou_identical(self):
        box_a = [10.0, 10.0, 50.0, 50.0]
        box_b = [10.0, 10.0, 50.0, 50.0]
        iou = calculate_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 1.0)

    def test_calculate_iou_disjoint(self):
        box_a = [0.0, 0.0, 10.0, 10.0]
        box_b = [20.0, 20.0, 30.0, 30.0]
        iou = calculate_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 0.0)

    def test_calculate_iou_half_overlap(self):
        # 10x10 boxes overlapping by 5x10 (area 50)
        # box_a: 0..10 x 0..10 (area 100)
        # box_b: 5..15 x 0..10 (area 100)
        # union = 100 + 100 - 50 = 150
        # iou = 50 / 150 = 1/3
        box_a = [0.0, 0.0, 10.0, 10.0]
        box_b = [5.0, 0.0, 15.0, 10.0]
        iou = calculate_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 1.0 / 3.0)

    def test_pixel_to_chess_square(self):
        # Center of board cell e4
        # offset = 40, cell_size = 52
        # col 'e' has index 4 -> px = 40 + (4 + 0.5) * 52 = 40 + 234 = 274
        # rank '4' has index 3 -> py = 500 - (40 + (3 + 0.5) * 52) = 500 - (40 + 182) = 278
        col, row, sq = pixel_to_chess_square(px=274, py=278, img_size=500, offset=40, cell_size=52)
        self.assertEqual(sq, "e4")
        self.assertEqual(col, 4)
        self.assertEqual(row, 3)


if __name__ == "__main__":
    unittest.main()
