"""The lab scores correction cost only for objects within 50 m (BEV) of the LiDAR."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vf_pp.core import MAX_RANGE, correction_cost, within_range  # noqa: E402


def box(x, y, label="Car"):
    return {"x": x, "y": y, "z": 0.0, "length": 4.0, "width": 1.8, "height": 1.5, "yaw": 0.0, "label": label}


class TestScoringRange(unittest.TestCase):
    def test_max_range_is_50_m(self):
        self.assertEqual(MAX_RANGE, 50.0)

    def test_boundary_is_inclusive(self):
        on_edge = box(30.0, 40.0)            # exactly 50 m
        just_out = box(30.0, 40.01)
        self.assertEqual(within_range([on_edge, just_out]), [on_edge])

    def test_uses_bev_distance_not_x_only(self):
        self.assertTrue(within_range([box(-45.0, 0.0)]))          # behind the car, 45 m
        self.assertFalse(within_range([box(40.0, -35.0)]))        # 53 m diagonally

    def test_far_label_is_not_an_add(self):
        near, far = box(10.0, 0.0), box(60.0, 0.0)
        c = correction_cost([near, far], [box(10.0, 0.0)])
        self.assertEqual((c["gt"], c["match"], c["add"], c["delete"], c["edit"], c["cost"]), (1, 1, 0, 0, 0, 0.0))

    def test_far_prediction_is_not_a_delete(self):
        c = correction_cost([box(10.0, 0.0)], [box(10.0, 0.0), box(0.0, 55.0)])
        self.assertEqual((c["pred"], c["delete"], c["cost"]), (1, 0, 0.0))

    def test_only_far_objects_cost_nothing(self):
        far = [box(70.0, 0.0), box(0.0, -60.0)]
        self.assertEqual(correction_cost(far, [box(51.0, 0.0)])["cost"], 0.0)

    def test_near_errors_still_counted(self):
        gt = [box(10.0, 0.0), box(20.0, 5.0), box(80.0, 0.0)]
        pred = [box(10.0, 0.0), box(-15.0, -15.0)]       # one hit, one extra, one miss within 50 m
        c = correction_cost(gt, pred)
        self.assertEqual((c["add"], c["delete"], c["cost"]), (1, 1, 1.3))

    def test_compare_view_hides_far_boxes(self):
        from scripts import pp_compare

        drawn = pp_compare.classify([box(10.0, 0.0), box(60.0, 0.0)], [box(10.0, 0.0), box(0.0, 70.0)])
        self.assertTrue(all(math.hypot(b["x"], b["y"]) <= MAX_RANGE for b, _ in drawn))
        self.assertEqual([c for _, c in drawn], [pp_compare.GREEN])

    def test_compare_bev_displays_boxes_at_50m_boundary(self):
        from scripts import pp_compare

        img = pp_compare.bev([], [(box(0.0, -48.0), pp_compare.BLUE)], "50m")
        colors = {c for _, c in img.getcolors(maxcolors=img.width * img.height)}
        self.assertIn((41, 121, 255), colors)

    def test_qa_bev_tile_displays_boxes_at_50m_boundary(self):
        from scripts import vf_qa

        b = {**box(0.0, -48.0, label="Car"), "idx": 1}
        img = vf_qa._bev_tile([], [b], (640, 340))
        colors = {c for _, c in img.getcolors(maxcolors=img.width * img.height)}
        self.assertIn((255, 23, 68), colors)


if __name__ == "__main__":
    unittest.main()
