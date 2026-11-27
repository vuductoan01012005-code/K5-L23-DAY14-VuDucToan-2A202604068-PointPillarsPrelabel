"""Edge cases and sequential pass inference tests for vf_pp.core.Detector."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vf_pp.core import Config, Detector  # noqa: E402


class TestDetectorEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.det = Detector(Config())

    def _run_with_spy(self, pts: np.ndarray, mock_box: bool = False):
        real_model = self.det.model
        batch_sizes: list[int] = []

        def _spy(batch, mode="test"):
            batch_sizes.append(len(batch))
            if mock_box:
                return [
                    {
                        "lidar_bboxes": np.array([[10.0, 0.0, -1.5, 1.8, 4.2, 1.5, 0.0]], dtype=np.float32),
                        "labels": np.array([2], dtype=np.int64),
                        "scores": np.array([0.9], dtype=np.float32),
                    }
                    for _ in batch
                ]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="torch.meshgrid", category=UserWarning)
                return real_model(batch, mode=mode)

        _spy.pc_range = real_model.pc_range
        self.det.model = _spy
        try:
            return self.det.raw_detect(pts), batch_sizes
        finally:
            self.det.model = real_model

    def test_empty_point_cloud_returns_empty_list(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        self.assertEqual(self.det.detect(empty), [])

    def test_out_of_range_point_cloud_returns_empty_list(self) -> None:
        pts = np.array([[200.0, 200.0, 0.0, 0.5]], dtype=np.float32)
        self.assertEqual(self.det.detect(pts), [])

    def test_partial_sector_cloud_does_not_crash_on_empty_rotated_passes(self) -> None:
        # Front-far sector: x_m in [42, 50], y_m in [1, 2] -> only pass 0 is in pc_range;
        # passes 1, 2, 3 all have 0 in-range points.
        rng = np.random.default_rng(42)
        n = 50
        pts = np.column_stack([
            rng.uniform(1.0, 2.0, n),
            rng.uniform(-50.0, -42.0, n),
            rng.uniform(-1.5, 0.5, n),
            rng.uniform(0.1, 0.9, n),
        ]).astype(np.float32)
        boxes, batch_sizes = self._run_with_spy(pts)
        self.assertIsInstance(boxes, list)
        self.assertEqual(batch_sizes, [1])

    def test_raw_detect_runs_single_pass_per_model_call(self) -> None:
        # Four points across all four quadrants so every rotated pass has in-range points.
        pts = np.array([
            [10.0, -10.0, 0.0, 0.5],
            [-10.0, -10.0, 0.0, 0.5],
            [-10.0, 10.0, 0.0, 0.5],
            [10.0, 10.0, 0.0, 0.5],
        ], dtype=np.float32)
        _, batch_sizes = self._run_with_spy(pts)
        self.assertEqual(batch_sizes, [1, 1, 1, 1])

    def test_skipped_empty_passes_preserve_pass_index(self) -> None:
        # Only pass 2 (angle = pi) lands in pc_range: x_m = -y_vf in [-50, -42], y_m = x_vf in [-2, -1].
        pts = np.array([[-1.5, 45.0, 0.0, 0.5]], dtype=np.float32)
        boxes, batch_sizes = self._run_with_spy(pts, mock_box=True)
        self.assertEqual(batch_sizes, [1])
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["pass"], 2)


if __name__ == "__main__":
    unittest.main()
