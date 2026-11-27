"""Regression checks for the Day 14 learner workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import vf_cvat
from scripts import vf_d14


class TestDay14ReleaseRegressions(unittest.TestCase):
    def test_frame_assignment_rejects_missing_or_short_sequences(self):
        complete = {seq: [f"{frame:04}" for frame in range(60)] for seq in range(1, 6)}
        cases = (
            ({seq: frames for seq, frames in complete.items() if seq != 5}, "sequence_05"),
            ({**complete, 2: complete[2][:-1]}, "sequence_02"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "d14_QA14001"
            args = SimpleNamespace(student="QA14001", seed="QA-SEED", data=tmp)
            for sequences, bad_sequence in cases:
                with self.subTest(sequence=bad_sequence):
                    with mock.patch("scripts.vf_common.list_sequences", return_value=sequences), \
                         mock.patch("scripts.vf_cvat.d14_dir", return_value=output):
                        with self.assertRaisesRegex(SystemExit, bad_sequence):
                            vf_cvat.cmd_day14(args)
                    self.assertFalse(output.exists())

    def test_required_reference_rejects_missing_frames_but_allows_empty_box_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref_A.json"
            names = ["s01f00", "s02f00", "s03f00"]
            present = {"s01f00": []}
            path.write_text(json.dumps({"frames": present}), encoding="utf-8-sig")
            with self.assertRaisesRegex(SystemExit, "s02f00.*s03f00"):
                vf_d14.load_reference(str(path), None, names, require_all=True)
            self.assertEqual(vf_d14.load_reference(str(path), None, ["s01f00"], require_all=True), present)
            with self.assertRaisesRegex(SystemExit, "--frames"):
                vf_d14.load_reference(str(path), None, [], require_all=True)

    def test_prelabel_rejects_unready_existing_tasks_without_overwriting(self):
        frame = SimpleNamespace(name="pointcloud/s01f00.pcd")
        other = SimpleNamespace(name="pointcloud/s01f01.pcd")
        cases = (
            ([], [frame], [], "chưa có job"),
            ([object()], [other], [], "không khớp"),
            ([object()], [frame], [], "chưa có nhãn"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "pred_A.json"
            pred.write_text(json.dumps({"frames": {"s01f00": [{"label": "Car"}]}}), encoding="utf-8")
            args = SimpleNamespace(student="QA14001", task="A", data=tmp, boxes=str(pred), frames="s01f00",
                                   host="http://localhost:8080", user="qa", org=None, bundle_out=None,
                                   dry_run=False, replace=False)
            for jobs, frames, shapes, message in cases:
                with self.subTest(message=message):
                    task = mock.Mock(id=7)
                    task.get_jobs.return_value = jobs
                    task.get_meta.return_value = SimpleNamespace(frames=frames)
                    task.get_annotations.return_value = SimpleNamespace(shapes=shapes, tracks=[])
                    with mock.patch("scripts.vf_d14.frame_rows", return_value=[{"name": "s01f00"}]), \
                         mock.patch("scripts.vf_cvat.connect"), \
                         mock.patch("scripts.vf_cvat.find_tasks", return_value=[task]), \
                         mock.patch("scripts.vf_cvat.build_bundle") as bundle:
                        with self.assertRaisesRegex(SystemExit, message):
                            vf_cvat.cmd_prelabel(args)
                        bundle.assert_not_called()
                        task.update_annotations.assert_not_called()
                        task.remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
