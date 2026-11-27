"""CVAT cuboid points and CVAT safety regressions."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import vf_common as vf  # noqa: E402
from scripts import vf_cvat  # noqa: E402
from scripts.vf_cvat import cuboid_points  # noqa: E402


class TestCvatCuboid(unittest.TestCase):
    def test_cuboid_has_16_values_and_round_trips(self):
        box = {"x": 3.5, "y": -12.0, "z": 0.8, "length": 4.4, "width": 1.9, "height": 1.6, "yaw": 1.2}
        pts = cuboid_points(box)
        self.assertEqual(len(pts), 16)
        self.assertEqual(pts[9:], [0.0] * 7)
        back = vf.cuboid_from_cvat(pts, "Car")
        for k in box:
            self.assertLess(abs(back[k] - box[k]), 1e-9, k)


class _FakeApiError(Exception):
    def __init__(self, status: int, body: str, message: str = "HTTP Error"):
        super().__init__(message)
        self.status = status
        self.body = body


class TestCvatSafety(unittest.TestCase):
    def test_warn_if_tracks_prints_warning_to_stderr_when_tracks_present(self):
        ann = SimpleNamespace(shapes=[], tracks=[SimpleNamespace(id=1)])
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            count = vf_cvat.warn_if_tracks("D14-STU01-A", ann)
        err = buf.getvalue()
        self.assertEqual(count, 1)
        self.assertIn("Track", err)
        self.assertIn("Shape", err)
        self.assertIn("D14-STU01-A", err)

    def test_warn_if_tracks_silent_when_no_tracks(self):
        for ann in (SimpleNamespace(shapes=[], tracks=[]), SimpleNamespace(shapes=[])):
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                count = vf_cvat.warn_if_tracks("D14-STU01-A", ann)
            self.assertEqual(count, 0)
            self.assertEqual(buf.getvalue(), "")

    def test_task_boxes_warns_when_tracks_present(self):
        task = SimpleNamespace(
            id=7,
            name="D13-STU01",
            get_labels=lambda: [SimpleNamespace(id=1, name="Car")],
            get_meta=lambda: SimpleNamespace(frames=[SimpleNamespace(name="pointcloud/s01f00.pcd")]),
            get_annotations=lambda: SimpleNamespace(shapes=[], tracks=[SimpleNamespace(id=9)]),
        )
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rows = vf_cvat.task_boxes(None, task)
        err = buf.getvalue()
        self.assertEqual(rows, [])
        self.assertIn("Track", err)
        self.assertIn("Shape", err)
        self.assertIn("D13-STU01", err)

    def test_cmd_reference_rejects_tracks_and_preserves_last_export(self):
        task = SimpleNamespace(
            id=11,
            name="D14-STU01-A",
            get_labels=lambda: [SimpleNamespace(id=1, name="Car")],
            get_meta=lambda: SimpleNamespace(frames=[SimpleNamespace(name="pointcloud/s01f00.pcd")]),
            get_annotations=lambda: SimpleNamespace(
                shapes=[],
                tracks=[SimpleNamespace(id=1), SimpleNamespace(id=2)],
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ref_A.json"
            out_path.write_text("previous export", encoding="utf-8")
            args = SimpleNamespace(
                student="STU01",
                task="A",
                host="http://localhost:8080",
                user="alice",
                org=None,
                out=str(out_path),
                scale_order="lwh",
            )
            with mock.patch("scripts.vf_cvat.connect", return_value=SimpleNamespace()), \
                 mock.patch("scripts.vf_cvat.find_tasks", return_value=[task]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    vf_cvat.cmd_reference(args)
            err = str(ctx.exception)
            self.assertIn("Track", err)
            self.assertIn("Shape", err)
            self.assertIn("D14-STU01-A", err)
            self.assertEqual(out_path.read_text(encoding="utf-8"), "previous export")

    def test_cmd_reference_rejects_duplicate_tasks_before_reading_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ref_B.json"
            args = SimpleNamespace(student="STU01", task="B", host="http://localhost:8080",
                                   user="alice", org=None, out=str(out_path), scale_order="lwh")
            tasks = [mock.Mock(id=11), mock.Mock(id=12)]
            with mock.patch("scripts.vf_cvat.connect"), mock.patch("scripts.vf_cvat.find_tasks", return_value=tasks):
                with self.assertRaisesRegex(SystemExit, r"D14-STU01-B.*11, 12"):
                    vf_cvat.cmd_reference(args)
            for task in tasks:
                task.get_annotations.assert_not_called()
            self.assertFalse(out_path.exists())

    def test_find_tasks_turns_404_org_into_friendly_system_exit(self):
        def _raise_404():
            raise _FakeApiError(status=404, body='{"detail":"badorg organization does not exist."}')

        client = SimpleNamespace(
            organization_slug="badorg",
            tasks=SimpleNamespace(list=_raise_404),
        )
        with self.assertRaises(SystemExit) as ctx:
            vf_cvat.find_tasks(client, "D14-STU01-A")
        msg = str(ctx.exception)
        self.assertIn("--org", msg)
        self.assertIn("badorg", msg)
        self.assertIn("Docker", msg)

    def test_find_tasks_reraises_other_errors_unchanged(self):
        for err in (
            _FakeApiError(status=500, body='{"detail":"badorg organization exploded"}'),
            _FakeApiError(status=404, body='{"detail":"task not found"}'),
        ):
            client = SimpleNamespace(
                organization_slug="badorg",
                tasks=SimpleNamespace(list=lambda e=err: (_ for _ in ()).throw(e)),
            )
            with self.assertRaises(_FakeApiError) as ctx:
                vf_cvat.find_tasks(client, "D14-STU01-A")
            self.assertIs(ctx.exception, err)

    def _make_prelabel_workspace(self, tmp: str) -> tuple[Path, Path]:
        base = Path(tmp)
        data = base / "data"
        lidar = data / "sequence_01" / "lidar"
        lidar.mkdir(parents=True)
        (lidar / "0000.bin").write_bytes(b"")
        frames_file = base / "frames_A.txt"
        frames_file.write_text("s01f00\n", encoding="utf-8")
        return data, frames_file

    def test_cmd_prelabel_checks_existing_task_before_building_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, frames_file = self._make_prelabel_workspace(tmp)
            args = SimpleNamespace(
                student="STU01",
                task="A",
                data=str(data),
                boxes=None,
                frames=str(frames_file),
                host="http://localhost:8080",
                user="alice",
                org=None,
                bundle_out=None,
                dry_run=False,
                replace=False,
            )
            task = mock.Mock(id=42)
            task.get_jobs.return_value = [object()]
            task.get_meta.return_value = SimpleNamespace(frames=[SimpleNamespace(name="pointcloud/s01f00.pcd")])
            with mock.patch("scripts.vf_cvat.connect", return_value=SimpleNamespace()) as m_connect, \
                 mock.patch("scripts.vf_cvat.find_tasks", return_value=[task]) as m_find, \
                 mock.patch("scripts.vf_cvat.build_bundle", wraps=vf_cvat.build_bundle) as m_bundle, \
                 contextlib.redirect_stdout(io.StringIO()):
                rc = vf_cvat.cmd_prelabel(args)
            self.assertEqual(rc, 0)
            self.assertEqual(m_connect.call_count, 1)
            self.assertEqual(m_find.call_count, 1)
            self.assertEqual(m_bundle.call_count, 0)

    def test_cmd_prelabel_rejects_changed_prediction_without_touching_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, frames_file = self._make_prelabel_workspace(tmp)
            prediction = Path(tmp) / "pred_B.json"
            args = SimpleNamespace(student="STU01", task="B", data=str(data), boxes=str(prediction),
                                   frames=str(frames_file), host="http://localhost:8080",
                                   user="alice", org=None, bundle_out=None, dry_run=False, replace=False)
            task = mock.Mock(id=11)
            task.get_jobs.return_value = [object()]
            task.get_meta.return_value = SimpleNamespace(frames=[SimpleNamespace(name="pointcloud/s01f00.pcd")])
            task.get_annotations.return_value = SimpleNamespace(shapes=[object()])
            client = mock.Mock()
            client.tasks.create_from_data.return_value = task
            doc = {"meta": {"config": {"score_thr": 0.1}}, "frames": {"s01f00": [{"label": "Car"}]}}
            prediction.write_text(json.dumps(doc), encoding="utf-8")
            state = Path(tmp) / "state"
            with mock.patch("scripts.vf_cvat.connect", return_value=client), \
                 mock.patch("scripts.vf_cvat.find_tasks", side_effect=[[], [task], [task], [task]]), \
                 mock.patch("scripts.vf_cvat.PRELABEL_STATE", state, create=True), \
                 mock.patch("scripts.vf_cvat.upload_prelabels") as upload, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(vf_cvat.cmd_prelabel(args), 0)
                prediction.write_text(json.dumps(doc, sort_keys=True, indent=2), encoding="utf-8-sig")
                self.assertEqual(vf_cvat.cmd_prelabel(args), 0)
                doc["meta"]["config"]["score_thr"] = 0.4
                prediction.write_text(json.dumps(doc), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "file dự đoán.*đã đổi"):
                    vf_cvat.cmd_prelabel(args)
                with mock.patch("scripts.vf_cvat.PRELABEL_STATE", Path(tmp) / "legacy", create=True), \
                     contextlib.redirect_stderr(io.StringIO()) as warning:
                    self.assertEqual(vf_cvat.cmd_prelabel(args), 0)
                self.assertIn("chưa thể đối chiếu", warning.getvalue())
            upload.assert_called_once()
            client.tasks.create_from_data.assert_called_once()
            task.remove.assert_not_called()

    def test_cmd_prelabel_validates_host_and_user_before_building_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, frames_file = self._make_prelabel_workspace(tmp)
            args = SimpleNamespace(
                student="STU01",
                task="A",
                data=str(data),
                boxes=None,
                frames=str(frames_file),
                host=None,
                user=None,
                org=None,
                bundle_out=None,
                dry_run=False,
                replace=False,
            )
            with mock.patch("scripts.vf_cvat.build_bundle", wraps=vf_cvat.build_bundle) as m_bundle, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    vf_cvat.cmd_prelabel(args)
            self.assertEqual(m_bundle.call_count, 0)

    def test_cmd_prelabel_dry_run_still_builds_bundle_without_connecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, frames_file = self._make_prelabel_workspace(tmp)
            bundle_out = Path(tmp) / "out.zip"
            args = SimpleNamespace(
                student="STU01",
                task="A",
                data=str(data),
                boxes=None,
                frames=str(frames_file),
                host=None,
                user=None,
                org=None,
                bundle_out=str(bundle_out),
                dry_run=True,
                replace=False,
            )
            with mock.patch("scripts.vf_cvat.connect") as m_connect, \
                 mock.patch("scripts.vf_cvat.find_tasks") as m_find, \
                 mock.patch("scripts.vf_cvat.build_bundle", wraps=vf_cvat.build_bundle) as m_bundle, \
                 contextlib.redirect_stdout(io.StringIO()):
                rc = vf_cvat.cmd_prelabel(args)
            self.assertEqual(rc, 0)
            self.assertEqual(m_connect.call_count, 0)
            self.assertEqual(m_find.call_count, 0)
            self.assertEqual(m_bundle.call_count, 1)
            self.assertTrue(bundle_out.exists())


if __name__ == "__main__":
    unittest.main()
