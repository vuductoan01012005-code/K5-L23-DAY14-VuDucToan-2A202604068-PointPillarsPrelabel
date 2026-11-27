"""Windows cp1252 stdio, UTF-8 / UTF-8-BOM file I/O, and CLI message regressions."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))

from scripts import vf_common as vf  # noqa: E402
from scripts import vf_d14  # noqa: E402


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(ROOT)
    return env


def _run_py(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        env=_clean_env(),
        capture_output=True,
    )


def _make_stub_data(base: Path) -> Path:
    data = base / "data_vf"
    for seq in range(1, 6):
        lidar = data / f"sequence_{seq:02d}" / "lidar"
        lidar.mkdir(parents=True, exist_ok=True)
        for idx in range(60):
            (lidar / f"{idx:04d}.bin").write_bytes(b"")
    return data


class TestWindowsEncoding(unittest.TestCase):
    def test_read_frame_list_vietnamese_utf8_under_cp1252(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.txt"
            path.write_text("# Đợt 1 - Thời điểm\ns02f07\n", encoding="utf-8")
            res = _run_py(["-c", "import sys; from scripts import vf_d14; print(vf_d14.read_frame_list(sys.argv[1]))", str(path)])
            self.assertEqual(res.returncode, 0, res.stderr.decode("utf-8", errors="replace"))
            self.assertIn("['s02f07']", res.stdout.decode("utf-8", errors="replace"))

    def test_read_frame_list_notepad_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames_bom.txt"
            path.write_text("# D14-001-A seed_tag=abcdef1234\n  # comment\ns02f07\n", encoding="utf-8-sig")
            self.assertEqual(vf_d14.read_frame_list(str(path)), ["s02f07"])

    def test_load_config_vietnamese_utf8_under_cp1252(self):
        with tempfile.TemporaryDirectory() as tmp:
            for enc in ("utf-8", "utf-8-sig"):
                with self.subTest(encoding=enc):
                    cfg_path = Path(tmp) / f"cfg_{enc}.json"
                    cfg_path.write_text('{"score_thr": 0.4, "extra": {"note": "Đợt 1"}}', encoding=enc)
                    res = _run_py([
                        "-c",
                        (
                            "import sys; from scripts import pp_prelabel; "
                            "c = pp_prelabel.load_config(sys.argv[1], []); "
                            "assert c.score_thr == 0.4 and c.extra['note'] == 'Đợt 1'"
                        ),
                        str(cfg_path),
                    ])
                    self.assertEqual(res.returncode, 0, res.stderr.decode("utf-8", errors="replace"))

    def test_cli_scripts_print_vietnamese_without_unicode_encode_error_under_cp1252(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            stub_data = _make_stub_data(tmp_dir)
            invocations: list[tuple[str, list[str], Path, bool, str | None]] = [
                ("pp_selftest.py", [str(SCRIPTS / "pp_selftest.py")], ROOT, True, None),
                (
                    "vf_cvat.py day14",
                    [str(SCRIPTS / "vf_cvat.py"), "day14", "--student", "STU01", "--seed", "LABSEED", "--data", str(stub_data)],
                    tmp_dir,
                    True,
                    None,
                ),
                ("pp_prelabel.py --help", [str(SCRIPTS / "pp_prelabel.py"), "--help"], ROOT, True, None),
                ("pp_sweep.py --help", [str(SCRIPTS / "pp_sweep.py"), "--help"], ROOT, True, None),
                ("pp_compare.py --help", [str(SCRIPTS / "pp_compare.py"), "--help"], ROOT, True, None),
                ("setup_pp.py --help", [str(SCRIPTS / "setup_pp.py"), "--help"], ROOT, True, None),
                (
                    "pp_prelabel.py --set invalid",
                    [str(SCRIPTS / "pp_prelabel.py"), "--data", str(stub_data), "--frames", "s02f07", "--set", "bad_key=1"],
                    tmp_dir,
                    False,
                    "không có tham số này",
                ),
                (
                    "pp_sweep.py --grid invalid",
                    [
                        str(SCRIPTS / "pp_sweep.py"),
                        "--data", str(stub_data),
                        "--frames", "s02f07",
                        "--gt", "missing.json",
                        "--grid", "bad_key=1",
                    ],
                    tmp_dir,
                    False,
                    "không có tham số này",
                ),
                (
                    "pp_compare.py missing gt",
                    [
                        str(SCRIPTS / "pp_compare.py"),
                        "--data", str(stub_data),
                        "--frames", "s02f07",
                        "--gt", "missing.json",
                        "--tuned", "missing.json",
                        "--out", str(tmp_dir / "cmp"),
                    ],
                    tmp_dir,
                    False,
                    "cần file",
                ),
            ]
            if (ROOT / "third_party" / "PointPillars" / "pretrained" / "epoch_160.pth").exists():
                invocations.append(("setup_pp.py", [str(SCRIPTS / "setup_pp.py")], ROOT, True, None))

            for label, args, cwd, expect_zero, expect_stderr_substr in invocations:
                with self.subTest(cmd=label):
                    res = _run_py(args, cwd=cwd)
                    self.assertNotIn(
                        b"UnicodeEncodeError",
                        res.stderr,
                        f"{label}: {res.stderr.decode('utf-8', errors='replace')}",
                    )
                    if expect_zero:
                        self.assertEqual(
                            res.returncode,
                            0,
                            f"{label} exited {res.returncode}: {res.stderr.decode('utf-8', errors='replace')}",
                        )
                    else:
                        self.assertNotEqual(res.returncode, 0)
                        if expect_stderr_substr:
                            self.assertIn(expect_stderr_substr, res.stderr.decode("utf-8", errors="replace"))

    def test_normalize_student_single_char_mentions_length_and_parse_frame_name_not_day13(self):
        with self.assertRaises(SystemExit) as ctx:
            vf.normalize_student("1")
        msg = str(ctx.exception)
        self.assertIn("2", msg)
        self.assertIn("32", msg)

        with self.assertRaises(ValueError) as ctx2:
            vf.parse_frame_name("bad")
        self.assertNotIn("Day 13", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
