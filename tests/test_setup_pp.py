"""Windows read-only .git recovery tests for setup_pp.py (BUG-03)."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import setup_pp  # noqa: E402


def _make_writable_tree(root: Path) -> None:
    if not root.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        os.chmod(dp, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        for name in filenames:
            os.chmod(dp / name, stat.S_IWRITE | stat.S_IREAD)


def _create_readonly_git_pack(root: Path, filename: str) -> Path:
    pack_dir = root / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_file = pack_dir / filename
    pack_file.write_bytes(b"pack")
    os.chmod(pack_file, stat.S_IREAD)
    return pack_file


class TestSetupPpWindowsRecovery(unittest.TestCase):
    def test_force_rm_deletes_readonly_git_pack_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "partial_clone"
            try:
                _create_readonly_git_pack(target, "pack-test.idx")
                setup_pp._force_rm(target)
                self.assertFalse(target.exists())
            finally:
                _make_writable_tree(target)

    def test_from_zip_replaces_existing_dir_with_readonly_git_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            dest = tmp_dir / "third_party" / "PointPillars"
            dest.parent.mkdir(parents=True, exist_ok=True)
            zip_path = tmp_dir / "PointPillars.zip"
            try:
                _create_readonly_git_pack(dest, "test.idx")
                with zipfile.ZipFile(zip_path, "w") as zf:
                    zf.writestr("PointPillars-abc/pretrained/epoch_160.pth", b"weights")

                with mock.patch.object(setup_pp, "DEST", dest):
                    setup_pp.from_zip(zip_path)

                self.assertTrue((dest / "pretrained" / "epoch_160.pth").exists())
            finally:
                _make_writable_tree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
