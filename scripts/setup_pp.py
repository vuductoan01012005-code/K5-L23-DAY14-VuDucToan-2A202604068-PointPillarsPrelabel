"""Fetch the pinned upstream PointPillars code + KITTI weights into third_party/ (no compiling).

    python scripts/setup_pp.py              # git clone (cần git + mạng)
    python scripts/setup_pp.py --zip FILE   # máy không có git: dùng file zip GitHub của commit đã ghim
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = "https://github.com/zhulf0804/PointPillars"
COMMIT = "620e6b0d07e4cb37b7b0114f26b934e8be92a0ba"
DEST = Path(__file__).resolve().parents[1] / "third_party" / "PointPillars"
WEIGHTS = DEST / "pretrained" / "epoch_160.pth"


def _handle_readonly(func, path: str, _exc: object) -> None:
    """Clear read-only attribute (e.g. Windows .git pack files) and retry removal."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _force_rm(path: Path) -> None:
    """Remove a directory tree, clearing Windows read-only attributes if needed."""
    if not path.exists() and not path.is_symlink():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_handle_readonly)
    else:
        shutil.rmtree(path, onerror=_handle_readonly)


def from_zip(path: Path) -> None:
    tmp = DEST.parent / "_unzip"
    _force_rm(tmp)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    (inner,) = [p for p in tmp.iterdir() if p.is_dir()]
    _force_rm(DEST)
    inner.rename(DEST)
    _force_rm(tmp)


def main() -> int:
    from scripts import vf_common  # stdlib-only, safe before dependencies are installed
    vf_common.configure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", help="đường dẫn zip đã tải sẵn")
    ap.add_argument("--download-zip", action="store_true", help="tải zip của commit qua HTTPS (không cần git)")
    args = ap.parse_args()
    if WEIGHTS.exists():
        print(f"Đã có {WEIGHTS}. Bỏ qua.")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if args.zip:
        from_zip(Path(args.zip))
    elif args.download_zip or not shutil.which("git"):
        target = DEST.parent / f"PointPillars-{COMMIT[:7]}.zip"
        print(f"Tải {REPO}/archive/{COMMIT}.zip ...")
        urllib.request.urlretrieve(f"{REPO}/archive/{COMMIT}.zip", target)
        from_zip(target)
        target.unlink()
    else:
        _force_rm(DEST)
        subprocess.run(["git", "clone", "--quiet", REPO, str(DEST)], check=True)
        subprocess.run(["git", "-C", str(DEST), "checkout", "--quiet", COMMIT], check=True)
    if not WEIGHTS.exists():
        print(f"Không thấy {WEIGHTS} sau khi tải.", file=sys.stderr)
        return 1
    print(f"Xong: {DEST} @ {COMMIT[:10]}, weights {WEIGHTS.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
