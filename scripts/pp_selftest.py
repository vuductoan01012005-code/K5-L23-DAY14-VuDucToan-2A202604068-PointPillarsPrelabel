"""Check the install in ~30 s: model loads, CPU ops agree with a brute-force reference, one frame runs.

    python scripts/pp_selftest.py [--data <Data-VF>]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="torch.meshgrid")


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'LỖI'}] {name} {detail}")
    return ok


def main() -> int:
    from scripts import vf_common  # stdlib-only, safe before dependencies are installed
    vf_common.configure_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    args = ap.parse_args()
    import numpy as np
    import torch
    from vf_pp import upstream
    from vf_pp.core import Config, Detector, bev_iou

    ok = check("python >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0])
    ok &= check("torch", True, torch.__version__)
    # voxelization: every point lands in its own pillar, counts add up
    rng = np.random.default_rng(0)
    pts = torch.from_numpy(rng.uniform([0, -39, -2.9, 0], [69, 39, 0.9, 1], (5000, 4)).astype(np.float32))
    vox = upstream.Voxelization([0.16, 0.16, 4], [0, -39.68, -3, 69.12, 39.68, 1], 32, (16000, 40000))
    vox.eval()
    v, c, n = vox(pts)
    cell = torch.floor((pts[:, :2] - torch.tensor([0, -39.68])) / 0.16).long()
    ok &= check("voxelize", int(n.sum()) == len(pts) and len(c) == len(torch.unique(cell, dim=0)),
                f"{len(c)} pillar")
    # rotated IoU: square vs itself rotated 45 deg -> known value
    a = {"x": 0, "y": 0, "length": 2, "width": 2, "yaw": 0}
    b = dict(a, yaw=math.pi / 4)
    expect = (8 * (math.sqrt(2) - 1)) / (8 - 8 * (math.sqrt(2) - 1))
    ok &= check("rotated IoU", abs(bev_iou(a, b) - expect) < 1e-6, f"{bev_iou(a, b):.4f}")
    start = time.time()
    det = Detector(Config())
    ok &= check("model + weights", True, f"{time.time() - start:.1f}s")
    if args.data:
        from scripts import vf_d14
        row = vf_d14.frame_rows(Path(args.data), ["s05f40"])[0]
        start = time.time()
        boxes = det.detect(vf_d14.load_points(row))
        ok &= check("s05f40", len(boxes) > 5, f"{len(boxes)} box, {time.time() - start:.1f}s")
    print("TẤT CẢ OK" if ok else "CÓ LỖI: xem dòng LỖI ở trên")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
