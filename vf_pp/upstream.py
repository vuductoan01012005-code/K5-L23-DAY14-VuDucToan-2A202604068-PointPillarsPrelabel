"""Load the unmodified zhulf0804/PointPillars model on any CPU, without compiling.

The upstream repo has two compiled extensions (voxelization in C++/CUDA and the
rotated NMS in CUDA). This module registers pure-PyTorch replacements under the
same import names, then imports ``pointpillars/model/*.py`` straight from the
pinned checkout made by ``scripts/setup_pp.py``. No upstream file is edited.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party" / "PointPillars"
WEIGHTS = UPSTREAM / "pretrained" / "epoch_160.pth"


# ------------------------------------------------------------ voxelize

class Voxelization(torch.nn.Module):
    """Hard voxelization equal in output to mmdet3d ``hard_voxelize`` (deterministic).

    Points are kept in input order: the first ``max_points`` of each pillar and
    the first ``max_voxels`` pillars (in order of first appearance) survive.
    """

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, deterministic=True):
        super().__init__()
        self.voxel_size = [float(v) for v in voxel_size]
        self.pc_range = [float(v) for v in point_cloud_range]
        self.max_points = int(max_num_points)
        self.max_voxels = max_voxels if isinstance(max_voxels, (tuple, list)) else (max_voxels, max_voxels)

    @torch.no_grad()
    def forward(self, points: torch.Tensor):
        max_voxels = self.max_voxels[0] if self.training else self.max_voxels[1]
        vs = torch.tensor(self.voxel_size, device=points.device)
        lo = torch.tensor(self.pc_range[:3], device=points.device)
        grid = torch.round((torch.tensor(self.pc_range[3:], device=points.device) - lo) / vs).long()
        coor = torch.floor((points[:, :3] - lo) / vs).long()  # (N, 3) x, y, z
        inside = ((coor >= 0) & (coor < grid)).all(dim=1)
        pts, coor = points[inside], coor[inside]
        key = (coor[:, 2] * grid[1] + coor[:, 1]) * grid[0] + coor[:, 0]
        uniq, first, inverse = _unique_first(key)
        order = torch.argsort(first)                      # voxels by first appearance
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(order), device=points.device)
        vid = rank[inverse]                               # voxel id per point
        keep_vox = vid < max_voxels
        pts, vid = pts[keep_vox], vid[keep_vox]
        n_vox = int(min(len(uniq), max_voxels))
        # slot of each point inside its voxel = running count in input order
        sort_idx = torch.argsort(vid, stable=True)
        sorted_vid = vid[sort_idx]
        starts = torch.searchsorted(sorted_vid, torch.arange(n_vox, device=points.device))
        slot_sorted = torch.arange(len(sorted_vid), device=points.device) - starts[sorted_vid]
        slot = torch.empty_like(slot_sorted)
        slot[sort_idx] = slot_sorted
        ok = slot < self.max_points
        voxels = points.new_zeros((n_vox, self.max_points, points.shape[1]))
        voxels[vid[ok], slot[ok]] = pts[ok]
        counts = torch.bincount(vid[ok], minlength=n_vox).int()
        vox_coor = coor[inside.nonzero().squeeze(1)[:0]] if n_vox == 0 else None
        coors = torch.zeros((n_vox, 3), dtype=torch.int, device=points.device)
        first_kept = first[order[:n_vox]]
        coors[:] = coor[first_kept].int()                  # (x, y, z) order, as upstream returns after flip
        del vox_coor
        return voxels, coors, counts


def _unique_first(key: torch.Tensor):
    """``torch.unique`` plus the index of the first occurrence of each value."""
    uniq, inverse = torch.unique(key, return_inverse=True)
    first = torch.full((len(uniq),), len(key), dtype=torch.long, device=key.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(key), device=key.device), reduce="amin")
    return uniq, first, inverse


# ----------------------------------------------------------------- NMS

def _bev_polygon(box5: np.ndarray) -> np.ndarray:
    """Corners of an upstream NMS box ``[x1, y1, x2, y2, ry]`` (rotated about its centre)."""
    x1, y1, x2, y2, ry = box5
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    c, s = math.cos(ry), math.sin(ry)
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    # same rotation sense as the mmdet3d CUDA kernel (rotate_around_center with -angle)
    rot = np.array([[c, s], [-s, c]])
    return local @ rot + np.array([cx, cy])


def _poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman clipping of convex polygons (counter-clockwise clipper)."""
    if _signed(clipper) < 0:
        clipper = clipper[::-1]
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        prev = inp[-1]
        for cur in inp:
            cin = _side(a, b, cur) >= 0
            pin = _side(a, b, prev) >= 0
            if cin:
                if not pin:
                    out.append(_cross_point(prev, cur, a, b))
                out.append(cur)
            elif pin:
                out.append(_cross_point(prev, cur, a, b))
            prev = cur
    return np.array(out) if out else np.zeros((0, 2))


def _signed(p):
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _side(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _cross_point(p, q, a, b):
    d1, d2 = q - p, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return q
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
    return p + t * d1


def rotated_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    inter = _poly_area(_clip(poly_a, poly_b))
    union = _poly_area(poly_a) + _poly_area(poly_b) - inter
    return inter / union if union > 0 else 0.0


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """CPU drop-in for upstream ``nms_cuda``: rotated BEV NMS, returns kept indices."""
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    arr = boxes.detach().cpu().numpy()[order.cpu().numpy()]
    polys = [_bev_polygon(b) for b in arr]
    centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in arr]).reshape(-1, 2)
    radius = np.array([math.hypot(b[2] - b[0], b[3] - b[1]) / 2 for b in arr])
    suppressed = np.zeros(len(arr), dtype=bool)
    keep = []
    for i in range(len(arr)):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, len(arr)):
            if suppressed[j]:
                continue
            if np.hypot(*(centers[i] - centers[j])) > radius[i] + radius[j]:
                continue  # circles do not touch, IoU is 0
            if rotated_iou(polys[i], polys[j]) > thresh:
                suppressed[j] = True
    keep_t = order[torch.as_tensor(keep, dtype=torch.long, device=order.device)]
    if post_max_size is not None:
        keep_t = keep_t[:post_max_size]
    return keep_t


def limit_period(val, offset=0.5, period=np.pi):
    """Same as upstream ``pointpillars.utils.limit_period``."""
    return val - torch.floor(val / period + offset) * period if torch.is_tensor(val) \
        else val - np.floor(val / period + offset) * period


def _training_only(*_a, **_k):
    raise NotImplementedError("vf_pp only runs inference; training needs the compiled upstream ops")


# --------------------------------------------------------------- load

def _register_stubs() -> None:
    """Make ``pointpillars.ops`` and ``pointpillars.utils`` resolve to the CPU versions."""
    pkg = types.ModuleType("pointpillars")
    pkg.__path__ = [str(UPSTREAM / "pointpillars")]
    ops = types.ModuleType("pointpillars.ops")
    ops.Voxelization, ops.nms_cuda = Voxelization, nms_cuda
    utils = types.ModuleType("pointpillars.utils")
    utils.limit_period = limit_period
    utils.iou2d_nearest = _training_only
    model_pkg = types.ModuleType("pointpillars.model")
    model_pkg.__path__ = [str(UPSTREAM / "pointpillars" / "model")]
    sys.modules.update({
        "pointpillars": pkg, "pointpillars.ops": ops,
        "pointpillars.utils": utils, "pointpillars.model": model_pkg,
    })


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_installed() -> None:
    missing = [p for p in (UPSTREAM / "pointpillars" / "model" / "pointpillars.py", WEIGHTS) if not p.exists()]
    if missing:
        raise SystemExit("Chưa có mã PointPillars. Chạy trước: python scripts/setup_pp.py\n  thiếu: "
                         + ", ".join(str(p) for p in missing))


def load_model(voxel_size: float = 0.16, device: str = "cpu", max_voxels: int = 40000):
    """Build upstream ``PointPillars`` with KITTI weights; ``voxel_size`` changes the pillar size.

    A different pillar size keeps the same weights but changes the BEV grid, so
    detections usually get worse; that is a lesson of the lab, not a bug.
    """
    check_installed()
    _register_stubs()
    _import_file("pointpillars.model.anchors", UPSTREAM / "pointpillars/model/anchors.py")
    mod = _import_file("pointpillars.model.pointpillars", UPSTREAM / "pointpillars/model/pointpillars.py")
    x_max = 69.12
    y_half = 39.68
    # keep the grid a multiple of 8 cells so the backbone/neck shapes line up
    nx = int(round(x_max / voxel_size / 8)) * 8
    ny = int(round(2 * y_half / voxel_size / 8)) * 8
    pc_range = [0.0, -ny * voxel_size / 2, -3.0, nx * voxel_size, ny * voxel_size / 2, 1.0]
    model = mod.PointPillars(nclasses=3, voxel_size=[voxel_size, voxel_size, 4],
                             point_cloud_range=pc_range, max_voxels=(16000, max_voxels))
    if abs(voxel_size - 0.16) > 1e-9:
        # anchors span the detection range; keep them on the new grid
        for r in model.anchors_generator.ranges:
            r[0], r[1], r[3], r[4] = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    state = torch.load(WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    model.grid = (nx, ny)
    model.pc_range = pc_range
    return model
