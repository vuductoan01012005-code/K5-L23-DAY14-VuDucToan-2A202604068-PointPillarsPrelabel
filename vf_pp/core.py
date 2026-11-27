"""PointPillars (KITTI weights) -> VF pre-label boxes, with every knob of the lab exposed.

Frames
------
* VF lidar: x left, y back, z up (front of the car is -y); yaw in [0, 2*pi) from +x, CCW.
* Model (KITTI lidar): x front, y left, z up. ``x_m = -y_vf``, ``y_m = x_vf``.
* Model box: (x, y, z_bottom, w, l, h, theta). Converted heading: ``yaw_vf = pi - theta``.

The model only looks forward (x in [0, 69.12] m, |y| < 39.68 m), so a 360-degree
frame is covered with ``passes`` rotated copies run sequentially, then merged with
a class-agnostic BEV NMS.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import upstream

MODEL_TO_VF = {0: 5, 1: 4, 2: 1}  # Pedestrian, Cyclist, Car  ->  VF class ids
VF_NAMES = {1: "Car", 2: "Truck", 3: "Bus", 4: "Cyclist", 5: "Pedestrian"}

# plausible (l, w, h) ranges per VF class (labeling guideline section 4.3)
SIZE_RANGE = {
    1: ((3.2, 5.6), (1.5, 2.2), (1.2, 2.2)),
    2: ((5.6, 12.0), (1.9, 2.8), (2.0, 4.0)),
    3: ((9.0, 15.0), (2.3, 2.9), (2.8, 4.0)),
    4: ((1.4, 2.2), (0.4, 1.0), (1.0, 2.0)),
    5: ((0.3, 1.2), (0.3, 1.2), (1.0, 2.1)),
}


@dataclass
class Config:
    # A: input
    voxel_size: float = 0.16
    intensity_mode: str = "clip"      # clip | log | const
    intensity_scale: float = 1.0
    ground_mode: str = "fixed"        # fixed (use delta_z) | auto (move ground to -1.73 m)
    delta_z: float = 0.0
    passes: int = 4
    # B: model output
    score_thr: float = 0.1
    thr_car: float | None = None
    thr_cyc: float | None = None
    thr_ped: float | None = None
    nms_thr: float = 0.01
    nms_pre: int = 100
    max_num: int = 50
    # C: VF post-processing
    range_m: float = 50.0
    min_points: int = 0
    merge_nms_thr: float = 0.1
    size_pad_m: float = 0.1
    truck_l_min: float = 5.6
    truck_h_min: float = 2.4
    bus_l_min: float = 9.0
    bus_h_min: float = 2.8
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ------------------------------------------------------------- inputs

def prepare_points(raw: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    """VF float32 (N, 4) -> model frame (N, 4); returns the points and the z shift used."""
    pts = np.empty_like(raw, dtype=np.float32)
    pts[:, 0] = -raw[:, 1]
    pts[:, 1] = raw[:, 0]
    dz = cfg.delta_z
    if cfg.ground_mode == "auto":
        r = np.hypot(raw[:, 0], raw[:, 1])
        near = raw[(r > 3) & (r < 15), 2]
        if len(near) > 100:
            dz = -1.73 - float(np.percentile(near, 5))
    pts[:, 2] = raw[:, 2] + dz
    i = raw[:, 3].astype(np.float32)
    if cfg.intensity_mode == "log":
        pts[:, 3] = np.clip(np.log1p(i) / math.log1p(255.0) * cfg.intensity_scale, 0, 1)
    elif cfg.intensity_mode == "const":
        pts[:, 3] = cfg.intensity_scale if cfg.intensity_scale <= 1 else 0.3
    else:
        pts[:, 3] = np.clip(i * cfg.intensity_scale, 0, 1)
    return pts, dz


def _rot(xy: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return xy @ np.array([[c, s], [-s, c]], dtype=xy.dtype)  # rotate by +a


# ------------------------------------------------------------ inference

class Detector:
    def __init__(self, cfg: Config | None = None, device: str = "cpu"):
        self.cfg = cfg or Config()
        self.device = device
        self.model = upstream.load_model(self.cfg.voxel_size, device)
        self._sync()

    def _sync(self):
        m, c = self.model, self.cfg
        thr = [c.thr_ped, c.thr_cyc, c.thr_car]
        m.score_thr = min([c.score_thr] + [t for t in thr if t is not None])
        m.nms_thr, m.nms_pre, m.max_num = c.nms_thr, c.nms_pre, c.max_num

    def set_config(self, cfg: Config):
        if abs(cfg.voxel_size - self.cfg.voxel_size) > 1e-9:
            self.model = upstream.load_model(cfg.voxel_size, self.device)
        self.cfg = cfg
        self._sync()

    @torch.no_grad()
    def raw_detect(self, raw: np.ndarray) -> list[dict]:
        """All model boxes in the VF frame, before VF post-processing."""
        if len(raw) == 0:
            return []
        cfg = self.cfg
        pts, dz = prepare_points(raw, cfg)
        x0, y0, z0, x1, y1, z1 = self.model.pc_range
        angles = [2 * math.pi * k / cfg.passes for k in range(cfg.passes)]
        out = []
        for pass_idx, a in enumerate(angles):
            p = pts.copy()
            p[:, :2] = _rot(pts[:, :2], a)
            has_pts = bool(
                ((p[:, 0] >= x0) & (p[:, 0] < x1)
                 & (p[:, 1] >= y0) & (p[:, 1] < y1)
                 & (p[:, 2] >= z0) & (p[:, 2] < z1)).any()
            )
            if not has_pts:
                continue
            res = self.model([torch.from_numpy(p).to(self.device)], mode="test")[0]
            if not len(res) or not isinstance(res, dict):
                continue
            for box, lab, sc in zip(res["lidar_bboxes"], res["labels"], res["scores"]):
                x, y, zb, w, l, h, th = (float(v) for v in box)
                cx, cy = _rot(np.array([[x, y]]), -a)[0]
                psi_m = -th - math.pi / 2 - a                    # heading in model frame
                out.append({
                    "x": float(cy), "y": float(-cx),             # model -> VF
                    "z": zb + h / 2 - dz,
                    "length": l, "width": w, "height": h,
                    "yaw": (psi_m - math.pi / 2) % (2 * math.pi),
                    "model_class": int(lab), "score": float(sc), "pass": pass_idx,
                })
        return out

    def detect(self, raw: np.ndarray) -> list[dict]:
        return postprocess(self.raw_detect(raw), raw, self.cfg)


# -------------------------------------------------------- VF post-step

def _corners(b: dict) -> np.ndarray:
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    hl, hw = b["length"] / 2, b["width"] / 2
    loc = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return loc @ np.array([[c, s], [-s, c]]) + np.array([b["x"], b["y"]])


def bev_iou(a: dict, b: dict) -> float:
    if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > (math.hypot(a["length"], a["width"])
                                                        + math.hypot(b["length"], b["width"])) / 2:
        return 0.0
    return upstream.rotated_iou(_corners(a), _corners(b))


def points_in_box(raw: np.ndarray, b: dict) -> int:
    d = raw[:, :2] - np.array([b["x"], b["y"]])
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    u, v = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    dz = np.abs(raw[:, 2] - b["z"])
    return int(((np.abs(u) <= b["length"] / 2) & (np.abs(v) <= b["width"] / 2)
                & (dz <= b["height"] / 2)).sum())


def vf_class(b: dict, cfg: Config) -> int:
    base = MODEL_TO_VF[b["model_class"]]
    if base != 1:
        return base
    if b["length"] >= cfg.bus_l_min and b["height"] >= cfg.bus_h_min:
        return 3
    if b["length"] >= cfg.truck_l_min or b["height"] >= cfg.truck_h_min:
        return 2
    return 1


def size_warning(b: dict) -> str:
    rng = SIZE_RANGE[b["label"]]
    bad = [name for name, v, (lo, hi) in zip("lwh", (b["length"], b["width"], b["height"]), rng)
           if not lo <= v <= hi]
    return ",".join(bad)


def postprocess(boxes: list[dict], raw: np.ndarray, cfg: Config) -> list[dict]:
    per_class = {0: cfg.thr_ped, 1: cfg.thr_cyc, 2: cfg.thr_car}
    kept = []
    for b in boxes:
        thr = per_class[b["model_class"]]
        if b["score"] < (thr if thr is not None else cfg.score_thr):
            continue
        if math.hypot(b["x"], b["y"]) > cfg.range_m:
            continue
        kept.append(dict(b))
    # merge the rotated passes: class-agnostic NMS by score
    kept.sort(key=lambda b: -b["score"])
    merged = []
    for b in kept:
        if all(bev_iou(b, m) <= cfg.merge_nms_thr for m in merged):
            merged.append(b)
    out = []
    for b in merged:
        b["npts"] = points_in_box(raw, b)
        if b["npts"] < cfg.min_points:
            continue
        b["label"] = vf_class(b, cfg)
        b["length"] += cfg.size_pad_m
        b["width"] += cfg.size_pad_m
        b["warn"] = size_warning(b)
        out.append(b)
    for i, b in enumerate(out):
        b["pp_id"] = i
    return out


# ----------------------------------------------------------- scoring

def needs_edit(g: dict, p: dict, pos_tol: float = 0.3, yaw_tol: float = math.radians(10),
               size_tol: float = 0.2) -> bool:
    """True when a matched prediction still has to be fixed by hand (see ``correction_cost``)."""
    dyaw = abs((g["yaw"] - p["yaw"] + math.pi / 2) % math.pi - math.pi / 2)
    return (g["label"] != VF_NAMES.get(p["label"], p["label"]) or math.hypot(g["x"] - p["x"], g["y"] - p["y"]) > pos_tol
            or dyaw > yaw_tol or any(abs(g[k] - p[k]) > size_tol for k in ("length", "width", "height")))


MAX_RANGE = 50.0  # metres; the lab (like the Day 13 grader) scores only objects within this BEV range


def within_range(boxes: list[dict], max_range: float = MAX_RANGE) -> list[dict]:
    """Boxes whose centre is at most ``max_range`` m from the LiDAR in BEV."""
    return [b for b in boxes if math.hypot(b["x"], b["y"]) <= max_range]


def correction_cost(gt: list[dict], pred: list[dict], min_iou: float = 0.3,
                    pos_tol: float = 0.3, yaw_tol: float = math.radians(10),
                    size_tol: float = 0.2) -> dict:
    """Annotator effort to turn ``pred`` into ``gt``: 0.3*delete + 1.0*add + 0.5*edit.

    A matched box needs an edit when its class differs, the centre is off by more
    than ``pos_tol`` m, the heading by more than ``yaw_tol`` or any size by more
    than ``size_tol`` m. Only boxes within ``MAX_RANGE`` m count. Heading is compared modulo pi (front/back flips are edits
    only through the class/position checks, as CVAT users fix them with one key).
    """
    from scripts.vf_common import match_boxes  # shared with the Day 13 grader

    gt, pred = within_range(gt), within_range(pred)
    pairs = match_boxes(gt, pred, min_iou)
    edit = sum(needs_edit(gt[gi], pred[pj], pos_tol, yaw_tol, size_tol) for gi, pj, _ in pairs)
    tp = len(pairs)
    delete, add = len(pred) - tp, len(gt) - tp
    return {"gt": len(gt), "pred": len(pred), "match": tp, "delete": delete, "add": add,
            "edit": edit, "cost": round(0.3 * delete + 1.0 * add + 0.5 * edit, 2)}
