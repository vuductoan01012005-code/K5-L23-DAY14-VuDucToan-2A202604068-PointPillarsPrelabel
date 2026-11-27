"""Day 14 shared helpers: frame sets (lecture and lab), box files and frame loading.

Lab frames depend on the student id AND a session seed announced by the coach
when the afternoon lab starts. The seed is never stored in this repository, so
nobody can pre-compute or pre-label their lab frames before the session.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

if __package__:
    from . import vf_common as vf
else:
    # The coach checker loads this module directly from its --lab directory.
    import vf_common as vf

ROOT = Path(__file__).resolve().parents[1]
LECTURE_FRAMES = [line.split()[0] for line in (ROOT / "demo" / "frames_lec.txt").read_text(encoding="utf-8-sig").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
SEED_ENV = "D14_SEED"


def get_seed(cli_value: str | None) -> str:
    seed = (cli_value or os.environ.get(SEED_ENV, "")).strip()
    if not seed:
        raise SystemExit(f"Thiếu SEED của buổi lab. Coach công bố SEED lúc bắt đầu lab; chạy lại với "
                         f"--seed <SEED> hoặc đặt biến môi trường {SEED_ENV}.")
    return seed


def seed_tag(seed: str) -> str:
    """Short fingerprint written next to the frame lists so the coach can check which seed was used."""
    return hashlib.sha256(f"d14-seed:{seed}".encode()).hexdigest()[:10]


def _blocked(student: str, counts: dict[int, int]) -> dict[int, set[int]]:
    """Indices a student must not get: sparse frames, lecture frames and own Day 13 frames, each +-2."""
    blocked: dict[int, set[int]] = {s: set(vf.SPARSE_FRAMES.get(s, set())) for s in counts}
    near = [vf.parse_frame_name(n) for n in LECTURE_FRAMES]
    near += [vf.parse_frame_name(n) for n in vf.select_frames(student, counts)]
    for seq, idx in near:
        blocked.setdefault(seq, set()).update(range(idx - 2, idx + 3))
    return blocked


def select_d14(student: str, seed: str, counts: dict[int, int],
               n_tune: int = 3, n_seq_b: int = 4, per_seq_b: int = 3, min_gap: int = 15,
               tune_gap: int = 6) -> dict[str, list[str]]:
    """Pick tuning set A (``n_tune`` frames, distinct sequences) and submission set B.

    B has ``per_seq_b`` frames in each of ``n_seq_b`` sequences, at least
    ``min_gap`` apart. A always includes the sequence left out of B; its other
    frames stay ``tune_gap`` indices from any B frame of the same sequence, so
    tuning never sees a near-copy of a submission frame.
    """
    student = vf.normalize_student(student)
    digest = hashlib.sha256(f"d14:{seed}:{student}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    blocked = _blocked(student, counts)
    seqs = sorted(counts)
    b_seqs = sorted(rng.sample(seqs, n_seq_b))
    chosen_b: dict[int, list[int]] = {}
    for seq in b_seqs:
        pool = [i for i in range(counts[seq]) if i not in blocked[seq]]
        for _ in range(10000):
            picks = sorted(rng.sample(pool, per_seq_b))
            if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
                break
        else:
            raise SystemExit(f"sequence_{seq:02d}: không đủ frame cho tập B.")
        chosen_b[seq] = picks
    tune = []
    spare = [s for s in seqs if s not in b_seqs]
    a_seqs = spare + rng.sample(b_seqs, n_tune - len(spare))
    for seq in sorted(a_seqs):
        pool = [i for i in range(counts[seq]) if i not in blocked[seq]
                and all(abs(i - j) >= tune_gap for j in chosen_b.get(seq, []))]
        if not pool:
            raise SystemExit(f"sequence_{seq:02d}: không còn frame cho tập A.")
        tune.append(vf.frame_name(seq, rng.choice(pool)))
    sub = [vf.frame_name(s, i) for s in b_seqs for i in chosen_b[s]]
    return {"A": tune, "B": sub}


# ------------------------------------------------------------- frames I/O

def read_frame_list(arg: str) -> list[str]:
    """``s02f07,s03f15`` or a text file with one frame name per line (``#`` comments)."""
    path = Path(arg)
    if path.exists():
        names = [ln.split()[0] for ln in path.read_text(encoding="utf-8-sig").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        names = [n for n in arg.replace(" ", ",").split(",") if n]
    return [vf.frame_name(*vf.parse_frame_name(n)) for n in names]


def frame_rows(data: Path, names: list[str]) -> list[dict]:
    """Frame dicts compatible with ``vf_cvat.build_bundle`` and ``vf_qa.render_frame``."""
    seqs = vf.list_sequences(data)
    rows = []
    for name in names:
        seq, index = vf.parse_frame_name(name)
        if seq not in seqs or index >= len(seqs[seq]):
            raise SystemExit(f"{name}: không có trong {data}.")
        stamp = seqs[seq][index]
        folder = Path(data) / f"sequence_{seq:02d}"
        cams = [(cam, folder / "cameras" / cam / f"{stamp}.jpg") for cam in vf.CAMERAS]
        rows.append({"name": name, "seq": seq, "index": index, "stamp": stamp,
                     "lidar": folder / "lidar" / f"{stamp}.bin",
                     "cameras": [(c, p) for c, p in cams if p.exists()]})
    return rows


def load_points(row: dict):
    import numpy as np
    return np.fromfile(row["lidar"], dtype=np.float32).reshape(-1, 4)


# --------------------------------------------------------------- box files

def as_vf_box(b: dict) -> dict:
    """Detector box -> plain VF box (class name, metres, radians)."""
    from vf_pp.core import VF_NAMES
    label = VF_NAMES.get(b["label"], b["label"]) if isinstance(b["label"], int) else b["label"]
    out = {k: round(float(b[k]), 4) for k in ("x", "y", "z", "length", "width", "height", "yaw")}
    out["label"] = label
    for k in ("pp_id", "score", "warn", "npts"):
        if k in b:
            out[k] = round(b[k], 4) if isinstance(b[k], float) else b[k]
    return out


def save_boxes(path: Path, boxes_by_frame: dict[str, list[dict]], meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"meta": meta or {}, "frames": {f: [as_vf_box(b) for b in bs] for f, bs in boxes_by_frame.items()}}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def load_reference(arg: str, data: Path | None, names: list[str], *,
                   require_all: bool = False) -> dict[str, list[dict]]:
    """Read reference boxes, optionally requiring an entry for every requested frame."""
    if require_all and not names:
        raise SystemExit("--frames không có frame nào. Kiểm file danh sách frame.")
    path = Path(arg)
    if path.is_file():
        frames = json.loads(path.read_text(encoding="utf-8-sig"))["frames"]
    elif path.is_dir() and data is not None:
        frames = {}
        for row in frame_rows(data, names):
            label = path / f"sequence_{row['seq']:02d}" / f"{row['stamp']}.txt"
            if label.exists():
                frames[row["name"]] = vf.load_gt(label)
    else:
        raise SystemExit(f"--gt {arg}: cần file ref_*.json hoặc thư mục nhãn.")
    if require_all:
        missing = [name for name in names if name not in frames]
        if missing:
            raise SystemExit(f"--gt {arg} thiếu frame: {', '.join(missing)}. "
                             "Kiểm --frames và xuất lại đúng tập A hoặc B từ CVAT.")
    return {name: frames[name] for name in names if name in frames}
