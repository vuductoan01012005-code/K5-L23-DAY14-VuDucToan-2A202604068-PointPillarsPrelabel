"""Run pre-trained PointPillars on VF frames and write pre-label boxes.

Examples::

    # buổi giảng: 6 frame chung
    python scripts/pp_prelabel.py --data <Data-VF> --frames demo/frames_lec.txt --out out/lec.json --plot out/lec_png

    # lab: frame tập A của bạn, cấu hình đã tune, so với ref_A.json bạn tự sửa trên CVAT
    python scripts/pp_prelabel.py --data <Data-VF> --frames d14_<ID>/frames_A.txt --config my_config.json \
        --gt d14_<ID>/ref_A.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="torch.meshgrid")

from scripts import vf_common as vf  # noqa: E402
from scripts import vf_d14  # noqa: E402
from vf_pp.core import VF_NAMES, Config, Detector, correction_cost  # noqa: E402

OPTIONAL_FIELDS = {"thr_car", "thr_cyc", "thr_ped"}
VALID_CHOICES = {
    "intensity_mode": ("clip", "log", "const"),
    "ground_mode": ("fixed", "auto"),
}
_DEFAULT_CFG = Config()


def _cast_override(key: str, value: str):
    """Parse a single ``--set KEY=VALUE`` string using ``Config`` field defaults."""
    if value.lower() in ("none", "null", ""):
        if key not in OPTIONAL_FIELDS:
            allowed = ", ".join(sorted(OPTIONAL_FIELDS))
            raise SystemExit(f"--set {key}={value!r}: chỉ các tham số tùy chọn ({allowed}) mới nhận giá trị none/null.")
        return None
    default_val = getattr(_DEFAULT_CFG, key)
    if isinstance(default_val, str):
        return value
    if isinstance(default_val, int) and not isinstance(default_val, bool):
        try:
            return int(value)
        except ValueError as exc:
            raise SystemExit(f"--set {key}={value!r}: phải là số nguyên ({exc}).")
    if key in OPTIONAL_FIELDS or isinstance(default_val, float):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise SystemExit(f"--set {key}={value!r}: phải là số thực ({exc}).")
        if not math.isfinite(parsed):
            raise SystemExit(f"--set {key}={value!r}: giá trị số không hợp lệ (nan/inf).")
        return parsed
    raise SystemExit(f"--set {key}: không hỗ trợ ghi đè kiểu {type(default_val).__name__}.")


def _validate_config(cfg: Config) -> None:
    """Validate types, enum choices, and numeric bounds on ``cfg``."""
    for key in Config.__dataclass_fields__:
        val = getattr(cfg, key)
        if key == "extra":
            if not isinstance(val, dict):
                raise SystemExit(f"Cấu hình không hợp lệ: extra phải là dict/object JSON, nhận {type(val).__name__}.")
            continue
        default_val = getattr(_DEFAULT_CFG, key)
        if isinstance(default_val, str):
            if not isinstance(val, str):
                raise SystemExit(f"Cấu hình không hợp lệ: {key}={val!r} (phải là chuỗi).")
        elif isinstance(default_val, int) and not isinstance(default_val, bool):
            if isinstance(val, bool) or not isinstance(val, int):
                raise SystemExit(f"Cấu hình không hợp lệ: {key}={val!r} (phải là số nguyên).")
        elif key in OPTIONAL_FIELDS or isinstance(default_val, float):
            if val is None:
                if key not in OPTIONAL_FIELDS:
                    raise SystemExit(f"Cấu hình không hợp lệ: {key} không được là null/None.")
            elif isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise SystemExit(f"Cấu hình không hợp lệ: {key}={val!r} (phải là số thực hợp lệ).")
            elif isinstance(val, int):
                setattr(cfg, key, float(val))

    for key, choices in VALID_CHOICES.items():
        val = getattr(cfg, key)
        if val not in choices:
            raise SystemExit(f"Cấu hình không hợp lệ: {key}={val!r} (chỉ nhận {', '.join(choices)}).")

    bounds = (
        ("passes", cfg.passes >= 1, "phải >= 1"),
        ("voxel_size", cfg.voxel_size > 0, "phải > 0"),
        ("range_m", cfg.range_m > 0, "phải > 0"),
        ("min_points", cfg.min_points >= 0, "phải >= 0"),
    )
    for key, ok, rule in bounds:
        if not ok:
            raise SystemExit(f"Cấu hình không hợp lệ: {key}={getattr(cfg, key)} ({rule}).")


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``KEY=VALUE`` CLI overrides to a copy of ``cfg`` and validate all fields."""
    cfg = replace(cfg)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set {item!r}: thiếu dấu '=' (ví dụ --set score_thr=0.4).")
        key, _, value = item.partition("=")
        if key not in Config.__dataclass_fields__:
            raise SystemExit(f"--set {key}: không có tham số này. Xem vf_pp/core.py:Config.")
        if key == "extra":
            raise SystemExit("--set extra: không thể ghi đè 'extra' qua --set (chỉ cấu hình qua JSON).")
        setattr(cfg, key, _cast_override(key, value))
    _validate_config(cfg)
    return cfg


def load_config(path: str | None, overrides: list[str]) -> Config:
    if path:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"--config {path!r}: không đọc được file JSON ({exc}).")
        if not isinstance(data, dict):
            raise SystemExit(f"--config {path!r}: file cấu hình phải là JSON object.")
        cfg = Config.from_dict(data)
    else:
        cfg = Config()
    return apply_overrides(cfg, overrides)


def main(argv=None) -> int:
    vf.configure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="thư mục Data-VF (đã tải ở Day 13)")
    ap.add_argument("--frames", required=True, help="file danh sách frame hoặc s02f07,s03f15")
    ap.add_argument("--config", help="file JSON cấu hình (mặc định: cấu hình gốc của model)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="ghi đè 1 tham số, ví dụ --set score_thr=0.3 (lặp được)")
    ap.add_argument("--out", help="ghi box ra JSON (đưa vào scripts/vf_cvat.py prelabel)")
    ap.add_argument("--gt", help="ref_*.json hoặc thư mục nhãn để tính chi phí sửa")
    ap.add_argument("--plot", help="thư mục ghi ảnh 7 camera + BEV cho từng frame")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    data = Path(args.data)
    names = vf_d14.read_frame_list(args.frames)
    rows = vf_d14.frame_rows(data, names)
    gt = vf_d14.load_reference(args.gt, data, names, require_all=True) if args.gt else {}
    det = Detector(cfg)
    results, total = {}, Counter()
    print(f"{'frame':8} {'box':>4}  lớp")
    for row in rows:
        start = time.time()
        boxes = det.detect(vf_d14.load_points(row))
        results[row["name"]] = boxes
        classes = Counter(VF_NAMES[b["label"]] for b in boxes)
        warns = sum(1 for b in boxes if b["warn"])
        line = f"{row['name']:8} {len(boxes):4}  {dict(classes)}  cảnh báo kích thước={warns}  {time.time() - start:.1f}s"
        if row["name"] in gt:
            c = correction_cost(gt[row["name"]], boxes)
            total.update({k: v for k, v in c.items()})
            line += f"  | xóa {c['delete']} thêm {c['add']} sửa {c['edit']} → chi phí {c['cost']}"
        print(line)
    if total:
        cost = round(0.3 * total["delete"] + total["add"] + 0.5 * total["edit"], 2)
        print(f"TỔNG ({len(gt)} frame có nhãn): GT {total['gt']}, dự đoán {total['pred']}, khớp {total['match']}, "
              f"xóa {total['delete']}, thêm {total['add']}, sửa {total['edit']} → chi phí {cost}")
    if args.out:
        vf_d14.save_boxes(Path(args.out), results, {"config": cfg.to_dict(), "frames": names})
        print(f"Đã ghi {args.out}")
    if args.plot:
        from scripts import vf_qa
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        calibs = {}
        for row in rows:
            if row["seq"] not in calibs:
                calibs[row["seq"]] = vf_qa.load_calib(data / f"sequence_{row['seq']:02d}")
            boxes = [dict(vf_d14.as_vf_box(b), idx=b["pp_id"]) for b in results[row["name"]]]
            sheet, _ = vf_qa.render_frame(row, boxes, calibs[row["seq"]])
            sheet.save(out / f"{row['name']}.jpg", quality=85)
        print(f"Đã vẽ {len(rows)} ảnh vào {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
