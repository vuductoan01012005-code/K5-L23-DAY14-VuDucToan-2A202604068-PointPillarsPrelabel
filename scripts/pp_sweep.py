"""Grid-search the pre-label config on your tuning frames (set A) against ref_A.json.

    python scripts/pp_sweep.py --data <Data-VF> --frames d14_<ID>/frames_A.txt --gt d14_<ID>/ref_A.json \
        --grid score_thr=0.1,0.2,0.3,0.4 --grid min_points=0,5,10 --grid range_m=40,50 --out d14_<ID>/sweep.csv

Model inference reruns only when a group-A/B input knob changes; group-C knobs
reuse cached raw boxes, so a grid over C is fast. Best config -> best_config.json.
Leave-one-out: for each frame, the config picked on the other frames is scored
on the held-out one, so you can see whether 3 frames are enough to trust the pick.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="torch.meshgrid")

from scripts import vf_common as vf  # noqa: E402
from scripts import vf_d14  # noqa: E402
from scripts.pp_prelabel import apply_overrides, load_config  # noqa: E402
from vf_pp.core import Config, Detector, correction_cost, postprocess  # noqa: E402

MODEL_KNOBS = {"voxel_size", "intensity_mode", "intensity_scale", "ground_mode", "delta_z", "passes",
               "nms_thr", "nms_pre", "max_num"}


def parse_grid(items):
    grid = {}
    for item in items:
        key, _, vals = item.partition("=")
        if key not in Config.__dataclass_fields__:
            raise SystemExit(f"--grid {key}: không có tham số này.")
        grid[key] = vals.split(",")
    return grid


def main(argv=None) -> int:
    vf.configure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gt", required=True, help="ref_A.json (box bạn đã sửa trên CVAT)")
    ap.add_argument("--config", help="cấu hình xuất phát")
    ap.add_argument("--grid", action="append", default=[], metavar="KEY=v1,v2,...")
    ap.add_argument("--out", default="sweep.csv")
    ap.add_argument("--best", default=None, help="mặc định: cạnh --out, tên best_config.json")
    args = ap.parse_args(argv)

    base = load_config(args.config, [])
    grid = parse_grid(args.grid) or {"score_thr": ["0.1", "0.2", "0.3", "0.4", "0.5"]}
    names = vf_d14.read_frame_list(args.frames)
    rows = vf_d14.frame_rows(Path(args.data), names)
    gt = vf_d14.load_reference(args.gt, Path(args.data), names, require_all=True)
    points = {r["name"]: vf_d14.load_points(r) for r in rows}
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"{len(combos)} cấu hình × {len(points)} frame")
    det, raw_cache, results = None, {}, []
    for combo in combos:
        cfg = load_config_from(base, dict(zip(keys, combo)))
        model_key = tuple((k, getattr(cfg, k)) for k in sorted(MODEL_KNOBS))
        # the model's internal score filter must stay below every threshold tried
        run_cfg = replace(cfg, score_thr=0.05, thr_car=None, thr_cyc=None, thr_ped=None)
        if model_key not in raw_cache:
            if det is None:
                det = Detector(run_cfg)
            else:
                det.set_config(run_cfg)
            raw_cache[model_key] = {n: det.raw_detect(p) for n, p in points.items()}
        per_frame = {n: correction_cost(gt[n], postprocess(raw_cache[model_key][n], points[n], cfg))
                     for n in points}
        total = {k: sum(c[k] for c in per_frame.values()) for k in ("delete", "add", "edit")}
        cost = round(0.3 * total["delete"] + total["add"] + 0.5 * total["edit"], 2)
        results.append({"combo": dict(zip(keys, combo)), "cfg": cfg, "cost": cost, **total,
                        "per_frame": {n: c["cost"] for n, c in per_frame.items()}})
        print(f"  {dict(zip(keys, combo))}  → chi phí {cost}")
    results.sort(key=lambda r: r["cost"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["cost", "delete", "add", "edit"] + list(points))
        for r in results:
            w.writerow([r["combo"][k] for k in keys] + [r["cost"], r["delete"], r["add"], r["edit"]]
                       + [r["per_frame"][n] for n in points])
    print(f"\nTop {min(10, len(results))} (ghi đủ ở {out}):")
    for r in results[:10]:
        print(f"  {r['cost']:7.2f}  xóa {r['delete']:3} thêm {r['add']:3} sửa {r['edit']:3}  {r['combo']}")
    best = Path(args.best or out.with_name("best_config.json"))
    best.write_text(json.dumps(results[0]["cfg"].to_dict(), indent=1), encoding="utf-8")
    print(f"Cấu hình tốt nhất → {best}")
    if len(points) > 1:
        print("\nLeave-one-out (chọn trên các frame còn lại, chấm trên frame bỏ ra):")
        for held in points:
            pick = min(results, key=lambda r: sum(v for n, v in r["per_frame"].items() if n != held))
            oracle = min(r["per_frame"][held] for r in results)
            print(f"  {held}: cấu hình chọn {pick['combo']} → {pick['per_frame'][held]} (tốt nhất có thể {oracle})")
    return 0


def load_config_from(cfg: Config, values: dict) -> Config:
    """Apply parameter overrides from ``values`` to a copy of ``cfg`` without temp files."""
    return apply_overrides(replace(cfg), [f"{k}={v}" for k, v in values.items()])


if __name__ == "__main__":
    raise SystemExit(main())
