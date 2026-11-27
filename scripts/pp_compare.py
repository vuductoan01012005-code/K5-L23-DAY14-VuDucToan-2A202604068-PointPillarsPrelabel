"""Before / after: default config vs your tuned config, scored against your reference labels.

    python scripts/pp_compare.py --data <Data-VF> --frames d14_<ID>/frames_A.txt --gt d14_<ID>/ref_A.json \
        --tuned d14_<ID>/best_config.json --out d14_<ID>/compare

For each frame it writes one image: BEV of the default config on the left, your
config on the right. Colours show what an annotator must do with each box:

* green  = correct, keep
* yellow = right object, needs an edit (class, centre, heading or size off)
* red    = extra box, delete
* blue   = object in your reference that the model missed, add by hand

It also prints how much correction cost your config saves and which knobs differ.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="torch.meshgrid")

from scripts import vf_common as vf  # noqa: E402
from scripts import vf_d14  # noqa: E402
from scripts.pp_prelabel import load_config  # noqa: E402
from vf_pp.core import Config, Detector, correction_cost, needs_edit, within_range  # noqa: E402

GREEN, YELLOW, RED, BLUE = (0, 230, 118), (255, 234, 0), (255, 23, 68), (41, 121, 255)


def classify(gt: list[dict], pred: list[dict]) -> list[tuple[dict, tuple]]:
    """Every box to draw with its colour: matched predictions, extra predictions, missed references."""
    gt, pred = within_range(gt), within_range(pred)  # same 50 m scope as correction_cost
    pairs = vf.match_boxes(gt, pred, 0.3)
    out, hit_g, hit_p = [], set(), set()
    for gi, pj, _ in pairs:
        hit_g.add(gi)
        hit_p.add(pj)
        out.append((pred[pj], YELLOW if needs_edit(gt[gi], pred[pj]) else GREEN))
    out += [(p, RED) for j, p in enumerate(pred) if j not in hit_p]
    out += [(g, BLUE) for i, g in enumerate(gt) if i not in hit_g]
    return out


def bev(points, boxes, title, size=640, metres=52.0):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (12, 12, 16))
    px = img.load()
    scale, c = size / (2 * metres), size / 2
    for x, y, _z in points:
        u, v = int(c - x * scale), int(c + y * scale)
        if 0 <= u < size and 0 <= v < size:
            px[u, v] = (110, 110, 120)
    draw = ImageDraw.Draw(img)
    for r in (10, 20, 30, 40, 50):
        outline = (120, 45, 45) if r == 50 else (50, 50, 60)
        draw.ellipse([c - r * scale, c - r * scale, c + r * scale, c + r * scale], outline=outline)
    draw.polygon([(c, c - 6), (c - 4, c + 4), (c + 4, c + 4)], fill=(255, 255, 255))
    for box, color in boxes:
        ring = [(c - x * scale, c + y * scale) for x, y in vf.rectangle_corners(box)]
        draw.polygon(ring, outline=color, width=3 if color == BLUE else 2)
    draw.text((8, 6), title, fill=(240, 240, 240))
    return img


def changed_knobs(base: Config, tuned: Config) -> list[str]:
    a, b = base.to_dict(), tuned.to_dict()
    return [f"{k}: {a[k]} -> {b[k]}" for k in a if k != "extra" and a[k] != b[k]]


def main(argv=None) -> int:
    vf.configure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gt", required=True, help="ref_A.json bạn đã sửa trên CVAT")
    ap.add_argument("--tuned", required=True, help="cấu hình của bạn, ví dụ best_config.json")
    ap.add_argument("--base", help="cấu hình để so (mặc định: cấu hình gốc của model)")
    ap.add_argument("--out", required=True, help="thư mục ghi ảnh so sánh")
    args = ap.parse_args(argv)

    from PIL import Image, ImageDraw
    from scripts import vf_qa

    data, out = Path(args.data), Path(args.out)
    names = vf_d14.read_frame_list(args.frames)
    gt = vf_d14.load_reference(args.gt, data, names, require_all=True)
    out.mkdir(parents=True, exist_ok=True)
    configs = {"gốc": load_config(args.base, []), "của bạn": load_config(args.tuned, [])}
    print("Tham số đã đổi:", "; ".join(changed_knobs(*configs.values())) or "(không có)")

    det = Detector(configs["gốc"])
    preds: dict[str, dict[str, list[dict]]] = {}
    for label, cfg in configs.items():
        det.set_config(cfg)
        preds[label] = {r["name"]: [vf_d14.as_vf_box(b) for b in det.detect(vf_d14.load_points(r))]
                        for r in vf_d14.frame_rows(data, names)}

    totals = {label: 0.0 for label in configs}
    print(f"\n{'frame':8} {'chi phí gốc':>12} {'của bạn':>9}")
    for row in vf_d14.frame_rows(data, names):
        name = row["name"]
        points = vf_qa.points_xyz(row["lidar"])
        tiles, costs = [], []
        for label in configs:
            c = correction_cost(gt[name], preds[label][name])
            totals[label] += c["cost"]
            costs.append(c["cost"])
            title = f"{'goc' if label == 'gốc' else 'cua ban'}: xoa {c['delete']} them {c['add']} sua {c['edit']} -> {c['cost']}"
            tiles.append(bev(points, classify(gt[name], preds[label][name]), title))
        sheet = Image.new("RGB", (tiles[0].width * 2, tiles[0].height + 26), (0, 0, 0))
        for n, t in enumerate(tiles):
            sheet.paste(t, (n * t.width, 26))
        ImageDraw.Draw(sheet).text(
            (8, 6), f"{name}   xanh la=dung  vang=can sua  do=thua (xoa)  xanh duong=bi sot (them)", fill=(255, 255, 255))
        sheet.save(out / f"{name}.jpg", quality=88)
        print(f"{name:8} {costs[0]:12.1f} {costs[1]:9.1f}")

    base, mine = totals["gốc"], totals["của bạn"]
    saved = (base - mine) / base * 100 if base else 0.0
    print(f"{'TỔNG':8} {base:12.1f} {mine:9.1f}")
    print(f"\nCấu hình của bạn giảm {saved:.0f}% công sửa so với cấu hình gốc trên {len(gt)} frame." if saved >= 0
          else f"\nCấu hình của bạn TĂNG {-saved:.0f}% công sửa so với cấu hình gốc. Xem lại tham số.")
    print(f"Ảnh so sánh: {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
