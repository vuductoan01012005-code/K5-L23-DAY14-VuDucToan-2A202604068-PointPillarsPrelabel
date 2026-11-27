"""Tính metric thao tác sửa nhãn từ file log của cvat-telemetry-addon.user.js.

    python bonus/telemetry_metrics.py d14-telemetry-job*.json [--csv out.csv]

Mỗi file là một phiên (một job). In số click trên canvas, số phím tắt, trung vị
khoảng thời gian giữa hai click, khoảng cách chuột trung bình, tỉ lệ dùng bàn
phím và độ dài phiên. Đây là mô tả cách làm việc của một người, không phải thời
gian công đã được kiểm chứng.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def _ts(event: dict) -> datetime:
    return datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))


def session_metrics(events: list[dict]) -> dict:
    """Gom metric cho một phiên từ danh sách event của userscript."""
    clicks = [e for e in events if e.get("obj_name") == "canvas-click"]
    keys = [e for e in events if e.get("obj_name") == "keydown"]
    click_payload = [json.loads(e.get("payload") or "{}") for e in clicks]
    gaps = [p["dt_ms"] for p in click_payload if p.get("dt_ms") is not None]
    dists = [p["dist_px"] for p in click_payload if p.get("dist_px") is not None]
    span = (_ts(events[-1]) - _ts(events[0])).total_seconds() if len(events) > 1 else 0.0
    n_click, n_key = len(clicks), len(keys)
    return {
        "job_id": events[0].get("job_id") if events else None,
        "canvas_clicks": n_click,
        "shortcut_keys": n_key,
        "median_click_gap_ms": round(statistics.median(gaps)) if gaps else None,
        "mean_click_dist_px": round(statistics.fmean(dists)) if dists else None,
        "keyboard_ratio": round(n_key / max(n_click + n_key, 1), 2),
        "active_span_s": round(span),
        "top_keys": " ".join(f"{k}×{c}" for k, c in Counter(e.get("obj_val") for e in keys).most_common(5)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, help="ghi bảng metric ra CSV")
    args = ap.parse_args(argv)

    rows = []
    for path in args.logs:
        events = sorted(json.loads(path.read_text(encoding="utf-8")), key=_ts)
        if not events:
            print(f"{path}: rỗng, bỏ qua", file=sys.stderr)
            continue
        rows.append({"file": path.name, **session_metrics(events)})

    for row in rows:
        print("\n".join(f"{k:>20}: {v}" for k, v in row.items()), end="\n\n")
    if args.csv and rows:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
