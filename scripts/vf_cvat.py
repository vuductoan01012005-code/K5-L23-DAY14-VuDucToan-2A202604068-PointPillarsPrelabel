"""Learner tool for CVAT 3D: Day 13 task + table A, Day 14 pre-label tasks.

Commands::

    # Day 14 (SEED do coach công bố lúc vào lab)
    python scripts/vf_cvat.py day14 --student <ID> --seed <SEED> --data <Data-VF>          # ghi d14_<ID>/frames_A.txt, frames_B.txt
    python scripts/vf_cvat.py prelabel --student <ID> --task A --data <Data-VF> --boxes d14_<ID>/pred_A.json --host <URL> --user <name> [--org SLUG]
    python scripts/vf_cvat.py reference --student <ID> --task A --host <URL> --user <name>  # đọc box đã sửa -> d14_<ID>/ref_A.json

    python scripts/vf_cvat.py day13 --student <ID> --data <data-VF> --host <URL> --user <name> [--org SLUG] [--dry-run] [--replace]
    python scripts/vf_cvat.py report d13 --student <ID> --host <URL> --user <name> [--org SLUG] --out ./nop
    python scripts/vf_cvat.py preview d13 --student <ID> --data <data-VF> --table nop/D13-<ID>-bangA.csv --out ./xem
    python scripts/vf_cvat.py check d13 --student <ID> --data <data-VF> --table nop/D13-<ID>-bangA.csv

The CVAT password is read from CVAT_PASSWORD or typed at a prompt. It is never
written to disk.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import vf_common as vf  # noqa: E402

TABLE_A_FIELDS = [
    "student", "frame", "idx", "label", "x", "y", "z", "length", "width", "height",
    "yaw_rad", "yaw_deg", "z_bottom", "dist_m", "az_deg", "dist_bin", "occluded",
]
PRELABEL_STATE = Path("out") / "cvat-prelabel"


# ------------------------------------------------------------ bundle

def frame_sources(data: Path, student: str) -> list[dict]:
    """Return the student's 10 frames with their LiDAR file and camera images."""
    seqs = vf.list_sequences(data)
    names = vf.select_frames(student, {seq: len(stamps) for seq, stamps in seqs.items()})
    rows = []
    for name in names:
        seq, index = vf.parse_frame_name(name)
        stamp = seqs[seq][index]
        folder = Path(data) / f"sequence_{seq:02d}"
        cams = [(cam, folder / "cameras" / cam / f"{stamp}.jpg") for cam in vf.CAMERAS]
        rows.append({
            "name": name, "seq": seq, "index": index, "stamp": stamp,
            "lidar": folder / "lidar" / f"{stamp}.bin",
            "cameras": [(cam, path) for cam, path in cams if path.exists()],
        })
    return rows


def build_bundle(frames: list[dict], out_zip: Path) -> Path:
    """Write the CVAT 3D zip: ``pointcloud/<f>.pcd`` and ``related_images/<f>_pcd/camXX.jpg``."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as bundle:
        for frame in frames:
            bundle.writestr(f"pointcloud/{frame['name']}.pcd", vf.bin_to_pcd(frame["lidar"]))
            for cam, path in frame["cameras"]:
                bundle.write(path, f"related_images/{frame['name']}_pcd/{cam}.jpg")
    return out_zip


# -------------------------------------------------------------- CVAT

def connect(host: str, user: str, org: str | None):
    """Log in with the SDK; the password comes from CVAT_PASSWORD or a prompt."""
    try:
        from cvat_sdk import make_client
    except ImportError:
        raise SystemExit("Thiếu cvat-sdk. Chạy: pip install -r requirements.txt")
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass(f"Mật khẩu CVAT cho {user}: ")
    client = make_client(host, credentials=(user, password))
    if org:
        client.organization_slug = org
    return client


def find_tasks(client, name: str) -> list:
    """Tasks whose name is exactly ``name`` in the current workspace."""
    try:
        tasks = client.tasks.list()
    except Exception as exc:
        if getattr(exc, "status", None) == 404 and "organization" in f"{exc} {getattr(exc, 'body', '')}".lower():
            org = getattr(client, "organization_slug", "") or ""
            raise SystemExit(
                f"Không thấy tổ chức CVAT '{org}'. Nếu chạy CVAT local (Docker), hãy bỏ cờ --org."
            ) from None
        raise
    return [task for task in tasks if task.name == name]


def create_task(client, name: str, bundle: Path):
    """Create a 3D task with one 10-frame job and the five Day 13 cuboid labels."""
    from cvat_sdk.core.proxies.tasks import ResourceType

    spec = {
        "name": name,
        "labels": [
            {"name": label, "type": "cuboid", "color": vf.LABEL_COLORS[label]}
            for label in vf.CLASS_IDS
        ],
        "segment_size": 10,
    }
    return client.tasks.create_from_data(
        spec=spec,
        resource_type=ResourceType.LOCAL,
        resources=[str(bundle)],
        data_params={"image_quality": 70, "sorting_method": "lexicographical"},
    )


def warn_if_tracks(task_name: str, ann) -> int:
    """Warn on stderr if CVAT annotations contain 3D cuboid Tracks (which are ignored)."""
    tracks = getattr(ann, "tracks", None) or []
    if tracks:
        print(
            f"Cảnh báo: {task_name} có {len(tracks)} cuboid dạng Track nên sẽ bị bỏ qua "
            "(chỉ đọc Shape; hãy vẽ lại bằng Shape).",
            file=sys.stderr,
        )
    return len(tracks)


def task_boxes(client, task, scale_order: str = "lwh") -> list[dict]:
    """Read every cuboid of a task as table-A rows (without the student column)."""
    labels = {label.id: label.name for label in task.get_labels()}
    frame_names = [frame.name for frame in task.get_meta().frames]
    ann = task.get_annotations()
    warn_if_tracks(getattr(task, "name", None) or f"#{getattr(task, 'id', '?')}", ann)
    rows = []
    shapes = sorted(ann.shapes, key=lambda s: (s.frame, s.id or 0))
    counters: dict[str, int] = {}
    for shape in shapes:
        if str(shape.type) != "cuboid" and getattr(shape.type, "value", None) != "cuboid":
            continue
        frame = vf.frame_name(*vf.parse_frame_name(frame_names[shape.frame]))
        box = vf.cuboid_from_cvat(shape.points, labels.get(shape.label_id, "?"), scale_order, shape.occluded)
        counters[frame] = counters.get(frame, 0) + 1
        rows.append(table_a_row(frame, counters[frame], box))
    return rows


def table_a_row(frame: str, idx: int, box: dict) -> dict:
    distance = vf.dist_m(box)
    return {
        "frame": frame, "idx": idx, "label": box["label"],
        "x": f"{box['x']:.3f}", "y": f"{box['y']:.3f}", "z": f"{box['z']:.3f}",
        "length": f"{box['length']:.3f}", "width": f"{box['width']:.3f}", "height": f"{box['height']:.3f}",
        "yaw_rad": f"{box['yaw']:.4f}", "yaw_deg": f"{math.degrees(box['yaw']):.1f}",
        "z_bottom": f"{box['z'] - box['height'] / 2:.3f}",
        "dist_m": f"{distance:.2f}", "az_deg": f"{vf.azimuth_deg(box):.1f}", "dist_bin": vf.dist_bin(distance),
        "occluded": int(box.get("occluded", False)),
    }


def write_table_a(path: Path, student: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_A_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"student": student, **row})


# ---------------------------------------------------------- commands

def cmd_day13(args) -> int:
    student = vf.normalize_student(args.student)
    name = vf.task_name(student)
    frames = frame_sources(Path(args.data), student)
    print(f"Task: {name}")
    print("10 frame của bạn (chỉ số đếm từ 0):")
    for frame in frames:
        print(f"  {frame['name']}  sequence_{frame['seq']:02d}  ts={frame['stamp']}  camera={len(frame['cameras'])}")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(args.bundle_out) if args.bundle_out else Path(tmp) / f"{name}.zip"
        start = time.time()
        build_bundle(frames, bundle)
        size_mb = bundle.stat().st_size / 1e6
        print(f"Đã đóng gói {size_mb:.1f} MB trong {time.time() - start:.1f} s.")
        if args.dry_run:
            print("--dry-run: không kết nối CVAT.")
            return 0
        if not (args.host and args.user):
            raise SystemExit("Cần --host và --user (hoặc dùng --dry-run).")
        client = connect(args.host, args.user, args.org)
        existing = find_tasks(client, name)
        if existing and not args.replace:
            print(f"Đã có task {name}: {args.host.rstrip('/')}/tasks/{existing[0].id}")
            print("Mở task cũ. Chỉ dùng --replace khi chưa vẽ gì: --replace xóa task và nhãn cũ.")
            return 0
        for task in existing:
            task.remove()
            print(f"Đã xóa task cũ #{task.id}.")
        start = time.time()
        task = create_task(client, name, bundle)
        print(f"Tạo xong trong {time.time() - start:.0f} s: {args.host.rstrip('/')}/tasks/{task.id}")
    return 0


def cmd_report(args) -> int:
    student = vf.normalize_student(args.student)
    name = vf.task_name(student)
    client = connect(args.host, args.user, args.org)
    tasks = find_tasks(client, name)
    if not tasks:
        raise SystemExit(f"Không thấy task {name}. Kiểm --host, --user, --org.")
    rows = task_boxes(client, tasks[0], args.scale_order)
    out = Path(args.out) / f"{name}-bangA.csv"
    write_table_a(out, student, rows)
    per_frame: dict[str, int] = {}
    for row in rows:
        per_frame[row["frame"]] = per_frame.get(row["frame"], 0) + 1
    print(f"Đã ghi {out} ({len(rows)} cuboid).")
    for frame, count in sorted(per_frame.items()):
        print(f"  {frame}: {count}")
    if not rows:
        print("Bảng A trống: bấm Save trong CVAT rồi chạy lại.")
    return 0


def _qa_inputs(args):
    from scripts import vf_qa

    student = vf.normalize_student(args.student)
    table = Path(args.table or f"nop/{vf.task_name(student)}-bangA.csv")
    frames = frame_sources(Path(args.data), student)
    return vf_qa, student, vf_qa.sorted_frames(frames, vf_qa.read_table_a(table, student))


def cmd_preview(args) -> int:
    """Draw your cuboids on the 7 camera images (calib.json only, no ground truth)."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("Thiếu Pillow. Chạy: pip install -r requirements.txt")
    vf_qa, student, pairs = _qa_inputs(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calibs: dict[int, dict] = {}
    for frame, boxes in pairs:
        if frame["seq"] not in calibs:
            calibs[frame["seq"]] = vf_qa.load_calib(Path(args.data) / f"sequence_{frame['seq']:02d}")
        sheet, seen = vf_qa.render_frame(frame, boxes, calibs[frame["seq"]])
        path = out / f"{vf.task_name(student)}-{frame['name']}.jpg"
        sheet.save(path, quality=85)
        unseen = len(boxes) - len(seen)
        note = f", {unseen} hộp không camera nào thấy" if unseen else ""
        print(f"{path}  ({len(boxes)} cuboid{note})")
    print("Mở ảnh, tìm hộp lệch khỏi vật, ngược đầu, hoặc đặt nhầm vật. Sửa trong CVAT rồi xuất lại bảng A.")
    return 0


def cmd_check(args) -> int:
    """Self-check each cuboid against its own LiDAR points. No ground truth, no score."""
    vf_qa, student, pairs = _qa_inputs(args)
    rows = []
    for frame, boxes in pairs:
        if not boxes:
            print(f"{frame['name']}: chưa có cuboid nào.")
            continue
        for row in vf_qa.check_frame(boxes, vf_qa.points_xyz(frame["lidar"])):
            rows.append({"frame": frame["name"], **row})
    out = Path(args.out or f"nop/{vf.task_name(student)}-tukiem.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=vf_qa.CHECK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    flagged = [r for r in rows if r["flags"]]
    print(f"Đã ghi {out}: {len(rows)} cuboid, {len(flagged)} có cảnh báo.")
    for row in flagged:
        print(f"  {row['frame']} #{row['idx']} {row['label']}: {row['flags']}")
    print("Cảnh báo là gợi ý để xem lại, không phải lỗi chắc chắn và không tính điểm.")
    return 0


# ------------------------------------------------------------- Day 14

PRELABEL_ATTRS = ("pp_id", "score", "warn", "note")


def d14_dir(student: str) -> Path:
    return Path(f"d14_{vf.normalize_student(student)}")


def d14_task_name(student: str, part: str) -> str:
    return f"D14-{vf.normalize_student(student)}-{part}"


def cuboid_points(box: dict) -> list[float]:
    """VF box -> CVAT cuboid points ``[x,y,z, rx,ry,rz, l,w,h]`` (inverse of ``cuboid_from_cvat``).

    The CVAT API rejects a cuboid unless it has exactly 16 values; the 3D editor
    stores the 9 above followed by 7 zeros, so the list is padded the same way.
    """
    head = [float(box["x"]), float(box["y"]), float(box["z"]), 0.0, 0.0, float(box["yaw"]),
            float(box["length"]), float(box["width"]), float(box["height"])]
    return head + [0.0] * 7


def cmd_day14(args) -> int:
    from scripts import vf_d14

    student = vf.normalize_student(args.student)
    seed = vf_d14.get_seed(args.seed)
    counts = {s: len(st) for s, st in vf.list_sequences(Path(args.data)).items()}
    expected = dict.fromkeys(range(1, 6), 60)
    if counts != expected:
        details = [
            f"sequence_{seq:02d}: {counts.get(seq, 0)}/{expected.get(seq, 0)} frame"
            for seq in sorted(counts.keys() | expected.keys())
            if counts.get(seq, 0) != expected.get(seq, 0)
        ]
        raise SystemExit(f"Data-VF phải đủ 5 sequence, mỗi sequence 60 frame. "
                         f"Kiểm đường dẫn và giải nén lại: {', '.join(details)}.")
    sets = vf_d14.select_d14(student, seed, counts)
    out = d14_dir(student)
    out.mkdir(exist_ok=True)
    tag = vf_d14.seed_tag(seed)
    for part, names in sets.items():
        path = out / f"frames_{part}.txt"
        path.write_text(f"# {d14_task_name(student, part)} seed_tag={tag}\n" + "\n".join(names) + "\n", encoding="utf-8")
        print(f"{path}: {' '.join(names)}")
    print(f"seed_tag={tag} (coach dùng để đối chiếu SEED)")
    return 0


def _attr_specs():
    return [{"name": a, "mutable": True, "input_type": "text", "default_value": "", "values": [""]}
            for a in PRELABEL_ATTRS]


def prelabel_state_path(host: str, org: str | None, task_id: int, name: str) -> Path:
    """Keep the prediction fingerprint separate for each CVAT task."""
    identity = json.dumps([host.rstrip("/"), org or "", task_id, name])
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return PRELABEL_STATE / f"{key}.sha256"


def cmd_prelabel(args) -> int:
    from scripts import vf_d14

    student = vf.normalize_student(args.student)
    name = d14_task_name(student, args.task)
    frames_file = Path(args.frames or d14_dir(student) / f"frames_{args.task}.txt")
    names = vf_d14.read_frame_list(str(frames_file))
    rows = vf_d14.frame_rows(Path(args.data), names)
    prediction = json.loads(Path(args.boxes).read_text(encoding="utf-8-sig")) if args.boxes else None
    boxes = prediction["frames"] if prediction is not None else {}
    missing = [n for n in names if n not in boxes]
    if args.boxes and missing:
        raise SystemExit(f"{args.boxes} thiếu frame {missing}. Chạy scripts/pp_prelabel.py với --frames {frames_file}.")
    fingerprint = hashlib.sha256(json.dumps(prediction, sort_keys=True).encode("utf-8")).hexdigest()
    box_count = sum(len(boxes.get(n, [])) for n in names)
    print(f"Task: {name}  ({len(rows)} frame, {box_count} box pre-label)")
    client = None
    existing = []
    if not args.dry_run:
        if not (args.host and args.user):
            raise SystemExit("Cần --host và --user (hoặc dùng --dry-run).")
        client = connect(args.host, args.user, args.org)
        existing = find_tasks(client, name)
        if existing and not args.replace:
            if len(existing) > 1:
                ids = ", ".join(str(task.id) for task in existing)
                raise SystemExit(f"Có nhiều task tên {name} (ID: {ids}). Kiểm trên CVAT trước khi chạy lại.")
            task = existing[0]
            link = f"{args.host.rstrip('/')}/tasks/{task.id}"
            if not task.get_jobs():
                raise SystemExit(f"Task {name} chưa có job: {link}. Chờ CVAT xử lý rồi chạy lại.")
            actual = [Path(frame.name).stem for frame in task.get_meta().frames]
            if actual != sorted(names):
                raise SystemExit(f"Task {name} không khớp bộ frame được giao: {link}. "
                                 "Kiểm SEED và file frames. Báo coach trước khi tạo lại task đã sửa.")
            if box_count and not task.get_annotations().shapes:
                raise SystemExit(f"Task {name} chưa có nhãn dạng Shape: {link}. "
                                 "Mở task để kiểm. Nếu đã xóa hộp có chủ đích, dùng reference để xuất.")
            if args.boxes:
                state = prelabel_state_path(args.host, args.org, task.id, name)
                try:
                    original = state.read_text(encoding="ascii").strip()
                except FileNotFoundError:
                    print(f"Cảnh báo: chưa thể đối chiếu file dự đoán với task {name}. "
                          "Hãy kiểm tra và dùng lại file đã nạp lúc tạo task.", file=sys.stderr)
                else:
                    if original != fingerprint:
                        raise SystemExit(f"Task {name}: file dự đoán {args.boxes} đã đổi từ lúc tạo task. "
                                         "CVAT vẫn giữ nhãn cũ. Dùng lại file đã nạp hoặc trao đổi với coach trước khi tạo lại task.")
            print(f"Đã có task {name}: {link}. Mở task để tiếp tục sửa nhãn.")
            print("Chỉ dùng --replace khi muốn xóa task và nhãn cũ để tạo lại.")
            return 0
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(args.bundle_out) if args.bundle_out else Path(tmp) / f"{name}.zip"
        build_bundle(rows, bundle)
        print(f"Đã đóng gói {bundle.stat().st_size / 1e6:.1f} MB.")
        if args.dry_run:
            print("--dry-run: không kết nối CVAT.")
            return 0
        for task in existing:
            task.remove()
        from cvat_sdk.core.proxies.tasks import ResourceType
        spec = {"name": name, "segment_size": len(rows),
                "labels": [{"name": label, "type": "cuboid", "color": vf.LABEL_COLORS[label],
                            "attributes": _attr_specs()} for label in vf.CLASS_IDS]}
        task = client.tasks.create_from_data(spec=spec, resource_type=ResourceType.LOCAL, resources=[str(bundle)],
                                             data_params={"image_quality": 70, "sorting_method": "lexicographical"})
        upload_prelabels(task, boxes)
        state = prelabel_state_path(args.host, args.org, task.id, name)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(fingerprint, encoding="ascii")
        print(f"Tạo xong: {args.host.rstrip('/')}/tasks/{task.id}")
    return 0


def upload_prelabels(task, boxes: dict[str, list[dict]]) -> int:
    from cvat_sdk import models

    labels = {lab.name: lab for lab in task.get_labels()}
    frame_idx = {vf.frame_name(*vf.parse_frame_name(f.name)): i for i, f in enumerate(task.get_meta().frames)}
    shapes = []
    for frame, items in boxes.items():
        if frame not in frame_idx:
            continue
        for b in items:
            lab = labels[b["label"]]
            spec_ids = {a.name: a.id for a in lab.attributes}
            vals = {"pp_id": str(b.get("pp_id", "")), "score": f"{b.get('score', 0):.2f}",
                    "warn": str(b.get("warn", "")), "note": ""}
            shapes.append(models.LabeledShapeRequest(
                type="cuboid", frame=frame_idx[frame], label_id=lab.id, points=cuboid_points(b),
                attributes=[models.AttributeValRequest(spec_id=spec_ids[k], value=v) for k, v in vals.items()
                            if k in spec_ids]))
    task.update_annotations(models.PatchedLabeledDataRequest(shapes=shapes))
    print(f"Đã đẩy {len(shapes)} cuboid pre-label.")
    return len(shapes)


def cmd_reference(args) -> int:
    student = vf.normalize_student(args.student)
    name = d14_task_name(student, args.task)
    client = connect(args.host, args.user, args.org)
    tasks = find_tasks(client, name)
    if not tasks:
        raise SystemExit(f"Không thấy task {name}.")
    if len(tasks) > 1:
        ids = ", ".join(str(task.id) for task in tasks)
        raise SystemExit(f"Có nhiều task tên {name} (ID: {ids}). "
                         "Kiểm ID trên CVAT và đổi tên các task còn lại trước khi xuất.")
    task = tasks[0]
    ann = task.get_annotations()
    tracks = getattr(ann, "tracks", None) or []
    if tracks:
        ids = ", ".join(str(track.id) for track in tracks)
        link = f"{args.host.rstrip('/')}/tasks/{task.id}"
        raise SystemExit(f"Task {name} có {len(tracks)} cuboid dạng Track (ID: {ids}): {link}. "
                         "Vẽ lại các cuboid đó bằng Shape, xóa Track, bấm Save rồi xuất lại.")
    labels = {lab.id: lab.name for lab in task.get_labels()}
    frames = [vf.frame_name(*vf.parse_frame_name(f.name)) for f in task.get_meta().frames]
    out = {f: [] for f in frames}
    for s in ann.shapes:
        if getattr(s.type, "value", str(s.type)) != "cuboid":
            continue
        b = vf.cuboid_from_cvat(s.points, labels.get(s.label_id, "?"), args.scale_order, s.occluded)
        out[frames[s.frame]].append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})
    path = Path(args.out or d14_dir(student) / f"ref_{args.task}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": {"task": name}, "frames": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {path}: " + ", ".join(f"{f}={len(v)}" for f, v in out.items()))
    return 0


def main(argv=None) -> int:
    vf.configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    day = sub.add_parser("day13", help="tạo task CVAT D13-<ID> với 10 frame của bạn")
    day.add_argument("--student", required=True)
    day.add_argument("--data", required=True, help="thư mục data-VF")
    day.add_argument("--host")
    day.add_argument("--user")
    day.add_argument("--org", help="slug tổ chức CVAT để coach xem được task")
    day.add_argument("--dry-run", action="store_true", help="chỉ in frame và đóng gói, không kết nối")
    day.add_argument("--replace", action="store_true", help="xóa task cùng tên (mất nhãn) rồi tạo lại")
    day.add_argument("--bundle-out", help="giữ file zip tại đường dẫn này (để coach kiểm)")
    day.set_defaults(func=cmd_day13)

    rep = sub.add_parser("report", help="xuất bảng A")
    rep.add_argument("lab", choices=["d13"])
    rep.add_argument("--student", required=True)
    rep.add_argument("--host", required=True)
    rep.add_argument("--user", required=True)
    rep.add_argument("--org")
    rep.add_argument("--out", default="./nop")
    rep.add_argument("--scale-order", choices=["lwh", "wlh"], default="lwh", help=argparse.SUPPRESS)
    rep.set_defaults(func=cmd_report)

    for name, func, help_text in (
        ("preview", cmd_preview, "vẽ cuboid của bạn lên 7 ảnh camera (không dùng GT)"),
        ("check", cmd_check, "tự kiểm cuboid bằng point cloud (không dùng GT, không chấm điểm)"),
    ):
        qa = sub.add_parser(name, help=help_text)
        qa.add_argument("lab", choices=["d13"])
        qa.add_argument("--student", required=True)
        qa.add_argument("--data", required=True, help="thư mục data-VF")
        qa.add_argument("--table", help="bảng A (mặc định nop/D13-<ID>-bangA.csv)")
        qa.add_argument("--out", default="./xem" if name == "preview" else None)
        qa.set_defaults(func=func)

    d14 = sub.add_parser("day14", help="chọn frame tập A (tune) và B (nộp) theo ID + SEED")
    d14.add_argument("--student", required=True)
    d14.add_argument("--seed", help="SEED coach công bố (hoặc biến môi trường D14_SEED)")
    d14.add_argument("--data", required=True)
    d14.set_defaults(func=cmd_day14)

    pre = sub.add_parser("prelabel", help="tạo task CVAT D14-<ID>-A/B kèm box pre-label")
    pre.add_argument("--student", required=True)
    pre.add_argument("--task", choices=["A", "B"], required=True)
    pre.add_argument("--data", required=True)
    pre.add_argument("--boxes", help="JSON từ scripts/pp_prelabel.py --out (bỏ trống = task rỗng)")
    pre.add_argument("--frames", help="mặc định d14_<ID>/frames_<task>.txt")
    for a in ("--host", "--user", "--org", "--bundle-out"):
        pre.add_argument(a)
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--replace", action="store_true")
    pre.set_defaults(func=cmd_prelabel)

    ref = sub.add_parser("reference", help="đọc box đã sửa trên CVAT thành ref_<task>.json")
    ref.add_argument("--student", required=True)
    ref.add_argument("--task", choices=["A", "B"], required=True)
    ref.add_argument("--host", required=True)
    ref.add_argument("--user", required=True)
    ref.add_argument("--org")
    ref.add_argument("--out")
    ref.add_argument("--scale-order", choices=["lwh", "wlh"], default="lwh")
    ref.set_defaults(func=cmd_reference)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
