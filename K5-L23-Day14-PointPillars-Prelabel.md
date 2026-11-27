---
day: "D14"
title: "Day 14 Lab — Pre-label 3D bằng PointPillars pre-trained (VF LiDAR)"
description: "Dùng PointPillars để tạo nhãn 3D trên dữ liệu VF. Sửa nhãn tập A trong CVAT, ghi dự đoán trước khi chạy sweep và áp dụng cấu hình đã chọn cho tập B."
outcomes:
  - "Cài đặt sạch môi trường PyTorch CPU (hoặc Docker fallback) không cần GPU/biên dịch CUDA, tải đúng weights KITTI ghim commit `620e6b0` bằng `scripts/setup_pp.py` và chạy qua `python scripts/pp_selftest.py --data <Data-VF>` với đầu ra `TẤT CẢ OK`."
  - "Chạy `scripts/pp_prelabel.py` trên 6 frame buổi giảng (`demo/frames_lec.txt`), giải thích được phép đổi trục VF ↔ KITTI, vùng phủ 4 lượt xoay (`passes=4`) và lý do mất cân bằng chi phí `0.3·xóa + 1.0·thêm + 0.5·sửa` trong bán kính `50 m`."
  - "Sinh bộ frame cá nhân theo `<ID>` và `<SEED>` qua `scripts/vf_cvat.py day14`, đẩy tập A (`3` frame) lên CVAT `D14-<ID>-A`, sửa tay bằng `Shape` (không bấm `Track`) và kéo về `d14_<ID>/ref_A.json` khớp số `Items`."
  - "Ghi mục 1 báo cáo `report/individual/<MSSV>_<HoTen>.md` (Bước 5b) rồi commit và push trước khi chạy `scripts/pp_sweep.py`, đối chiếu `sweep.csv`, `best_config.json`, leave-one-out và `scripts/pp_compare.py`, sửa đủ 12 frame tập B (`ref_B.json`) và nộp link commit SHA repo `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel` lên LMS trước 12:00 trưa hôm sau (GMT+7)."
prerequisites:
  - "Đã hoàn thành thao tác tạo Task 3D, xoay/chỉnh cuboid theo 3 khung Top/Side/Front và xuất nhãn trên CVAT ở Day 13."
  - "Có Python 3.9+, Git (hoặc chạy `scripts/setup_pp.py --download-zip`), bộ dữ liệu `<Data-VF>` (5 sequence từ Day 13) và mã học viên cá nhân."
  - "Nhận `<SEED>` từ giảng viên khi bắt đầu phần lab buổi chiều để sinh danh sách frame."
requiredTools:
  - "Python 3.9+ với `numpy>=1.24`, `torch>=2.1` (bản CPU), `Pillow>=10` và `cvat-sdk>=2.20` (`requirements.txt`)."
  - "CVAT 3D (`http://localhost:8080` khi chạy Docker local như Day 13 hoặc server CVAT của lớp)."
  - "Tùy chọn Docker (`Dockerfile` + `.dockerignore`) khi môi trường Python máy gặp sự cố."
commonErrors:
  - "PowerShell 5.1 báo lỗi khi gõ `&&`, `source`, hoặc chặn `.venv\\Scripts\\Activate.ps1` → Chạy từng dòng riêng, bật `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` và đặt `$env:PYTHONUTF8 = \"1\"` trước khi chạy."
  - "Nếu JSON không đọc được, tạo lại dự đoán bằng `scripts/pp_prelabel.py` hoặc xuất lại nhãn từ CVAT bằng `scripts/vf_cvat.py reference`."
  - "Lệnh `scripts/vf_cvat.py reference` dừng nếu còn Track. Vẽ lại các hộp đó bằng Shape, xóa Track, bấm Save rồi xuất lại. Nếu trùng tên task, kiểm các ID và đổi tên task bạn sẽ không xuất."
  - "Truyền `--org <SLUG>` khi chạy CVAT local Docker gây lỗi 404 không thấy tổ chức → Bỏ hẳn cờ `--org` trên CVAT local (`http://localhost:8080`); chỉ truyền `--org` thống nhất ở cả Bước 3, 5, 7 và 9 trên server lớp có organization."
  - "Docker báo `exit 137` (OOM) hoặc thoát container bị mất `d14_<ID>/` → Tắt CVAT Docker (`docker compose stop`) khi chạy model nếu WSL2 RAM <= 4 GB, và luôn mount `-v \"${PWD}\\work:/work\" -w /work`."
  - "Khi khai báo `--grid` cho `scripts/pp_sweep.py`, dùng đúng tên tham số và dạng `--grid score_thr=0.1,0.2,0.3,0.4,0.5` và giữ mức gốc `0.1` trong lưới."
  - "Mất điểm mục 3 hoặc mục 4 dù đủ file → Phải commit/push mục 1 báo cáo (5b) *trước* `scripts/pp_sweep.py`, tự kiểm sửa 12 frame tập B để `ref_B.json` không trùng `pred_B.json`, và nộp đúng link commit SHA (`git rev-parse HEAD`) lên LMS trước 12:00 trưa hôm sau."
requiresSubmission: true
workMode: "individual"
---

# Day 14 Lab — Pre-label 3D bằng PointPillars pre-trained (VF LiDAR)

Bài cá nhân này có **245 phút thực hành**. Bạn dành 45 phút ở cuối buổi lý thuyết sáng (09:00–13:00), gồm 10 phút cài đặt và 35 phút chạy thử. Buổi chiều (16:00–19:30) có 200 phút làm tập A và B. Bạn chạy PointPillars đã huấn luyện trên KITTI trên dữ liệu LiDAR 360° của Day 13. Mô hình dùng CPU và mã PyTorch thay cho hai phép toán CUDA, nên máy không cần GPU. Mã PointPillars gốc giữ nguyên tại commit `620e6b0`.

Tạo cuboid dự đoán, đưa lên CVAT rồi sửa 3 frame tập A để có nhãn tham chiếu. Ghi dự đoán và push báo cáo ở bước 5b trước khi thử nhiều cấu hình. Sau đó dùng cấu hình đã chọn cho 12 frame tập B, sửa nhãn và xuất `ref_B.json`.

Đẩy bài lên repo `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel`. Nộp link commit có SHA lên LMS trước **12:00 trưa hôm sau (GMT+7)**. Xem [SUBMISSION.md](SUBMISSION.md) để biết file cần nộp và [RUBRIC.md](RUBRIC.md) để biết cách chấm. Theo dõi tiến độ bằng [CHECKPOINTS.md](CHECKPOINTS.md), đọc [RULES.md](RULES.md) và dùng [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).

**Sao chép lệnh.** Lấy lệnh từ bản Markdown hoặc dùng nút **Copy** trong bản HTML. Khi sao chép từ PDF, lệnh dài có thể bị tách thành nhiều dòng.

## Pha 0 — Cài đặt và kiểm tra môi trường (Buổi sáng · 0:00 → 0:10 · 10 phút)

Cuối buổi lý thuyết sáng, clone repo và tạo môi trường Python 3.9+. Chuẩn bị đường dẫn `<Data-VF>` chứa trực tiếp `sequence_01` đến `sequence_05`. Xem [CHECKPOINTS.md](CHECKPOINTS.md) và [RULES.md](RULES.md) trước khi bắt đầu.

:::repository{url="https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel" label="Repo gốc Day 14 — K5-L23-Day14-PointPillars-Prelabel"}
- [Repository gốc Day 14](https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel)
:::

### Cài Python theo hệ điều hành

Khối cài nhanh trong HTML chỉ dành cho **Windows PowerShell**. Nếu dùng macOS hoặc Linux, chọn tab **macOS / Linux (Bash)** bên dưới. Dùng tab tương ứng để cài PyTorch CPU và các gói trong `requirements.txt`. `scripts/setup_pp.py` tải mã PointPillars ở commit `620e6b0` cùng trọng số `epoch_160.pth` (~20 MB). Nếu Git không tải được mã nguồn, chạy lại với `--download-zip`. Trên PowerShell 5.1, chạy từng dòng trong tab Windows.

:::code-tabs{title="Cài đặt môi trường Python CPU và tự kiểm tra cài đặt"}
```powershell {title="Windows (PowerShell)"}
git clone https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel.git k5-day14-lab
cd k5-day14-lab
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_pp.py
python scripts/pp_selftest.py --data "<Data-VF>"
```
```bash {title="macOS / Linux (Bash)"}
git clone https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel.git k5-day14-lab
cd k5-day14-lab
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_pp.py
python scripts/pp_selftest.py --data "<Data-VF>"
```
:::

1. Chạy các lệnh trên; nếu bản clone cũ bị lỗi read-only trong `third_party/PointPillars/.git`, xóa `third_party` rồi chạy lại `python scripts/setup_pp.py`.
2. Chạy `python scripts/pp_selftest.py --data "<Data-VF>"` và kiểm tra mọi dòng đều in `[OK ]` kèm `TẤT CẢ OK`.
3. Nếu môi trường máy gặp lỗi, dùng `Dockerfile` (tắt CVAT bằng `docker compose stop` nếu WSL2 RAM `<= 4 GB` tránh OOM `exit 137`, gắn `-v "${PWD}\work:/work" -w /work` giữ `d14_<ID>/`, và dùng `--host http://host.docker.internal:8080`).

:::callout{tone="info" title="Chạy qua Docker khi môi trường Python cục bộ lỗi"}
Build một lần bằng `docker build -t k5-day14-lab .`, sau đó chạy `docker run --rm -v "C:\path\Data-VF:/data" -v "${PWD}\work:/work" -w /work k5-day14-lab python /lab/scripts/pp_selftest.py --data /data` và kiểm tra `TẤT CẢ OK`.
:::

:::checklist{title="Tín hiệu sẵn sàng trước Pha 1" tone="info"}
- [ ] Đã clone đúng repo và kích hoạt `.venv` có Python 3.9+.
- [ ] `scripts/setup_pp.py` tải xong `third_party/PointPillars` (`620e6b0`) và `epoch_160.pth`.
- [ ] `scripts/pp_selftest.py --data "<Data-VF>"` in `[OK ]` ở mọi mục và `TẤT CẢ OK`.
- [ ] CVAT 3D (`http://localhost:8080` hoặc server lớp) đăng nhập bình thường.
:::

## Pha 1 — Thực hành 6 frame buổi sáng, hệ trục VF ↔ KITTI và chi phí sửa (0:10 → 0:45 · 35 phút)

### Hệ trục và lớp vật thể của KITTI và VF

Một mô hình phát hiện 3D được huấn luyện trên bộ dữ liệu KITTI không thể dự đoán chính xác ngay lập tức khi đưa sang bộ dữ liệu VF của Day 13 vì sự khác biệt cả về hệ tọa độ cảm biến, vùng quét không gian lẫn danh sách lớp đối tượng. Trên bộ KITTI, cảm biến chỉ gán nhãn trong vùng hình nón phía trước xe (`x` từ `0` đến `69.12 m`, `|y| < 39.68 m`) và chỉ phân biệt ba lớp là `Car`, `Cyclist` và `Pedestrian`. Trong khi đó, đám mây điểm VF LiDAR thu thập toàn cảnh 360° quanh xe với hệ trục đặt khác hẳn: trục `x` hướng sang trái xe, trục `y` hướng về phía sau (nên đầu xe là hướng `−y`), trục `z` hướng lên, còn góc `yaw` trong `[0, 2π)` đo ngược chiều kim đồng hồ từ `+x`. Đồng thời, dữ liệu VF có thêm hai lớp phương tiện cỡ lớn là `Truck` và `Bus` mà trọng số KITTI chưa từng được học trực tiếp.

:::image{src="walkthrough_images/diagram_01_vf_axes_kitti.jpg" alt="Sơ đồ đối chiếu hệ trục LiDAR VF và khung tọa độ KITTI của PointPillars" caption="Phép đổi trục trong vf_pp/core.py đưa đám mây điểm VF về hướng nhìn của model KITTI."}
:::

### Bốn lượt xoay và cách tính công sửa

Vì model chỉ nhìn vùng 90° phía trước (`x` trong `0–69.12 m`, `|y| < 39.68 m`), hàm `Detector.raw_detect` trong `vf_pp/core.py` thực hiện **4 lượt xoay tuần tự (`passes = 4`)** ở `0°`, `90°`, `180°`, `270°` (bỏ qua an toàn lượt không có điểm), rồi gộp hộp bằng NMS không phân lớp (`merge_nms_thr = 0.1`). Vòng tròn chấm điểm bán kính `50.0 m` (`MAX_RANGE = 50.0`, cố định bất kể tham số `range_m`) nằm gọn bên trong hợp bốn khung xoay này.

:::image{src="walkthrough_images/diagram_02_model_range_50m.jpg" alt="Sơ đồ vùng nhìn BEV bốn lượt xoay bao trọn vòng tròn đỏ 50 m" caption="Hợp 4 lượt xoay phủ kín vòng tròn chấm điểm 50 m; giảm passes=1 sẽ sót hai bên xe."}
:::

Chất lượng pre-label được đo trong bán kính `50.0 m` bằng **`correction_cost = 0.3·xóa + 1.0·thêm + 0.5·sửa`** sau khi ghép 1–1 theo Hungarian (`BEV IoU >= 0.3`). Hàm `needs_edit` kiểm tra `yaw` theo chu kỳ `π`, nên box ngược đầu đúng 180° không tính là "sửa" nếu lớp, tâm và cạnh đã đạt.

| Thao tác (trọng số) | Điều kiện tính trong `correction_cost` (`MAX_RANGE = 50.0 m`) | Tác động khi sửa trên CVAT |
| --- | --- | --- |
| **Xóa (`0.3`)** | Box dự đoán dư không ghép được cặp với `gt` (`BEV IoU < 0.3`) | Bấm `Remove` hoặc `Delete` rất nhanh nên phạt nhẹ nhất |
| **Thêm (`1.0`)** | Box thật trong `gt` bị model bỏ sót hoàn toàn | Phải vẽ cuboid mới và kéo lại 3 mặt cắt nên phạt nặng nhất |
| **Sửa (`0.5`)** | Cặp khớp (`BEV IoU >= 0.3`) nhưng sai lớp, lệch tâm `> 0.3 m`, lệch cạnh `> 0.2 m`, hoặc lệch `yaw > 10°` (mod `π`) | Kéo khung `Top/Side/Front` hoặc gõ `L/W/H`, đổi nhãn trên `DETAILS` |

### Các tham số dùng ở buổi lab

Các tham số trong `vf_pp/core.py:Config` được tách thành ba nhóm để bạn can thiệp đúng tầng khi tinh chỉnh. Nhóm **A** đưa điểm VF về lưới KITTI: luôn giữ `voxel_size = 0.16` khớp weights đã huấn luyện và giữ `passes = 4` để phủ đủ `360°`, vì thay đổi hai tham số này làm tụt chất lượng ngay lập tức. Nhóm **B** đặt ngưỡng tin cậy chung (`score_thr`) hoặc theo lớp (`thr_car`, `thr_cyc`, `thr_ped`) cùng NMS nội bộ. Nhóm **C** hậu xử lý trên không gian VF: lọc `min_points`, nới hộp `size_pad_m` và đổi `Car` cỡ lớn thành `Truck`/`Bus` qua `vf_class`. Khi thử nhanh, dùng cờ `--set KEY=VALUE`; khi chốt cấu hình ổn định, lưu file JSON và truyền qua `--config`.

| Nhóm | Tham số chính (mặc định) | Ý nghĩa và phạm vi tác động |
| --- | --- | --- |
| A. Đầu vào model | `voxel_size` (`0.16`), `intensity_mode` (`clip`/`log`/`const`), `intensity_scale` (`1.0`), `ground_mode` (`fixed`/`auto`), `delta_z` (`0.0`), `passes` (`4`) | Quy đổi điểm VF về miền KITTI; đổi `voxel_size` khỏi `0.16` làm sai lệch lưới mà weights đã học, giảm `passes` làm mất vùng nhìn |
| B. Đầu ra model | `score_thr` (`0.1`), `thr_car`/`thr_cyc`/`thr_ped` (`None`), `nms_thr` (`0.01`), `nms_pre` (`100`), `max_num` (`50`) | Lọc box theo độ tin cậy chung hoặc riêng từng lớp và đặt ngưỡng NMS bên trong mạng |
| C. Hậu xử lý VF | `range_m` (`50.0`), `min_points` (`0`), `merge_nms_thr` (`0.1`), `size_pad_m` (`0.1`), `truck_l_min` (`5.6`), `truck_h_min` (`2.4`), `bus_l_min` (`9.0`), `bus_h_min` (`2.8`) | Cắt bán kính giữ box, lọc box quá ít điểm LiDAR (`min_points`), nới kích thước (`size_pad_m`) và đổi nhãn `Car` dài/cao thành `Truck` hoặc `Bus` theo luật kích thước |

### Chạy thử trên 6 frame buổi giảng

Chạy PointPillars trên 6 frame trong `demo/frames_lec.txt`. Lượt đầu dùng cấu hình gốc. Lượt hai tăng ngưỡng điểm lên `0.4` và giữ hộp có ít nhất 5 điểm LiDAR. So sánh xem lượt hai lọc được hộp thừa nào và bỏ mất vật thể thật nào.

```bash
python scripts/pp_prelabel.py --data "<Data-VF>" --frames demo/frames_lec.txt --out out/lec.json --plot out/lec_png
python scripts/pp_prelabel.py --data "<Data-VF>" --frames demo/frames_lec.txt --set score_thr=0.4 --set min_points=5 --out out/lec_filtered.json --plot out/lec_filtered_png
```

1. Chạy lệnh đầu tiên để xuất danh sách box `out/lec.json` và vẽ 6 ảnh tổng hợp `out/lec_png/*.jpg` (mỗi ảnh gồm 7 góc camera và bản đồ BEV bán kính `50 m`).
2. Mở ảnh `out/lec_png/s05f40.jpg` cùng các frame còn lại để đối chiếu box trên camera với vị trí vòng tròn BEV: ghi nhận model nhận xe con gần khá ổn, nhưng dễ gán nhầm xe buýt lớn thành nhiều hộp nhỏ, vẽ hộp thừa ở hàng cây hoặc bụi rậm, hoặc bỏ sót `Pedestrian` và `Cyclist` ở xa.

:::image{src="walkthrough_images/cli_03_lec_prelabel_s05f40.jpg" alt="Ảnh 7 camera và BEV 50 m của frame s05f40 với 35 cuboid pre-label" caption="Ảnh out/lec_png/s05f40.jpg hiển thị 7 camera và BEV vòng đỏ 50 m kèm mã pp_id."}
:::

3. Chạy lệnh thứ hai để lưu `out/lec_filtered.json` và 6 ảnh trong `out/lec_filtered_png/`. So sánh số hộp trên terminal (ví dụ `s05f40` giảm từ `35` xuống `20` hộp). Mở ảnh cùng mã frame trong hai thư mục để xem hộp bị lọc là hộp nhiễu hay đang bao một vật thể thật.

### Kiểm tra kết quả buổi sáng

Kiểm tra cả hai lượt đều tạo đủ 6 ảnh. Sau khi đối chiếu ảnh, giải thích vì sao thêm một vật thể bị sót tốn công hơn xóa một hộp thừa. Công thức tính lần lượt là `1.0` và `0.3` cho mỗi hộp. Buổi chiều (`16:00–19:30`) bạn dùng các số này để đánh giá cấu hình trên dữ liệu riêng.

## Pha 2 — Tạo và sửa nhãn tập A (bước 1–5) (0:45 → 1:45 · 60 phút)

### Cách script chọn tập A và tập B

Coach công bố `<SEED>` ở đầu buổi chiều. Lệnh `scripts/vf_cvat.py day14` chọn frame cố định theo `<ID>` và `<SEED>`. Chạy lại với cùng mã và SEED sẽ ra cùng danh sách. Hai học viên có thể nhận một số frame trùng nhau.

- **Tập A** có 3 frame thuộc 3 sequence để chọn cấu hình.
- **Tập B** có 12 frame thuộc 4 sequence để kiểm tra cấu hình và nộp bài. Trong mỗi sequence, các frame B cách nhau ít nhất 15 chỉ số.

Script tránh các frame buổi giảng, các frame Day 13 của bạn và các chỉ số lân cận. Nếu hai tập dùng chung sequence, mỗi frame tập A cách các frame tập B ít nhất 6 chỉ số.

### Kiểm tra mã học viên, file và tùy chọn CVAT

Giữ đúng ba quy ước kỹ thuật sau trước khi chạy lệnh ở Pha 2:
- **Giữ số `0` đầu mã `<ID>`:** `vf_common.normalize_student` chỉ viết hoa chữ, nên `"001"` và `"01"` tạo hai bộ frame và hai thư mục (`d14_001/` vs `d14_01/`) khác nhau; trên PowerShell luôn bọc `--student "001"`. Không sửa tay `frames_A.txt` hay `frames_B.txt`.
- **Giữ nguyên file do script tạo.** Script đọc được BOM UTF-8 trong `frames_*.txt`, cấu hình, `pred_*.json` và `ref_*.json`. Nếu file JSON bị hỏng, tạo lại dự đoán bằng `scripts/pp_prelabel.py` hoặc xuất lại nhãn từ CVAT bằng `scripts/vf_cvat.py reference`.
- **Cờ `--org <SLUG>` là tùy chọn:** bỏ hẳn `--org` khi chạy CVAT Docker cá nhân (`http://localhost:8080`, tránh lỗi 404); chỉ truyền thống nhất ở Bước 3, 5, 7 và 9 khi server lớp có organization.

### Sửa cuboid trên CVAT

Dành khoảng 45 phút kiểm đủ 3 frame theo [hướng dẫn gán nhãn 3D Cuboid](docs/GUIDELINE.md). Giữ hộp đúng, chỉnh hộp sai, xóa hộp thừa và thêm vật bị sót. Dùng `pp_id` để đối chiếu hộp với ảnh BEV. Các thuộc tính `score`, `warn` và `note` giúp bạn tìm hộp cần xem kỹ. Rubric mục 1 tính cả chất lượng các hộp bạn vẽ bổ sung.

:::callout{tone="warning" title="Thêm cuboid bằng Shape"}
Lệnh `reference` dừng nếu task còn **Track**. Vẽ lại các hộp đó bằng **Shape**, xóa Track cũ rồi bấm **Save** và xuất lại. Đối chiếu số hộp từng frame với **Items** trên CVAT.
:::

### Chạy lệnh cho tập A

Chạy ba lệnh dưới đây để chọn frame, tạo nhãn dự đoán và nạp tập A lên CVAT. Nhập mật khẩu ở dấu nhắc CVAT hoặc dùng biến `CVAT_PASSWORD`. Sau khi task xuất hiện, mở job và làm bước 4.

```bash
python scripts/vf_cvat.py day14 --student "<ID>" --seed "<SEED>" --data "<Data-VF>"
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --out "d14_<ID>/pred_A.json" --plot "d14_<ID>/png_A"
python scripts/vf_cvat.py prelabel --student "<ID>" --task A --data "<Data-VF>" --boxes "d14_<ID>/pred_A.json" --host http://localhost:8080 --user "<ten>"
```

1. **Bước 1 — Sinh danh sách frame:** Chạy `scripts/vf_cvat.py day14` tạo `d14_<ID>/frames_A.txt` và `frames_B.txt`; giữ nguyên dòng `seed_tag=` để coach đối chiếu SEED.
2. **Bước 2 — Pre-label Tập A:** Chạy `scripts/pp_prelabel.py` tạo `pred_A.json` và 3 ảnh `d14_<ID>/png_A/*.jpg` để xem trước vùng thừa/sót.

:::image{src="walkthrough_images/cli_01_prelabel_7cam_bev_s02f45.jpg" alt="Ảnh pre-label tập A trên s02f45 gồm 7 camera và BEV 50 m" caption="Ảnh d14_001/png_A/s02f45.jpg ở Bước 2 giúp rà nhanh 7 camera và BEV trước khi lên CVAT."}
:::

3. **Bước 3 — Đẩy task lên CVAT:** Chạy `scripts/vf_cvat.py prelabel --task A` (bỏ `--org` trên Docker local). Mở trang **Tasks**, vào `D14-<ID>-A` kiểm tra đủ 5 nhãn (`Car`, `Truck`, `Bus`, `Cyclist`, `Pedestrian`) và đúng 3 frame.

:::image{src="walkthrough_images/cvat_01_tasks_list_A.png" alt="Trang Tasks trên CVAT hiển thị task D14-001-A đã được tạo" caption="Trang Tasks sau Bước 3: task D14-001-A xuất hiện kèm trạng thái 1 annotating."}
:::

:::image{src="walkthrough_images/cvat_02_task_details_A.png" alt="Chi tiết Task D14-001-A với 5 nhãn cuboid và 1 Job 3 frame" caption="Trang chi tiết Task: 5 nhãn cuboid và Job gồm đúng 3 frame tập A."}
:::

4. **Bước 4 — Sửa tay 3 frame trên CVAT 3D (~45 phút):** Mở Job ở `Standard 3D` và làm lần lượt các thao tác dưới đây:
   - **Kiểm tra thuộc tính:** Mở `Objects` để xem số `Items`. Mở `DETAILS` để xem kích thước `L/W/H` và các thuộc tính của hộp.

:::image{src="walkthrough_images/cvat_03_3d_editor_initial.png" alt="Không gian CVAT 3D trên s02f45.pcd với 46 cuboid và 3 mặt cắt" caption="Giao diện CVAT 3D ở Bước 4: khung 3D trên cùng và 3 mặt cắt Top/Side/Front bên dưới."}
:::

:::image{src="walkthrough_images/cvat_04_object_details_attrs.png" alt="Thẻ đối tượng Car mở DETAILS hiển thị L W H và 4 thuộc tính" caption="Mục DETAILS hiển thị L/W/H cùng bốn thuộc tính pp_id, score, warn, note do tool tạo sẵn."}
:::

   - **Chỉnh hộp sai.** Dùng `Top` để căn tâm và hướng, `Side` và `Front` để kiểm đáy và chiều cao. Đổi lớp nếu cần, ví dụ `Car` thành `Truck` hoặc `Bus`.

:::image{src="walkthrough_images/cvat_05_edit_length_4_23.png" alt="Chỉnh chiều dài L của cuboid Car từ 3.78 thành 4.23 trong DETAILS" caption="Sửa kích thước hộp bằng cách gõ số vào L/W/H hoặc kéo các điểm điều chỉnh trên khung cắt."}
:::

   - **Xóa box thừa:** Bấm `⋮ → Remove` trên thẻ đối tượng (hoặc `Delete`), kiểm tra `Items` giảm đi 1.

:::image{src="walkthrough_images/cvat_06_delete_menu_obj2.png" alt="Mở menu ba chấm trên đối tượng số 2 chọn Remove xóa hộp thừa" caption="Bấm nút ba chấm ở góc thẻ đối tượng và chọn Remove để xóa hộp dự đoán sai."}
:::

:::image{src="walkthrough_images/cvat_07_after_delete_45_items.png" alt="Bảng Objects sau khi xóa hiển thị chỉ số Items giảm từ 46 xuống 45" caption="Sau khi xóa một box thừa trên s02f45, chỉ số Items giảm ngay từ 46 xuống 45."}
:::

   - **Thêm vật bị sót.** Chọn biểu tượng lập phương rồi chọn **Shape**. Bấm đúp vào khung 3D để đặt hộp, căn theo ba khung chiếu và bấm **Save**.

:::image{src="walkthrough_images/cvat_08_draw_cuboid_shape_vs_track.png" alt="Popup Draw new cuboid hiển thị hai nút Shape và Track cạnh nhau" caption="Chọn Shape khi vẽ hộp mới. Lệnh reference dừng nếu task còn Track."}
:::

:::image{src="walkthrough_images/cvat_09_add_bus_shape_46_items.png" alt="Thêm thành công đối tượng #101 Bus dạng CUBOID SHAPE đạt 46 Items" caption="Đối tượng #101 nhãn Bus tạo bằng CUBOID SHAPE làm tổng Items tăng trở lại 46."}
:::

:::image{src="walkthrough_images/cvat_10_saved_clean_46_items.png" alt="Màn hình CVAT sau khi bấm Save với đủ 46 cuboid dạng Shape" caption="Bấm Save tới khi nút Save mờ đi trước khi chạy lệnh kéo nhãn tham chiếu về máy."}
:::

### Xuất nhãn và đo công sửa tập A (bước 5)

Sửa đủ 3 frame rồi bấm **Save** trong CVAT. Quay lại terminal và xuất nhãn tham chiếu.

```bash
python scripts/vf_cvat.py reference --student "<ID>" --task A --host http://localhost:8080 --user "<ten>"
```

**Chỉ chạy lệnh đo sau khi lệnh xuất in Đã ghi** và liệt kê đủ 3 frame. Đối chiếu số cuboid của từng frame với **Items** trên CVAT. Nếu lệnh xuất báo lỗi, xử lý rồi xuất lại để tránh đo bằng `ref_A.json` cũ. Nếu server xuất đảo chiều dài và chiều rộng, thêm `--scale-order wlh` khi xuất lại.

```bash
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json"
```

### Kiểm tra file và số đo tập A (1:45 → 1:55 · 10 phút)

Kiểm tra có đủ `frames_A.txt`, `frames_B.txt`, `pred_A.json` và `ref_A.json` trong `d14_<ID>/`. Số hộp xuất ra phải khớp **Items** trên từng frame CVAT. Ghi lại số xóa, thêm, sửa và chi phí của cấu hình gốc để dùng ở bước 5b. Trong 10 phút nghỉ, sao chép [mẫu báo cáo](report/individual/MSSV_HoTen.md) thành `report/individual/<MSSV>_<HoTen>.md`.

## Pha 3 — Ghi dự đoán, chạy sweep và so sánh trên tập A (1:55 → 2:25 · 30 phút)

### Bước 5b. Ghi và đẩy dự đoán

Điền mã học viên và `seed_tag` ở đầu mẫu báo cáo. Trong mục 1, chỉ ghi tham số bạn nghĩ sẽ giảm công sửa nhiều nhất và lý do. Dùng ảnh BEV trong `png_A/` cùng số xóa, thêm và sửa đã đo ở bước 5.

Ví dụ, nếu ảnh cho thấy nhiều hộp thừa có điểm tin cậy thấp, hãy giải thích vì sao bạn dự đoán tăng `score_thr` sẽ giảm số hộp phải xóa.

Tạo [repo cá nhân trống](SUBMISSION.md#1-repo-nộp) rồi chạy các lệnh sau. Thay `<tai-khoan>` và tên repo mẫu bằng thông tin của bạn.

```bash
git remote set-url origin "https://github.com/<tai-khoan>/K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel.git"
git add "report/individual/<MSSV>_<HoTen>.md"
git commit -m "5b: dự đoán"
git branch -M main
git push -u origin main
```

**Mở repo cá nhân trên GitHub. Chỉ sang bước 6 khi đã thấy commit `5b: dự đoán`.** Nếu push lỗi, xử lý lỗi và xác nhận lại trên GitHub. Giữ nguyên mục 1 sau khi đã chạy sweep.

### Bước 6. Thử 15 cấu hình trên tập A

Chọn tab theo hệ điều hành. Lệnh dưới đây thử 5 mức `score_thr` và 3 mức `min_points`, rồi ghi `sweep.csv` và `best_config.json`. Viết mỗi mục `--grid` theo dạng `key=vals`. Giữ cặp `score_thr=0.1` và `min_points=0` trong lưới để có cấu hình gốc đối chiếu.

:::code-tabs{title="Chạy sweep sau khi đã push dự đoán"}
```powershell {title="Windows (PowerShell)"}
python scripts/pp_sweep.py --data "<Data-VF>" `
  --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" `
  --grid score_thr=0.1,0.2,0.3,0.4,0.5 --grid min_points=0,5,10 `
  --out "d14_<ID>/sweep.csv"
```
```bash {title="macOS / Linux (Bash)"}
python scripts/pp_sweep.py --data "<Data-VF>" \
  --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" \
  --grid score_thr=0.1,0.2,0.3,0.4,0.5 --grid min_points=0,5,10 \
  --out "d14_<ID>/sweep.csv"
```
:::

Script dùng lại dự đoán thô khi chỉ đổi ngưỡng điểm hoặc lọc số điểm LiDAR. Vì vậy, thử nhiều tổ hợp hậu xử lý không đồng nghĩa với việc chạy lại mô hình cho từng tổ hợp.

Khi đọc bảng, xem cả chi phí và số hộp cần xóa, thêm, sửa. Ví dụ, bớt 3 hộp thừa giúp chi phí giảm 0,9. Nếu cấu hình đó đồng thời bỏ sót thêm 1 vật thể, chi phí tăng 1,0. Vì vậy, ít hộp dự đoán hơn vẫn có thể tốn công sửa hơn.

Đọc phần **leave-one-out** ở cuối terminal. Mỗi lượt chọn cấu hình trên 2 frame rồi đo công sửa ở frame còn lại. `oracle` là chi phí thấp nhất trong các cấu hình đã thử trên chính frame kiểm tra đó. Nếu chi phí cao hơn nhiều so với `oracle`, cấu hình có thể hợp với hai frame dùng để chọn hơn là với frame còn lại. Ghi số đo của cấu hình gốc, cấu hình đã chọn và nhận xét leave-one-out vào mục 2 báo cáo.

### Bước 8. So sánh cấu hình gốc và cấu hình đã chọn

Chạy so sánh trên tập A sau khi có `best_config.json`. Xem kết quả công sửa trên terminal và mở ảnh trong thư mục `compare/`.

```bash
python scripts/pp_compare.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" --tuned "d14_<ID>/best_config.json" --out "d14_<ID>/compare"
```

Ảnh BEV đặt cấu hình gốc bên trái và cấu hình đã chọn bên phải. Khung ảnh kéo tới ±52 m, với vòng đỏ 50 m đánh dấu phạm vi tính công sửa. Màu xanh lá là hộp giữ nguyên, vàng là hộp cần sửa, đỏ là hộp cần xóa và xanh dương là vật bị sót.

:::image{src="walkthrough_images/cli_02_compare_52m_bev_s02f45.jpg" alt="Ảnh scripts/pp_compare.py frame s02f45 phạm vi 52 m với vòng đỏ 50 m" caption="Ảnh d14_001/compare/s02f45.jpg: bản đồ BEV 52 m kèm vòng đỏ 50 m đối chiếu gốc và tuned."}
:::

Đưa một ảnh so sánh vào mục 3 của báo cáo. Dùng số xóa, thêm và sửa để giải thích kết quả khớp hay lệch dự đoán ban đầu. Giữ ảnh đã dùng trong `compare/` để link trong báo cáo mở được trên GitHub. Dùng đường dẫn tương đối từ file báo cáo, ví dụ `../../d14_<ID>/compare/<frame>.jpg`. Trước khi chuyển sang tập B, kiểm tra đã có `sweep.csv` và `best_config.json`, đồng thời mục 1–3 đã đủ nội dung.

## Pha 4 — Sửa nhãn tập B và nộp bài (bước 7, 9) (2:25 → 4:05 · 100 phút)

### Kiểm tra cấu hình trên 12 frame tập B

Tập B có 12 frame từ 4 sequence và chiếm 30 điểm trong [RUBRIC.md](RUBRIC.md). Dùng `best_config.json` để tạo `pred_B.json`, nạp task B và tự kiểm đủ 12 frame. Sau khi sửa và lưu trên CVAT, xuất `ref_B.json` để đo công sửa cho mục 4 báo cáo. Nếu toàn bộ hộp trong `ref_B.json` trùng dự đoán về vị trí, kích thước, góc và lớp, mục này mất 30 điểm. Nêu ít nhất hai giới hạn của cách làm trong mục 5.

### File cần giữ để coach chấm bài

Vì dùng CVAT Docker cá nhân (`http://localhost:8080`), coach **không mở CVAT cá nhân của bạn** mà chấm hoàn toàn từ các file trong repo tại **commit SHA** bạn nộp trên LMS ([SUBMISSION.md](SUBMISSION.md)). Đầu tiên, coach `git checkout <SHA>` và chạy `check_student.py` với `SEED` buổi lab trên `d14_<ID>/frames_A.txt` và `frames_B.txt`; nếu báo **CÓ VẤN ĐỀ** (sai SEED, sai ID hoặc sửa tay frame), các mục 1–5 không được chấm.

Commit các file bắt buộc theo [SUBMISSION.md](SUBMISSION.md). `.gitignore` đã loại thư mục `d14_*/png_*/`. Giữ ảnh so sánh dùng trong báo cáo. Để dữ liệu VF, thư mục `third_party/`, mật khẩu CVAT và `.env` ngoài bài nộp.

Nếu repo là Private, mời tài khoản GitHub của coach. Nộp link dạng `https://github.com/<tai-khoan>/<repo>/commit/<SHA>` lên LMS trước **12:00 trưa hôm sau (GMT+7)**. Giữ commit đã nộp theo [RULES.md](RULES.md).

### Tạo task B và sửa cuboid (bước 7)

```bash
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --out "d14_<ID>/pred_B.json"
python scripts/vf_cvat.py prelabel --student "<ID>" --task B --data "<Data-VF>" --boxes "d14_<ID>/pred_B.json" --host http://localhost:8080 --user "<ten>"
```

1. **Bước 7a–7b.** Tạo `pred_B.json` từ `frames_B.txt` bằng cấu hình `best_config.json`. Nạp dự đoán lên CVAT để tạo task `D14-<ID>-B`. Kiểm tra trang **Tasks** có task A và task B. Khi tiếp tục trên task B đã tạo, dùng lại đúng file dự đoán đã nạp. Nếu script phát hiện file đã đổi, lệnh dừng và giữ nguyên nhãn cũ. Nếu script báo chưa thể đối chiếu, kiểm tra lại file đã nạp lúc tạo task. Trao đổi với coach trước khi tạo lại task vì `--replace` xóa toàn bộ nhãn đã sửa.

:::image{src="walkthrough_images/cvat_11_tasks_A_and_B.png" alt="Trang Tasks trên CVAT hiển thị đầy đủ cả task D14-001-A và D14-001-B" caption="Trang Tasks trên CVAT sau khi tạo thành công cả task D14-001-A (3 frame) và D14-001-B (12 frame)."}
:::

2. **Bước 7c.** Kiểm đủ 12 frame của `D14-<ID>-B` theo [hướng dẫn gán nhãn 3D Cuboid](docs/GUIDELINE.md). Rà lớp và hình hộp, xóa hộp thừa và dùng **Shape** để thêm vật bị sót trong 50 m. Bấm **Save**.
### Xuất nhãn và đo công sửa tập B (bước 9)

Sau khi sửa đủ 12 frame và bấm **Save**, quay lại terminal để xuất nhãn.

```bash
python scripts/vf_cvat.py reference --student "<ID>" --task B --host http://localhost:8080 --user "<ten>"
```

**Chỉ chạy lệnh đo sau khi lệnh xuất in Đã ghi** và liệt kê đủ 12 frame. Đối chiếu số hộp từng frame với **Items**. Nếu lệnh xuất báo lỗi, xử lý rồi xuất lại để tránh dùng `ref_B.json` cũ. Xem thêm [bước 9 trong README](README.md#xuất-nhãn-và-đo-công-sửa-tập-b-bước-9).

```bash
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --gt "d14_<ID>/ref_B.json"
```

Ghi số hộp cần xóa, thêm, sửa và chi phí vào mục 4 của báo cáo. Nêu ít nhất hai giới hạn ở mục 5.

### Commit và nộp link (bước 9)

Chỉ commit sau khi đã kiểm đủ file theo [SUBMISSION.md](SUBMISSION.md#1-repo-nộp) và điền xong báo cáo. Chạy từ gốc repo, thay `<ID>` bằng mã trong tên thư mục đã tạo ở bước 1.

```bash
git add "d14_<ID>" report
git commit -m "Day 14: bài nộp"
git push -u origin main
git rev-parse HEAD
```

Khi push thành công, mở repo cá nhân trên GitHub và xác nhận commit `Day 14: bài nộp` có đủ file. Nộp link commit đó lên LMS trước **12:00 trưa hôm sau (GMT+7)**.

```text
K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel/
├── d14_<ID>/                  # giữ số 0 đầu, ví dụ d14_001/ (png_*/ đã bị .gitignore bỏ qua)
│   ├── frames_A.txt           # 3 frame tập A kèm seed_tag= (không sửa tay)
│   ├── frames_B.txt           # 12 frame tập B kèm seed_tag= (không sửa tay)
│   ├── pred_A.json            # pre-label gốc trên tập A
│   ├── ref_A.json             # nhãn tham chiếu A đã sửa trên CVAT
│   ├── sweep.csv              # bảng quét lưới tham số Bước 6
│   ├── best_config.json       # cấu hình tốt nhất từ scripts/pp_sweep.py
│   ├── pred_B.json            # pre-label dùng best_config.json (B)
│   ├── compare/*.jpg          # ảnh BEV trước/sau 52 m (ít nhất ảnh dùng trong báo cáo)
│   └── ref_B.json             # nhãn Tập B 12 frame đã sửa xong (sản phẩm chính)
└── report/individual/<MSSV>_<HoTen>.md   # mục 1 đã commit trước sweep; đủ mục 1–5
```

### Kiểm tra trước khi nộp

Mở commit bài nộp trên GitHub và kiểm tra theo [SUBMISSION.md](SUBMISSION.md) và [CHECKPOINTS.md](CHECKPOINTS.md). Xác nhận số frame, các file cần nộp và lịch sử commit của bước 5b. Rà lại để tránh đưa dữ liệu gốc, mật khẩu hoặc SEED lên GitHub.

:::checklist{title="Checklist bàn giao cuối bài Lab Day 14" tone="success"}
- [ ] `frames_A.txt` và `frames_B.txt` giữ nguyên `seed_tag=` và số `0` đầu nếu mã có.
- [ ] Mục 1 báo cáo (`5b: dự đoán`) đã commit và push trước `sweep.csv`.
- [ ] Số hộp trong `ref_A.json` và `ref_B.json` khớp **Items** trên CVAT.
- [ ] Đã chạy `scripts/pp_prelabel.py --gt ref_B.json` và điền đủ 5 mục báo cáo.
- [ ] Repo `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel` đủ file theo `SUBMISSION.md`, đã bỏ qua `png_*/`.
- [ ] Đã nộp link commit `git rev-parse HEAD` lên LMS trước 12:00 trưa hôm sau.
:::
