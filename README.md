# Day 14 — Pre-label 3D bằng PointPillars pre-trained (VF LiDAR)

**Bài cá nhân.** Cuối buổi lý thuyết sáng, bạn có 10 phút cài đặt và 35 phút chạy thử. Buổi chiều, bạn làm tập A và B. Nộp link commit GitHub (có SHA) lên LMS trước **12:00 trưa hôm sau** (GMT+7).

Tài liệu: [SUBMISSION.md](SUBMISSION.md) (nộp gì, nộp ở đâu) · [RUBRIC.md](RUBRIC.md) (chấm điểm) · [CHECKPOINTS.md](CHECKPOINTS.md) (tự kiểm từng bước) · [RULES.md](RULES.md) (quy định) · [report/individual/MSSV_HoTen.md](report/individual/MSSV_HoTen.md) (template báo cáo)

Chạy mô hình PointPillars đã huấn luyện trên KITTI ([zhulf0804/PointPillars](https://github.com/zhulf0804/PointPillars), MIT, commit `620e6b0`) trên dữ liệu VF của Day 13. Mô hình tạo cuboid dự đoán để bạn sửa trực tiếp trên CVAT. Bạn sẽ thử nhiều cấu hình trên tập A (sweep), rồi dùng cấu hình đã chọn cho tập B.
Code chạy bằng PyTorch trên CPU, không cần GPU và không cần biên dịch: hai op CUDA của upstream (voxelize, rotated NMS) được thay bằng bản Python thuần trong `vf_pp/upstream.py`. File của upstream giữ nguyên, không sửa dòng nào.

Chạy các lệnh từ gốc repo. Các công cụ dòng lệnh của lab nằm trong thư mục `scripts/`.

## 0. Cài đặt cuối buổi lý thuyết sáng (10 phút)

**Windows (PowerShell):**

```powershell
git clone https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel.git k5-day14-lab
cd k5-day14-lab
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_pp.py                    # tải code upstream + weights (~20 MB); không có git: python scripts/setup_pp.py --download-zip
$DataVF = "C:\duong-dan\Data-VF"
python scripts/pp_selftest.py --data "$DataVF"
```

**macOS / Linux (bash/zsh):**

```bash
git clone https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel.git k5-day14-lab && cd k5-day14-lab
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_pp.py                    # tải code upstream + weights (~20 MB); không có git: python scripts/setup_pp.py --download-zip
DataVF="/duong-dan/Data-VF"
python scripts/pp_selftest.py --data "$DataVF"
```
Đặt `DataVF` thành đường dẫn thực tế tới thư mục VF từ Day 13. Thư mục này phải chứa trực tiếp `sequence_01` đến `sequence_05`, mỗi sequence có đủ 60 frame LiDAR. Khi mở terminal mới, đặt lại biến trước khi chạy lệnh. Selftest in `TẤT CẢ OK` khi cài đặt thành công. Nếu dùng Docker, xem hướng dẫn trong `Dockerfile`.

## 1. Buổi giảng: 6 frame chung (mọi người giống nhau)

```bash
python scripts/pp_prelabel.py --data "$DataVF" --frames demo/frames_lec.txt --out out/lec.json --plot out/lec_png
python scripts/pp_prelabel.py --data "$DataVF" --frames demo/frames_lec.txt --set score_thr=0.4 --set min_points=5 --out out/lec_filtered.json --plot out/lec_filtered_png
```
Lệnh đầu lưu `out/lec.json` và 6 ảnh trong `out/lec_png/`. Lệnh thứ hai lưu `out/lec_filtered.json` và 6 ảnh trong `out/lec_filtered_png/`. Mở cùng một frame trong hai thư mục để đối chiếu các hộp qua 7 góc camera và hình nhìn từ trên xuống (BEV). Xem hộp nào được lọc là hộp thừa và vật thể thật nào có thể bị sót.

## 2. Lab buổi chiều: frame riêng theo mã học viên + SEED

SEED **chỉ được coach công bố lúc bắt đầu lab**. Thay `<ID>`, `<SEED>` và `<tên>` bằng giá trị thực tế. Với CVAT local, thay `<URL>` bằng `http://localhost:8080`. Cú pháp tùy chọn `[--org <SLUG>]` áp dụng khi dùng tổ chức CVAT.

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Chọn frame | `python scripts/vf_cvat.py day14 --student "<ID>" --seed "<SEED>" --data "$DataVF"` | `d14_<ID>/frames_A.txt` (3 frame tune), `frames_B.txt` (12 frame nộp) |
| 2. Pre-label A | `python scripts/pp_prelabel.py --data "$DataVF" --frames "d14_<ID>/frames_A.txt" --out "d14_<ID>/pred_A.json" --plot "d14_<ID>/png_A"` | box của cấu hình gốc + ảnh để xem |
| 3. Lên CVAT | `python scripts/vf_cvat.py prelabel --student "<ID>" --task A --data "$DataVF" --boxes "d14_<ID>/pred_A.json" --host "<URL>" --user "<tên>"` | task `D14-<ID>-A` có sẵn cuboid (thuộc tính `pp_id`, `score`, `warn`, `note`) |
| 4. Sửa tay A | Mở từng frame trên CVAT. Xóa hộp thừa, thêm vật bị sót và chỉnh hộp sai theo [hướng dẫn gán nhãn 3D Cuboid](docs/GUIDELINE.md). Khi vẽ hộp mới, chọn **Shape**. Bấm **Save**. | nhãn tham chiếu của bạn |
| 5. Lấy tham chiếu | `python scripts/vf_cvat.py reference --student "<ID>" --task A --host "<URL>" --user "<tên>"` | `d14_<ID>/ref_A.json` |
| 5b. Dự đoán trước sweep | Làm theo [lệnh ghi và đẩy dự đoán](#đẩy-dự-đoán-trước-sweep-bước-5b). | Mục 1 báo cáo đã được push, làm bằng chứng khi đối chiếu ở bước 8 |
| 6. Tune | `python scripts/pp_sweep.py --data "$DataVF" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" --grid score_thr=0.1,0.2,0.3,0.4,0.5 --grid min_points=0,5,10 --out "d14_<ID>/sweep.csv"` | `sweep.csv`, `best_config.json`, top 10, leave-one-out |
| 7. Pre-label B | `python scripts/pp_prelabel.py --data "$DataVF" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --out "d14_<ID>/pred_B.json"` rồi `python scripts/vf_cvat.py prelabel --student "<ID>" --task B --data "$DataVF" --boxes "d14_<ID>/pred_B.json" --host "<URL>" --user "<tên>"` | task `D14-<ID>-B` để sửa đủ 12 frame trên CVAT theo [hướng dẫn gán nhãn 3D Cuboid](docs/GUIDELINE.md). Khi thêm hộp, chọn **Shape**. |
| 8. Trước/sau | `python scripts/pp_compare.py --data "$DataVF" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" --tuned "d14_<ID>/best_config.json" --out "d14_<ID>/compare"` | Mức thay đổi chi phí sửa và ảnh so sánh từng frame. Đưa một ảnh vào báo cáo và giải thích vì sao kết quả khớp hoặc lệch dự đoán ở bước 5b. |
| 9. Xuất nhãn B, nộp | Làm theo [bước 9 bên dưới](#xuất-nhãn-và-đo-công-sửa-tập-b-bước-9). | `d14_<ID>/ref_B.json` và số đo cho mục 4 báo cáo |

### Đẩy dự đoán trước sweep (bước 5b)

Tạo repo cá nhân trống theo [mục Repo nộp](SUBMISSION.md#1-repo-nộp). Sao chép mẫu báo cáo thành `report/individual/<MSSV>_<HoTen>.md`. Ở mục 1, ghi tham số bạn dự đoán sẽ giảm công sửa và lý do dựa trên ảnh BEV tập A. Chạy các lệnh sau từ gốc repo. Mở GitHub và kiểm tra commit dự đoán đã xuất hiện trước khi chạy sweep.

```bash
git remote set-url origin "https://github.com/<tài-khoản>/<repo-cá-nhân>.git"
git add "report/individual/<MSSV>_<HoTen>.md"
git commit -m "5b: dự đoán"
git push -u origin main
```

### Xuất nhãn và đo công sửa tập B (bước 9)

Sửa đủ 12 frame trên CVAT và bấm **Save**. Xuất nhãn đã sửa bằng lệnh sau.

```bash
python scripts/vf_cvat.py reference --student "<ID>" --task B --host "<URL>" --user "<tên>"
```

Chỉ chạy lệnh đo sau khi lệnh xuất in **Đã ghi**. Nếu lệnh xuất báo lỗi, xử lý lỗi ngay để tránh đo bằng bản nhãn cũ.

```bash
python scripts/pp_prelabel.py --data "$DataVF" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --gt "d14_<ID>/ref_B.json"
```

Ghi số hộp cần xóa, thêm và sửa vào mục 4 báo cáo, rồi làm theo [hướng dẫn nộp bài](SUBMISSION.md).

### Đọc số đo công sửa

Script tính chi phí bằng `0.3·xóa + 1.0·thêm + 0.5·sửa`. Script chỉ tính hộp có tâm cách LiDAR tối đa 50 m trên mặt phẳng ngang. Hai hộp được ghép nếu độ chồng lấp khi nhìn từ trên (BEV IoU) đạt ít nhất 0,3.

Một hộp đã ghép vẫn tính là cần sửa nếu sai lớp, tâm lệch ngang quá 0,3 m, một kích thước lệch quá 0,2 m hoặc phương trục dài lệch quá 10°. Hai góc cách nhau 180° có cùng phương trục dài. Trên CVAT, vẫn căn đúng đầu xe và đặt đáy hộp sát mặt đường theo [hướng dẫn gán nhãn 3D Cuboid](docs/GUIDELINE.md#2-quy-tắc-cuboid).

### Xử lý lỗi CVAT

- Nếu `reference` báo có **Track**, mở task, vẽ lại các hộp đó bằng **Shape** rồi xóa Track. Bấm **Save** và chạy lại lệnh xuất.
- Nếu có nhiều task trùng tên, kiểm tra các ID trong thông báo. Giữ tên của task bạn sẽ xuất và đổi tên các task còn lại.
- Các lệnh đo chi phí cần nhãn của đủ frame trong tập đang chạy. Nếu thiếu frame, kiểm tra bạn đang mở đúng task A hoặc B, bấm **Save** rồi xuất lại.
- Khi chạy lại `prelabel`, dùng file dự đoán đã nạp lúc tạo task. Giữ thư mục `out/cvat-prelabel/` để script đối chiếu file. Script dừng nếu phát hiện file đã đổi. Nếu script báo chưa thể đối chiếu với task cũ, hãy tự kiểm file trước khi tiếp tục. `--replace` xóa task và toàn bộ nhãn đã sửa.

CVAT ở đây là CVAT cá nhân chạy Docker trên máy bạn (như Day 13). Mật khẩu CVAT đọc từ biến `CVAT_PASSWORD` hoặc được hỏi khi chạy.

## 3. Các tham số (`vf_pp/core.py:Config`)

| Nhóm | Tham số | Ý nghĩa |
|---|---|---|
| A. Đầu vào | `voxel_size` (0.16), `intensity_mode` clip/log/const, `intensity_scale`, `ground_mode` fixed/auto, `delta_z`, `passes` (4) | Đưa điểm VF về miền KITTI. Model chỉ nhìn 90° phía trước nên mỗi frame chạy 4 lượt xoay để phủ 360° |
| B. Đầu ra model | `score_thr` (0.1), `thr_car`/`thr_cyc`/`thr_ped`, `nms_thr`, `nms_pre`, `max_num` | Lọc theo độ tin cậy và NMS |
| C. Hậu xử lý VF | `range_m` (50), `min_points` (0), `merge_nms_thr`, `size_pad_m`, `truck_l_min`/`truck_h_min`, `bus_l_min`/`bus_h_min` | KITTI không có Truck/Bus: box Car dài/cao sẽ được đổi lớp theo luật kích thước |

Ghi đè nhanh: `--set KEY=VALUE`. Lưu cấu hình: sửa `best_config.json` rồi dùng `--config`.

## 4. Hệ trục

VF: x trái, y sau, z lên (phía trước xe là −y), yaw 0…2π tính từ +x, ngược chiều kim đồng hồ. KITTI: x trước, y trái. Phép đổi trục nằm trong `vf_pp/core.py` (`x_kitti = −y_vf`, `y_kitti = x_vf`).
Đã kiểm trên 6 frame của buổi giảng: với box khớp nhãn, trung vị sai lệch là 13 cm (tâm), 2.3° (yaw), dưới 6 cm (kích thước).

## Điểm cộng (tùy chọn)

Sau khi nộp xong phần bắt buộc, xem [`bonus/README.md`](bonus/README.md): đo thao tác sửa nhãn bằng userscript (hướng chính), hoặc thử chạy PointPillars trong CVAT (thử thách mở). Mỗi hướng tối đa 45 phút.

## 5. Lưu ý trung thực khi báo cáo

- Model học trên KITTI (LiDAR 64 tia, chỉ phía trước, 3 lớp). Trên VF nó bỏ sót nhiều Pedestrian/Cyclist và không biết Truck/Bus. Pre-label là để **giảm** công sửa, không thay người gán nhãn.
- 3 frame tune là ít. Hãy đọc phần leave-one-out của `scripts/pp_sweep.py` trước khi tin cấu hình "tốt nhất".
- Mọi box nộp phải được người kiểm. Không nộp thẳng box của model.

## Góp ý bài lab (30 giây)

Mở [form góp ý bài lab](https://docs.google.com/forms/d/e/1FAIpQLSd8Mh7YkGEn-hnaePgbli4KqJ8-rkzCrFOIjr3YyCcrDs1UHA/viewform?usp=publish-editor) và chọn **L2 · Day14** hoặc **L3 · Day14** theo lớp của bạn. Form không yêu cầu email, họ tên hoặc MSSV.
