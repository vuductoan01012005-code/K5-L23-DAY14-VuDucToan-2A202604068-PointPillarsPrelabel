# Checkpoints Day 14

Lệnh đầy đủ ở [README.md](README.md) mục 2. `<ID>` là mã học viên của bạn.

| # | Việc làm | Sản phẩm | Cần hiểu | Tự kiểm |
|---|---|---|---|---|
| 0 | Cài đặt, chạy `scripts/pp_selftest.py` | Môi trường chạy được | Model chạy CPU, phủ 360° bằng 4 lượt xoay | Selftest in `TẤT CẢ OK` |
| 1 | Chọn frame theo SEED | `frames_A.txt`, `frames_B.txt` | Frame phụ thuộc mã học viên và SEED, không làm trước được | Mỗi file có dòng `seed_tag=`; A có 3 frame, B có 12 frame |
| 2 | Pre-label tập A | `pred_A.json`, `png_A/` | Model KITTI hay sót box và không biết Truck/Bus | Mở ảnh BEV, đếm sơ bộ box thừa và box sót |
| 3 | Đưa lên CVAT cá nhân | Task `D14-<ID>-A` | Box pre-label kèm thuộc tính `pp_id`, `score`, `warn` | Task mở được ở chế độ 3D, có cuboid |
| 4 | Sửa tay tập A | Nhãn đã Save trên CVAT | Xóa, thêm, sửa đều có giá trong công thức chi phí | Số Items thay đổi sau khi thêm/xóa; đã bấm Save |
| 5 | Lấy tham chiếu A | `ref_A.json` | Đây là "đáp án" của riêng bạn để tune | `scripts/pp_prelabel.py ... --gt d14_<ID>/ref_A.json` in ra số xóa/thêm/sửa |
| 5b | Đoán trước | Mục 1 báo cáo | Thêm box sót tốn 1 điểm, xóa box thừa chỉ 0.3 | Đã commit và push mục 1 báo cáo trước khi chạy sweep |
| 6 | Tune | `sweep.csv`, `best_config.json` | Tune trên 3 frame dễ overfit | Đọc phần leave-one-out: cấu hình chọn có ổn định không |
| 7 | Pre-label và sửa tập B | `pred_B.json`, task `D14-<ID>-B` đã sửa | Tập B tách khỏi tập A | Đã sửa đủ 12 frame và Save |
| 8 | So sánh trước/sau | `compare/*.jpg`, mục 3 báo cáo | Chỉ chấm box trong 50 m; xanh lá đúng, vàng cần sửa, đỏ thừa, xanh dương bị sót | Biết tổng chi phí của bạn so với gốc; dự đoán 5b đúng hay sai |
| 9 | Xuất nhãn B và nộp | `ref_B.json`, repo nộp | Coach chấm từ file trong repo, không mở CVAT của bạn | Làm hết checklist trong [SUBMISSION.md](SUBMISSION.md); link trên LMS là link commit có SHA |
