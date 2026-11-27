# Rubric Day 14 — Pre-label 3D bằng PointPillars

Điểm chính 100. Điểm cộng tách riêng, tối đa +10 (xem [bonus/README.md](bonus/README.md)). Mọi bằng chứng phải nằm trong repo nộp (xem [SUBMISSION.md](SUBMISSION.md)); coach không mở CVAT cá nhân của bạn.

## Điều kiện bắt buộc

Coach checkout đúng commit bạn nộp trên LMS rồi dùng `check_student.py` và SEED của buổi lab để kiểm tra `d14_<ID>/frames_A.txt` và `frames_B.txt`. Khi chạy checker, coach trỏ `--lab` vào thư mục `scripts/` của repo đề. Nếu kết quả là **CÓ VẤN ĐỀ** (sai SEED, sai mã học viên, hoặc danh sách frame bị sửa tay), các mục 1–5 bên dưới không được chấm.

## Điểm chính

| # | Tiêu chí | Điểm | Bằng chứng | Mất điểm khi |
|---|---|---:|---|---|
| 1 | Nhãn tham chiếu tập A hợp lý | 20 | `ref_A.json`, so với `pred_A.json` | Thiếu file; `ref_A.json` gần như trùng `pred_A.json` (chỉ xóa, không thêm box bị sót); box sai lớp hoặc sai kích thước rõ ràng theo guideline Day 13 |
| 2 | Sweep và lựa chọn cấu hình có lý do | 20 | `sweep.csv`, `best_config.json`, mục 2 báo cáo | Thiếu file; báo cáo không nhắc kết quả leave-one-out; chọn cấu hình không giải thích |
| 3 | Dự đoán 5b và đối chiếu ở bước 8 | 20 | Mục 1 và 3 báo cáo, ảnh `compare/*.jpg` | Không có dự đoán; lịch sử commit cho thấy mục 1 được thêm hoặc sửa sau khi đã có `sweep.csv` (thứ tự trình bày trong báo cáo không đủ làm bằng chứng); không có ảnh; không nói dự đoán đúng hay sai. Đoán sai nhưng giải thích đúng vẫn được đủ điểm |
| 4 | Tập B đã sửa xong | 30 | `ref_B.json` (đủ 12 frame), `pred_B.json`, mục 4 báo cáo | Thiếu `ref_B.json` hoặc thiếu frame: mất cả mục. Mọi box trong mọi frame của `ref_B.json` trùng box tương ứng của `pred_B.json` (so x, y, z, kích thước, yaw, lớp; bỏ qua phần `meta` và các thuộc tính `score`/`pp_id`): coi là nộp thẳng box của model, mất cả mục. Chi phí sửa bằng 0 tự nó không đủ để kết luận điều này. Còn box thừa/sót rõ ràng trong 50 m: trừ theo mức độ |
| 5 | Báo cáo trung thực | 10 | Báo cáo | Phóng đại % giảm; gọi chi phí sửa là "tiết kiệm X giờ công"; không nêu giới hạn (3 frame tune, model KITTI) |
| | **Tổng** | **100** | | |

## Điểm cộng (tối đa +10)

Chỉ chấm khi phần bắt buộc đã đủ. Hướng 1 (đo thao tác sửa nhãn): tối đa +7. Hướng 2 (PointPillars chạy trong CVAT): tối đa +3. Mức điểm chi tiết ở [bonus/README.md](bonus/README.md).
