# Điểm cộng Day 14 (tùy chọn, tối đa +10)

Chỉ làm sau khi đã có đủ phần bắt buộc theo checklist trong [`SUBMISSION.md`](../SUBMISSION.md) (`ref_B.json` của 12 frame tập B và báo cáo). Mỗi hướng làm **tối đa 45 phút**; hết giờ thì dừng và viết lại mình đã làm tới đâu. Điểm cộng không bù cho phần bắt buộc còn thiếu.

## Hướng 1 (chính, tối đa +7): đo thao tác sửa nhãn

Công thức chi phí của lab `0.3·xóa + 1.0·thêm + 0.5·sửa` là do người ra đề đặt, chưa được đo. Hướng này dùng log thao tác thật của bạn để kiểm tra công thức đó.

Log gốc của CVAT chỉ ghi click vào nút. Nó không ghi click trên vùng vẽ và không ghi phím tắt ([cvat#7988](https://github.com/cvat-ai/cvat/issues/7988)). Userscript trong thư mục này ghi thêm hai thứ đó.

**Cài đặt:**
1. Cài Tampermonkey cho trình duyệt. Tạo script mới, dán nội dung `cvat-telemetry-addon.user.js` vào rồi lưu.
2. Mở job `D14-<ID>-B` trên CVAT và sửa nhãn như bình thường. Script tự gửi event lên `/api/events` của CVAT và giữ một bản sao trong trình duyệt.
3. Sửa xong, mở Console (F12) và gõ `__d14Telemetry.download()`. Trình duyệt sẽ tải về file `d14-telemetry-job<id>-….json`. Hãy tải trước khi đóng tab.
4. Tính metric: `python bonus/telemetry_metrics.py d14-telemetry-job*.json --csv d14_<ID>/telemetry.csv`

| Mức | Nộp gì | Điểm |
|---|---|---|
| Cơ bản | Bảng metric của phiên sửa tập B (số click, số phím tắt, độ dài phiên) và ghi chú đã sửa bao nhiêu box | +3 |
| Khá | Thêm khoảng thời gian và khoảng cách giữa các click, phân tích phím tắt nào dùng nhiều, ghi theo từng frame (một phiên cho mỗi frame) | +5 |
| Wow | Đối chiếu với số xóa/thêm/sửa do `scripts/pp_prelabel.py --gt` in ra. Ví dụ: thêm một box tốn bao nhiêu click hoặc giây so với sửa một box? Trọng số 0.3/1/0.5 có hợp lý không, nên đổi thành bao nhiêu? Phải nêu rõ giới hạn của dữ liệu | +7 |

**Ràng buộc:**
- Metric thao tác không tính vào điểm chính. Không có lợi gì khi cố click ít hơn cho đẹp số.
- Chỉ ghi log của chính bạn, trên tài khoản của bạn. Không cài script lên máy người khác. Muốn gộp dữ liệu với bạn khác thì phải hỏi người đó trước.
- Một người sửa vài frame chỉ là bằng chứng sơ bộ. Không viết là "tiết kiệm X giờ công".
- Tắt script khi không làm lab (trong Tampermonkey, gạt tắt).

## Hướng 2 (thử thách mở, tối đa +3): model 3D trong CVAT

Lệnh `scripts/vf_cvat.py prelabel` đã đẩy pre-label vào CVAT, và đó là phần bắt buộc, không tính điểm cộng. Thử thách ở đây là chạy PointPillars **bên trong** CVAT dưới dạng serverless function (Nuclio), để CVAT tự sinh cuboid.

- Hướng dẫn từng bước: mở [`huong-dan-model-3d.html`](huong-dan-model-3d.html) bằng trình duyệt.
- Dùng chính CVAT Docker bạn đang chạy, chỉ cần chạy lại kèm `-f components/serverless/docker-compose.serverless.yml` để có Nuclio.
- Các model mẫu của CVAT chỉ hỗ trợ ảnh 2D. Với task 3D, hộp thoại Automatic annotation báo "No models available", vì CVAT đã từ chối mở khóa ô này (bộ chọn sẽ hiện cả model 2D), nên phải gọi API: `POST /api/lambda/requests {"function": "<tên>", "task": <id>, "threshold": 0.3}`.
- Dashboard Nuclio (cổng 8070) không có đăng nhập. Chỉ bind vào `127.0.0.1`, không mở ra mạng.
- Cuboid vẽ dạng shape có thể không xuất ra khi export ([cvat#9144](https://github.com/cvat-ai/cvat/issues/9144)). Hãy kiểm tra file export.

| Kết quả | Điểm |
|---|---|
| Function chạy được trên task của bạn, có video ngắn, và có số so khớp với `pred_*.json` offline (bao nhiêu box trùng tâm trong 0.5 m) | +3 |
| Chưa chạy được, nhưng có báo cáo nêu đã làm những bước nào, vướng ở bước nào, log lỗi ra sao, và nếu làm tiếp sẽ thử gì | +1 |
