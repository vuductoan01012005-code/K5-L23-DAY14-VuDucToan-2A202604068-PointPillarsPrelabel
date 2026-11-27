# Quy định Day 14

## Làm bài

- Bài cá nhân. Mỗi người làm trên frame của mình (theo mã học viên và SEED) và CVAT cá nhân của mình.
- Không sửa tay `frames_A.txt`, `frames_B.txt`. Không dùng SEED khác SEED coach công bố.
- Mọi box nộp phải được người kiểm. Không nộp thẳng box của model làm `ref_B.json`.

## Dùng AI

- Được dùng AI để hỏi về lệnh, lỗi cài đặt, giải thích tham số và sửa câu chữ báo cáo.
- Không dùng AI để viết dự đoán 5b hay kết luận đối chiếu thay bạn; đó là phần được chấm.
- Nhãn trên CVAT do bạn sửa. Nếu dùng công cụ tự động nào khác ngoài pre-label của lab, ghi rõ trong báo cáo.

## Hợp tác và sao chép

- Được trao đổi cách dùng tool và cách sửa box trên CVAT.
- Không chép `ref_*.json`, `sweep.csv` hay báo cáo của người khác. Frame mỗi người khác nhau nên file chép sẽ không khớp frame của bạn.
- Không xin, không dùng nhãn Labels-VF của coach.

## Nộp muộn và sửa sau deadline

- Deadline: 12:00 trưa ngày hôm sau ngày lab (GMT+7), xem [SUBMISSION.md](SUBMISSION.md).
- Bài được chấm theo commit có SHA bạn nộp trên LMS. Coach checkout đúng SHA đó; commit đẩy lên sau mà không cập nhật link LMS không được tính.
- Link commit nộp hoặc cập nhật trên LMS sau deadline là bài nộp muộn: chấm commit đó và trừ điểm theo quy định chung. Commit có thời điểm sau deadline cũng tính là muộn dù link LMS cập nhật trước.
- Không force-push hay xóa commit đã nộp. Nếu coach không mở được SHA trên LMS, bài coi như chưa nộp đúng.

## Bảo mật

- Không commit mật khẩu CVAT. `scripts/vf_cvat.py` đọc mật khẩu từ biến môi trường `CVAT_PASSWORD` hoặc hỏi khi chạy.
- Không commit dữ liệu `Data-VF` hay file `.env`. Không chia sẻ link và mật khẩu dữ liệu ra ngoài lớp.
