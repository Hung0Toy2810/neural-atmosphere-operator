# Bản báo cáo rút gọn — 30 slide

Bản cũ 61 slide được giữ tại `archive_61_slides/`. Bản mới giữ kiến trúc, loss, training curves và bổ sung benchmark rollout đến 7 ngày. Notebook và speaker notes khớp thứ tự slide.

Thời lượng mục tiêu: 12 phút 10 giây, chưa tính chuyển slide.

## Thay đổi

- Bỏ slide chuyển phần, mục lục dài, bảng kích thước tensor riêng, cấu hình lặp và danh sách hướng nghiên cứu dàn trải.
- Gộp thông tin lưới Stage 1 vào kiến trúc; corrector E84/L4 và lưới gốc vào sơ đồ Stage 2.
- Thay phần kết quả tập trung smoke test bằng benchmark gốc 60 khởi tạo, 6–168 giờ. Không trộn hai phép kiểm thử.
- Sửa nhận định sai rằng chưa test rollout dài; train K=1 vẫn có thể test autoregressive 28 bước.
- Phổ thực nghiệm Z500 có vùng dư/thiếu năng lượng; không dùng nhận định cũ rằng không suy hao hay nổ phổ.
- Không dùng các giá trị training loss từng bị gắn nhãn validation trong báo cáo benchmark cũ; validation lấy từ history đã kiểm tra.

## Thứ tự mới

| Slide | Nội dung | Nguồn | Giây |
|---:|---|---|---:|
| 1 | Neural Atmosphere Operator | Slide cũ 1 | 20 |
| 2 | Bài toán dự báo khí quyển | Slide cũ 4 | 20 |
| 3 | Nguồn dữ liệu ERA5 / WeatherBench2 | Slide cũ 9 | 20 |
| 4 | Các biến khí tượng đầu vào | Slide cũ 10 | 20 |
| 5 | Chia dữ liệu và hai phép kiểm thử | Mới / gộp | 20 |
| 6 | Chuẩn hóa state và tendency | Slide cũ 13 | 20 |
| 7 | Đọc các chỉ số đánh giá | Mới / gộp | 20 |
| 8 | Đặc điểm của lưới khí quyển trên mặt cầu | Slide cũ 17 | 20 |
| 9 | Cấu trúc khối SFNO | Slide cũ 19 | 20 |
| 10 | Loss function | Slide cũ 21 | 20 |
| 11 | Kiến trúc Stage 1 | Slide cũ 25 | 20 |
| 12 | Cấu hình huấn luyện Stage 1 | Slide cũ 27 | 20 |
| 13 | Stage 1 — training và validation loss | Slide cũ 29 | 35 |
| 14 | Nguyên lý thiết kế Stage 2 | Slide cũ 36 | 20 |
| 15 | Static và analytic forcing | Slide cũ 37 | 20 |
| 16 | Luồng dữ liệu đầy đủ của Stage 2 | Slide cũ 39 | 20 |
| 17 | Fine-tuning theo hai giai đoạn | Slide cũ 41 | 20 |
| 18 | Stage 2 — training và validation loss | Slide cũ 43 | 35 |
| 19 | Benchmark rollout: 28 bước đến 7 ngày | Mới / gộp | 20 |
| 20 | RMSE theo thời hạn dự báo | Mới / gộp | 35 |
| 21 | ACC theo thời hạn dự báo | Mới / gộp | 35 |
| 22 | Kết quả tại 7 ngày: Stage 1 và Stage 2 | Mới / gộp | 40 |
| 23 | MSE skill so với persistence và climatology | Mới / gộp | 30 |
| 24 | Bias khi rollout: sai lệch trung bình có dấu | Mới / gộp | 30 |
| 25 | Phổ Z500: năng lượng không được giữ đồng đều | Mới / gộp | 40 |
| 26 | Kiểm tra trên toàn bộ 71 kênh | Mới / gộp | 30 |
| 27 | Giới hạn và phạm vi kết luận | Mới / gộp | 20 |
| 28 | Thí nghiệm tiếp theo: ưu tiên kiểm chứng | Mới / gộp | 20 |
| 29 | Kết luận | Mới / gộp | 30 |
| 30 | Cảm ơn và hỏi đáp | Slide cũ 61 | 10 |

## Phạm vi bằng chứng

Các CSV tại `evidence/rollout_2020/` được lọc từ bản export, chỉ giữ Stage 1, Stage 2 và baselines. Dữ liệu gốc không bị sửa. Biểu đồ tính từ CSV và spectral arrays lưu trữ, không chạy inference mới. Tỷ số phổ chỉ có Z500 và bậc 0–179. MSE skill được tính lại từ RMSE model và baseline, hiển thị phần trăm. CI là CI của từng model, chưa phải CI cho chênh lệch ghép cặp. Metadata gốc lưu test year và chỉ số mẫu, chưa có đầy đủ timestamp/hash checkpoint. Evaluator cũ dùng cos(vĩ độ), không phải exact latitude-cell-boundary weights như evaluator mới.
