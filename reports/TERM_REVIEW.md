# Rà soát thuật ngữ trong slide

Phiên bản hiện tại có 30 slide. Bản rút gọn tiếp tục dùng các thuật ngữ đã
rà soát bên dưới, bao gồm chữ trong biểu đồ mới. Phân biệt MSE skill với
phần trăm giảm RMSE; spectral ratio với spatial variance ratio; số bước train
với thời hạn test. Benchmark lưu trữ 60 khởi tạo không bị gộp với smoke test
12 khởi tạo. Phần dưới là lịch sử rà soát bản 61 slide.

Phạm vi: chữ trong slide, sơ đồ và ảnh biểu đồ. Giữ nguyên công thức,
cấu hình huấn luyện, kết quả và các nguồn dữ liệu của bản 61 trang.
Đã xem riêng ảnh của cả 61 trang trong `review_terms/`.

| Cách dùng ở bản trước | Cách dùng sau khi sửa |
|---|---|
| Gia số từ Stage 1 | Stage 1 tendency |
| Gia số gần nhất | Recent tendency |
| Bộ hiệu chỉnh | Corrector |
| Phần hiệu chỉnh | Correction |
| Ghép các trường | Concatenate |
| Điều kiện và vĩ độ | Forcing và latitude features |
| Hệ số gia số | Tendency scale |
| Suy luận trên GPU | Inference trên GPU |
| Thí nghiệm tách thành phần | Ablation study |
| Chuyển trọng số | Transfer learning |
| Liên kết dư | Residual connection |
| Gộp ô | Pooling |
| Nội suy song tuyến để tăng lưới | Bilinear interpolation |
| Độ lệch phương sai, tại biểu đồ vẽ tỷ số | \|variance ratio − 1\| |

Giữ các cách gọi tiếng Việt đã rõ nghĩa trong ngữ cảnh: nhiệt độ, độ ẩm riêng,
địa thế vị, vĩ độ, kinh độ, diện tích ô lưới, chuẩn hóa và hàm băm.
Không thêm các từ hồi cứu, truy hồi hoặc truy xuất để thay cho inference.

Các ảnh biểu đồ được dựng lại từ mã nguồn; thay nhãn trước khi xuất PNG/PDF.
Series số liệu lấy từ các CSV cũ. Nhãn objective loss thống nhất giữa ô số liệu
và biểu đồ; dùng loss cho giá trị số, loss function cho hàm mục tiêu.

Sơ đồ Stage 2 vẫn ghi: history input 142 kênh, backbone 71 kênh,
corrector 222 kênh đầu vào và 71 kênh đầu ra.
Slide learning rate vẫn được loại khỏi báo cáo. Chỉ giữ Stage 1 và Stage 2.
