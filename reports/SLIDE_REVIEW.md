# Kiểm tra trực quan báo cáo Stage 1 / Stage 2

## Phiên bản hiện tại: 30 slide

Bản rút gọn giữ kiến trúc, loss và đường training/validation loss của hai stage,
bổ sung benchmark lưu trữ 60 khởi tạo, rollout đến 168 giờ. Đã mở riêng ảnh
của từng slide trong `review_compact/`; sửa khoảng cách legend trên biểu đồ
RMSE/ACC/bias và mở trục skill xuống vùng âm để không cắt mất kết quả U10.
Notebook mới có 30 Markdown cells và speaker notes cùng thứ tự. Bản 61 slide
được giữ trong `archive_61_slides/`. Chi tiết nguồn số liệu, phạm vi kết luận
và ánh xạ slide nằm trong `COMPACT_REPORT_REVIEW.md`.

## Lịch sử rà soát: bản 61 slide

Đã bỏ slide 44 của bản 62 trang (đồ thị learning rate). Các trang cũ 45–62
chuyển thành 44–61. Bảng kiểm tra phía dưới lưu lại vòng đánh giá bản 62 trang,
không phải số trang của bản hiện tại.

Vòng sửa văn phong: đã mở riêng cả 61 ảnh trong `review_wording/`.
Tiêu đề và nội dung dùng tiếng Việt học thuật; các số liệu thực nghiệm được giữ.
Các công thức dùng Computer Modern, giữ cỡ chữ của renderer, xuất thêm SVG/PDF
trong `equations/`. Đây là MathText với phông Computer Modern, không phải một
phương trình vật lý bổ sung từ ảnh minh họa của người dùng.

Trong vòng rà soát đã sửa thêm nhãn tỷ số phương sai bị chạm caption,
heading ở bảng nhóm biến bị chạm bullet và khoảng cách title/body của các card.

Đã mở riêng từng ảnh của bản cũ (57 trang), sau đó mở riêng từng ảnh
của bản sửa (62 trang, 1200 × 675 px). Sau vòng sửa cuối, đã mở lại các
trang có thay đổi công thức, sơ đồ, typography, legend và nguồn số liệu.
Ảnh render đầy đủ nằm trong `review/slide_01.png` đến `slide_62.png`.

## Lỗi đáng kể ở bản cũ

- Slide 12: panel che hoàn toàn sơ đồ regridding.
- Slide 16, 20, 35: panel che công thức; không thể đọc nội dung toán học.
- Slide 18: công thức quá cao; chỉ số tổng bị panel đè lên.
- Slide 43: caption trong metric card chồng giá trị.
- Không có training-loss curves dù history.csv đã có trong export.
- Sơ đồ thiếu arrowheads; Stage 2 có các đường chéo không rõ luồng dữ liệu.
- Chữ nhỏ, khung quá lớn và font bị thay thế thành serif khi xuất PDF.
- Kết luận về bão hòa và attribution vượt quá bằng chứng hiện có.

## Nguồn dữ liệu

- `evidence/stage1_history.csv`: 25 epoch, sao từ export Stage 1 baseline.
- `evidence/stage2_history.csv`: 4 epoch, sao từ export spherical corrector.
- Best validation trong history khớp metadata checkpoint đến sai số < 1e-10.
- RMSE/ACC/bias/variance: `results/stage_smoke_tests` của submission.
- Kích thước internal grid được đối chiếu source backend: 181 × 360.
- Biểu đồ phổ ở slide 33 là minh họa; không được xem là phổ đo của forecast.

SHA-256:

```text
db7232930235a14c555ec7d2d80c8c4fb8ee30bb71d42ee9aa68f7552c0a5e3e  stage1_history.csv
f170bc69fe032f8cbb49e7961d4d95fe85104ac93f8e788cf8b057eb0b8ba36a  stage2_history.csv
```

## Đánh giá từng slide của bản sửa

| Slide | Nội dung và bằng chứng | Định dạng / hình / biểu đồ đã kiểm tra |
|---:|---|---|
| 01 | Tên đề tài, sinh viên, hai stage | Logo thật từ template; font Arial; kiểm tra title không tràn |
| 02 | Outline năm phần | Các nhãn và mô tả cùng hàng; bỏ khẳng định bão hòa ở outline |
| 03 | Động lực nghiên cứu | Body 18 pt; các dòng wrap nằm trong panel |
| 04 | Ánh xạ operator | Render công thức giữ aspect ratio, tách khỏi ba khối giải thích |
| 05 | Sáu mục tiêu | Text wrap trong card; nhãn G1–G6 tương phản rõ |
| 06 | Đóng góp dự án | Tăng body font; không để khung chiếm hết trang với chữ li ti |
| 07 | Pipeline end-to-end | Arrowheads chỉ rõ chiều; kiểm tra nhãn leakage |
| 08 | Phân cách dữ liệu | Logo, title, subtitle và số trang màu sáng |
| 09 | Nguồn, cadence, resolution | Giá trị đọc được; không làm tràn card có tên nguồn dài |
| 10 | 6 + 5×13 = 71 channels | Tách số kênh, nhóm biến và pressure range |
| 11 | Train 1983–2018, valid 2019, test 2020 | Timeline rõ; 95 cửa sổ là protocol đầy tháng, không phải N của smoke test |
| 12 | Regridding | Dựng lại; sơ đồ không bị panel che; mũi tên rõ |
| 13 | State và tendency normalization | Công thức không chạm card; s và z giải thích riêng |
| 14 | RMSE/ACC/bias/variance | Nêu lead 6 giờ, K=1; metric không đồng nhất về ý nghĩa |
| 15 | Phân cách phương pháp | Title/contrast/số trang |
| 16 | Neural operator | Công thức phía trên; panel phía dưới, không chồng |
| 17 | Hình học cầu | Label dài có đủ chiều rộng; hai phương pháp phân biệt |
| 18 | SHT | Chỉ số tổng đầy đủ; thêm khoảng cách giữa hai tổng |
| 19 | SFNO block | Dựng lại nhánh spectral và residual/local; sơ đồ chức năng được ghi rõ |
| 20 | Residual tendency | Công thức không bị panel che; bullet dễ đọc |
| 21 | Standardized tendency MSE | Render giữ đúng aspect ratio; chỉ số c không bị che |
| 22 | Channel và area weighting | Panel đủ chỗ cho wrap; bottom note không che bullet |
| 23 | Training stack | Backend version hai dòng không chồng subtitle |
| 24 | Phân cách Stage 1 | Title, subtitle và contrast |
| 25 | Backbone Stage 1 | 71→192→71; arrowheads rõ; nhận một state |
| 26 | Tensor shapes | Sửa internal grid thành 181×360, đối chiếu backend |
| 27 | Hyperparameters Stage 1 | Epoch hiển thị 25, checkpoint index 24 để tránh lệch convention |
| 28 | Optimization/model selection | Bullet không bị bottom note đè lên |
| 29 | Loss Stage 1 thực đo | 25 điểm train/valid; axis, legend, best-point annotation; khớp CSV |
| 30 | Stage 1 vs persistence | Các metric đọc được; chỉ kết luận cho smoke-test 6 giờ |
| 31 | RMSE vật lý S1/S2 | Thay bảng nhỏ bằng 9 panels có đơn vị riêng; axes độc lập |
| 32 | Hội tụ Stage 1 | Zoom 10 epoch cuối và mức giảm/epoch; sửa title; chưa chứng minh plateau |
| 33 | Spectral bottleneck | Có nhãn minh họa; không trình bày như energy đo từ model |
| 34 | Giả thuyết giới hạn | Đổi title để không khẳng định nguyên nhân nhân quả |
| 35 | Phân cách Stage 2 | Logo và số trang tương phản |
| 36 | Residual corrector equation | Panel không che equation; phần giữ/bổ sung rõ |
| 37 | Physical forcing | 7 channel từ 5 nhóm; đây là proxy thiên văn, không phải radiation measurement |
| 38 | Corrector architecture | 220 covariates + 2 geometry = 222; E84/L4; full-resolution |
| 39 | Luồng Stage 2 | Dựng lại đường có arrowheads; 71/142/222 tách rõ; stop-gradient ghi nhãn |
| 40 | Transfer và zero-init | Sửa zero output thành trước optimizer update đầu tiên |
| 41 | Fine-tuning timeline | Dùng epoch 1–2 / 3–4; best epoch 4, checkpoint index 3 |
| 42 | Hyperparameters Stage 2 | Caption checkpoint khớp metadata; 1 block + decoder được mở |
| 43 | Loss Stage 2 thực đo | Train/valid 4 epoch; baseline S1; unfreeze boundary; legend không che đường loss |
| 44 | Learning rate | Ghi rõ epoch-end LR; tail inactive trong hai epoch đầu |
| 45 | Gradient | Mean/max trước clipping; log scale; clip threshold; dữ liệu finite |
| 46 | Phân cách kết quả | Logo, title, subtitle và số trang |
| 47 | Aggregate comparison | Metric card không chồng chữ; phân biệt validation với smoke-test |
| 48 | Priority RMSE reduction | 9/9 biến ưu tiên giảm RMSE; tỷ lệ tính từ CSV |
| 49 | 71-channel distribution | 46 tốt hơn / 25 tệ hơn; median 0,69%; không che regression |
| 50 | Nhóm biến | Sửa nhận định: U/V/T trung vị âm; không quy lợi ích cho forcing khi thiếu ablation |
| 51 | Pressure heatmap | 65 cells có số, pressure và colorbar; màu dương/tốt, âm/tệ rõ |
| 52 | ACC/variance | Label axes, legend và bottom note; variance không tốt hơn ở mọi biến |
| 53 | Bias vật lý | Thay bảng thiếu đơn vị bằng 9 panels; dấu và đơn vị đầy đủ |
| 54 | Giả thuyết đóng góp | Thứ tự là suy luận, không phải kết quả ablation |
| 55 | Giới hạn tính hợp lệ | Thêm chưa xác minh khởi tạo ghép cặp; N=12/stage |
| 56 | Ablation đề xuất | Ghi là việc còn thiếu, không trình bày như đã chạy |
| 57 | Hướng cải tiến loss/sampling | Đề xuất nghiên cứu; không đưa số cải thiện giả |
| 58 | Hướng kiến trúc | Các hướng thử tiếp; chưa khẳng định hiệu quả trên dự án |
| 59 | Bài học | Text wrap nằm trong card; tránh suy luận performance từ số dòng code |
| 60 | Tái lập | Đường dẫn artifacts đúng; kết quả audit/type check kỹ thuật |
| 61 | Kết luận | Bỏ khẳng định corrector tốt nhất; ghi scope K=1 và hạn chế extreme skill |
| 62 | Q&A | Logo, title, màu sáng; số trang đúng |

## Giới hạn cần giữ khi thuyết trình

Smoke-test có 12 khởi tạo/stage. Persistence của hai stage khác nhau; chưa
xác minh cùng các initialization timestamps. Vì vậy tỷ lệ giảm RMSE smoke-test
là mô tả kết quả hai lần đánh giá, không phải phép kiểm định ghép cặp.
Các slide kết quả có chú thích điều này, và speaker notes cũng lưu nguồn/scope.

History chỉ chứa 25 epoch Stage 1 và 4 epoch Stage 2. Loss Stage 1 vẫn giảm ở
cuối run. Spectral cutoff là giả thuyết về giới hạn representation; chưa có
ablation số mode để xác lập nó là nguyên nhân bão hòa.

PDF được kiểm tra bounding boxes: không phát hiện span chữ nằm ngoài trang.
Kiểm tra đó không thay thế đọc ảnh; các ảnh từng slide đã được mở trực tiếp.
