# Nội dung phần của Khang — Slide 4–6

## Slide 4 — MLP xấp xỉ Gaussian 2D

- Đầu vào: \((x_1,x_2)\); đầu ra: một giá trị \(\hat y\).
- Kiến trúc baseline: `2 → 16 → 16 → 1`.
- Hai tầng ẩn dùng `tanh`; tầng đầu ra tuyến tính.
- Biến thể thực nghiệm thay đổi số nơ-ron (B) hoặc số tầng ẩn (C).
- Hình gợi ý: sơ đồ MLP kèm công thức Gaussian của đề.

## Slide 5 — Huấn luyện và chọn mô hình

- PyTorch: `nn.Module`, `nn.Linear`, `DataLoader`, `MSELoss`.
- Baseline: Adam, learning rate \(10^{-3}\), batch size 32, tối đa 1000 epoch.
- Mỗi epoch: cập nhật trên train, đo validation loss ở chế độ `eval()` và `no_grad()`.
- Early stopping: patience 100; khôi phục checkpoint có validation loss thấp nhất.
- Seed huấn luyện: 0, 1, 2; ghi loss theo epoch và thời gian bằng `time.perf_counter()`.
- Tập test được giữ riêng, không dùng để chọn epoch hoặc cấu hình.
- Hình gợi ý: sơ đồ luồng train → validation → checkpoint tốt nhất.

## Slide 6 — Đánh giá và minh chứng

- Chỉ số: MSE, MAE, RMSE trên thang nhãn gốc.
- Đồ thị: \(y\) thật so với \(\hat y\), cùng đường tham chiếu \(y=\hat y\).
- Bề mặt 3D: Gaussian thật đặt cạnh bề mặt MLP dự đoán.
- Sau khi chạy xong: chèn kết quả test của mô hình đã chọn và dẫn tên log/checkpoint.
- Không điền số liệu trước khi có log thực nghiệm.
