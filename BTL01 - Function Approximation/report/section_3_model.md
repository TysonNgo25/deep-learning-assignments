# 3. Mô hình và cài đặt

## 3.1. Kiến trúc mạng

Bài toán nhận một điểm đầu vào \(\mathbf{x}=(x_1,x_2)\) và dự đoán một giá trị vô hướng \(\hat y\), xấp xỉ mật độ Gaussian hai chiều đã cho. Mô hình được cài đặt bằng PyTorch dưới dạng một lớp kế thừa `torch.nn.Module`; các phép biến đổi tuyến tính sử dụng `torch.nn.Linear`.

Với \(L\) tầng ẩn, phép lan truyền thuận được viết:

\[
\mathbf{h}_0=\mathbf{x}, \qquad
\mathbf{h}_\ell=\tanh(W_\ell\mathbf{h}_{\ell-1}+\mathbf{b}_\ell),
\quad \ell=1,\ldots,L,
\]
\[
\hat y=W_{\mathrm{out}}\mathbf{h}_L+b_{\mathrm{out}}.
\]

Tầng đầu ra không dùng hàm kích hoạt để mô hình dự đoán trực tiếp giá trị thực. Cấu hình baseline dùng hai tầng ẩn, mỗi tầng 16 nơ-ron (`hidden = [16, 16]`). Cùng một lớp MLP cho phép thay đổi danh sách số nơ-ron để thực hiện các cấu hình thực nghiệm B và C mà không phải viết lại kiến trúc.

## 3.2. Huấn luyện và chọn checkpoint

Dữ liệu đầu vào được chuyển sang kiểu `float32`, tương thích với trọng số mặc định của PyTorch. Mỗi batch được lấy bằng `DataLoader` có xáo trộn; hàm mất mát là sai số bình phương trung bình:

\[
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2.
\]

Baseline sử dụng Adam với learning rate \(10^{-3}\), batch size 32 và tối đa 1000 epoch. Mỗi epoch, mô hình cập nhật trọng số trên tập train rồi chuyển sang chế độ đánh giá để tính validation loss trong `torch.no_grad()`. Early stopping dừng khi validation loss không cải thiện trong 100 epoch liên tiếp. Trọng số có validation loss thấp nhất được khôi phục và lưu làm checkpoint; tập test không tham gia chọn epoch hay cấu hình.

Seed huấn luyện được đặt trước khi khởi tạo mạng và xáo trộn batch. Các thí nghiệm dùng các seed 0, 1, 2 theo baseline. Thời gian được đo bằng `time.perf_counter()` quanh vòng lặp huấn luyện. Mỗi lần chạy có thể lưu log CSV gồm `epoch`, `train_loss`, `val_loss`; checkpoint lưu trọng số tốt nhất cùng kiến trúc, seed, epoch tốt nhất, validation loss và cấu hình huấn luyện.

## 3.3. Đánh giá

Sau khi chọn cấu hình bằng validation, mô hình được đánh giá trên tập test độc lập bằng:

\[
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|,
\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
\]

MSE, MAE và RMSE được tính trên thang \(y\) gốc; không chuẩn hóa nhãn. Hàm đánh giá tạo đồ thị giá trị thật–dự đoán với đường tham chiếu \(y=\hat y\), đồng thời tạo hai bề mặt 3D đặt cạnh nhau: hàm Gaussian thật và bề mặt do MLP dự đoán. Các con số và hình kết quả cuối cùng sẽ được bổ sung sau khi các thí nghiệm hoàn tất và phải truy xuất được về log tương ứng.

## 3.4. Hiện thực trong mã nguồn

- `src/model.py`: định nghĩa `MLP`, nhận danh sách kích thước tầng ẩn và tên hàm kích hoạt.
- `src/train.py`: tạo `DataLoader`, huấn luyện với Adam/SGD, log loss từng epoch, early stopping, checkpoint theo validation loss và đo thời gian.
- `src/evaluate.py`: tính MSE/MAE/RMSE, suy luận và vẽ đồ thị \(y\)–\(\hat y\), bề mặt thật–xấp xỉ.

Các kết quả thực nghiệm, cấu hình được chọn cuối cùng và số liệu kiểm thử sẽ được điền sau khi pipeline chạy; bản thảo này không giả định trước kết quả.
