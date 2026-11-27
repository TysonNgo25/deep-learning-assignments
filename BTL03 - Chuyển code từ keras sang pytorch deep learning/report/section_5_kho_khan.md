# 5. Khó khăn và cách xử lý

Người viết: Ngô Nguyễn Thành Nhân

## 5.1. Lỗi kỹ thuật gặp khi chuyển

Bảng 5.1 liệt kê các lỗi gặp trong quá trình chuyển, theo thứ tự gặp. Phần lớn không làm chương trình báo lỗi mà chỉ cho kết quả sai hoặc chạy chậm, nên khó phát hiện.

Bảng 5.1. Lỗi gặp khi chuyển và cách xử lý.

| Lỗi | Biểu hiện | Nguyên nhân | Cách xử lý |
| --- | --- | --- | --- |
| Train bằng CPU trên Kaggle | lần chạy GPU đầu tiên mất gần 12 giờ mà chưa xong, không có thông báo lỗi | hàm `summary()` truyền `device="cpu"` cho `torchinfo`, `torchinfo` gọi `model.to("cpu")` nên mô hình bị kéo khỏi GPU; `fit` đi theo thiết bị của mô hình | `summary()` dùng thiết bị đang chứa mô hình; `fit` in thiết bị ở dòng đầu; notebook chạy Kaggle kiểm mô hình còn ở `cuda` trước khi chạy và đặt giới hạn 2 giờ mỗi ô. Lần chạy lại mất 44 phút cho 7 notebook |
| Nhãn sai shape với MSE | loss không giảm như mong đợi | nhãn `(N,)`, đầu ra `(N, 1)`: `nn.MSELoss` broadcast thành ma trận N×N | đưa nhãn về `(N, 1)` (mục 3.1) |
| Mảng có stride âm | `torch.as_tensor` báo lỗi với dữ liệu Gender | `np.rot90` trả về mảng nhìn ngược (view) | `np.ascontiguousarray` trong `to_nchw` |
| Đường dẫn nhãn SegNet | không đọc được ảnh nhãn | `fn.replace('depth', 'labels')` của gốc thay cả tên thư mục `depth_hand/` | ghép đường dẫn nhãn riêng |
| Dữ liệu Gender xếp theo nhãn | chạy thử với 1.000 ảnh đầu chỉ có một lớp | file `.npy` xếp male trước, female sau | lấy mẫu rải đều khi chạy thử; khi train thật `shuffle=True` |
| Hai notebook ghi cùng một file | có thể mất dòng trong `so_sanh.csv` khi chạy 2 luồng trên 2 GPU | đọc-sửa-ghi file cùng lúc | khóa file `.lock` trong `ghi_so_sanh` |
| `jupyter nbconvert` chạy nhầm Python | `NoSuchKernel: python3` | máy có một Python 3.12 ngoài conda nằm trước trong PATH | gọi `python -m nbconvert` của env |

## 5.2. Những chỗ phải đọc kỹ mã nguồn Keras

Một số hành vi của Keras không thấy được trong code notebook mà nằm trong giá trị mặc định của thư viện. Muốn bản PyTorch giống bản gốc phải viết lại đúng các hành vi này:

- **Khởi tạo**: glorot uniform và bias 0 (PyTorch mặc định Kaiming uniform); LSTM có kernel hồi quy orthogonal và bias cổng quên bằng 1.
- **BatchNorm**: `momentum` của Keras (0.99) và PyTorch (0.01) định nghĩa ngược nhau; `epsilon` mặc định khác (1e-3 so với 1e-5).
- **LSTM**: cổng dùng `hard_sigmoid` trong Keras 2.0.9; PyTorch chỉ có sigmoid. Đây là chỗ duy nhất không giữ được.
- **"accuracy" với loss MSE**: Keras hiểu là `binary_accuracy`, so `round(dự đoán)` với nhãn (mục 3.10).
- **EarlyStopping**: dừng sau đúng `patience` epoch xấu liên tiếp và không khôi phục trọng số tốt nhất. Đã đối chiếu với hai log gốc.
- **ImageDataGenerator**: xoay quanh tâm, nội suy điểm gần nhất, lấp viền bằng giá trị biên; `torchvision` lấp bằng 0 nên phải viết lại bằng `scipy`.
- **`Reshape` trên tensor kênh cuối**: chỉ cắt lại dãy số trong bộ nhớ, không tách kênh. Chỗ này gây lỗi trong chính notebook SegNet gốc (mục 3.9).

## 5.3. Nhận xét về dữ liệu

- **Gender lệch lớp.** 78.85% ảnh là male ở cả train và val, nên đoán toàn "male" đã đạt 78.85%; log gốc epoch 1 có val accuracy đúng bằng số này. Bản PyTorch đạt 83.88% nhưng recall lớp female chỉ 35.74% (451/1262 ảnh), male 96.79%. Accuracy chung che mất điểm yếu này; với dữ liệu lệch lớp nên báo cáo thêm recall từng lớp hoặc F1.
- **Gender có ảnh hỏng.** 662/5968 ảnh val đen hoàn toàn (612 male, 50 female). Mô hình đoán male cho cả 662 ảnh. Bài giữ nguyên các ảnh này để so sánh công bằng với gốc.
- **Không chuẩn hóa đầu vào.** Gender, Emotion và SegNet đưa ảnh giá trị 0–255 thẳng vào mạng, như gốc. Bản PyTorch giữ nguyên để so sánh công bằng. Với lr nhỏ (1e-4) mạng vẫn học được, nhưng loss những batch đầu rất lớn; chuẩn hóa về [0, 1] là lựa chọn nên làm nếu không cần bám bản gốc.
- **Nhãn SegNet có mức xám lạ.** 4 mức xám ngoài 7 lớp chiếm 0,06% số pixel train và 0,37% số pixel test; bản PyTorch xử lý chúng giống hệt Keras (mục 3.9).
- **Phát hiện xe chia ngẫu nhiên không seed.** Mỗi lần chạy gốc là một cách chia train/test khác, nên số 95.23% của gốc không lặp lại được chính xác.
