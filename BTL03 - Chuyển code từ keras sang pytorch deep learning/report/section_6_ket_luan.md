# 6. Kết luận

Người viết: Huỳnh Ngọc Trúc

**Kết quả đạt được.**

- Chuyển đủ 11 notebook Keras sang PyTorch, giữ nguyên thứ tự mục, kiến trúc, siêu tham số và tiền xử lý của bản gốc. Hai notebook không có Keras (Iris dùng scikit-learn, `colab_getting_started`) không chuyển.
- Số tham số của cả 10 notebook có mô hình đúng như dự tính: 8 mô hình khớp tuyệt đối với Keras, CNN+LSTM lệch đúng 1.024 do bias kép của LSTM, SegNet khớp cả số tham số train được lẫn tổng.
- Bảy trong tám notebook đo bằng accuracy chênh dưới 1.5 điểm so với gốc, không lệch theo một chiều cố định. Hai bản Training Utilities gần trùng gốc (+0.36 và +0.03 điểm).
- Phát hiện và giải thích được một lỗi trong notebook SegNet gốc (`Reshape` + `Permute` viết cho kênh trước nhưng chạy với kênh sau). Bản PyTorch tính đúng và cao hơn gốc 3.76 điểm pixel accuracy.
- `pytorch/common.py` gom vòng huấn luyện, đánh giá, callback và lưu kết quả, nên 11 notebook dùng chung một cách huấn luyện và một bảng so sánh `results/so_sanh.csv`.

**Bài học khi chuyển Keras sang PyTorch.** PyTorch bắt người viết làm rõ những gì Keras làm ngầm: kích thước đầu vào mỗi lớp, chế độ train/eval, thiết bị, cách khởi tạo, cách tính chỉ số. Chính những chỗ ngầm định đó là nơi dễ sai nhất, và phần lớn lỗi không báo gì mà chỉ cho kết quả lệch hoặc chạy chậm. Kiểm số tham số bằng `assert`, đối chiếu log từng epoch với bản gốc và in thiết bị đang dùng là ba cách phát hiện lỗi hiệu quả nhất trong bài.

**Hạn chế.**

- Mỗi notebook chỉ chạy một lần với seed 0, nên chênh lệch dưới 1 điểm không đủ để kết luận bản nào tốt hơn.
- LSTM của PyTorch không có `hard_sigmoid`, khác bản gốc ở hàm kích hoạt cổng.
- Bản gốc Func_apxm_MLP chỉ có hình, số so sánh phải đọc gần đúng từ hình.
- Phần cứng và phiên bản thư viện giữa hai bản khác nhau, nên không so được tốc độ.
- Bài giữ nguyên các điểm yếu của dữ liệu gốc (Gender lệch lớp và có ảnh đen; ảnh không chuẩn hóa) để so sánh công bằng, chưa thử bản đã xử lý dữ liệu.
