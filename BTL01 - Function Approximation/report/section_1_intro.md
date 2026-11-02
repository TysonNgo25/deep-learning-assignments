# 1. Giới thiệu

Nháp ngày 02/11/2026 (việc theo kế hoạch ngày 01/11), người viết: Huỳnh Ngọc Trúc. Chỗ ghi [...] là thông tin chỉ có sau khi nhóm chọn mô hình cuối (03/11) và chốt tỷ lệ đóng góp (07/11).

## 1.1. Mục tiêu

Nhóm 5 dùng mạng perceptron nhiều tầng (MLP) viết bằng PyTorch để học lại một hàm mật độ Gaussian hai chiều từ các điểm lấy mẫu, rồi đo sai số thay đổi ra sao khi đổi số mẫu train, số nơ-ron và số tầng ẩn. Công thức của hàm đã biết trước, nên tại bất kỳ điểm nào trong miền nhóm cũng tính được giá trị đúng để đặt cạnh dự đoán của mạng. Dữ liệu không có nhiễu, vì vậy mọi chênh lệch giữa hai bề mặt là sai số của mạng.

Đề bài bắt buộc tự viết mô hình và vòng huấn luyện bằng PyTorch (`nn.Module`, `nn.Linear`, `DataLoader`, `MSELoss`, Adam) và không chấp nhận TensorFlow hay Keras. Code mẫu và báo cáo mẫu đi kèm đề viết bằng TensorFlow/Keras, dùng ma trận hiệp phương sai đơn vị và lấy tập test làm validation. Nhóm chỉ tham khảo cách chia hàm của chúng (sinh dữ liệu, đọc dữ liệu, vẽ 3D), phần mô hình và vòng train viết lại từ đầu.

## 1.2. Phát biểu bài toán

Đầu vào là điểm x = (x₁, x₂) lấy đều trong miền −2 ≤ x₁ ≤ 2, −4 ≤ x₂ ≤ 4. Nhãn là y = f(x), với f là mật độ Gaussian hai chiều có kỳ vọng μ = (0, 0) và ma trận hiệp phương sai C = [[0.3, −0.5], [−0.5, 2]] (công thức đầy đủ ở mục 2.1). Mạng nhận hai số x₁, x₂ và trả về một số thực ŷ. Huấn luyện nhằm đưa ŷ về gần y theo sai số bình phương trung bình (MSE).

Hàm đạt đỉnh f(μ) ≈ 0.269 tại gốc tọa độ rồi giảm nhanh khi ra xa, đến mức hơn 60% điểm lấy mẫu có y < 0.01. MSE của các mô hình vì thế rất nhỏ, trong 81 lần chạy của nhóm nằm trong khoảng từ cỡ 10⁻³ (mạng 1 nơ-ron) xuống cỡ 10⁻⁷ (baseline với 2000 mẫu train). Nhóm báo cáo thêm MAE và RMSE, hai chỉ số có cùng đơn vị với y, để đọc sai số so với đỉnh 0.269.

Dữ liệu gồm ba tập độc lập, sinh một lần với seed 42 và lưu thành CSV: pool train 2000 điểm, validation 500 điểm, test 100 điểm. Validation quyết định khi nào dừng sớm, giữ checkpoint nào và cấu hình nào tốt hơn. Tập test chỉ chạm tới một lần, khi đánh giá mô hình cuối, vì đề cấm dùng test để chọn siêu tham số.

Bảng 1.1. Ba nhóm thực nghiệm. Mỗi nhóm chỉ đổi một yếu tố, các tham số khác giữ theo baseline.

| Nhóm | Yếu tố thay đổi | Giá trị | Số lần chạy (3 seed) |
| --- | --- | --- | --- |
| A | Số mẫu train | 200, 400, ..., 2000 | 30 |
| B | Số nơ-ron của mạng 1 tầng ẩn | 1 đến 10, thêm 16, 32, 64 | 39 |
| C | Số tầng ẩn, mỗi tầng 16 nơ-ron | 1, 2, 3, 4 | 12 |

Baseline là mạng 2 tầng ẩn × 16 nơ-ron, kích hoạt tanh, đầu ra tuyến tính, Adam với learning rate 10⁻³, batch 32, tối đa 1000 epoch, dừng sớm khi val loss 100 epoch liên tiếp không giảm, và 1000 mẫu train cho B, C (bảng đầy đủ ở mục 3.3). Đề gợi ý B từ 1 đến 10 và cho phép mở rộng. Nhóm thêm 16, 32 và 64 nơ-ron để có những mạng đủ rộng mà xét chuyện quá khớp, vì với tối đa 10 nơ-ron mạng có thể chỉ thiếu khớp.

## 1.3. Cách nhóm giữ cho phép so sánh công bằng

Mọi tham số baseline và danh sách giá trị của A, B, C nằm trong một file `configs/baseline.yaml`. Các script thực nghiệm đều đọc file này, nên không lần chạy nào lệch tham số so với lần khác. Kiểm tra cột `config_json` trong 81 file log cho thấy cả 81 lần chạy dùng cùng optimizer, learning rate, batch, số epoch tối đa, patience và hàm kích hoạt.

Ở thực nghiệm A, tập 200 mẫu là 200 dòng đầu của pool, tập 400 mẫu là 400 dòng đầu, và cứ thế đến 2000. Hai cấu hình liền nhau chỉ khác phần mẫu thêm vào. Validation và test giữ nguyên cho cả ba nhóm thực nghiệm.

Mỗi cấu hình chạy với ba seed huấn luyện 0, 1, 2. Seed quyết định trọng số khởi tạo và thứ tự xáo mini-batch, nên kết quả báo cáo dưới dạng trung bình ± độ lệch chuẩn qua ba seed thay vì một lần chạy. Để kiểm tra tính tái lập, ngày 31/10 nhóm chạy lại cấu hình baseline (seed 0) bằng script riêng. Val loss của cả 1000 epoch trùng khớp với log A (A = 1000, seed 0) và log C (C = 2, seed 0) đã chạy trước đó, chênh lệch bằng 0.

## 1.4. Phân công và đóng góp

Mỗi thành viên giữ một phần code, tự chạy phần thực nghiệm của mình và viết phần báo cáo tương ứng. Người viết số liệu cũng là người có log sinh ra số liệu đó. Ngày 31/10 nhóm review chéo toàn bộ code, sửa lỗi runner ghi đè bảng tổng hợp khi chạy một phần cấu hình và lỗi chồng nhãn trục ở hình bề mặt.

Bảng 1.2. Thành viên, nhiệm vụ và tỷ lệ đóng góp.

| Thành viên | MSSV | Vai trò | Code phụ trách | Phần báo cáo | Đóng góp |
| --- | --- | --- | --- | --- | --- |
| Huỳnh Ngọc Trúc | 3123411312 | Trưởng nhóm, dữ liệu, ghép báo cáo | `src/data.py`, `scripts/plot_data.py`, `scripts/run_baseline.py`, `README.md`, `requirements.txt` | 1, 2, 6 + ghép, định dạng | [...] |
| Phạm Trần Minh Khang | 3123411143 | Lõi PyTorch | `src/model.py`, `src/train.py`, `src/evaluate.py` | 3 + đánh giá chung | [...] |
| Tạ Quang Trung | 3123411318 | Thực nghiệm A, B | `experiments/run_A.py`, `experiments/run_B.py`, `experiments/common.py` | 4.1, 4.2 + thảo luận A, B | [...] |
| Ngô Nguyễn Thành Nhân | 3123411205 | Thực nghiệm C, slide | `experiments/run_C.py`, `src/plots.py` | 4.3 + thảo luận C, hạn chế | [...] |

[Tỷ lệ đóng góp chốt ngày 07/11 theo việc thực tế, tổng 100%.]

## 1.5. Kết quả chính

[Điền sau 03/11: cấu hình được chọn theo validation, sai số của nó trên test (MSE, MAE, RMSE) và một câu kết luận cho mỗi nhóm A, B, C.]

## 1.6. Bố cục báo cáo

- Mục 2. Dữ liệu: công thức Gaussian, cách lấy mẫu và chia tập, hình và thống kê.
- Mục 3. Mô hình và cài đặt: kiến trúc MLP, công thức forward và loss, quy trình train, validation, checkpoint.
- Mục 4. Thực nghiệm: bảng siêu tham số, kết quả A, B, C kèm bảng và đồ thị.
- Mục 5. Thảo luận: giải thích xu hướng, trường hợp không như dự đoán, hạn chế và hướng cải tiến.
- Mục 6. Kết luận, tài liệu tham khảo và lệnh tái lập.
