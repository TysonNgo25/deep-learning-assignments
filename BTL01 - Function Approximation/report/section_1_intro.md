# 1. Giới thiệu

## 1.1. Mục tiêu

Nhóm 5 dùng mạng perceptron nhiều tầng (MLP) viết bằng PyTorch để học lại một hàm mật độ Gaussian hai chiều từ các điểm lấy mẫu, rồi đo sai số thay đổi ra sao khi đổi số mẫu train, số nơ-ron và số tầng ẩn. Công thức của hàm đã biết trước, nên tại bất kỳ điểm nào trong miền nhóm cũng tính được giá trị đúng để đặt cạnh dự đoán của mạng. Dữ liệu không có nhiễu, vì vậy mọi chênh lệch giữa hai bề mặt là sai số của mạng.

Đề bài [1] bắt buộc tự viết mô hình và vòng huấn luyện bằng PyTorch [2] (`nn.Module`, `nn.Linear`, `DataLoader`, `MSELoss`, Adam [4]) và không chấp nhận TensorFlow hay Keras. Code mẫu và báo cáo mẫu đi kèm đề viết bằng TensorFlow/Keras, dùng ma trận hiệp phương sai đơn vị và lấy tập test làm validation. Nhóm chỉ tham khảo cách chia hàm của chúng (sinh dữ liệu, đọc dữ liệu, vẽ 3D), phần mô hình và vòng train viết lại từ đầu.

## 1.2. Phát biểu bài toán

Đầu vào là điểm x = (x₁, x₂) lấy đều trong miền −2 ≤ x₁ ≤ 2, −4 ≤ x₂ ≤ 4. Nhãn là y = f(x), với f là mật độ Gaussian hai chiều có kỳ vọng μ = (0, 0) và ma trận hiệp phương sai C = [[0.3, −0.5], [−0.5, 2]] (công thức đầy đủ ở mục 2.1). Mạng nhận hai số x₁, x₂ và trả về một số thực ŷ. Huấn luyện nhằm đưa ŷ về gần y theo sai số bình phương trung bình (MSE).

Hàm đạt đỉnh f(μ) ≈ 0.269 tại gốc tọa độ rồi giảm nhanh khi ra xa, đến mức hơn 60% điểm lấy mẫu có y < 0.01. MSE của các mô hình vì thế rất nhỏ, trong 81 lần chạy của nhóm nằm trong khoảng từ cỡ 10⁻³ (mạng 1 nơ-ron) xuống cỡ 10⁻⁷ (baseline với 2000 mẫu train). Nhóm báo cáo thêm MAE và RMSE, hai chỉ số có cùng đơn vị với y, để đọc sai số so với đỉnh 0.269.

Dữ liệu gồm ba tập độc lập, sinh một lần với seed 42 và lưu thành CSV: pool train 2000 điểm, validation 500 điểm, test 100 điểm. Validation quyết định khi nào dừng sớm, giữ checkpoint nào và cấu hình nào tốt hơn. Tập test không tham gia lựa chọn nào trong số đó, vì đề cấm dùng test để chọn siêu tham số. Nhóm chỉ tính sai số test ở hai chỗ. Chỗ thứ nhất là từng lượt chạy của thực nghiệm C, vì đề yêu cầu báo test error theo số tầng. Chỗ thứ hai là mô hình cuối, sau khi đã chọn xong theo validation.

Bảng 1.1 liệt kê ba nhóm thực nghiệm và số lần chạy của mỗi nhóm.

Bảng 1.1. Ba nhóm thực nghiệm. Mỗi nhóm chỉ đổi một yếu tố, các tham số khác giữ theo baseline.

| Nhóm | Yếu tố thay đổi | Giá trị | Số lần chạy (3 seed) |
| --- | --- | --- | --- |
| A | Số mẫu train | 200, 400, ..., 2000 | 30 |
| B | Số nơ-ron của mạng 1 tầng ẩn | 1 đến 10, thêm 16, 32, 64 | 39 |
| C | Số tầng ẩn, mỗi tầng 16 nơ-ron | 1, 2, 3, 4 | 12 |

Baseline là mạng 2 tầng ẩn × 16 nơ-ron, kích hoạt tanh, đầu ra tuyến tính, Adam với learning rate 10⁻³, batch 32, tối đa 1000 epoch, dừng sớm khi val loss 100 epoch liên tiếp không giảm, và 1000 mẫu train cho B, C (Bảng 3.1 ở mục 3.3). Đề gợi ý B từ 1 đến 10 và cho phép mở rộng. Nhóm thêm 16, 32 và 64 nơ-ron để có những mạng đủ rộng mà xét chuyện quá khớp, vì với tối đa 10 nơ-ron mạng có thể chỉ thiếu khớp.

## 1.3. Cách nhóm giữ cho phép so sánh công bằng

Mọi tham số baseline và danh sách giá trị của A, B, C nằm trong một file `configs/baseline.yaml`. Các script thực nghiệm đều đọc file này, nên không lần chạy nào lệch tham số so với lần khác. Kiểm tra cột `config_json` trong 81 file log cho thấy cả 81 lần chạy dùng cùng optimizer, learning rate, batch, số epoch tối đa, patience và hàm kích hoạt.

Ở thực nghiệm A, tập 200 mẫu là 200 dòng đầu của pool, tập 400 mẫu là 400 dòng đầu, và cứ thế đến 2000. Hai cấu hình liền nhau chỉ khác phần mẫu thêm vào. Validation và test giữ nguyên cho cả ba nhóm thực nghiệm.

Mỗi cấu hình chạy với ba seed huấn luyện 0, 1, 2. Seed quyết định trọng số khởi tạo và thứ tự xáo mini-batch, nên kết quả báo cáo dưới dạng trung bình ± độ lệch chuẩn qua ba seed thay vì một lần chạy. Để kiểm tra tính tái lập, ngày 31/10 nhóm chạy lại cấu hình baseline (seed 0) bằng script riêng. Val loss của cả 1000 epoch trùng khớp với log A (A = 1000, seed 0) và log C (C = 2, seed 0) đã chạy trước đó, chênh lệch bằng 0.

## 1.4. Phân công và đóng góp

Mỗi thành viên giữ một phần code, tự chạy phần thực nghiệm của mình và viết phần báo cáo tương ứng. Người viết số liệu cũng là người có log sinh ra số liệu đó. Ngày 31/10 nhóm review chéo toàn bộ code, sửa lỗi runner ghi đè bảng tổng hợp khi chạy một phần cấu hình và lỗi chồng nhãn trục ở hình bề mặt. Bảng 1.2 ghi phần việc và tỷ lệ đóng góp của từng người.

Bảng 1.2. Thành viên, nhiệm vụ và tỷ lệ đóng góp.

| Thành viên | MSSV | Vai trò | Code phụ trách | Phần báo cáo | Đóng góp |
| --- | --- | --- | --- | --- | --- |
| Huỳnh Ngọc Trúc | 3123411312 | Trưởng nhóm, dữ liệu, ghép báo cáo | `src/data.py`, `scripts/plot_data.py`, `scripts/run_baseline.py`, `scripts/build_report.py`, `README.md`, `requirements.txt` | 1, 2, 4.1, 6 + ghép, định dạng | 25% |
| Phạm Trần Minh Khang | 3123411143 | Lõi PyTorch | `src/model.py`, `src/train.py`, `src/evaluate.py`, `scripts/run_final.py`, `scripts/plot_final_evaluation.py`, `scripts/plot_mlp_architecture.py` | 3 + đánh giá chung | 25% |
| Tạ Quang Trung | 3123411318 | Thực nghiệm A, B | `experiments/run_A.py`, `experiments/run_B.py`, `experiments/common.py` | 4.2 + thảo luận A, B | 25% |
| Ngô Nguyễn Thành Nhân | 3123411205 | Thực nghiệm C, slide | `experiments/run_C.py`, `experiments/export_C.py`, `src/plots.py` | 4.3 + thảo luận C, hạn chế; dựng slide | 25% |

Tổng tỷ lệ đóng góp của bốn thành viên là 100%.

## 1.5. Kết quả chính

Nhóm xếp hạng cả 27 cấu hình của A, B, C theo val MSE trung bình qua ba seed. Đứng đầu là baseline 2 tầng ẩn × 16 nơ-ron train với đủ 2000 mẫu của pool (A = 2000), val MSE 1.68 × 10⁻⁷ ± 5.89 × 10⁻⁸. Cấu hình tốt nhất của B (64 nơ-ron) chỉ đạt 2.81 × 10⁻⁶, của C (4 tầng) đạt 4.10 × 10⁻⁷ với 1000 mẫu train. Trong ba seed của A = 2000, seed 1 có val MSE thấp nhất (1.05 × 10⁻⁷). Nhóm train lại đúng cấu hình và seed này, lưu trọng số ở epoch có val loss nhỏ nhất vào `results/final/selected_model.pt` (337 tham số), rồi nạp lại file đó để tính sai số trên test một lần. Bảng 1.3 ghi sai số của checkpoint này trên validation và test.

Bảng 1.3. Sai số của mô hình cuối (checkpoint ở epoch 980). Nguồn: `results/final/final_metrics.json`.

| Tập | MSE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Validation (500 điểm) | 1.081954e-07 | 2.450498e-04 | 3.289307e-04 |
| Test (100 điểm) | 8.651794e-08 | 2.163969e-04 | 2.941393e-04 |

RMSE trên test bằng khoảng 0.11% giá trị đỉnh 0.269 của hàm. Con số test cần đọc kèm một giới hạn. Runner C đã tính sai số trên chính 100 điểm test này trước khi nhóm chọn mô hình, nên đây là điểm đo trên một tập nhóm đã từng xem kết quả, không phải tập test hoàn toàn mới. Sai số test của C không tham gia vào việc chọn cấu hình.

Checkpoint cuối train trên một máy khác máy đã chạy A, nên val MSE của nó (1.08 × 10⁻⁷ ở epoch 980) lệch nhẹ so với log A (1.05 × 10⁻⁷ ở epoch 999). Hai log khác nhau ngay từ epoch 1, ở chữ số có nghĩa thứ tám. Nhóm chạy lại cấu hình này trên một máy thứ ba và thu được log trùng log A ở cả 1000 epoch, tức chênh lệch đến từ môi trường tính toán của máy train checkpoint, không đến từ code. Log của lần chạy lại lưu ở `results/final/logs/rerun_A_2000_seed1.csv`.

Ở A, val MSE giảm đều qua cả mười cỡ mẫu, từ 1.22 × 10⁻⁵ với 200 mẫu xuống 1.68 × 10⁻⁷ với 2000 mẫu, khoảng 72.5 lần. Ở B, val MSE giảm từ 2.92 × 10⁻³ với 1 nơ-ron xuống 2.81 × 10⁻⁶ với 64 nơ-ron và chững lại ở 9, 10 nơ-ron. Train và val MSE của B sát nhau ở mọi độ rộng, chưa cho thấy quá khớp rõ ràng. Ở C, tầng ẩn thứ hai đưa val MSE từ 1.19 × 10⁻⁵ xuống 4.35 × 10⁻⁷. Tầng thứ ba và thứ tư không giảm thêm rõ rệt, trong khi thời gian train trung bình tăng từ 15.2 lên 25.7 giây.

## 1.6. Bố cục báo cáo

- Mục 2. Dữ liệu: công thức Gaussian, cách lấy mẫu và chia tập, hình và thống kê.
- Mục 3. Mô hình và cài đặt: kiến trúc MLP, công thức forward và loss, quy trình train, validation, checkpoint.
- Mục 4. Thực nghiệm: bảng siêu tham số, kết quả A, B, C kèm bảng và đồ thị.
- Mục 5. Thảo luận: giải thích xu hướng, trường hợp không như dự đoán, hạn chế và hướng cải tiến.
- Mục 6. Kết luận, tài liệu tham khảo và lệnh tái lập.
