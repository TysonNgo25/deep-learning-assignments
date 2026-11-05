# 6. Kết luận

Nháp ngày 06/11/2026 (việc theo kế hoạch ngày 05/11), người viết: Huỳnh Ngọc Trúc. Mọi số trong mục này đã có ở mục 1.5, 4 và 5. Nguồn ghi ở cuối mục.

## 6.1. Kết quả đạt được

Nhóm đã viết bằng PyTorch một MLP có số tầng ẩn và số nơ-ron tùy chỉnh, cùng vòng train/validation tự viết có dừng sớm (early stopping) và lưu checkpoint theo val loss [2], [3]. Pipeline này dùng cho cả 81 lượt chạy của ba nhóm thực nghiệm. Mỗi lượt ghi log loss theo epoch thành CSV, và mọi bảng, hình trong báo cáo được sinh lại từ các log đó.

Với hàm Gaussian, miền và ngân sách 1000 epoch của bài này, ba nhóm thực nghiệm cho các kết quả sau:

1. **Số mẫu train (A).** Tăng từ 200 lên 2000 mẫu làm val MSE trung bình giảm từ 1.22 × 10⁻⁵ xuống 1.68 × 10⁻⁷, khoảng 72.5 lần. Sai số vẫn giảm ở 2000 mẫu, nên chưa thấy điểm bão hòa trong khoảng đã thử.
2. **Độ rộng một tầng ẩn (B).** Mạng 1 nơ-ron có val MSE 2.92 × 10⁻³, tức RMSE khoảng 0.054, bằng 20% đỉnh 0.269 của hàm: mạng thiếu khớp (underfitting) rõ. Lên 64 nơ-ron, val MSE còn 2.81 × 10⁻⁶. Xu hướng giảm này phù hợp với định lý xấp xỉ vạn năng [6]: MLP một tầng ẩn đủ rộng xấp xỉ được hàm liên tục trên miền đóng tới độ chính xác tùy ý, dù định lý không nói cần bao nhiêu nơ-ron hay tối ưu có tìm ra trọng số đó không. Train và val MSE sát nhau ở mọi độ rộng (Bảng 4.3), nên nhóm không thấy quá khớp (overfitting) theo nghĩa train error giảm còn val error tăng [5]. Đoạn 8 đến 10 nơ-ron đi ngang và dao động mạnh giữa các seed.
3. **Số tầng ẩn (C).** Thêm tầng ẩn thứ hai giảm val MSE khoảng 27 lần (1.19 × 10⁻⁵ xuống 4.35 × 10⁻⁷). Tầng thứ ba và thứ tư không giảm thêm rõ rệt: 4 tầng chỉ thấp hơn 2 tầng 2.5 × 10⁻⁸, nhỏ hơn nhiều so với độ lệch chuẩn 1.85 × 10⁻⁷ của chính cấu hình 4 tầng qua ba seed. Thời gian train trung bình tăng từ 15.24 giây (1 tầng) lên 25.72 giây (4 tầng).

Mô hình cuối được chọn chỉ theo val MSE: mạng baseline 2 tầng ẩn × 16 nơ-ron, tanh, train với 2000 mẫu, seed 1 (337 tham số). Trên 100 điểm test, checkpoint đạt MSE 8.65 × 10⁻⁸, MAE 2.16 × 10⁻⁴, RMSE 2.94 × 10⁻⁴ (Bảng 1.3). RMSE này bằng khoảng 0.11% đỉnh hàm. Đồ thị y–ŷ và bề mặt xấp xỉ ở mục 5.3 cho thấy mạng tái hiện được đỉnh hẹp quanh gốc tọa độ và phần mặt gần 0 ở xa đỉnh.

Trong các cấu hình đã thử, thêm dữ liệu giảm sai số nhiều hơn thêm nơ-ron hay thêm tầng. Mạng 2 tầng × 16 nơ-ron với 2000 mẫu (val MSE 1.68 × 10⁻⁷) tốt hơn mạng 4 tầng (4.10 × 10⁻⁷) và mạng 64 nơ-ron (2.81 × 10⁻⁶), hai mạng này cùng train với 1000 mẫu. So sánh này chỉ là thứ hạng giữa các cấu hình đã chạy. Nhóm chưa chạy các tổ hợp như 4 tầng với 2000 mẫu, nên chưa tách được phần đóng góp của từng yếu tố.

## 6.2. Hạn chế

- **Ba seed.** Mỗi cấu hình chỉ có ba lượt chạy. Độ lệch chuẩn mô tả được biến động nhưng chưa đủ để kiểm định các chênh lệch nhỏ, ví dụ giữa C = 2 và C = 4 hay giữa B = 8, 9, 10.
- **Tập test đã được xem trước khi chọn.** Runner C tính sai số test cho 12 lượt theo yêu cầu đề, trước khi nhóm chọn mô hình cuối. Test không tham gia bước chọn nào, nhưng số test của mô hình cuối không phải ước lượng trên tập chưa ai xem.
- **Ngân sách epoch.** Nhiều lượt chạy đủ 1000 epoch mà chưa kích hoạt dừng sớm, gồm cả ba seed của C = 1 và C = 2. Một phần sai số vì vậy có thể do mạng chưa hội tụ hết, không chỉ do kiến trúc hay số mẫu.
- **Dữ liệu không nhiễu.** Nhãn tính đúng từ công thức, nên thực nghiệm không cho biết mạng chịu nhiễu đo đến đâu, và cũng ít có cơ hội quan sát quá khớp.
- **Chỉ đánh giá trong miền.** Mọi điểm train, val, test và lưới vẽ bề mặt đều nằm trong [−2, 2] × [−4, 4]. Báo cáo không có kết luận nào về dự đoán ngoài miền.
- **Môi trường tính toán.** Cùng cấu hình và seed, checkpoint train trên máy khác cho val MSE 1.08 × 10⁻⁷ thay vì 1.05 × 10⁻⁷ như log A (mục 1.5). File `requirements.txt` chỉ ghi phiên bản tối thiểu nên chưa bảo đảm trùng log đến từng chữ số giữa các máy.

## 6.3. Hướng cải tiến

- Tăng số seed lên 5 đến 10 cho các cấu hình có chênh lệch nhỏ và báo thêm khoảng tin cậy.
- Thêm nhiễu Gaussian vào nhãn train với vài mức độ lệch chuẩn để xem B và C có xuất hiện quá khớp không.
- Tăng giới hạn epoch hoặc dùng lịch giảm learning rate, để tách sai số do kiến trúc khỏi sai số do dừng train sớm theo ngân sách.
- So sánh các mạng có cùng số tham số nhưng khác độ sâu, để tách ảnh hưởng của độ sâu khỏi ảnh hưởng của số tham số.
- Chạy kết hợp số mẫu với kiến trúc (ví dụ 3 hoặc 4 tầng với 2000 mẫu) theo một protocol chọn bằng validation đã định trước.
- Sinh một tập test mới với seed khác và chỉ đánh giá mô hình cuối trên đó, để có ước lượng test chưa bị xem trước.
- Ghim phiên bản thư viện trong `requirements.txt` để log tái lập trùng nhau giữa các máy.

## 6.4. Nguồn số liệu

- Kết quả A, B: `results/A/summary.csv`, `results/B/summary.csv` (mục 4.2).
- Kết quả C và thời gian train: `results/C/summary_mean_std.csv`, `results/C/table_C_mean_std.csv` (mục 4.3).
- Mô hình cuối: `results/final/final_metrics.json`, checkpoint `results/final/selected_model.pt` (mục 1.5, 5.3).

## Tài liệu tham khảo

[1] Đề bài "Bài tập xấp xỉ hàm số dùng mạng MLP" (`MLP_Exercise.pdf`, `MLP_Exercise - Yeu Cau.pdf`), học phần Học sâu, kèm `Sample_Code.html` và `Sample_Report.pdf` chỉ dùng để tham khảo cách tổ chức.

[2] A. Paszke và cộng sự, "PyTorch: An Imperative Style, High-Performance Deep Learning Library", *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, tr. 8024–8035, 2019. Tài liệu API: https://pytorch.org/docs/stable/ (`torch.nn.Module`, `torch.nn.Linear`, `torch.nn.MSELoss`, `torch.utils.data.DataLoader`, `torch.optim.Adam`).

[3] L. Prechelt, "Early Stopping – But When?", trong G. B. Orr, K.-R. Müller (biên tập), *Neural Networks: Tricks of the Trade*, LNCS 1524, Springer, 1998, tr. 55–69.

[4] D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", *International Conference on Learning Representations (ICLR)*, 2015. arXiv:1412.6980.

[5] I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016. Chương 6 (mạng truyền thẳng sâu) và mục 7.8 (dừng sớm).

[6] K. Hornik, M. Stinchcombe, H. White, "Multilayer feedforward networks are universal approximators", *Neural Networks*, tập 2, số 5, tr. 359–366, 1989.

[7] C. R. Harris và cộng sự, "Array programming with NumPy", *Nature*, tập 585, tr. 357–362, 2020.

[8] J. D. Hunter, "Matplotlib: A 2D Graphics Environment", *Computing in Science & Engineering*, tập 9, số 3, tr. 90–95, 2007.

## Phụ lục. Lệnh tái lập

Chạy trong Anaconda Prompt, đứng ở thư mục `BTL01 - Function Approximation`, sau khi cài môi trường theo `README.md`. Các lệnh chạy theo thứ tự. Các bước sinh dữ liệu và huấn luyện đọc tham số từ `configs/baseline.yaml`. Bảng 6.1 ghi lệnh và file mà mỗi bước sinh ra.

Bảng 6.1. Lệnh sinh lại dữ liệu, kết quả và hình của báo cáo.

| Bước | Lệnh | Sinh ra |
| --- | --- | --- |
| 1 | `python -m src.data` | `data/train_pool.csv`, `val.csv`, `test.csv`, `stats.csv` |
| 2 | `python -m scripts.plot_data` | Hình 2.1 đến 2.3 trong `report/figures/` |
| 3 | `python -m experiments.run_A` | `results/A/logs/`, `summary.csv`, Hình 4.1 |
| 4 | `python -m experiments.run_B` | `results/B/logs/`, `summary.csv`, Hình 4.2 |
| 5 | `python -m experiments.run_C` | `results/C/logs/`, `summary.csv`, `summary_mean_std.csv`, Hình 4.3 đến 4.5 |
| 6 | `python -m experiments.export_C` | `results/C/table_C_mean_std.csv`, `.md` (kiểm 12 log rồi xuất bảng) |
| 7 | `python -m scripts.run_final` | `results/final/selected_model.pt`, `final_metrics.json` |
| 8 | `python -m scripts.plot_final_evaluation` | Hình 5.1, 5.2 trong `results/final/figures/` |
| 9 | `python -m scripts.build_report` | `report/BTL01_Nhom05.md` (ghép các mục báo cáo) |

Bước 7 từ chối chạy nếu `results/final/final_metrics.json` đã tồn tại, để test không bị đánh giá lần thứ hai. Muốn chạy lại bước này thì phải xóa file JSON một cách có chủ ý. Các bước 3 đến 5 mất từ vài phút đến vài chục phút trên CPU tùy máy. Thời gian train ghi trong cột `elapsed_seconds_mean` của `results/A/summary.csv`, `results/B/summary.csv` và cột `train_time_s` của `results/C/summary.csv`.
