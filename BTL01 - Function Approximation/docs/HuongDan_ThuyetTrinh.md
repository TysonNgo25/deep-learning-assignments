# BTL01 – Hướng dẫn thuyết trình và bản đồ code (Nhóm 5)

Dùng cho buổi trình bày 10–15 phút và phần vấn đáp. Thầy sẽ chỉ định ngẫu nhiên **một** thành viên trình bày toàn bộ 15 slide, nên cả 4 người phải nói được trọn bài từ slide 1 đến 15 và trả lời được mọi phần code, không chia phần theo người. Mọi đường dẫn tính từ thư mục `BTL01 - Function Approximation/`. Số dòng ghi theo code hiện tại, nếu file đổi thì tìm lại theo tên hàm.

## 1. Bài toán trong 30 giây

- Xấp xỉ hàm mật độ Gaussian 2 chiều `f(x)` với μ = (0, 0), C = [[0.3, −0.5], [−0.5, 2]], miền x₁ ∈ [−2, 2], x₂ ∈ [−4, 4].
- Dùng MLP viết bằng PyTorch: 2 đầu vào (x₁, x₂), 1 đầu ra (ŷ). Không dùng TensorFlow/Keras.
- Đỉnh của hàm khoảng 0.269 (tại gốc), nên MSE cỡ 10⁻⁷ nghĩa là RMSE cỡ 3 × 10⁻⁴, tức khoảng 0.11% đỉnh. Luôn nói kèm RMSE để người nghe hình dung được.
- Ba thực nghiệm, mỗi lần chỉ đổi một biến: **A** số mẫu train (200…2000), **B** số nơ-ron của 1 tầng ẩn (1…10, 16, 32, 64), **C** số tầng ẩn (1…4, mỗi tầng 16 nơ-ron). Mỗi cấu hình chạy 3 seed (0, 1, 2), báo cáo mean ± std.

## 2. Kịch bản trình bày (15 slide, khoảng 13 phút)

Slide nằm ở `slides/BTL01_Nhom05.pptx`, bản PDF ở `docs/BTL01_Nhom05_Slide.pdf`. Một người nói liền mạch cả bài, tổng khoảng 13 phút. Cột "Thời gian" là mức nên giữ để không vượt 15 phút.

| Slide | Nội dung | Thời gian | Ý phải nói được | Code / số liệu để mở khi bị hỏi |
| --- | --- | --- | --- | --- |
| 1 | Tiêu đề, thành viên | 0:30 | Tên đề, nhóm 5, 4 thành viên, đóng góp mỗi người 25% | – |
| 2 | Hàm Gaussian mục tiêu | 1:00 | Công thức, μ, C, det C = 0.35, đỉnh 0.269, vì sao MSE rất nhỏ | `src/data.py:25` `gaussian_2d` |
| 3 | Tập train / val / test | 1:00 | Pool train 2000, val 500, test 100, seed 42; test chỉ dùng đánh giá cuối | `src/data.py:45` `generate_splits`, `configs/baseline.yaml` |
| 4 | Kiến trúc MLP | 1:00 | 2 → 16 → 16 → 1, tanh, đầu ra tuyến tính, 337 tham số | `src/model.py:15` `MLP` |
| 5 | Vòng huấn luyện, checkpoint | 1:30 | Adam 1e-3, batch 32, 1000 epoch, early stopping 100, lưu trọng số tại epoch val loss nhỏ nhất | `src/train.py:67` `train_model` |
| 6 | Đánh giá, y–ŷ | 1:00 | MSE, MAE, RMSE trên y gốc; đường y = ŷ càng sát càng tốt | `src/evaluate.py:58` `regression_metrics` |
| 7 | Thực nghiệm A | 1:15 | Val MSE giảm từ 1.22 × 10⁻⁵ (200 mẫu) xuống 1.68 × 10⁻⁷ (2000 mẫu), khoảng 72.5 lần | `experiments/run_A.py`, `results/A/summary.csv` |
| 8 | Thực nghiệm B | 1:15 | 1 nơ-ron thiếu khớp (2.92 × 10⁻³); 64 nơ-ron đạt 2.81 × 10⁻⁶; train và val sát nhau nên chưa quá khớp rõ | `experiments/run_B.py`, `results/B/summary.csv` |
| 9 | Đối chiếu A và B | 0:45 | Giá trị thấp nhất mỗi nhóm theo val MSE trung bình 3 seed | `results/A\|B/summary.csv` |
| 10 | Thiết lập C | 0:45 | Đổi số tầng ẩn, giữ 16 nơ-ron, 1000 mẫu | `experiments/run_C.py:43` |
| 11 | Đường loss, dừng sớm | 1:00 | 2 tầng hội tụ nhanh hơn 1 tầng; đọc best_epoch | `src/plots.py:84` `plot_loss_curves`, `results/C/figures/C_loss_curves.png` |
| 12 | Sai số theo số tầng ẩn | 1:00 | Tầng thứ hai đưa val MSE từ 1.19 × 10⁻⁵ xuống 4.35 × 10⁻⁷; tầng 3, 4 không giảm thêm rõ | `results/C/summary_mean_std.csv` |
| 13 | Thời gian train | 0:45 | Trung bình 15.2 s (1 tầng) lên 25.7 s (4 tầng) | `src/train.py:153` đo bằng `time.perf_counter()` |
| 14 | Mô hình cuối | 1:00 | A = 2000, 2×16, seed 1, epoch 980; test MSE 8.65 × 10⁻⁸, RMSE 2.94 × 10⁻⁴ | `results/final/final_metrics.json`, `scripts/run_final.py` |
| 15 | Kết luận, hạn chế | 1:00 | Kết luận chỉ đúng cho hàm, miền và cấu hình đã chạy; test đã bị C xem trước | báo cáo mục 5.3.3 và 6.2 |

### 2.1. Mạch kể và câu chuyển slide

Người nói một mình nên cần một mạch xuyên suốt: **dữ liệu → mô hình → ba thí nghiệm → chọn mô hình → kết luận**. Câu chuyển gợi ý:

- 1 → 2: "Bài toán là xấp xỉ một hàm Gaussian hai chiều bằng mạng MLP viết trên PyTorch."
- 3 → 4: "Có dữ liệu chia sạch rồi, giờ đến mạng sẽ học hàm này."
- 6 → 7: "Có mô hình và cách đo, nhóm làm ba thí nghiệm, mỗi lần chỉ đổi một biến. Đầu tiên là số mẫu."
- 9 → 10: "A và B cho biết số mẫu và độ rộng ảnh hưởng thế nào, còn C hỏi mạng sâu hơn có tốt hơn không."
- 13 → 14: "Từ 27 cấu hình, nhóm chọn mô hình theo val MSE, không dùng test."
- 14 → 15: "Cuối cùng là những gì kết luận này đúng và chưa đúng."

Mẹo khi chỉ một người nói: ở mỗi slide kết quả, nói con số trước rồi mới giải thích; nếu chậm thì rút slide 9 và 13 còn một câu, giữ đủ thời gian cho slide 5 và 14.

## 2.2. Từng slide: nói gì và code nằm ở đâu

Mỗi slide có 4 dòng: **Trên slide** (người nghe thấy gì), **Nói** (ý và câu gợi ý), **Code / số liệu** (mở file nào, dòng nào), **Có thể bị hỏi**.

### Slide 1. Tiêu đề

- **Trên slide:** Nhóm 05, Bài tập lớn 01, xấp xỉ hàm mật độ Gaussian hai chiều bằng MLP với PyTorch, 4 tên thành viên.
- **Nói:** "Chào thầy, em xin trình bày bài tập lớn 01 của nhóm 5: dùng mạng MLP viết bằng PyTorch để xấp xỉ hàm mật độ Gaussian hai chiều." Giới thiệu nhanh bố cục: dữ liệu, mô hình, ba thí nghiệm A, B, C, kết luận.
- **Code / số liệu:** không có. Đóng góp 25% mỗi người ở báo cáo Bảng 1.2.
- **Có thể bị hỏi:** ai làm phần nào. Trả lời theo Bảng 1.2: Trúc dữ liệu và ghép báo cáo, Khang lõi PyTorch (`model.py`, `train.py`, `evaluate.py`), Trung thực nghiệm A, B, Nhân thực nghiệm C và slide.

### Slide 2. Hàm Gaussian mục tiêu

- **Trên slide:** mạng nhận (x₁, x₂), dự đoán mật độ; miền x₁ ∈ [−2, 2], x₂ ∈ [−4, 4]; μ = (0, 0), Var(x₁) = 0.3, Var(x₂) = 2, Cov = −0.5.
- **Nói:** nêu công thức, nói ma trận hiệp phương sai C có det = 0.35 nên đỉnh hàm ≈ 0.269 tại gốc. Miền rộng 4 × 8 nhưng hàm chỉ cao ở giữa, nên phần lớn điểm có y gần 0 và MSE sẽ rất nhỏ. Vì thế nhóm báo thêm RMSE.
- **Code / số liệu:** `src/data.py:25` `gaussian_2d` (dạng toàn phương bằng `np.einsum`); tham số ở `configs/baseline.yaml`; hình `report/figures/fig_data_true_surface.png`.
- **Có thể bị hỏi:** vì sao lấy mẫu đều chứ không lấy theo phân phối? Vì đề yêu cầu lấy đều trong miền; đây là bài hồi quy hàm số, không phải sinh mẫu từ Gaussian.

### Slide 3. Tập train, validation, test

- **Trên slide:** bảng train pool 2000 (cập nhật trọng số), validation 500 (dừng sớm, chọn checkpoint và cấu hình), test 100 (không dùng để chọn); ghi chú mẫu train nhỏ nằm trong mẫu train lớn ở thực nghiệm A.
- **Nói:** ba tập sinh độc lập từ seed 42. Validation dùng để chọn, test chỉ để báo cáo cuối. Ở A, lấy n dòng đầu của pool nên các cấu hình lồng nhau, chỉ khác số mẫu.
- **Code / số liệu:** `src/data.py:45` `generate_splits` (`default_rng(seed).spawn(3)` cho mỗi tập một bộ sinh riêng, không rò rỉ); `src/data.py:70` `load_train(n)`; file `data/train_pool.csv`, `val.csv`, `test.csv`; hình `fig_data_scatter3d.png`, `fig_data_histograms.png`.
- **Có thể bị hỏi:** có rò rỉ test không? Trung thực: test không dùng chọn mô hình, nhưng runner C đã tính sai số test từng lượt (slide 14 và 15 có nói rõ).

### Slide 4. Kiến trúc MLP

- **Trên slide:** đầu vào 2; tầng ẩn 1 và 2 mỗi tầng 16, Linear rồi tanh; đầu ra 1, Linear; 337 tham số tính cả bias.
- **Nói:** 2 → 16 → 16 → 1. Đếm tham số: 2×16+16 = 48, 16×16+16 = 272, 16×1+1 = 17, tổng 337. Đầu ra tuyến tính vì y là số thực không bị chặn.
- **Code / số liệu:** `src/model.py:15` class `MLP`; vòng dựng tầng ở dòng 45–48; `forward` dòng 54; sơ đồ `report/figures/mlp_baseline_architecture.png`.
- **Có thể bị hỏi:** vì sao tanh? Hàm Gaussian trơn, tanh trơn nên hợp; baseline đã chốt ngày 29/10 trong `configs/baseline.yaml`.

### Slide 5. Vòng huấn luyện và checkpoint

- **Trên slide:** Adam lr 0.001, batch 32, tối đa 1000 epoch; 3 bước: mini-batch, cuối epoch, dừng sớm.
- **Nói:** mỗi batch: `zero_grad` → forward → `MSELoss` → `backward` → `step`. Cuối epoch chuyển `model.eval()` và `torch.no_grad()` để đo val loss; nếu val loss giảm thì copy trọng số. Dừng sau 100 epoch không cải thiện và nạp lại trọng số tốt nhất.
- **Code / số liệu:** `src/train.py:67` `train_model`: DataLoader dòng 132, loss và Adam 141–142, đo giờ 153, train 155–171, eval và no_grad 175–176, lưu `best_state` 190–200, dừng sớm 201, nạp lại 210, `torch.save` 222. Log: `results/<nhóm>/logs/*.csv`.
- **Có thể bị hỏi:** vì sao phải `clone` trọng số? Optimizer còn sửa tham số ở epoch sau, nếu không copy thì "bản tốt nhất" bị ghi đè. Vì sao `no_grad`? Bỏ đồ thị gradient khi chỉ đánh giá.

### Slide 6. Đánh giá giá trị dự đoán

- **Trên slide:** MSE, MAE, RMSE trên thang y gốc; đồ thị y–ŷ với đường chéo; validation RMSE 3.29 × 10⁻⁴ trên 500 mẫu.
- **Nói:** điểm nằm sát đường chéo y = ŷ nghĩa là dự đoán đúng. RMSE 3.29 × 10⁻⁴ so với đỉnh 0.269 là khoảng 0.12%. Không chuẩn hóa y nên các chỉ số đọc trực tiếp được.
- **Code / số liệu:** `src/evaluate.py:58` `regression_metrics`, `:80` `predict`, `:104` `evaluate_model`, `:116` `plot_y_vs_yhat`; hình `results/final/figures/selected_model_validation_y_yhat.png`.
- **Có thể bị hỏi:** MSE, MAE, RMSE khác gì? MSE phạt nặng sai số lớn, MAE là sai số tuyệt đối trung bình, RMSE = √MSE cùng đơn vị với y.

### Slide 7. Thực nghiệm A: số mẫu train

- **Trên slide:** giữ MLP 2×16, tăng mẫu từ 200 đến 2000; val MSE trung bình giảm từ 1.22 × 10⁻⁵ xuống 1.68 × 10⁻⁷ (kèm đồ thị).
- **Nói:** nhiều dữ liệu hơn thì sai số giảm đều, khoảng 72.5 lần. 10 cỡ mẫu × 3 seed = 30 lần chạy. Chỉ đổi đúng một biến.
- **Code / số liệu:** `experiments/run_A.py:51` `run_experiment`, `:75` `run_one`; gộp mean ± std ở `experiments/common.py:15` và `:48`; kết quả `results/A/summary.csv`; hình `results/A/figures/A_validation_metrics.png`; log `results/A/logs/A_<n>_seed<k>.csv`.
- **Có thể bị hỏi:** vì sao 3 seed? Để thấy độ biến thiên do khởi tạo ngẫu nhiên; báo mean ± std. Hạn chế: 3 seed vẫn ít (slide 15).

### Slide 8. Thực nghiệm B: số nơ-ron

- **Trên slide:** 1 tầng ẩn, 1000 mẫu, độ rộng 1–10, 16, 32, 64; val MSE giảm từ 2.92 × 10⁻³ xuống 2.81 × 10⁻⁶.
- **Nói:** 1 nơ-ron thiếu khớp rõ vì không đủ năng lực biểu diễn đỉnh Gaussian. Thêm nơ-ron thì sai số giảm, tốt nhất ở 64. 13 độ rộng × 3 seed = 39 lần chạy.
- **Code / số liệu:** `experiments/run_B.py:65` `run_experiment`, `:94` `run_one` (biến là `hidden = [n]`); `results/B/summary.csv`; hình `results/B/figures/B_metrics_vs_neurons.png`.
- **Có thể bị hỏi:** có quá khớp không? Slide sau trả lời: train và val sát nhau nên chưa thấy quá khớp rõ.

### Slide 9. Đối chiếu A và B

- **Trên slide:** A tốt nhất: 2000 mẫu, 2×16, (1.68 ± 0.589) × 10⁻⁷. B tốt nhất: 1×64, (2.81 ± 1.48) × 10⁻⁶. Ghi chú: ở B, độ rộng 9–10 train và val cùng tăng so với 8, chưa phải quá khớp rõ.
- **Nói:** A tốt hơn B khoảng một bậc nhưng không so công bằng tuyệt đối, vì A dùng 2000 mẫu và 2 tầng, B dùng 1000 mẫu và 1 tầng. Chỉ rút ra xu hướng trong từng nhóm.
- **Code / số liệu:** `results/A/summary.csv` (dòng n_train = 2000) và `results/B/summary.csv` (dòng 64).
- **Có thể bị hỏi:** vì sao không kết luận quá khớp? Quá khớp cần val tăng trong khi train tiếp tục giảm; ở đây cả hai cùng tăng.

### Slide 10. Thiết lập thực nghiệm C

- **Trên slide:** số tầng ẩn 1–4, mỗi tầng 16, tham số 65 / 337 / 609 / 881; 4 cấu hình × 3 seed = 12 lượt.
- **Nói:** C đổi độ sâu, giữ 16 nơ-ron mỗi tầng và 1000 mẫu train. Số tham số tăng theo tầng nên khi so sánh phải nhớ mạng sâu hơn cũng lớn hơn.
- **Code / số liệu:** `experiments/run_C.py:43` `run_experiment`, `:21` `_evaluate` (ghi thêm test error và thời gian từng lượt vì đề yêu cầu); `experiments/export_C.py:23` `audit_runs` kiểm đủ 12 log; cấu hình `configs/baseline.yaml` mục `experiments.C`.
- **Có thể bị hỏi:** vì sao C có tính test mà A, B không? Đề yêu cầu báo test error theo số tầng ở C; test không tham gia chọn cấu hình.

### Slide 11. Đường loss và dừng sớm

- **Trên slide:** đồ thị loss; 2 tầng giảm nhanh hơn 1 tầng ở giai đoạn đầu; 1 và 2 tầng cả 3 seed chạy đủ 1000 epoch; 3 và 4 tầng mỗi cấu hình có 2 seed dừng sớm.
- **Nói:** đường loss của cấu hình sâu hơn đi xuống nhanh hơn lúc đầu. Đoạn cuối đường trung bình chỉ lấy từ các seed còn chạy nên đọc cẩn thận.
- **Code / số liệu:** `src/plots.py:84` `plot_loss_curves`; hình `results/C/figures/C_loss_curves.png`; log `results/C/logs/`.
- **Có thể bị hỏi:** dừng sớm là gì và dựa vào đâu? Dừng khi val loss không cải thiện 100 epoch liền (`patience: 100`), theo val chứ không theo test.

### Slide 12. Sai số theo số tầng ẩn

- **Trên slide:** bảng val MSE, test MSE, test RMSE trung bình ± std: 1 tầng val 1.19 × 10⁻⁵; 2 tầng 4.35 × 10⁻⁷; 3 tầng 5.50 × 10⁻⁷; 4 tầng 4.10 × 10⁻⁷. Ghi chú: 4 tầng có mean val MSE thấp nhất trong C, 2 tầng ít dao động hơn.
- **Nói:** tầng ẩn thứ hai làm sai số giảm khoảng 27 lần. Tầng 3 và 4 không giảm thêm một cách rõ rệt: 4 tầng nhỉnh hơn 2 tầng về trung bình nhưng std lớn (± 1.85 × 10⁻⁷), nên chênh lệch nằm trong độ nhiễu. Đừng nói "4 tầng tốt nhất" như kết luận chắc chắn.
- **Code / số liệu:** `results/C/summary_mean_std.csv`, `results/C/table_C_mean_std.md`; hình `C_test_mse.png`, `C_test_rmse.png`; xuất bảng bằng `experiments/export_C.py:61`.
- **Có thể bị hỏi:** vì sao 4 tầng thấp nhất mà nhóm vẫn chọn 2 tầng? Mô hình cuối được chọn trên cả 27 cấu hình A, B, C theo val MSE; cấu hình 2×16 với 2000 mẫu của A đứng đầu (1.68 × 10⁻⁷), tốt hơn 4 tầng ở C (4.10 × 10⁻⁷, dùng 1000 mẫu).

### Slide 13. Thời gian huấn luyện theo độ sâu

- **Trên slide:** hai tầng trung bình 21.55 s, bốn tầng 25.72 s; lưu ý dừng sớm làm số epoch khác nhau giữa các lượt.
- **Nói:** thời gian đo vòng lặp train gồm cả validation mỗi epoch, trên cùng máy. Mạng sâu hơn tốn nhiều thời gian hơn (1 tầng 15.2 s, 4 tầng 25.7 s), trong khi lợi ích sai số từ tầng 3 trở đi không rõ.
- **Code / số liệu:** `src/train.py:153` và `:205` (`time.perf_counter()`); cột `train_time_s_mean` trong `results/C/summary_mean_std.csv`; hình `results/C/figures/C_train_time.png`, `C_error_and_time.png`; `src/plots.py:153`, `:164`.
- **Có thể bị hỏi:** thời gian có đáng tin không? Chỉ so sánh được trong cùng máy và cùng lần chạy; số epoch khác nhau do dừng sớm cũng ảnh hưởng.

### Slide 14. Mô hình cuối

- **Trên slide:** A với 2000 mẫu, 2×16, seed 1, checkpoint epoch 980; val MSE 1.08 × 10⁻⁷, test MSE 8.65 × 10⁻⁸, test RMSE 2.94 × 10⁻⁴; ghi chú test đã bị truy vấn ở C.
- **Nói:** chọn theo val MSE trên 27 cấu hình, lấy seed 1 vì val MSE thấp nhất trong ba seed, train lại, lưu checkpoint, nạp lại và đo test đúng một lần. Nói thẳng giới hạn: test đã được xem ở 12 lượt C nên không phải ước lượng độc lập hoàn toàn.
- **Code / số liệu:** `scripts/run_final.py` (từ chối chạy lần hai ở dòng 24–27); `results/final/final_metrics.json`, `results/final/selected_model.pt`, `src/train.py:246` `load_checkpoint`; hình `results/final/figures/selected_model_surface_comparison.png`.
- **Có thể bị hỏi:** vì sao val 1.08e-7 lệch log A 1.05e-7? Checkpoint train trên máy khác; chạy lại trên máy thứ ba ra log trùng A (`results/final/logs/rerun_A_2000_seed1.csv`), nên lệch do môi trường, không do code.

### Slide 15. Kết luận, hạn chế, hướng cải tiến

- **Trên slide:** kết luận (val MSE giảm ≈ 72.5 lần từ 200 lên 2000 mẫu; 1 nơ-ron thiếu khớp rõ; tầng thứ hai giảm ≈ 27 lần, tầng 3, 4 không giảm thêm rõ; tốt nhất 2×16 với 2000 mẫu); hạn chế (3 seed, test đã xem ở C, nhiều lượt chạm 1000 epoch, dữ liệu không nhiễu, chỉ đánh giá trong miền); hướng cải tiến (thêm seed và khoảng tin cậy, thêm nhiễu nhãn, so mạng cùng số tham số, sinh tập test mới).
- **Nói:** chốt 3 ý của kết luận, nêu 2 hạn chế quan trọng nhất (test đã xem, chỉ 3 seed), kết thúc bằng một hướng cải tiến. Xin câu hỏi.
- **Code / số liệu:** báo cáo mục 6.1, 6.2 và 5.3.3 (`report/section_6_conclusion.md`).
- **Có thể bị hỏi:** nếu làm lại thì làm gì khác? Sinh tập test mới chỉ dùng cho mô hình cuối, tăng số seed, tăng max_epochs cho các lượt chạm 1000 epoch.

## 3. Bản đồ code theo luồng dữ liệu

```text
configs/baseline.yaml
   │  (seed, miền, μ, C, kích thước tập, hyper-parameter)
   ▼
src/data.py ──► data/train_pool.csv, val.csv, test.csv, stats.csv
   │  load_split / load_train(n)
   ▼
src/model.py (MLP) ◄── src/train.py (train_model) ──► results/<nhóm>/logs/*.csv  + checkpoint .pt
                              │
                              ▼
                      src/evaluate.py (MSE, MAE, RMSE, y–ŷ, bề mặt)
                              │
experiments/run_A|B|C.py ─────┴──► results/<nhóm>/summary.csv ──► src/plots.py ──► figures/*.png
                                                   │
scripts/run_final.py ──► results/final/selected_model.pt + final_metrics.json
scripts/build_report.py ──► report/BTL01_Nhom05.md ──► docs/ (Word, PDF)
```

### 3.1. Dữ liệu – `src/data.py` (Trúc)

| Việc | Hàm | Dòng | Giải thích ngắn |
| --- | --- | --- | --- |
| Đọc cấu hình | `load_config` | 20 | Đọc `configs/baseline.yaml` |
| Hàm Gaussian | `gaussian_2d` | 25 | Dạng toàn phương `(x−μ)ᵀC⁻¹(x−μ)` tính bằng `np.einsum` cho cả n điểm |
| Lấy mẫu đều | `sample_split` | 37 | `rng.uniform` trên từng trục rồi gắn nhãn y |
| Chia 3 tập | `generate_splits` | 45 | `default_rng(seed).spawn(3)`: mỗi tập một bộ sinh riêng, nên đổi cỡ tập này không làm đổi tập kia, không rò rỉ giữa train, val, test |
| Lưu và đọc CSV | `save_splits`, `load_split` | 56, 62 | `load_split` trả `float32` để đưa thẳng vào torch |
| Lấy n mẫu train | `load_train` | 70 | Lấy **n dòng đầu** của pool, nên A = 200 nằm trọn trong A = 400. Các cấu hình A chỉ khác nhau ở số mẫu |
| Thống kê, kiểm tra | `describe`, `check` | 78, 92 | Số liệu bảng thống kê ở mục 2.4 của báo cáo |

Hình mục 2: `scripts/plot_data.py` (`plot_scatter3d` dòng 51, `plot_true_surface` 68, `plot_histograms` 96), ra `report/figures/fig_data_*.png`.

### 3.2. Mô hình – `src/model.py` (Khang)

- `MLP` (dòng 15) nhận danh sách số nơ-ron mỗi tầng ẩn, ví dụ `[16, 16]`. Vòng `for` ở dòng 45–48 chồng `nn.Linear` + activation, tầng cuối `nn.Linear(…, 1)` không activation.
- `forward` (dòng 54) chạy `nn.Sequential`.
- Vì sao đầu ra tuyến tính: bài toán hồi quy, y không bị chặn bởi tanh.
- Sơ đồ vẽ bởi `scripts/plot_mlp_architecture.py` → `report/figures/mlp_baseline_architecture.png`.

### 3.3. Vòng huấn luyện – `src/train.py` (Khang). Phần thầy dễ hỏi nhất

`train_model` (dòng 67) làm theo thứ tự:

| Bước | Dòng | Ý nghĩa |
| --- | --- | --- |
| `_seed_everything` | 51 | Cố định seed để chạy lại ra cùng log |
| `DataLoader(TensorDataset…)` | 132 | Chia mini-batch 32, xáo trộn mỗi epoch |
| `nn.MSELoss()`, `torch.optim.Adam` | 141–142 | Loss và optimizer theo đề |
| `time.perf_counter()` | 153, 205 | Đo thời gian vòng lặp train (không tính lưu file, vẽ hình) cho thực nghiệm C |
| `model.train()` + vòng batch | 155–171 | `zero_grad` → forward → loss → `backward` → `step` |
| `model.eval()` + `torch.no_grad()` | 175–176 | Tính val loss, không tạo đồ thị gradient, tắt hành vi train |
| Ghi `history` theo epoch | 185 | Sinh log CSV `epoch, train_loss, val_loss` |
| Lưu `best_state` khi val loss giảm | 190–200 | Copy độc lập (`clone`) vì optimizer còn cập nhật tham số ở epoch sau |
| Early stopping | 201 | Dừng nếu 100 epoch liền không cải thiện val loss |
| `load_state_dict(best_state)` | 210 | Trả mô hình về epoch tốt nhất, không phải epoch cuối |
| `torch.save` | 222 | Checkpoint theo val loss |
| `load_checkpoint` | 246 | Nạp lại mô hình để đánh giá |

Câu cốt lõi để nói: **test không xuất hiện trong `train_model`**. Chọn epoch và chọn cấu hình đều dựa vào val loss.

### 3.4. Đánh giá – `src/evaluate.py` (Khang)

- `regression_metrics` (58): MSE, MAE, RMSE, kiểm tra kích thước và giá trị hữu hạn.
- `predict` (80) dùng `eval()` + `no_grad()`; `evaluate_model` (104) gọi `predict` rồi tính chỉ số.
- `plot_y_vs_yhat` (116) và `plot_surface_comparison` (155): đồ thị y–ŷ và bề mặt thật so với xấp xỉ.

### 3.5. Thực nghiệm – `experiments/` (Trung: A, B; Nhân: C)

- `common.py`: `run_seeded_experiments` (15) chạy từng cấu hình với seed 0, 1, 2; `summarize_seed_runs` (48) gộp mean ± std.
- `run_A.py`: `run_experiment` (51), hàm con `run_one` (75) gọi `load_train(n)` rồi `train_model`. Ghi `results/A/summary.csv`.
- `run_B.py`: `run_experiment` (65), `run_one` (94). Biến là `hidden = [n]`.
- `run_C.py`: `run_experiment` (43). Có thêm `_evaluate` (21) để ghi sai số test và thời gian từng lượt, vì đề yêu cầu test error theo số tầng.
- `export_C.py`: `audit_runs` (23) kiểm đủ 12 log, `export_tables` (61) xuất bảng mean ± std (`results/C/table_C_mean_std.md`).
- Log từng lượt: `results/<nhóm>/logs/<nhóm>_<giá trị>_seed<k>.csv`, ví dụ `results/A/logs/A_800_seed1.csv`.

### 3.6. Vẽ hình – `src/plots.py` (Nhân)

`plot_loss_curves` (84), `plot_error_vs_param` (138), `plot_train_time` (153), `plot_error_and_time` (164), `make_group_figures` (201). Đọc từ `summary.csv` và log, không train lại.

### 3.7. Mô hình cuối – `scripts/` (Khang, Trúc)

- `scripts/run_baseline.py` (Trúc): chạy thử baseline end-to-end, không dùng test.
- `scripts/run_final.py` (Khang): xếp hạng 27 cấu hình theo val MSE, train lại cấu hình thắng (A = 2000, seed 1), lưu `results/final/selected_model.pt`, nạp lại, tính test một lần. **Từ chối chạy lần hai** nếu `final_metrics.json` đã có (dòng 24–27).
- `scripts/plot_final_evaluation.py`: hình y–ŷ và bề mặt của mô hình cuối.
- `scripts/build_report.py` (Trúc): ghép `report/section_*.md`, kiểm đánh số bảng/hình (`PROBLEMS: 0`).

## 4. Demo code (nếu thầy yêu cầu), khoảng 1–2 phút

Chỉ chạy lệnh **không ghi đè kết quả nộp**. Mở Anaconda Prompt tại thư mục BTL01:

```powershell
conda activate btl01
python -c "from src.train import load_checkpoint; from src.data import load_split; from src.evaluate import evaluate_model; m,_=load_checkpoint('results/final/selected_model.pt'); X,y=load_split('val'); print(evaluate_model(m,X,y))"
```

Kết quả đúng: MSE ≈ 1.08e-07, RMSE ≈ 3.29e-04 trên validation (khớp Bảng 1.3 của báo cáo). Không chạy `run_A/B/C`, `src.data`, `plot_final_evaluation` trên máy nộp vì chúng ghi đè `data/` và `results/`. Muốn thử thì chép thư mục sang chỗ khác.

## 5. Con số phải thuộc

| Đại lượng | Giá trị | Nguồn |
| --- | --- | --- |
| Đỉnh của hàm | ≈ 0.269 (det C = 0.35) | `src/data.py` |
| Kích thước tập | train pool 2000, val 500, test 100, seed 42 | `configs/baseline.yaml` |
| Baseline | 2×16, tanh, Adam 1e-3, batch 32, 1000 epoch, patience 100 | `configs/baseline.yaml` |
| Tổng số lần chạy | A 30 + B 39 + C 12 = 81 | Kế hoạch mục 2 |
| A: val MSE | 1.22e-5 (200) → 1.68e-7 (2000), giảm ≈ 72.5 lần | `results/A/summary.csv` |
| B: val MSE | 2.92e-3 (1 nơ-ron) → 2.81e-6 (64 nơ-ron) | `results/B/summary.csv` |
| C: val MSE | 1.19e-5 (1 tầng) → 4.35e-7 (2 tầng); 3, 4 tầng không giảm rõ | `results/C/summary_mean_std.csv` |
| C: thời gian | 15.2 s → 21.5 → 21.9 → 25.7 s | `results/C/summary_mean_std.csv` |
| Mô hình cuối | 337 tham số, epoch 980, test MSE 8.65e-8, MAE 2.16e-4, RMSE 2.94e-4 | `results/final/final_metrics.json` |

## 6. Câu hỏi thầy hay hỏi và cách trả lời

1. **Vì sao MSE nhỏ thế, có đáng tin không?** Phần lớn điểm trong miền 4 × 8 có y gần 0 (đỉnh chỉ 0.269), nên MSE nhỏ. Nhóm báo thêm RMSE ≈ 3 × 10⁻⁴, bằng khoảng 0.11% đỉnh. Chỉ số này đọc được ngay.
2. **Test có bị dùng để chọn mô hình không?** Không. Chọn bằng val MSE. Nhưng runner C đã tính sai số test trước khi chọn, nên test không còn "chưa ai xem". Nhóm ghi rõ giới hạn này ở báo cáo 5.3.3. Nói thẳng, đừng giấu.
3. **Vì sao dùng `model.eval()` và `torch.no_grad()`?** `eval()` đổi chế độ các lớp như dropout/batch-norm (MLP này không có nhưng là thói quen đúng); `no_grad()` bỏ việc lưu đồ thị gradient, nhanh và đỡ tốn bộ nhớ khi chỉ đánh giá. `src/train.py:175–176`.
4. **Checkpoint lưu epoch nào?** Epoch có val loss nhỏ nhất, không phải epoch cuối. Mô hình cuối là epoch 980 trên 1000. `src/train.py:190–210`.
5. **Vì sao A dùng "n dòng đầu" của pool?** Để các cấu hình A lồng nhau, val và test giữ nguyên, nên khác biệt chỉ đến từ số mẫu. `src/data.py:70`.
6. **Hai tầng đã đủ, vì sao thêm tầng không tốt hơn?** Hàm Gaussian trơn, 2 tầng × 16 đã đủ năng lực. Thêm tầng làm thời gian train tăng (15 → 26 s) mà val MSE không giảm rõ, và std giữa các seed lớn. Kết luận chỉ áp dụng cho cấu hình đã chạy.
7. **Có quá khớp không?** Ở B, train và val MSE sát nhau ở mọi độ rộng nên chưa thấy quá khớp rõ. Thiếu khớp thấy rõ ở 1–2 nơ-ron. Đừng khẳng định có quá khớp nếu số liệu không cho thấy.
8. **Vì sao val MSE của mô hình cuối (1.08e-7) lệch log A (1.05e-7)?** Checkpoint train trên máy khác máy chạy A. Chạy lại trên máy thứ ba ra log trùng log A ở cả 1000 epoch (`results/final/logs/rerun_A_2000_seed1.csv`), nên chênh lệch đến từ môi trường tính toán, không phải code.
9. **Vì sao không chuẩn hóa y?** Giữ y gốc để MSE/MAE/RMSE có nghĩa trực tiếp so với đỉnh 0.269. `normalize_y: false` trong cấu hình.
10. **Làm sao chạy lại?** README: tạo môi trường `btl01`, `pip install -r requirements.txt` (torch 2.15.0 bản CPU), chạy `python -m …` từ thư mục BTL01. Đường dẫn đều tương đối qua `src/paths.py`.

Thầy chọn ngẫu nhiên người nói nên ai cũng phải giải thích được `data.py`, `model.py`, `train.py`, `evaluate.py` (mục 3.1 đến 3.4) và đọc được cả ba nhóm kết quả. Tên trong ngoặc ở mục 3 chỉ là người viết code, không phải người duy nhất trả lời được.

## 7. Checklist trước giờ trình bày

- [ ] Mở sẵn `docs/BTL01_Nhom05_Slide.pdf` và báo cáo PDF để tra số.
- [ ] Mở sẵn trong editor: `src/train.py`, `src/model.py`, `src/data.py`, `results/final/final_metrics.json`.
- [ ] Anaconda Prompt đã `conda activate btl01`, đứng đúng thư mục BTL01, lệnh demo ở mục 4 đã thử chạy được.
- [ ] Cả 4 người tập nói trọn 15 slide ít nhất một lần, bấm giờ, không quá 15 phút.
- [ ] Cả nhóm thuộc mục 5 (con số) và trả lời được mục 6.
