# BTL01 – Vấn đáp về code (Nhóm 5)

Bộ câu hỏi về code thầy có thể hỏi, kèm đáp án ngắn và chỗ để mở file. Thầy chọn ngẫu nhiên một người, nên ai cũng phải trả lời được toàn bộ. Đường dẫn tính từ thư mục `BTL01 - Function Approximation/`. Số dòng theo code hiện tại, nếu file đổi thì tìm theo tên hàm. Mỗi câu có mức: **[Cơ bản]** ai cũng phải biết, **[Sâu]** thầy hỏi để phân biệt người hiểu với người đọc thuộc.

## 1. Tổng quan dự án

**1.1. [Cơ bản] Chạy dự án theo thứ tự nào?**
`python -m src.data` (sinh CSV) → `python -m experiments.run_A`, `run_B`, `run_C` (mỗi lần gọi `train_model`) → `python -m experiments.export_C` → `python -m scripts.run_final` (mô hình cuối, test một lần) → `python -m scripts.plot_final_evaluation` → `python -m scripts.build_report`. Thứ tự này nằm ở README mục "Chạy".

**1.2. [Cơ bản] Vì sao chạy bằng `python -m ...` chứ không `python file.py`?**
Để `import src...` hoạt động từ thư mục gốc BTL01 mà không phải sửa `sys.path`. Mọi đường dẫn lấy từ `src/paths.py` nên không có đường dẫn tuyệt đối, máy khác chạy được.

**1.3. [Cơ bản] File nào chứa tham số thí nghiệm?**
`configs/baseline.yaml`: seed dữ liệu 42, miền, μ, C, kích thước tập 2000/500/100, seed huấn luyện 0/1/2, 2×16 tanh, Adam lr 1e-3, batch 32, tối đa 1000 epoch, patience 100, `normalize_y: false`, và danh sách giá trị của A, B, C.

**1.4. [Sâu] Làm sao đảm bảo mỗi thí nghiệm chỉ đổi một biến?**
Mọi thí nghiệm đọc cùng một baseline. A chỉ đổi `n_train`, B chỉ đổi `hidden = [n]`, C chỉ đổi độ sâu. Validation và test giữ nguyên ở mọi cấu hình, và A lấy n dòng đầu của cùng một pool (`src/data.py:70`).

## 2. Dữ liệu – `src/data.py`

**2.1. [Cơ bản] Hàm `gaussian_2d` (dòng 25) tính gì và vì sao dùng `einsum`?**
Tính f(x) = exp(−½ (x−μ)ᵀC⁻¹(x−μ)) / (2π√det C) cho cả n điểm một lúc. `np.einsum("ni,ij,nj->n", d, inv(C), d)` là dạng toàn phương cho từng dòng, tránh vòng lặp Python và kết quả có shape (n,).

**2.2. [Cơ bản] Ba tập được chia thế nào để không rò rỉ?**
`generate_splits` (dòng 45) dùng `np.random.default_rng(seed).spawn(3)`: mỗi tập (train pool, val, test) có bộ sinh số ngẫu nhiên riêng tách từ cùng seed 42. Đổi kích thước một tập không làm đổi hai tập còn lại, và các tập được sinh độc lập nên không trùng điểm.

**2.3. [Cơ bản] `load_train(n)` (dòng 70) làm gì và vì sao?**
Trả về n dòng đầu của `train_pool.csv`. Nhờ vậy A = 200 nằm trọn trong A = 400, v.v., và các cấu hình A chỉ khác nhau ở số mẫu, không khác nhau ở việc bốc ngẫu nhiên mẫu nào.

**2.4. [Cơ bản] Vì sao `load_split` trả `float32`?**
PyTorch mặc định dùng float32 cho tham số mô hình; đưa float64 vào `nn.Linear` sẽ báo lỗi lệch kiểu. Dữ liệu CSV vẫn sinh bằng float64 để nhãn chính xác.

**2.5. [Sâu] Vì sao MSE ở bài này nhỏ cỡ 10⁻⁷?**
Đỉnh của hàm chỉ ≈ 0.269 (det C = 0.35) và hàm gần bằng 0 ở phần lớn miền 4 × 8. Vì vậy báo cáo ghi dạng số mũ và thêm RMSE ≈ 3 × 10⁻⁴, bằng khoảng 0.11% đỉnh.

## 3. Mô hình – `src/model.py`

**3.1. [Cơ bản] Class `MLP` (dòng 15) dựng mạng thế nào?**
Nhận `hidden` là danh sách số nơ-ron mỗi tầng ẩn. Vòng `for` (dòng 45–48) chồng cặp `nn.Linear(vào, size)` + activation, tầng cuối `nn.Linear(…, 1)` không activation. Tất cả gom vào `nn.Sequential`.

**3.2. [Cơ bản] Baseline có bao nhiêu tham số, tính ra sao?**
2×16+16 = 48; 16×16+16 = 272; 16×1+1 = 17; tổng 337. Với 1 tầng 16 nơ-ron: 48 + 17 = 65. Với 3 và 4 tầng: 609 và 881.

**3.3. [Cơ bản] Vì sao đầu ra không có activation?**
Bài toán hồi quy, y là số thực dương. Activation như tanh sẽ chặn đầu ra trong (−1, 1) và làm khó việc học giá trị nhỏ cỡ 10⁻³.

**3.4. [Sâu] Tanh thay bằng ReLU được không?**
Được (`_ACTIVATIONS` trong `model.py` cho phép tanh, relu, sigmoid). Nhóm chọn tanh vì hàm mục tiêu trơn; ReLU cho xấp xỉ gồm các đoạn thẳng nên cần nhiều nơ-ron hơn mới mịn. Nhóm chưa chạy thực nghiệm so sánh activation, nên không kết luận.

**3.5. [Sâu] `forward` (dòng 54) kiểm tra gì?**
Kiểm đầu vào có shape (n, 2), rồi gọi `self.network(x)` trả shape (n, 1).

## 4. Huấn luyện – `src/train.py` (phần thầy hỏi nhiều nhất)

**4.1. [Cơ bản] Kể các bước của một batch.**
`optimizer.zero_grad()` → `predictions = model(batch_x)` → `loss = loss_fn(predictions, batch_y)` → `loss.backward()` → `optimizer.step()` (khoảng dòng 159–167). Thiếu `zero_grad` thì gradient cộng dồn qua batch.

**4.2. [Cơ bản] `model.train()` và `model.eval()` khác gì? `torch.no_grad()` để làm gì?**
`train()` bật chế độ huấn luyện (ảnh hưởng dropout, batch-norm), `eval()` bật chế độ suy luận. MLP này không có hai lớp đó nên kết quả không đổi, nhưng đây là thói quen đúng và đề yêu cầu. `no_grad()` (dòng 175–176) tắt việc dựng đồ thị tính gradient khi chỉ đo val loss, nhanh hơn và đỡ tốn bộ nhớ.

**4.3. [Cơ bản] Loss theo epoch được tính thế nào?**
Cộng `loss.item() * len(batch_x)` qua các batch rồi chia tổng số mẫu (khoảng dòng 169–173). Có nhân kích thước batch để batch cuối ngắn hơn (ví dụ 1000 mẫu, batch 32 dư 8 mẫu) được tính đúng tỷ trọng.

**4.4. [Cơ bản] Checkpoint tốt nhất được lưu khi nào?**
Khi `val_loss < best_val_loss` (dòng 190). Khi đó lưu `best_state` bằng `tensor.detach().cpu().clone()` cho từng tham số.

**4.5. [Sâu] Vì sao phải `clone()` khi lưu `best_state`?**
`state_dict()` trả tham chiếu tới chính tensor của mô hình. Nếu không copy, optimizer ở epoch sau vẫn sửa các tensor đó và "bản tốt nhất" bị ghi đè theo.

**4.6. [Cơ bản] Early stopping hoạt động ra sao?**
Đếm `epochs_without_improvement`; mỗi epoch val loss không giảm thì cộng 1, giảm thì về 0. Chạm `patience = 100` thì `break` (dòng 201) và `stopped_early = True`.

**4.7. [Cơ bản] Sau vòng lặp, mô hình trả về là epoch nào?**
Epoch có val loss nhỏ nhất, không phải epoch cuối: `model.load_state_dict(best_state)` (dòng 210). Mô hình cuối của nhóm có `best_epoch` = 980 trên 1000 epoch.

**4.8. [Cơ bản] Vì sao dừng sớm và chọn checkpoint theo validation, không theo test?**
Nếu dùng test để quyết định thì test không còn là phép đo độc lập. `train_model` thậm chí không nhận tham số test (docstring dòng 86).

**4.9. [Sâu] `_seed_everything` (dòng 51) khóa những gì?**
`random`, `numpy`, `torch.manual_seed`, CUDA nếu có, và một `torch.Generator` riêng truyền cho `DataLoader` để thứ tự xáo trộn batch lặp lại được. Nhờ vậy cùng seed cho cùng log, và 3 seed 0, 1, 2 khác nhau ở cả khởi tạo trọng số lẫn thứ tự batch.

**4.10. [Sâu] Thời gian train đo ở đâu, gồm những gì?**
`time.perf_counter()` ở dòng 153 và 205, bao quanh vòng epoch gồm cả validation mỗi epoch, không gồm ghi file hay vẽ hình. Do dừng sớm, số epoch khác nhau nên thời gian không tỷ lệ thuần với độ sâu.

**4.11. [Sâu] Vì sao `_as_xy` kiểm tra nhiều thế (shape, hữu hạn, cùng độ dài)?**
Để lỗi dữ liệu (NaN, y sai chiều) dừng ngay với thông báo rõ thay vì cho loss `nan` âm thầm. Trong vòng lặp còn kiểm `np.isfinite(train_loss)` và raise `RuntimeError` nếu loss không hữu hạn.

**4.12. [Cơ bản] Log và checkpoint nằm ở đâu, nạp lại thế nào?**
Log mỗi epoch (`epoch, train_loss, val_loss`) ở `results/<nhóm>/logs/<nhóm>_<giá trị>_seed<k>.csv`. Checkpoint là file `.pt` chứa `model_state_dict`, `hidden`, `activation`, `best_epoch`, `best_val_loss`, `elapsed_seconds`, `seed`, `train_config` (dòng 222–234). `load_checkpoint` (dòng 246) đọc bằng `torch.load(..., weights_only=True)`, dựng lại `MLP` đúng kiến trúc rồi `eval()`.

## 5. Đánh giá – `src/evaluate.py`

**5.1. [Cơ bản] `regression_metrics` (dòng 58) trả những gì?**
MSE = mean((y−ŷ)²), MAE = mean(|y−ŷ|), RMSE = √MSE, đều trên thang y gốc, kiểm cùng kích thước và hữu hạn.

**5.2. [Cơ bản] `predict` (dòng 80) khác vòng val trong `train_model` chỗ nào?**
Cùng ý: `eval()` và `no_grad()`. `predict` dùng để đánh giá sau huấn luyện trên bất kỳ tập nào, trả mảng NumPy shape (n, 1), kiểm giá trị hữu hạn, và khôi phục chế độ train/eval cũ của mô hình khi xong.

**5.3. [Cơ bản] Đồ thị y–ŷ đọc thế nào?**
Mỗi điểm là (y thật, ŷ dự đoán) của một mẫu; điểm nằm trên đường chéo y = ŷ là dự đoán đúng, lệch xa là sai nhiều. `plot_y_vs_yhat` ở dòng 116.

**5.4. [Sâu] Vì sao không chuẩn hóa y?**
Để MSE, MAE, RMSE đọc trực tiếp so với đỉnh 0.269. Chuẩn hóa sẽ phải đổi ngược khi báo cáo. Hệ quả: loss rất nhỏ nên Adam cần nhiều epoch, nhiều lượt chạm 1000 epoch (nhóm nêu ở hạn chế).

## 6. Thực nghiệm – `experiments/`

**6.1. [Cơ bản] `run_seeded_experiments` (`common.py:15`) làm gì?**
Lặp mọi cặp (giá trị, seed), gọi `run_one(value, seed)` và gom mỗi lượt thành một dòng. Có kiểm không trùng giá trị, không trùng seed, và kết quả không được ghi đè cột `seed`.

**6.2. [Cơ bản] `summarize_seed_runs` (`common.py:48`) tính mean ± std thế nào?**
`groupby(giá trị)` rồi `agg(["mean", "std"])`. `std` của pandas là độ lệch chuẩn mẫu (chia n−1), với n = 3 seed. Hàm từ chối nếu có metric không hữu hạn hoặc trùng seed.

**6.3. [Cơ bản] Số lần chạy mỗi thí nghiệm?**
A: 10 cỡ mẫu × 3 seed = 30. B: 13 độ rộng × 3 = 39. C: 4 độ sâu × 3 = 12. Tổng 81 lượt.

**6.4. [Cơ bản] `run_A.py:run_one` (dòng 75) làm gì?**
Gọi `load_train(n)`, đọc val, rồi `train_model(...)` với baseline 2×16, ghi log và trả về các chỉ số train/val cùng `best_epoch`, thời gian. Không động tới test.

**6.5. [Cơ bản] Vì sao runner C có tính test còn A, B thì không?**
Đề yêu cầu báo sai số test theo số tầng ở C. `_evaluate` (`run_C.py:21`) ghi test MSE/MAE/RMSE và thời gian từng lượt. Test không tham gia chọn cấu hình, nhưng vì đã bị truy vấn nên nhóm ghi rõ ở báo cáo rằng test của mô hình cuối không phải ước lượng độc lập hoàn toàn.

**6.6. [Sâu] `export_C.audit_runs` (dòng 23) kiểm gì?**
Kiểm đủ đúng 12 cặp (độ sâu, seed); mỗi log có epoch liên tục từ 1, seed khớp, metadata baseline khớp (16 nơ-ron mỗi tầng, 1000 mẫu, Adam, lr, batch), loss hữu hạn, epoch có val loss nhỏ nhất khớp với `summary.csv`, thời gian dương. Sai chỗ nào thì raise lỗi và không xuất bảng.

**6.7. [Sâu] Runner từng ghi đè `summary.csv` khi chạy một phần cấu hình. Đã sửa gì?**
Lỗi phát hiện ở buổi review chéo 31/10 (báo cáo mục 1.4): chạy thử vài giá trị làm mất các dòng còn lại. Đã sửa để bảng tổng hợp chứa đủ cấu hình.

## 7. Mô hình cuối và vẽ hình – `scripts/`, `src/plots.py`

**7.1. [Cơ bản] Mô hình cuối chọn thế nào?**
Xếp hạng 27 cấu hình A, B, C theo val MSE trung bình 3 seed; đứng đầu là A = 2000, 2×16 (1.68 × 10⁻⁷). Trong 3 seed lấy seed 1 vì val MSE thấp nhất. `scripts/run_final.py` train lại, lưu `results/final/selected_model.pt`, nạp lại và tính test.

**7.2. [Cơ bản] Làm sao chắc test chỉ đánh giá một lần cho mô hình cuối?**
`run_final.py` dòng 24–27: nếu `final_metrics.json` đã tồn tại thì raise `FileExistsError`, từ chối chạy. Muốn chạy lại phải xóa file đó một cách có chủ ý.

**7.3. [Sâu] Vì sao val của checkpoint cuối (1.08e-7) lệch log A (1.05e-7)?**
Checkpoint train trên máy khác máy chạy A. Chạy lại cùng cấu hình trên máy thứ ba cho log trùng log A ở cả 1000 epoch (`results/final/logs/rerun_A_2000_seed1.csv`). Kết luận: lệch do môi trường tính toán (thư viện số học), không do code. `requirements.txt` ghim Python 3.11 và torch 2.15.0 bản CPU.

**7.4. [Cơ bản] `src/plots.py` có train lại mô hình không?**
Không. Chỉ đọc `summary.csv` và log CSV rồi vẽ: `plot_loss_curves` (84), `plot_error_vs_param` (138), `plot_train_time` (153), `plot_error_and_time` (164); `make_group_figures` (201) gọi tất cả cho một nhóm.

**7.5. [Cơ bản] `scripts/build_report.py` làm gì?**
Ghép `report/section_*.md` thành `report/BTL01_Nhom05.md`, đánh số lại và kiểm bảng/hình đánh số liên tục, mỗi bảng/hình được nhắc trong văn trước khi xuất hiện. In `PROBLEMS: 0` nếu ổn.

## 8. Câu hỏi "bẫy" cần trả lời thẳng

**8.1. Test có bị dùng để chọn mô hình không?**
Không dùng để chọn. Nhưng runner C đã tính test cho 12 lượt trước khi chọn mô hình cuối, nên con số test cuối không phải ước lượng độc lập hoàn toàn. Hướng cải tiến: sinh tập test mới chỉ dùng cho mô hình cuối.

**8.2. 4 tầng có val MSE trung bình thấp hơn 2 tầng, sao không chọn 4 tầng?**
Chênh lệch nằm trong độ nhiễu (4 tầng: 4.10 ± 1.85 × 10⁻⁷, 2 tầng: 4.35 ± 0.245 × 10⁻⁷, chỉ 3 seed), và 4 tầng chậm hơn (25.7 s so với 21.5 s). Ngoài ra mô hình cuối chọn trên cả 27 cấu hình, trong đó A = 2000 tốt hơn hẳn C vì có thêm dữ liệu.

**8.3. Có quá khớp không?**
Ở B, train và val MSE sát nhau ở mọi độ rộng, nên không kết luận quá khớp rõ. Thiếu khớp thấy rõ ở 1 nơ-ron. Nhóm không khẳng định điều số liệu không cho thấy.

**8.4. Mô hình nhỏ nhất có thể đạt tốt không?**
Không so sánh trực tiếp giữa các nhóm vì khác số mẫu và số tham số; chỉ rút ra xu hướng trong từng nhóm. Đề xuất so các mạng cùng số tham số là hướng cải tiến.

**8.5. Nếu thầy đổi seed hoặc cấu hình, kết quả có giống không?**
Cùng seed và cùng môi trường thì log trùng. Khác máy có thể lệch ở chữ số có nghĩa thứ tám (xem 7.3). Khác seed thì giá trị dao động, nên nhóm báo mean ± std qua 3 seed.

## 9. Chuẩn bị trước buổi vấn đáp

- [ ] Mở sẵn trong editor: `src/train.py`, `src/model.py`, `src/data.py`, `src/evaluate.py`, `experiments/common.py`, `scripts/run_final.py`.
- [ ] Tự đọc lại vòng huấn luyện `src/train.py:154–210` và kể được từng dòng.
- [ ] Thuộc các con số: 337 tham số, 2000/500/100, 81 lượt chạy, val MSE cuối 1.08e-7, test MSE 8.65e-8, RMSE 2.94e-4.
- [ ] Thuộc câu trả lời thẳng cho mục 8, nhất là 8.1.
- [ ] Mỗi người tự hỏi nhau ít nhất 10 câu mức [Sâu] của mục 4 và mục 6.
