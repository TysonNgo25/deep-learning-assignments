# CodetoTruc – Nhật ký việc đã làm

Ghi lại từng ngày Trúc đã làm gì trong BTL01, code chạy thế nào và vì sao viết như vậy. Dùng để ôn lại trước buổi vấn đáp và khi viết báo cáo. Ngày mới thêm ở cuối file.

Kế hoạch gốc: [Kế hoạch.md](Kế%20hoạch.md).

---

## T4 28/10/2026 – Phân chia đồ án, dựng khung project BTL01

Mục này bổ sung tối 31/10: ngày 28/10 có commit nhưng nhật ký mới bắt đầu từ 29/10. Chèn ở đây (không ở cuối file) để các mục giữ đúng thứ tự ngày.

### Việc theo kế hoạch

Kế hoạch 28/10 của Trúc: "Đọc đề, gửi kế hoạch, tạo thư mục BTL01 trên GitHub + cấu trúc thư mục". Làm xong cả ba việc.

### Commit trong ngày

| Hash | Người | Message | Làm gì |
| --- | --- | --- | --- |
| `c6b5683` | Trúc | Phân chia đồ án BTL01 | Đưa `Kế hoạch.md` (đề bài phân tích, phân vai, lịch 28/10 → 8/11, baseline đề xuất, checklist nộp) và 4 file đề vào `Đề và yêu cầu/` (`MLP_Exercise - Yeu Cau.pdf`, `MLP_Exercise.pdf`, `Sample_Code.html`, `Sample_Report.pdf`). |
| `bc9ef1c` | Trúc | Dựng khung project BTL01 | Tạo cấu trúc thư mục thống nhất cho cả nhóm, `README.md`, `requirements.txt`, `configs/baseline.yaml` (bản đề xuất), `src/paths.py` và file rỗng có docstring ghi người phụ trách cho từng module. |

### Kết quả

Cây thư mục tạo trong `bc9ef1c`:

```text
configs/baseline.yaml            baseline ĐỀ XUẤT (chốt ngày 29/10)
data/  models/  report/  slides/ thư mục rỗng, giữ bằng .gitkeep
src/        paths.py (đã viết), data.py, model.py, train.py, evaluate.py, plots.py (chỉ có docstring)
scripts/    plot_data.py (chỉ có docstring)
experiments/ run_A.py, run_B.py, run_C.py (chỉ có docstring)
results/A|B|C/logs/, results/A|B|C/figures/
README.md, requirements.txt
```

### Giải thích

- **Vì sao dựng khung trước khi viết code?** Bốn người viết song song. Có sẵn tên file, tên thư mục và nơi ghi log thì mỗi người làm phần mình mà không đặt tên lệch nhau. Mỗi file rỗng có docstring ghi người phụ trách và hạn, ví dụ `src/data.py`: "Phụ trách: Trúc (hạn 29/10)", `run_A.py`: "Phụ trách: Trung".
- **`.gitkeep`**: git không lưu thư mục rỗng. Đặt một file rỗng tên `.gitkeep` vào để thư mục `results/A/logs/`… có sẵn trên máy mọi người sau khi clone.
- **`src/paths.py`**: `ROOT = Path(__file__).resolve().parent.parent` lấy gốc project từ vị trí file, rồi ghép `configs`, `data`, `results`, `models`. Cả nhóm import đường dẫn từ đây nên không ai phải ghi đường dẫn tuyệt đối kiểu `C:\Users\...`. Đề mục 5 yêu cầu "không phụ thuộc đường dẫn cá nhân". (Ngày 31/10 thêm `REPORT_DIR`.)
- **`configs/baseline.yaml` bản đề xuất**: ghi sẵn seed 42, miền, μ, C, số mẫu 2000/500/100, kiến trúc 2 × 16 tanh, Adam 1e−3, batch 32, 1000 epoch, patience 100. B lúc này chỉ có 1…10 nơ-ron. Ngày 29/10 nhóm chốt, thêm 16/32/64 cho B và `normalize_y: false` (xem mục 29/10).
- **`requirements.txt`**: numpy, pandas, matplotlib, pyyaml, torch, ipykernel. Không có scipy vì tự viết hàm Gaussian, không có tensorflow/keras vì đề cấm.
- **Đọc đề, rút ra chỗ dễ mất điểm** (ghi trong `Kế hoạch.md` mục 2): code mẫu dùng Keras và báo cáo mẫu dùng C = ma trận đơn vị, lấy test làm validation, nên không chép theo; pipeline phải xong trước 31/10 vì A/B/C đều phụ thuộc vào nó.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao không dùng code mẫu?* Code mẫu viết bằng Keras, đề cấm TensorFlow/Keras. Chỉ mượn cách chia hàm (sinh dữ liệu, đọc dữ liệu, vẽ 3D), phần mô hình và vòng train viết lại bằng PyTorch.
- *Vì sao baseline ghi trong file yaml?* Đề yêu cầu công bố baseline và chỉ đổi đúng một biến trong mỗi nhóm thực nghiệm. Cả nhóm đọc chung một file nên không ai vô tình dùng tham số khác.

### Còn treo

- [x] Chốt baseline với nhóm (làm ngày 29/10).
- [x] Viết `src/data.py` (làm ngày 29/10).

---

## T5 29/10/2026 – Viết `src/data.py`, sinh dữ liệu

**Việc theo kế hoạch:** viết `data.py` (hàm Gaussian, sinh pool train 2000 / val 500 / test 100, lưu CSV) và chốt baseline với nhóm.

### Commit trong ngày

| Hash | Người | Message | Làm gì |
| --- | --- | --- | --- |
| `dc76a8f` | Trúc | Viết `src/data.py`, sinh dữ liệu | Đưa `src/data.py`, 4 file `data/*.csv`, `configs/baseline.yaml` đã chốt, `Kế hoạch.md` đã sửa và nhật ký này lên `main`. |
| `0902fe5` | Khang | Lõi PyTorch | Đẩy cùng lúc `model.py`, `train.py`, `evaluate.py` và nháp mục 3. Sáng 30/10 Khang đã revert commit này (xem mục 30/10). |

### Kết quả

Chạy `python -m src.data` (đứng ở thư mục BTL01) sinh ra:

| File | Nội dung |
| --- | --- |
| `data/train_pool.csv` | 2000 mẫu, cột `x1, x2, y` |
| `data/val.csv` | 500 mẫu |
| `data/test.csv` | 100 mẫu |
| `data/stats.csv` | Thống kê x₁, x₂, y của từng tập (cho mục 2 báo cáo) |

Mọi tham số (seed 42, miền, μ, C, số mẫu) đọc từ `configs/baseline.yaml`, không ghi cứng trong code.

### Giải thích từng hàm trong `data.py`

**`load_config()`**: đọc `configs/baseline.yaml`. Muốn đổi seed hay số mẫu thì chỉ sửa file yaml.

**`gaussian_2d(x, mu, cov)`**: tính đúng công thức đề bài

$$f(\mathbf{x}) = \frac{1}{2\pi\sqrt{\det C}} \exp\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T C^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

- `d = x - mu` là (x − μ), tính cho cả n điểm một lúc.
- `np.einsum("ni,ij,nj->n", d, C⁻¹, d)` tính (x−μ)ᵀ C⁻¹ (x−μ) cho từng điểm. Đọc chuỗi `"ni,ij,nj->n"` như sau: với mỗi điểm n, nhân vector d với ma trận C⁻¹ rồi nhân tiếp với d, ra một số. Viết vòng `for` cũng ra kết quả đó nhưng chậm hơn.
- Dòng cuối là `exp(−½ · …) / (2π √det C)`.
- Tự viết thay vì dùng `scipy.stats.multivariate_normal` như code mẫu, vì rubric chấm "viết hàm Gaussian đúng công thức" (5 điểm). Thêm nữa, scipy không có trong `requirements.txt`.

**`sample_split(n, rng, ...)`**: lấy x₁ đều trong [−2, 2], x₂ đều trong [−4, 4], rồi gắn nhãn y = f(x).

**`generate_splits(cfg)`**: sinh 3 tập. Chỗ quan trọng là `default_rng(42).spawn(3)`: từ seed 42 tách ra 3 bộ sinh số ngẫu nhiên riêng, mỗi tập dùng một bộ.
- Nếu dùng chung một bộ, train rút số trước rồi val, test rút tiếp phần sau. Đổi `n_train_pool` từ 2000 sang 3000 sẽ làm val và test đổi theo, và các kết quả đã chạy không còn so được với nhau.
- Tách riêng giúp 3 tập độc lập hẳn. Đây là câu trả lời cho ý "chia tập và seed hợp lý, tránh rò rỉ" (3 điểm rubric).

**`save_splits()`**: ghi 3 tập ra `data/*.csv`.

**`load_split(name)`**: đọc CSV, trả về `X` shape **(n, 2)** và `y` shape **(n, 1)**, kiểu **float32**.
- PyTorch mặc định dùng float32. Đưa float64 vào sẽ bị lỗi kiểu dữ liệu với trọng số của `nn.Linear`.
- Mô hình xuất ra shape (n, 1). Nếu y có shape (n,), `MSELoss` tự broadcast thành ma trận n×n và tính ra loss sai. Lỗi này chỉ hiện một dòng warning nên rất dễ bỏ sót.

**`load_train(n)`**: lấy n dòng đầu của pool train. Tập A = 200 nằm trọn trong A = 400, A = 400 nằm trọn trong A = 600, v.v., nên các cấu hình A chỉ khác nhau đúng ở số mẫu.

**`describe()`**: tính mean, std, min, tứ phân vị, max và tỷ lệ y < 0.01 cho từng tập, lưu vào `stats.csv`.

**`check()`**: tự kiểm tra trước khi ghi file. Nó xác nhận đủ số mẫu, x nằm trong miền, 0 < y ≤ đỉnh, và không có điểm nào trùng giữa train, val, test. Sai chỗ nào thì chương trình dừng, không ghi ra CSV sai.

### Cách nhóm dùng dữ liệu

```python
from src.data import load_train, load_split

X_train, y_train = load_train(800)    # Thực nghiệm A: đổi số mẫu
X_train, y_train = load_train(1000)   # B, C: mặc định 1000 mẫu
X_val, y_val = load_split("val")      # chọn mô hình, early stopping
X_test, y_test = load_split("test")   # CHỈ dùng đánh giá cuối
```

### Đã kiểm tra

- det C = 0.35, đỉnh f(μ) = 0.269021.
- Tính tay tại (1, 1): C⁻¹ = (1/0.35)·[[2, 0.5], [0.5, 0.3]], dạng toàn phương = 3.3/0.35 = 9.4286, f = e^(−4.7143) / (2π√0.35) = 0.0024121140460877264. Hàm trả về 0.002412114046087728, chỉ lệch ở chữ số thứ 16 do làm tròn.
- Chạy lại lần hai, 4 file CSV giống hệt từng byte (so hash), nên dữ liệu tái lập được.
- `load_train(200)` trả về (200, 2) và (200, 1), float32. 200 dòng đầu của A = 400 trùng với A = 200.

### Cách tự kiểm tra lại

Mở Anaconda Prompt rồi chạy:

```powershell
cd /d "D:\Data\GitHub\deep-learning-assignments\BTL01 - Function Approximation"
conda activate btl01
python -m src.data                      # sinh lại dữ liệu; không có AssertionError là check() qua hết
python -c "from src.data import load_train; X, y = load_train(200); print(X.shape, y.shape, X.dtype)"
python -c "from src.data import gaussian_2d; print(gaussian_2d([[1, 1]], [0, 0], [[0.3, -0.5], [-0.5, 2]]))"
python -c "from src.data import load_config; c = load_config(); print(c['train']['normalize_y'], len(c['experiments']['B']['hidden']))"
```

Lưu ý: Anaconda Prompt là cmd. Thiếu `/d` thì `cd` sang ổ D không có tác dụng (dấu nhắc vẫn là `C:\Users\...`) và Python báo `No module named 'src'`. Trước khi chạy, nhìn dấu nhắc phải thấy `D:\...\BTL01 - Function Approximation>`.

Kết quả đúng: bảng thống kê, `(200, 2) (200, 1) float32`, `[0.00241211]`, `False 13`. Sau khi commit CSV, chạy lại `python -m src.data` rồi `git status`: không thấy `data/*.csv` bị sửa nghĩa là dữ liệu tái lập được.

### Số liệu dùng cho báo cáo mục 2

| Tập | Độ lệch chuẩn x₁ | Độ lệch chuẩn x₂ | Trung bình y | Max y | Tỷ lệ y < 0.01 |
| --- | --- | --- | --- | --- | --- |
| train_pool | 1.144 | 2.310 | 0.0318 | 0.2685 | 61.6% |
| val | 1.227 | 2.268 | 0.0277 | 0.2663 | 65.2% |
| test | 1.142 | 2.292 | 0.0332 | 0.2558 | 65.0% |

- Độ lệch chuẩn khớp lý thuyết phân bố đều (độ dài miền / √12): x₁ là 4/√12 = 1.155, x₂ là 8/√12 = 2.309.
- Khoảng 62–65% mẫu có y gần 0. MSE vì vậy sẽ rất nhỏ (cỡ 1e−5), nên bảng kết quả cần ghi dạng số mũ và báo cáo thêm RMSE để so với đỉnh 0.269.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao không dùng test để chọn mô hình?* Test phải giữ "chưa thấy" để ước lượng sai số thật. Chọn cấu hình theo test là để thông tin test rò vào quá trình chọn, kết quả sẽ lạc quan hơn thực tế. Việc chọn mô hình dùng val.
- *Vì sao y có shape (n, 1)?* Để khớp đầu ra của mô hình, tránh `MSELoss` broadcast thành ma trận n×n.
- *Vì sao các tập dùng bộ sinh ngẫu nhiên riêng?* Đổi kích thước một tập không làm thay đổi tập khác.

### Còn treo

- [x] Chốt baseline với nhóm (xem mục "Chốt baseline" bên dưới).
- [x] Soạn tin gửi nhóm cách dùng `load_train` / `load_split` (Trúc tự gửi lên nhóm chat).
- [x] Commit `src/data.py` + `data/*.csv` + `configs/baseline.yaml` + `Kế hoạch.md` (đã commit `dc76a8f`, đẩy thẳng lên `main`, không qua nhánh `data`).

### Chốt baseline (chiều 29/10)

Hai điểm còn mở đã quyết như sau:

**1. Không chuẩn hóa y** (`normalize_y: false` trong yaml)
- Adam chia gradient cho độ lớn trung bình của chính nó, nên y nhỏ (đỉnh 0.269) không làm mạng học chậm đi. Đây là lý do chính để giữ y gốc.
- Nếu chuẩn hóa thì phải nhớ đổi ngược về thang gốc trước khi tính MSE/MAE/RMSE. Chỉ cần một người quên là số liệu trong báo cáo sai. Giữ y gốc thì không có rủi ro này.
- Đổi lại, loss rất nhỏ (cỡ 1e−5), nên bảng phải ghi dạng số mũ và kèm RMSE để so với đỉnh 0.269.

**2. Thực nghiệm B chạy thêm 16, 32, 64 nơ-ron** (ngoài 1…10)
- Với 1…10 nơ-ron, mạng có thể chỉ thiếu khớp. Rubric B lại chấm phần nhận xét thiếu khớp *và* quá khớp, nên cần mạng rộng hơn để có cái mà so sánh.
- Khối lượng B tăng từ 30 lên 39 lần chạy, tổng cả bài là 81 lần. Đã sửa số này trong `Kế hoạch.md`.

File đã sửa:
- `configs/baseline.yaml`: bỏ chữ "ĐỀ XUẤT", thêm `normalize_y: false`, B thêm `[16], [32], [64]`.
- `Kế hoạch.md` mục 2 và 5, cùng sơ đồ gantt: cập nhật số lần chạy, ghi "đã chốt", thêm dòng "Chuẩn hóa y".

Baseline cuối cùng:

| Tham số | Giá trị |
| --- | --- |
| Seed dữ liệu / seed train | 42 / 0, 1, 2 |
| Kiến trúc baseline | 2 tầng ẩn × 16, tanh, đầu ra tuyến tính |
| Số mẫu train mặc định (B, C) | 1000 |
| Optimizer, lr, batch | Adam, 1e−3, 32 |
| Dừng | Tối đa 1000 epoch, early stopping patience 100 theo val loss |
| Chuẩn hóa y | Không |
| A | 200, 400, …, 2000 mẫu |
| B | 1 tầng ẩn, 1…10, 16, 32, 64 nơ-ron |
| C | 1, 2, 3, 4 tầng ẩn × 16 |

---

## T6 30/10/2026 – Nhóm đẩy `model.py`, `train.py`, runner A/B/C

Ngày này Trúc không có commit. Mục này ghi lại commit của các bạn để Trúc (trưởng nhóm) nắm tiến độ và chuẩn bị review chéo ngày 31/10. Phần việc của Trúc theo kế hoạch ngày 30/10 (hình dữ liệu, nháp mục 2) làm xong rạng sáng 31/10, ghi ở mục 31/10.

### Việc theo kế hoạch

| Người | Kế hoạch 30/10 | Thực tế |
| --- | --- | --- |
| Trúc | Scatter 3D, bề mặt hàm thật, histogram; nháp mục 2 | Chưa commit trong ngày, xong sáng 31/10 |
| Khang | Viết `train.py` | Xong (`f626842`); tiếp tục hoàn thiện `evaluate.py` trong ngày 31/10 (`1c73723`) |
| Trung | Hàm chạy nhiều seed + gộp mean ± std | Xong, và chạy luôn đủ A (30 lần) và B (39 lần), sớm hơn lịch |
| Nhân | Hoàn thiện `plots.py`, phác `run_C.py` | `run_C.py` xong (`5374531`). `plots.py` vẫn trống |

### Commit trong ngày

| Hash | Người | Message | Làm gì |
| --- | --- | --- | --- |
| `759cbb1` | Khang | Revert "Lõi PyTorch" | Gỡ toàn bộ commit `0902fe5` tối 29/10 (model, train, evaluate, nháp mục 3) để làm lại từng phần theo đúng ngày. |
| `8157831` | Khang | định nghĩa MLP tùy biến | Viết lại `src/model.py`: lớp `MLP(hidden, activation)` và file `tra_loi_van_dap_K.md`. |
| `9dcb237` | Trung | runner a , b | `experiments/run_A.py`, `run_B.py`, 69 file log, `summary.csv` và hình của A, B; sửa README phần chạy. |
| `5374531` | Nhân | Implement experiment C runner | `experiments/run_C.py`: chạy C = 1..4 tầng × 16 nơ-ron, xuất `summary.csv` và bảng mean ± std. |
| `f626842` | Khang | Training loop | `src/train.py`: hàm `train_model` và `load_checkpoint`, thêm phần 30/10 vào `tra_loi_van_dap_K.md`. |

### Tóm tắt code của các bạn (để Trúc trả lời được khi bị hỏi)

**`src/model.py` (Khang)**: `MLP(hidden=[16, 16], activation="tanh")` tạo mạng 2 → 16 → 16 → 1. Mỗi số trong `hidden` là một tầng ẩn gồm `nn.Linear` rồi hàm kích hoạt. Tầng cuối là `nn.Linear(…, 1)` không có kích hoạt, vì y là số thực (không ép vào khoảng (−1, 1) như tanh). Nhờ nhận danh sách, cùng một lớp dùng được cho B (`[1]`…`[64]`) và C (`[16]`…`[16,16,16,16]`).

**`src/train.py` (Khang)**: `train_model(X_train, y_train, X_val, y_val, …)` không nhận test. Mỗi epoch: `model.train()`, duyệt `DataLoader` (batch 32, xáo trộn theo seed), `zero_grad → forward → MSELoss → backward → step`; sau đó `model.eval()` + `torch.no_grad()` tính val loss. Val loss tốt hơn thì sao chép `state_dict` làm checkpoint, 100 epoch liên tiếp không tốt hơn thì dừng. Cuối cùng nạp lại checkpoint tốt nhất, nên mô hình trả về là mô hình tốt nhất trên val, không phải mô hình ở epoch cuối. Thời gian đo bằng `time.perf_counter()` bao quanh vòng epoch.

**Runner A, B, C (Trung, Nhân)**: lặp qua giá trị cần khảo sát × seed 0, 1, 2, gọi `train_model`, tính MSE/MAE/RMSE, ghi log từng lần chạy vào `results/<nhóm>/logs/` và bảng gộp vào `summary.csv`. A lấy `pool_x[:n_train]`, tương đương `load_train(n)`.

### Đã kiểm tra (Trúc, sáng 31/10)

Chạy `python -c "import experiments.run_A"` (và B, C) trên `main` hiện tại:

```text
ImportError: cannot import name 'evaluate_model' from 'src.evaluate'
```

Cả ba runner gọi `evaluate_model` trong `src/evaluate.py`, nhưng tại lần kiểm tra đầu tiên file này chỉ có docstring. Khang đã bổ sung hàm đánh giá trong commit `1c73723` lúc 22:24 ngày 31/10. Ở thời điểm kiểm tra ban đầu, `src/plots.py` cũng trống; Nhân sau đó hoàn thiện file này và chạy đủ C trong commit `48fbf36`.

- Sau commit `1c73723`, smoke test A đã chạy thành công. Ở cuối ngày 31/10, các log A/B cũ được kiểm tra tái lập bằng cách chạy lại một số cấu hình đại diện; theo lịch, Trung tiếp tục chạy lại đủ A vào ngày 1/11 (xem mục bên dưới).
- Log A/B ban đầu được commit lúc 15:09 ngày 30/10, trước khi `train.py` lên repo (23:47) và khi `evaluate.py` còn trống. Ngày 1/11, Trung đã chạy lại đủ 30 lượt A bằng code hiện tại, cập nhật log và summary; log B cũ vẫn được giữ vì các cấu hình kiểm tra đại diện đã tái lập khớp từng epoch.

Nhận xét sơ bộ từ `summary.csv` (chỉ để định hướng, sẽ thay bằng số chạy lại):

- Ở A, `best_epoch` trung bình từ 835 đến 994 trên tối đa 1000. Val loss gần như vẫn còn giảm khi chạm 1000 epoch, early stopping hầu như không kích hoạt. Cần nói rõ trong mục 4 rằng mô hình dừng vì giới hạn epoch.
- Ở B, val MSE nhìn chung giảm từ 1 đến 64 nơ-ron (2.9e−3 xuống 3e−6), chưa thấy quá khớp. *(Đính chính 31/10: không giảm đều. B = 8 → 9 → 10 val MSE tăng nhẹ 9.5e−5 → 1.07e−4 → 1.10e−4. Train MSE của B = 9, 10 (1.03e−4, 9.8e−5) cũng cao hơn B = 8 (8.3e−5), nên đây là do huấn luyện chưa hội tụ hết, không phải quá khớp. Quá khớp là train giảm mà val tăng.)* Dữ liệu không nhiễu và val dày (500 điểm) nên khó quá khớp. Phần thảo luận B cần nói thẳng điều này thay vì gượng ép "quá khớp".
- B = 1 nơ-ron có val MSE 2.9e−3, trong khi đoán ŷ = 0 cho mọi điểm đã được 3.7e−3 trên val (trung bình y²). Một nơ-ron gần như không học được hàm, ví dụ rõ cho thiếu khớp.
- `run_C.py` tính test cho mọi cấu hình C. Được phép nếu chỉ báo cáo và chọn cấu hình theo val. Khi viết 4.3 phải ghi rõ cột dùng để chọn là `val_mse`.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao Khang revert rồi commit lại?* Commit `0902fe5` gộp việc của ba ngày vào một lần. Tách ra theo từng ngày để lịch sử git khớp với phân công trong kế hoạch.
- *Mô hình trả về là ở epoch nào?* Epoch có val loss nhỏ nhất (`best_epoch`), không phải epoch cuối, vì `train_model` nạp lại checkpoint trước khi trả về.
- *Vì sao 64 nơ-ron không quá khớp?* Nhãn không có nhiễu, nên khớp sát train cũng là khớp sát hàm thật. Quá khớp xuất hiện rõ khi dữ liệu có nhiễu hoặc rất ít mẫu so với số tham số.

### Còn treo

- [x] Khắc phục lỗi import: Khang bổ sung `evaluate_model` trong `1c73723`; smoke test A xác nhận runner đã chạy được.
- [x] Nhân hoàn thiện `plots.py` và chạy đủ C (commit `48fbf36`, đã có log và hình trong repo).
- [x] Kiểm tra tái lập log A/B cũ: tối 31/10 Trúc train lại `A_1000_seed0`, `B_1_seed0`, `B_10_seed2` (và `C_2_seed0`) bằng code hiện tại trên repo; val loss trùng log cũ từng epoch, chênh 0.0. Theo lịch ngày 1/11, Trung còn chạy lại toàn bộ A; B tiếp tục dùng log cũ đã được kiểm chứng đại diện.

---

## T7 31/10/2026 – Vẽ hình dữ liệu, nháp mục 2; chạy baseline end-to-end và review chéo (Mốc 1)

### Commit trong ngày

| Hash | Người | Message | Làm gì |
| --- | --- | --- | --- |
| `2309c3e` | Trúc | Vẽ hình dữ liệu, nháp mục 2 | Thêm script vẽ ba hình dữ liệu, các hình kết quả và nháp mục 2; cập nhật README, đường dẫn báo cáo và nhật ký. |
| `9568ca7` | Trúc | Merge branch `main` | Gộp thay đổi runner A/B và hàm dùng chung chạy nhiều seed vào nhánh local. |
| `48fbf36` | Nhân | Complete experiment C and plots | Hoàn thiện `plots.py`, runner C và chạy đủ 12 tổ hợp C; log, bảng tổng hợp và hình đã có trong repo. |
| `86bd04b` | Nhân | Merge branch `main` | Gộp các thay đổi ngày 31/10; `main` hiện tại đã ở commit này. |
| `1c73723` | Khang | `evaluate.py`: tính MSE, MAE, RMSE | Bổ sung metric, hàm dự đoán và biểu đồ; khắc phục lỗi import ngăn runner A/B/C khởi động. |
| `e80cce7` | Trung | chạy thử A với 2 giá trị và chạy đầy đủ a b | Chỉ sửa `CodetoTruc.md` (ghi kết quả smoke test A, tick việc treo). **Không có file log hay `summary.csv` nào thay đổi**: log A/B trong repo vẫn là bản của `9dcb237` (30/10). Phần "chạy đầy đủ a b" trong message không khớp nội dung commit. |
| _(điền sau)_ | Trúc | _(commit tối 31/10)_ | `scripts/run_baseline.py`, `results/baseline/`, sửa bug sau review trong `run_A.py`, `run_B.py`, `run_C.py`, `evaluate.py`; README, `Kế hoạch.md` (B 13 giá trị), nhật ký này (thêm mục 28/10, đính chính). |

Ghi chú: lần kiểm tra đầu phiên, `main` local sau `origin/main` 2 commit; sau đó đồng bộ đến `86bd04b` và hiện `main`/`origin/main` cùng ở `e80cce7`.

**Việc theo kế hoạch:**

| Người | Kế hoạch 31/10 | Thực tế |
| --- | --- | --- |
| Trúc | Chạy thử baseline end-to-end; review code chéo | Xong cả hai (chi tiết bên dưới). Kèm việc của 30/10 (hình dữ liệu, nháp mục 2) làm xong rạng sáng 31/10 |
| Khang | Viết `evaluate.py`; sửa lỗi sau review | `evaluate.py` xong (`1c73723`). Sửa lỗi sau review: còn chờ các góp ý ở mục Review |
| Trung | Chạy thử A với 2 giá trị, 1 seed | Xong (A = 200, 400, seed 0) |
| Nhân | Chạy thử C với 1 giá trị, 1 seed | Làm vượt: chạy đủ C (4 × 3 seed) trong `48fbf36` |

→ **Mốc 1 (pipeline chạy được) đạt.** Đề mục 3.1 về hình dữ liệu được giải thích ở các phần đầu; phần baseline và review nằm ở các tiểu mục có chữ "baseline" và "Review".

Đề mục 3.1 yêu cầu: "Vẽ scatter 3D của dữ liệu, bề mặt hàm thật và mô tả phân bố của x₁, x₂, y". Rubric mục Dữ liệu có 2 điểm cho "trực quan hóa/kiểm tra dữ liệu".

### Kết quả

| File | Trạng thái | Nội dung |
| --- | --- | --- |
| `scripts/plot_data.py` | viết mới | Vẽ 3 hình từ `data/*.csv` |
| `report/figures/fig_data_scatter3d.png` | sinh ra | Hình 2.1: scatter 3D của train/val/test |
| `report/figures/fig_data_true_surface.png` | sinh ra | Hình 2.2: (a) bề mặt 3D, (b) đường đồng mức |
| `report/figures/fig_data_histograms.png` | sinh ra | Hình 2.3: phân bố x₁, x₂, y của từng tập |
| `report/section_2_data.md` | viết mới | Nháp mục 2 báo cáo |
| `src/paths.py` | sửa | Thêm `REPORT_DIR = ROOT / "report"` |
| `README.md` | sửa | Thêm lệnh `python -m scripts.plot_data`; tối 31/10 thêm `run_baseline` và `run_C` (trước đó README thiếu lệnh chạy C) |
| `scripts/run_baseline.py` | viết mới (tối 31/10) | Chạy thử baseline end-to-end |
| `results/baseline/logs/baseline_seed0.csv` | sinh ra | Loss train/val từng epoch (1000 dòng) |
| `results/baseline/baseline_seed0.pt` | sinh ra | Checkpoint tốt nhất theo val loss |
| `results/baseline/figures/baseline_seed0_val_y_yhat.png` | sinh ra | y–ŷ trên val |
| `results/baseline/figures/baseline_seed0_surface.png` | sinh ra | Bề mặt hàm thật cạnh bề mặt MLP |

Lệnh chạy (đứng ở thư mục BTL01, sau khi đã có `data/*.csv`):

```powershell
python -m scripts.plot_data
python -m scripts.run_baseline
```

### Giải thích từng hàm trong `plot_data.py`

**Đầu file**: `matplotlib.use("Agg")` chọn backend chỉ ghi file, không mở cửa sổ. Script chạy được cả trên máy không có màn hình và không bị treo chờ đóng cửa sổ.

**`STYLE`**: mỗi tập một màu cố định (train xanh dương, val cam, test xanh lục) *và* một hình marker (tròn, tam giác, vuông). Có hình marker thì in đen trắng hoặc người mù màu vẫn phân biệt được. Ba màu đã chạy qua bộ kiểm tra bảng màu cho người mù màu, đều đạt.

**`BLUES`**: thang màu một sắc xanh từ nhạt đến đậm cho độ lớn f(x). Giá trị càng lớn càng đậm, đọc được ngay mà không cần đoán như thang cầu vồng.

**`load_splits()`**: đọc 3 file CSV thành dict `{tên tập: DataFrame}`. Dùng `SPLITS` từ `src.data` để thứ tự và tên tập luôn khớp với `data.py`.

**`plot_scatter3d()`**: vẽ cả 2600 điểm (x₁, x₂, y) trên một trục 3D. Tập nhỏ vẽ marker to hơn và vẽ sau để không bị 2000 điểm train che mất. `depthshade=False` để màu không bị nhạt đi theo độ sâu, giữ đúng màu của từng tập.

**`plot_true_surface()`**: tính f trên lưới 201 × 201 điểm phủ toàn miền, dùng lại đúng hàm `gaussian_2d` của `data.py` (không viết lại công thức lần hai, tránh lệch nhau).
- Bên trái: `plot_surface` cho bề mặt 3D.
- Bên phải: `contourf` tô màu theo mức và `contour` vẽ 4 đường mức 0.01, 0.05, 0.10, 0.20. `set_aspect("equal")` giữ tỷ lệ 1:1 giữa hai trục. Không có dòng này, matplotlib kéo giãn trục x₁ và elip trông như không nghiêng.
- Lưới đều khác dữ liệu ngẫu nhiên: bề mặt này là hàm thật, không phải dữ liệu, nên phải vẽ trên lưới để mặt liền mạch.

**`plot_histograms()`**: 3 hình con cho x₁, x₂, y.
- `weights = 100 / len(df)` cho mỗi mẫu, nên chiều cao cột là **% số mẫu của tập đó**. Train 2000 mẫu và test 100 mẫu nhờ vậy vẽ chung một trục được. Nếu đếm số mẫu thì cột test thấp hơn cột train 20 lần, không so được.
- `histtype="step"` vẽ viền thay vì tô đặc, để 3 tập chồng lên nhau vẫn nhìn thấy cả 3.
- Với x₁, x₂: chia 10 khoảng bằng nhau, phân bố đều thì mỗi khoảng 10%, vẽ đường đứt nét ở 10% để so.
- Với y: ghi trực tiếp tỷ lệ y < 0.01 và giá trị đỉnh lên hình.

**`main()`**: gọi 3 hàm vẽ, lưu vào `report/figures/`, in đường dẫn từng file.

### Giải thích `scripts/run_baseline.py` (tối 31/10)

"End-to-end" nghĩa là đi hết một vòng từ đầu đến cuối bằng đúng code trong repo: đọc CSV → train → lưu `.pt` → nạp lại `.pt` → tính chỉ số → vẽ hình. Mỗi file do một người viết, nên chỉ chạy thật cả chuỗi mới biết các file có nối khớp với nhau không (tên hàm, shape mảng, kiểu dữ liệu).

- **`matplotlib.use("Agg")` đặt trước khi import `src.evaluate`**: `evaluate.py` import `pyplot` ngay đầu file. Nếu chọn backend sau đó thì quá muộn, máy không màn hình có thể lỗi.
- **`torch.set_num_threads(1)`**: giống runner A/B/C. Thời gian train đo trong cùng điều kiện, và dùng 1 luồng thì phép cộng số thực luôn theo cùng thứ tự, kết quả lặp lại được.
- **Đọc tham số từ `configs/baseline.yaml`**, không gõ tay số nào: `n_train = 1000`, `hidden = [16, 16]`, `tanh`, Adam, lr 1e−3, batch 32, 1000 epoch, patience 100. Đổi baseline chỉ cần sửa một chỗ.
- **`load_train(n_train)`** lấy 1000 dòng đầu của pool, đúng cách runner A/B/C lấy dữ liệu (`pool_x[:n_train]`).
- **`train_model(..., log_path=..., checkpoint_path=...)`**: runner A/B/C không truyền hai tham số này, nên phần ghi log/checkpoint của `train.py` chưa ai chạy. Script này chạy luôn hai nhánh đó.
- **Nạp lại checkpoint và so dự đoán**: `load_checkpoint` dựng lại `MLP` từ `hidden`, `activation` lưu trong file rồi nạp trọng số. So dự đoán trên val của mô hình nạp lại với mô hình trong bộ nhớ, lệch phải bằng 0. Ngày 3/11 nhóm sẽ lưu `best_model.pt` và nạp lại để chấm test, nên nạp lại phải đúng từ bây giờ.
- **Tính chỉ số bằng mô hình nạp lại** (không dùng mô hình trong bộ nhớ): mọi số in ra là số của chính file `.pt`.
- **`np.mean(y_val**2)`**: MSE nếu đoán ŷ = 0 cho mọi điểm, làm mốc so sánh. MSE của mô hình phải nhỏ hơn mốc này nhiều mới gọi là học được.
- **`plot_y_vs_yhat`, `plot_surface_comparison`**: hai hình đề yêu cầu ở mục "Đánh giá chung", lần đầu được chạy thật.
- **Không đọc `test.csv`**: chạy thử thì không được nhìn test, nếu không sẽ bị nghi đã dùng test để chỉnh mô hình.

### Đã kiểm tra

- Lần chạy đầu `python -m experiments.run_A --values 200 400 --seeds 0` bị dừng ở import vì `evaluate_model` chưa có. Sau khi Khang đẩy `1c73723`, chạy lại cùng lệnh thành công:
  - A=200, seed 0: val MSE `5.484254e-06`, MAE `0.00178050`, RMSE `0.00234185`, best epoch 991.
  - A=400, seed 0: val MSE `1.562960e-06`, MAE `0.000988967`, RMSE `0.00125018`, best epoch 999.
- Đây là smoke test theo yêu cầu ngày 31, chỉ 2 cấu hình và 1 seed; không phải kết quả A đầy đủ (10 cấu hình × 3 seed), cũng không thay thế baseline end-to-end hoặc chạy B. Các file kết quả A có sẵn từ trước được giữ nguyên sau khi thử để không làm mất bảng đủ cấu hình.
- Import cả ba module `experiments.run_A`, `experiments.run_B`, `experiments.run_C` sau commit `1c73723` đều thành công.
- Chạy 2 lần liên tiếp, 3 file PNG giống hệt từng byte (so MD5), nên hình tái lập được.
- Đã mở xem từng hình: nhãn trục không chồng nhau, chữ tiếng Việt hiện đúng dấu.
- Các số trong nháp mục 2 đều tính lại bằng code, không chép tay:
  - Trị riêng của C: 0.164 và 2.136. Trục dài hướng (−0.263, 0.965), lệch 15.2° so với trục x₂.
  - Tương quan ρ = −0.5 / √0.6 = −0.645.
  - Đường mức 0.01: x₂ từ −3.625 đến 3.625, tại x₂ = 0 thì x₁ từ −1.07 đến 1.07.
  - Trung bình y² (MSE nếu đoán ŷ = 0): train 0.00445, val 0.00368, test 0.00511.
  - Val có 13.6% mẫu ở x₁ ∈ [1.6, 2], lý do độ lệch chuẩn x₁ của val (1.227) cao hơn lý thuyết (1.155).

**Baseline end-to-end (tối 31/10, env `btl01`, `torch.__version__` in ra `2.15.0+cpu`):**

- `python -m src.data` sinh lại dữ liệu, `git status data` không thấy thay đổi: CSV tái lập được từ seed 42.
- `python -m scripts.run_baseline` chạy hết, không lỗi:

  ```text
  Kiến trúc 2 -> 16 -> 16 -> 1, tanh, 337 tham số; n_train = 1000, seed = 0
  Dừng ở epoch 1000, best epoch 998, early stopping: False, thời gian train 26.6 s
    train MSE 3.9627e-07  MAE 4.8292e-04  RMSE 6.2950e-04
    val   MSE 4.1756e-07  MAE 5.0138e-04  RMSE 6.4619e-04
    val   MSE khi đoán ŷ = 0: 3.6805e-03
  Checkpoint nạp lại lệch tối đa 0.0e+00 so với mô hình trong bộ nhớ
  ```

- Chạy lần 2: cùng val MSE `4.1756e-07`, cùng best epoch 998. Chỉ thời gian train khác (39.1 s), do máy đang bận việc khác.
- **So chéo với log của bạn**: baseline cùng cấu hình với `C_2_seed0` (Nhân) và `A_1000_seed0` (Trung). Val loss của cả 1000 epoch trùng nhau tuyệt đối (chênh lớn nhất `0.0`), best val loss đều là `4.175624e-07` ở epoch 998.
- **Kiểm tra log B cũ**: train lại bằng `train_model` (gọi thẳng, không qua runner để không ghi đè `summary.csv`) hai cấu hình rồi so với log trong repo:
  - B = 1, seed 0: dừng sớm ở epoch 148 cả hai bên, best `2.915562e-03`, chênh `0.0`.
  - B = 10, seed 2: 1000 epoch, best `9.955913e-05`, chênh `0.0`.
- Kết luận: **log A/B commit ngày 30/10 tái lập được từ code hiện tại trên repo** (4/4 cấu hình thử đều khớp từng epoch). Lo ngại "log A/B không truy vết được" ở mục 30/10 được giải tỏa bằng số liệu. Chỉ có thứ tự cột trong log A cũ khác bản `run_A.py` hiện tại (`n_train` đứng ngay sau `val_loss`), các con số không đổi.
- Hình y–ŷ: các điểm nằm sát đường chéo trên toàn khoảng 0 đến 0.27. Hình bề mặt: MLP trùng hàm thật bằng mắt thường, kể cả hướng nghiêng của elip.

### Review code chéo (tối 31/10)

Đã đọc `model.py`, `train.py`, `evaluate.py`, `experiments/common.py`, `run_A.py`, `run_B.py`, `run_C.py`, `plots.py`.

**Những chỗ đúng yêu cầu rubric** (để Trúc tự tin trả lời khi bị hỏi):

| Yêu cầu | Ở đâu |
| --- | --- |
| `nn.Module` + `nn.Linear`, tùy số tầng/nơ-ron | `MLP(hidden=[...])` trong `model.py` |
| `DataLoader`, `MSELoss`, Adam/SGD | `train.py` dòng 132–143 |
| `model.train()` / `model.eval()` + `torch.no_grad()` | `train.py` dòng 155, 175–177; `evaluate.predict` |
| Vòng train tự viết: `zero_grad → forward → loss → backward → step` | `train.py` dòng 163–167 |
| Log loss theo epoch | `history` trong `train.py`, ghi ra CSV |
| Checkpoint theo val loss, nạp lại bản tốt nhất | `train.py` dòng 189–210 |
| Đo thời gian bằng `time.perf_counter()` | `train.py` dòng 153, 205 |
| Seed cho khởi tạo trọng số và thứ tự xáo batch | `_seed_everything` gọi **trước** khi tạo `MLP`; `DataLoader(generator=...)` |
| Hàm train không nhận test | chữ ký `train_model(X_train, y_train, X_val, y_val, ...)` |

**Góp ý gửi nhóm** (xếp theo mức ảnh hưởng):

1. **Trung, Nhân: chạy `--values` một phần sẽ ghi đè `summary.csv` và hình đầy đủ.** `run_A/B/C.py` luôn ghi `summary.csv` và figure từ những giá trị vừa chạy. Ví dụ `python -m experiments.run_A --values 200 400 --seeds 0` làm `results/A/summary.csv` chỉ còn 2 dòng. Đề xuất: khi có `--values`/`--seeds` thì ghi ra `summary_partial.csv`, hoặc không ghi summary. Trước mắt: chạy thử xong thì `git checkout -- results/` để lấy lại bản đầy đủ. → **Đã sửa** (xem "Sửa bug sau review" bên dưới).
2. **Nhân: hình C chỉ vẽ sai số test, chưa có val.** `plots.py` vẽ `C_test_mse`, `C_test_rmse`; chưa có hình nào cho `val_mse`, mà cấu hình phải được chọn theo val. Cần thêm một hình val (hoặc vẽ val và test chung) để mục 4.3 chọn số tầng theo val. Việc này thuộc "xuất bảng + đồ thị C" ngày 2/11, chưa phải lỗi. Comment ở `run_C.py` dòng 94 ghi test được đánh giá "sau khi chọn mô hình", nhưng thực tế test được tính cho mọi lần chạy. → **Comment đã sửa.**
3. **Nhân: `run_C.py` tự viết `_evaluate` thay vì dùng `src.evaluate.evaluate_model`.** Lúc viết thì `evaluate.py` còn trống nên mới tự viết. → **Quyết định giữ `_evaluate`**: thử thì hai hàm lệch ở chữ số thứ 8 (`_evaluate` tính float32, `evaluate_model` float64), đổi sẽ làm `results/C/summary.csv` không còn khớp từng chữ số với code. Chỉ sửa docstring cho đúng lý do.
4. **Trung, Nhân: tên cột A/B khác C.** A/B dùng `n_train`/`n_neurons`, `elapsed_seconds`, `summary.csv` một dòng mỗi cấu hình (mean/std). C dùng `value`, `train_time_s`, `summary.csv` một dòng mỗi lần chạy kèm `summary_mean_std.csv`. Không sai, nhưng khi ghép bảng báo cáo phải nhớ cột nào là cột nào. Chốt trước 2/11.
5. **Cả nhóm: phần lớn lần chạy dừng vì đủ 1000 epoch, không phải nhờ early stopping.** Đếm số log ngắn hơn 1000 dòng: chỉ 20/81 lần chạy dừng sớm (A 6/30, B 10/39, C 4/12). Baseline seed 0 có best epoch 998/1000, val loss vẫn đang giảm khi chạm giới hạn. Mục 4 phải ghi rõ lần chạy nào dừng sớm, lần nào dừng vì đủ epoch. Không đổi baseline lúc này vì đã chốt. Có thể đưa vào phần hạn chế.
6. **Nhân, Trung: thời gian train phụ thuộc máy và tải.** Cùng một cấu hình (2 × 16, 1000 mẫu, seed 0) mà ra 20.9 s trong `results/C/summary.csv`, 26.6 s và 39.1 s ở hai lần chạy trên máy Trúc. Bảng thời gian ở C chỉ so được khi cả 12 lần chạy trên cùng một máy, cùng một phiên; báo cáo cần ghi cấu hình máy. Thời gian đo cũng gồm cả việc tính val loss mỗi epoch.
7. **Khang: hình bề mặt `plot_surface_comparison` bị chồng nhãn trục x₁** (−2.0, −1.5, −1.0 dính nhau). Thưa bớt vạch chia, ví dụ `axis.set_xticks([-2, -1, 0, 1, 2])`, trước khi đưa vào báo cáo. → **Đã sửa.**
8. **Trung: message `e80cce7` ghi "chạy đầy đủ a b" nhưng commit chỉ sửa nhật ký.** Lần sau message nên ghi đúng nội dung commit, vì thầy có thể xem lịch sử git.

### Sửa bug sau review (tối 31/10, Trúc là nhóm trưởng sửa thay)

Sửa vào file của các bạn, nên các bạn **pull về trước khi sửa tiếp** để không bị xung đột. Không file log hay bảng kết quả nào đã commit bị thay đổi.

| File (người viết) | Sửa gì | Vì sao |
| --- | --- | --- |
| `experiments/run_A.py` (Trung) | Thêm biến `is_full_run`: chạy đủ 10 giá trị × seed 0, 1, 2 thì ghi `summary.csv` + hình như cũ; chạy một phần thì ghi `summary_partial.csv`, không vẽ hình | Góp ý 1: tránh ghi đè bảng đầy đủ khi chạy thử |
| `experiments/run_B.py` (Trung) | Như trên, so với 13 giá trị của B | Như trên |
| `experiments/run_C.py` (Nhân) | Như trên: chạy một phần thì chỉ ghi `summary_partial.csv`, không ghi `summary.csv`, `summary_mean_std.csv`, không gọi `make_group_figures`. Sửa comment dòng 94 và docstring `_evaluate` | Góp ý 1, 2, 3 |
| `src/evaluate.py` (Khang) | `plot_surface_comparison` đặt 5 vạch chia mỗi trục: x₁ −2…2, x₂ −4…4 | Góp ý 7: nhãn x₁ bị chồng |
| `README.md` (Trúc) | Ghi rõ chạy một phần thì ra `summary_partial.csv` | Để cả nhóm biết hành vi mới |

`is_full_run` so **tập hợp** giá trị và seed với cấu hình trong yaml, nên `--values` liệt kê đủ cả 10 giá trị A vẫn được tính là chạy đầy đủ.

**Đã kiểm tra:**

- `python -m experiments.run_A --values 200 --seeds 0`: in `Partial run: wrote ...summary_partial.csv`, val MSE `5.484254e-06` (đúng số smoke test trước). `results/A/summary.csv` và hình không đổi.
- `python -m experiments.run_C --values 1 --seeds 0`: test MSE `1.101137e-05`, trùng dòng C = 1, seed 0 trong `summary.csv` đã commit. `summary.csv`, `summary_mean_std.csv`, 5 hình C không đổi; `C_1_seed0.csv` ghi lại giống hệt từng byte.
- Log `A_200_seed0.csv` ghi lại có số giống hệt bản đã commit, chỉ khác thứ tự cột (bản cũ đặt `n_train` sau `val_loss`) và thời gian chạy. Đã `git checkout` trả về bản đã commit và xóa hai file `summary_partial.csv` sinh ra khi thử.
- Chạy lại `python -m scripts.run_baseline`: số liệu không đổi, hình bề mặt mới có vạch x₁ −2, −1, 0, 1, 2, không còn chồng chữ.

**Câu hỏi vấn đáp có thể gặp:**

- *Vì sao `run_C.py` không dùng chung `evaluate_model`?* `_evaluate` tính MSE trên float32, `evaluate_model` trên float64; hai cách lệch nhau ở chữ số thứ 8 (val MSE C = 2 seed 0: `4.1756243e-07` so với `4.1756242e-07`). Giữ `_evaluate` để kết quả C đã commit tái lập được từng chữ số. Mức lệch này nhỏ hơn rất nhiều so với chênh lệch giữa các seed nên không ảnh hưởng kết luận.
- *Chạy thử một phần thì kết quả nằm ở đâu?* `results/<nhóm>/summary_partial.csv`. File này chỉ để xem nhanh, không commit, không đưa vào báo cáo.

### Cách tự kiểm tra lại

Mở Anaconda Prompt:

```powershell
cd /d "D:\Data\GitHub\deep-learning-assignments\BTL01 - Function Approximation"
conda activate btl01
python -m scripts.plot_data
git status
```

Kết quả đúng: in ra 3 dòng `-> ...\report\figures\fig_data_*.png`. Nếu đã commit hình, `git status` sau khi chạy lại không thấy file PNG nào bị sửa. Mở 3 hình trong `report\figures\`: Hình 2.2b phải thấy elip nghiêng (phía trên lệch sang trái), Hình 2.3 cột đầu tiên của y cao khoảng 70%.

Kiểm tra baseline end-to-end (mất khoảng 30–40 giây):

```powershell
python -m src.data
git status data
python -m scripts.run_baseline
```

Kết quả đúng:

- `git status data` không liệt kê file nào (dữ liệu sinh lại giống hệt).
- `run_baseline` in `best epoch 998`, `val MSE 4.1756e-07`, `RMSE 6.4619e-04` và `lệch tối đa 0.0e+00`. Thời gian train mỗi lần một khác, không sao.
- Mở `results\baseline\figures\baseline_seed0_val_y_yhat.png`: các điểm nằm trên đường chéo đỏ.

Nếu ra số khác ở chữ số thứ 4 trở đi thì gần như chắc chắn khác phiên bản PyTorch hoặc chưa `conda activate btl01`.

### Số liệu cho báo cáo

Đã đưa hết vào [report/section_2_data.md](report/section_2_data.md): Bảng 2.1 (chia tập), Bảng 2.2 (thống kê từ `stats.csv`), Bảng 2.3 (tỷ lệ y < 0.01, > 0.1, > 0.2), Hình 2.1 đến 2.3. Các số Trúc nên nhớ để vấn đáp:

| Đại lượng | Giá trị |
| --- | --- |
| det C | 0.35 |
| Đỉnh f(μ) | 0.2690 |
| Tương quan x₁, x₂ | −0.645 |
| Tỷ lệ y < 0.01 (train / val / test) | 61.6% / 65.2% / 65.0% |
| Trung vị y (train) | 0.0031 |
| MSE khi đoán ŷ = 0 (test) | 0.0051 |

Baseline (2 × 16, tanh, n_train 1000, seed 0), nguồn `results/baseline/logs/baseline_seed0.csv`. Số này trùng với dòng C = 2, seed 0 trong `results/C/summary.csv`, nên báo cáo có thể trích từ bất kỳ file nào:

| Đại lượng | Giá trị |
| --- | --- |
| Số tham số | 337 (2·16+16 + 16·16+16 + 16·1+1) |
| Best epoch / số epoch đã chạy | 998 / 1000 (không dừng sớm) |
| Train MSE / RMSE | 3.96e−07 / 6.30e−04 |
| Val MSE / MAE / RMSE | 4.18e−07 / 5.01e−04 / 6.46e−04 |
| Val MSE khi đoán ŷ = 0 | 3.68e−03 (gấp khoảng 8800 lần MSE của mô hình) |
| RMSE val so với đỉnh 0.269 | ≈ 0.24% |

### Câu hỏi vấn đáp có thể gặp

- *Vì sao elip đồng mức nghiêng?* Vì hiệp phương sai −0.5 khác 0, x₁ và x₂ tương quan âm (ρ = −0.645). Nếu C chéo (như báo cáo mẫu dùng ma trận đơn vị) thì elip thẳng theo trục.
- *Vì sao elip dài theo x₂?* Phương sai theo x₂ là 2, lớn hơn nhiều so với 0.3 của x₁. Trị riêng lớn 2.136 ứng với trục dài, hướng gần trục x₂.
- *Vì sao histogram vẽ theo % thay vì số mẫu?* Các tập có cỡ 2000, 500, 100. Vẽ % mới so được hình dạng phân bố.
- *Histogram test vì sao răng cưa nhiều?* Chỉ có 100 mẫu, mỗi khoảng trung bình 10 mẫu, lệch vài mẫu đã thành vài %. Phân bố vẫn phủ đủ miền nên không cần sinh lại.
- *Phân bố y lệch về 0 thì ảnh hưởng gì?* MSE rất nhỏ nên phải báo RMSE. Vùng đỉnh ít mẫu nhất nên sai số lớn nhất nhiều khả năng nằm quanh μ. Mốc so sánh: đoán toàn 0 đã được MSE ≈ 0.005, mô hình phải thấp hơn nhiều.
- *Vì sao bề mặt hàm thật vẽ trên lưới, không vẽ từ dữ liệu?* Dữ liệu ngẫu nhiên, rải không đều. Hàm thật tính được tại mọi điểm nên vẽ trên lưới đều cho mặt liền.
- *Làm sao biết kết quả của nhóm chạy lại được?* Baseline chạy trên máy Trúc trùng từng epoch với log C = 2 của Nhân và A = 1000 của Trung, hai người chạy trên máy khác. Train lại thêm 2 cấu hình B cũng khớp tuyệt đối. Lý do: mọi nguồn ngẫu nhiên đều gắn seed (dữ liệu seed 42; khởi tạo trọng số và thứ tự batch theo seed huấn luyện), cộng với `torch.set_num_threads(1)`.
- *Vì sao phải lưu rồi nạp lại checkpoint khi chạy thử?* Mô hình cuối được nộp dưới dạng `best_model.pt` và chấm test bằng file đó. Phải chắc rằng nạp lại cho đúng dự đoán như lúc train (lệch 0.0), nếu không số test sẽ không phải của mô hình đã chọn.
- *Checkpoint lưu những gì?* `state_dict` (trọng số và bias của từng `nn.Linear`) cộng với `hidden`, `activation` để dựng lại đúng kiến trúc, và best epoch, best val loss, seed, cấu hình train để truy vết.
- *337 tham số tính thế nào?* Tầng 2→16: 2·16 trọng số + 16 bias = 48. Tầng 16→16: 256 + 16 = 272. Tầng 16→1: 16 + 1 = 17. Tổng 48 + 272 + 17 = 337.
- *Vì sao val MSE nhỏ như vậy mà vẫn chưa dừng sớm?* Val loss vẫn còn giảm đến epoch 998, nên patience 100 chưa bao giờ hết. Mô hình dừng vì chạm giới hạn 1000 epoch. Dữ liệu không nhiễu nên train thêm vẫn còn cải thiện được.
- *Chạy thử có dùng tập test không?* Không. `run_baseline.py` chỉ đọc `train_pool` và `val`. Test để dành cho lần đánh giá duy nhất ngày 3/11, sau khi đã chọn mô hình theo val.

### Còn treo

- [x] Commit `scripts/plot_data.py`, `src/paths.py`, `README.md`, `report/figures/`, `report/section_2_data.md`, `CodetoTruc.md` (đã có trong `2309c3e`).
- [x] Chạy thử A của Trung (2 giá trị, 1 seed): hoàn tất sau khi `evaluate_model` được thêm.
- [x] Chạy baseline end-to-end và review code chéo (tối 31/10, xem các tiểu mục baseline và Review ở trên).
- [ ] Commit phần tối 31/10 rồi điền hash vào bảng commit ở đầu mục này.
- [x] Sửa bug góp ý 1 (ghi đè `summary.csv`), comment góp ý 2–3, góp ý 7 (nhãn trục) — Trúc sửa thay tối 31/10.
- [ ] Gửi nhóm chat: báo đã sửa `run_A.py`, `run_B.py`, `run_C.py`, `evaluate.py` (mọi người pull trước khi sửa tiếp), kèm các góp ý còn lại (2: hình `val_mse` cho C vào ngày 2/11; 4: thống nhất tên cột; 5, 6: ghi chú khi viết báo cáo; 8: message commit).
- [ ] Nhắn Khang: `tra_loi_van_dap_K.md` mục 8.5 ghi val MSE baseline `4.17493e-07`, còn log trong repo (baseline, `A_1000_seed0`, `C_2_seed0`) đều ra `4.17562e-07`. Có thể do khác phiên bản PyTorch (Khang chạy bằng `.venv`), chưa kiểm chứng. File của Khang nên Khang tự sửa; báo cáo lấy số từ log.
- [ ] Khi ghép báo cáo: chuyển công thức LaTeX trong `section_2_data.md` sang Equation của Word, đánh lại số hình/bảng nếu thứ tự mục thay đổi.

---

## CN 01/11/2026 – Chạy đủ thực nghiệm A

### Việc theo kế hoạch

Theo [Kế hoạch.md](Kế%20hoạch.md), ngày 1/11 Trung chạy đủ thực nghiệm A: 10 cỡ mẫu train (200 đến 2000, bước 200), mỗi cỡ mẫu chạy với 3 seed (0, 1, 2), tổng cộng 30 lượt. Giữ nguyên baseline: MLP `[16, 16]`, tanh, Adam, learning rate `1e-3`, batch size 32, tối đa 1000 epoch, patience 100 theo validation loss.

### Commit trong ngày

Chưa có commit ngày 1/11 trong lịch sử Git tại thời điểm ghi mục này. Trúc tự commit; khi commit sẽ bổ sung hash và message.

### Kết quả

Chạy `python -m experiments.run_A` từ thư mục `BTL01 - Function Approximation`. Runner đọc cấu hình và dữ liệu hiện có, chạy đủ 30 tổ hợp, ghi lại:

- `results/A/logs/A_<n_train>_seed<seed>.csv`: log theo epoch cho mỗi lượt chạy, gồm 30 file.
- `results/A/summary.csv`: bảng mean ± sample std theo từng cỡ mẫu, tổng cộng 10 dòng.
- `results/A/figures/A_validation_metrics.png`: đồ thị các metric validation theo số mẫu train.

Không dùng tập test trong thực nghiệm A; các chỉ số trong bảng dưới đây là trên validation.

### Giải thích các hàm liên quan

- `run_experiment()` trong `experiments/run_A.py` đọc baseline, kiểm tra cỡ mẫu/seed, cố định validation, rồi chạy lần lượt từng cỡ mẫu và seed.
- Hàm `run_one()` huấn luyện một cấu hình bằng `train_model()`, tính metric train/validation qua `evaluate_model()`, ghi lịch sử từng epoch và cấu hình vào log CSV.
- `run_seeded_experiments()` trong `experiments/common.py` gọi mỗi cặp giá trị/seed đúng một lần; `summarize_seed_runs()` tổng hợp trung bình và độ lệch chuẩn mẫu giữa ba seed.
- `_plot_summary()` vẽ các metric validation theo cỡ mẫu trên trục log để dễ so sánh sai số nhỏ.

### Đã kiểm tra

- Runner hoàn tất đủ 30 lượt; summary có đủ 10 cỡ mẫu và ghi seed `0,1,2` cho từng cỡ.
- Kiểm tra đủ 30 log: mỗi log có epoch liên tiếp bắt đầu từ 1, cấu hình trong log khớp tên file/cỡ mẫu/seed, metric hữu hạn. Số epoch mỗi lượt từ 617 đến 1000; lượt dừng trước 1000 là do early stopping.
- Toàn bộ lượt dùng đúng baseline đã chốt; `git diff --check` đạt.
- Validation MSE trung bình giảm từ `1.219636e-05` ở 200 mẫu xuống `1.681281e-07` ở 2000 mẫu, khoảng 72.5 lần. Với ba seed, kết quả vẫn dao động; vì vậy báo cáo mean ± std thay vì chỉ chọn một lần chạy.

### Số liệu cho báo cáo mục 4.1

| Số mẫu train | Validation MSE (mean ± std) | Validation RMSE (mean) | Best epoch (mean) |
| ---: | ---: | ---: | ---: |
| 200 | `1.219636e-05 ± 7.144094e-06` | `3.3857e-03` | 985.3 |
| 400 | `2.224003e-06 ± 1.091532e-06` | `1.4638e-03` | 994.3 |
| 600 | `1.222356e-06 ± 2.747918e-07` | `1.1012e-03` | 953.7 |
| 800 | `8.253509e-07 ± 1.502634e-07` | `9.0583e-04` | 853.0 |
| 1000 | `4.350544e-07 ± 2.451342e-08` | `6.5941e-04` | 988.0 |
| 1200 | `3.433571e-07 ± 4.001068e-08` | `5.8528e-04` | 965.3 |
| 1400 | `3.244289e-07 ± 1.587016e-07` | `5.5781e-04` | 858.3 |
| 1600 | `2.853109e-07 ± 1.335988e-07` | `5.2441e-04` | 835.0 |
| 1800 | `2.391381e-07 ± 3.670003e-08` | `4.8807e-04` | 856.0 |
| 2000 | `1.681281e-07 ± 5.886715e-08` | `4.0545e-04` | 892.0 |

### Cách tự kiểm tra lại

Trong Anaconda Prompt:

```powershell
cd /d "D:\Games\code cua c#\deep-learning-assignments\BTL01 - Function Approximation"
conda activate btl01
python -m experiments.run_A
```

Kết quả đúng: terminal in 30 dòng `A n_train=..., seed=...`; `results/A/summary.csv` có 10 dòng cấu hình; `results/A/logs/` có 30 log. Mỗi cấu hình chạy lại có thể khác thời gian đo, còn metric và lịch sử loss phải tái lập trong cùng môi trường/phiên bản thư viện.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao mỗi cỡ mẫu chạy ba seed?* Để đánh giá ảnh hưởng của khởi tạo và thứ tự mini-batch; mean ± std phản ánh cả xu hướng lẫn độ dao động, tránh kết luận từ một lần chạy.
- *Vì sao validation MSE nhìn chung giảm khi tăng số mẫu?* Mạng được cung cấp nhiều điểm huấn luyện hơn để xấp xỉ hàm trên miền; kết quả thực nghiệm cho thấy xu hướng này, dù các cỡ gần nhau có thể dao động giữa seed.
- *Có dùng test để chọn mô hình hoặc tính bảng A không?* Không. A so sánh và chọn theo validation; test được giữ riêng cho đánh giá cuối theo kế hoạch.

### Còn treo

- [ ] Ngày 2/11: chạy đủ B (13 cấu hình × 3 seed); đối chiếu bảng và hình A đã sinh, rồi xuất phần kết quả A/B cho báo cáo.
