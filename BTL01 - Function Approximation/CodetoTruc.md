# CodetoTruc – Nhật ký việc đã làm

Ghi lại từng ngày Trúc đã làm gì trong BTL01, code chạy thế nào và vì sao viết như vậy. Dùng để ôn lại trước buổi vấn đáp và khi viết báo cáo. Ngày mới thêm ở cuối file.

Kế hoạch gốc: [Kế hoạch.md](Kế%20hoạch.md).

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
| Khang | Viết `train.py` | Xong (`f626842`). `evaluate.py` theo lịch là ngày 31/10, hiện vẫn trống |
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

Cả ba runner gọi `evaluate_model` trong `src/evaluate.py`, nhưng file này trên `main` chỉ có docstring (Khang đã revert bản cũ và chưa đẩy lại). `run_C.py` còn import `make_group_figures` từ `src/plots.py`, file này cũng đang trống. Như vậy:

- Hiện **không ai chạy lại được A, B, C từ repo**.
- Log và `summary.csv` của A, B commit lúc 15:09 ngày 30/10, trước khi `train.py` lên repo (23:47) và khi `evaluate.py` vẫn trống. Tức là số liệu này sinh từ code nằm trên máy Trung, không có trong repo. Rubric chấm "số liệu truy vết được" và "chạy lại được", nên khi `evaluate.py` có trên `main`, Trung phải chạy lại A, B và commit log mới.

Nhận xét sơ bộ từ `summary.csv` (chỉ để định hướng, sẽ thay bằng số chạy lại):

- Ở A, `best_epoch` trung bình từ 835 đến 994 trên tối đa 1000. Val loss gần như vẫn còn giảm khi chạm 1000 epoch, early stopping hầu như không kích hoạt. Cần nói rõ trong mục 4 rằng mô hình dừng vì giới hạn epoch.
- Ở B, val MSE giảm đều từ 1 đến 64 nơ-ron (2.9e−3 xuống 3e−6), chưa thấy quá khớp. Dữ liệu không nhiễu và val dày (500 điểm) nên khó quá khớp. Phần thảo luận B cần nói thẳng điều này thay vì gượng ép "quá khớp".
- B = 1 nơ-ron có val MSE 2.9e−3, trong khi đoán ŷ = 0 cho mọi điểm đã được 3.7e−3 trên val (trung bình y²). Một nơ-ron gần như không học được hàm, ví dụ rõ cho thiếu khớp.
- `run_C.py` tính test cho mọi cấu hình C. Được phép nếu chỉ báo cáo và chọn cấu hình theo val. Khi viết 4.3 phải ghi rõ cột dùng để chọn là `val_mse`.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao Khang revert rồi commit lại?* Commit `0902fe5` gộp việc của ba ngày vào một lần. Tách ra theo từng ngày để lịch sử git khớp với phân công trong kế hoạch.
- *Mô hình trả về là ở epoch nào?* Epoch có val loss nhỏ nhất (`best_epoch`), không phải epoch cuối, vì `train_model` nạp lại checkpoint trước khi trả về.
- *Vì sao 64 nơ-ron không quá khớp?* Nhãn không có nhiễu, nên khớp sát train cũng là khớp sát hàm thật. Quá khớp xuất hiện rõ khi dữ liệu có nhiễu hoặc rất ít mẫu so với số tham số.

### Còn treo

- [ ] Báo nhóm: runner A/B/C đang lỗi `ImportError`, cần Khang đẩy `evaluate.py` (có `evaluate_model`) và Nhân đẩy `plots.py` (có `make_group_figures`).
- [ ] Trung chạy lại A, B từ repo sau khi `evaluate.py` lên `main`, commit log mới.

---

## T7 31/10/2026 – Vẽ hình dữ liệu (`scripts/plot_data.py`), nháp mục 2 báo cáo

**Việc theo kế hoạch:** đây là việc của ngày 30/10 ("Vẽ scatter 3D, bề mặt hàm thật, histogram x₁ x₂ y; viết nháp mục 2"), làm xong rạng sáng 31/10. Việc của ngày 31/10 ("chạy thử baseline end-to-end; review code chéo") đã bắt đầu: phát hiện runner đang lỗi import (xem mục 30/10).

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
| `README.md` | sửa | Thêm lệnh `python -m scripts.plot_data` |

Lệnh chạy (đứng ở thư mục BTL01, sau khi đã có `data/*.csv`):

```powershell
python -m scripts.plot_data
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

### Đã kiểm tra

- Chạy 2 lần liên tiếp, 3 file PNG giống hệt từng byte (so MD5), nên hình tái lập được.
- Đã mở xem từng hình: nhãn trục không chồng nhau, chữ tiếng Việt hiện đúng dấu.
- Các số trong nháp mục 2 đều tính lại bằng code, không chép tay:
  - Trị riêng của C: 0.164 và 2.136. Trục dài hướng (−0.263, 0.965), lệch 15.2° so với trục x₂.
  - Tương quan ρ = −0.5 / √0.6 = −0.645.
  - Đường mức 0.01: x₂ từ −3.625 đến 3.625, tại x₂ = 0 thì x₁ từ −1.07 đến 1.07.
  - Trung bình y² (MSE nếu đoán ŷ = 0): train 0.00445, val 0.00368, test 0.00511.
  - Val có 13.6% mẫu ở x₁ ∈ [1.6, 2], lý do độ lệch chuẩn x₁ của val (1.227) cao hơn lý thuyết (1.155).

### Cách tự kiểm tra lại

Mở Anaconda Prompt:

```powershell
cd /d "D:\Data\GitHub\deep-learning-assignments\BTL01 - Function Approximation"
conda activate btl01
python -m scripts.plot_data
git status
```

Kết quả đúng: in ra 3 dòng `-> ...\report\figures\fig_data_*.png`. Nếu đã commit hình, `git status` sau khi chạy lại không thấy file PNG nào bị sửa. Mở 3 hình trong `report\figures\`: Hình 2.2b phải thấy elip nghiêng (phía trên lệch sang trái), Hình 2.3 cột đầu tiên của y cao khoảng 70%.

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

### Câu hỏi vấn đáp có thể gặp

- *Vì sao elip đồng mức nghiêng?* Vì hiệp phương sai −0.5 khác 0, x₁ và x₂ tương quan âm (ρ = −0.645). Nếu C chéo (như báo cáo mẫu dùng ma trận đơn vị) thì elip thẳng theo trục.
- *Vì sao elip dài theo x₂?* Phương sai theo x₂ là 2, lớn hơn nhiều so với 0.3 của x₁. Trị riêng lớn 2.136 ứng với trục dài, hướng gần trục x₂.
- *Vì sao histogram vẽ theo % thay vì số mẫu?* Các tập có cỡ 2000, 500, 100. Vẽ % mới so được hình dạng phân bố.
- *Histogram test vì sao răng cưa nhiều?* Chỉ có 100 mẫu, mỗi khoảng trung bình 10 mẫu, lệch vài mẫu đã thành vài %. Phân bố vẫn phủ đủ miền nên không cần sinh lại.
- *Phân bố y lệch về 0 thì ảnh hưởng gì?* MSE rất nhỏ nên phải báo RMSE. Vùng đỉnh ít mẫu nhất nên sai số lớn nhất nhiều khả năng nằm quanh μ. Mốc so sánh: đoán toàn 0 đã được MSE ≈ 0.005, mô hình phải thấp hơn nhiều.
- *Vì sao bề mặt hàm thật vẽ trên lưới, không vẽ từ dữ liệu?* Dữ liệu ngẫu nhiên, rải không đều. Hàm thật tính được tại mọi điểm nên vẽ trên lưới đều cho mặt liền.

### Còn treo

- [ ] Commit `scripts/plot_data.py`, `src/paths.py`, `README.md`, `report/figures/`, `report/section_2_data.md`, `CodetoTruc.md`.
- [ ] Gửi nhóm chat: runner A/B/C lỗi `ImportError` (chi tiết ở mục 30/10).
- [ ] Chạy thử baseline end-to-end khi `evaluate.py` đã lên `main` (việc kế hoạch 31/10).
- [ ] Review code chéo: đọc `model.py`, `train.py` đủ để tự giải thích được.
- [ ] Khi ghép báo cáo: chuyển công thức LaTeX trong `section_2_data.md` sang Equation của Word, đánh lại số hình/bảng nếu thứ tự mục thay đổi.

---
