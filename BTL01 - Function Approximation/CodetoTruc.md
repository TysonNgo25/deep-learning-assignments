# CodetoTruc – Nhật ký việc đã làm

Ghi lại từng ngày Trúc đã làm gì trong BTL01, code chạy thế nào và vì sao viết như vậy. Dùng để ôn lại trước buổi vấn đáp và khi viết báo cáo. Ngày mới thêm ở cuối file.

Kế hoạch gốc: [Kế hoạch.md](Kế%20hoạch.md).

---

## T5 29/10/2026 – Viết `src/data.py`, sinh dữ liệu

**Việc theo kế hoạch:** viết `data.py` (hàm Gaussian, sinh pool train 2000 / val 500 / test 100, lưu CSV) và chốt baseline với nhóm.

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
- [ ] Commit `src/data.py` + `data/*.csv` + `configs/baseline.yaml` + `Kế hoạch.md` lên nhánh `data`.

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
