# Trả lời vấn đáp — Khang: phần việc ngày 29/10–3/11

## 1. Xác định phần việc theo ngày

Theo bảng tiến độ trong [`Kế hoạch.md`](Kế%20hoạch.md), phần Khang được giao đến hết ngày 30/10 gồm:

- **29/10:** viết `src/model.py` — MLP nhận danh sách số nơ-ron.
- **30/10:** viết `src/train.py` — vòng lặp huấn luyện, `DataLoader`, Adam, `MSELoss`, log, checkpoint và đo thời gian.

**Ngày 31/10** mới là phần `src/evaluate.py` — đánh giá MSE/MAE/RMSE và biểu đồ. Phần đó chưa thuộc phạm vi hoàn thành của hai ngày đang ghi trong tài liệu này.

## 2. Đề bài yêu cầu gì liên quan đến mô hình?

Trong `Đề và yêu cầu/MLP_Exercise - Yeu Cau.pdf`:

- **Trang 1, mục 2:** xây MLP có **hai đầu vào, một đầu ra** \(\hat y\), xấp xỉ nhãn \(y=f(x_1,x_2)\).
- **Trang 1, mục 3.2:** dùng PyTorch `torch.nn.Module`, `nn.Linear`, hàm kích hoạt phù hợp; cho phép tùy chỉnh số tầng ẩn và số nơ-ron mỗi tầng.
- **Trang 2, mục 4:** thực nghiệm B khảo sát số nơ-ron tầng ẩn; thực nghiệm C khảo sát số tầng ẩn. Do đó mô hình cần nhận được các cấu hình tầng ẩn khác nhau.
- **Trang 3, mục 6:** báo cáo cần có sơ đồ kiến trúc và công thức lan truyền thuận.

Đề mẫu tiếng Anh mô tả MLP một tầng ẩn với \(B\) nơ-ron, và MLP có \(C\) tầng ẩn. Cấu hình nào chạy trong thực nghiệm cụ thể do file baseline và runner của nhóm quy định; `model.py` cung cấp mô hình có thể cấu hình, không tự quyết định kết quả thí nghiệm.

## 3. Mô hình đã cài đặt

Lớp `MLP` kế thừa `torch.nn.Module`. Tham số `hidden` là dãy số nguyên, mỗi số là số nơ-ron của một tầng ẩn:

```python
MLP(hidden=[16, 16], activation="tanh")  # 2 đầu vào -> 16 -> 16 -> 1 đầu ra
MLP(hidden=[8], activation="tanh")       # 2 đầu vào -> 8 -> 1 đầu ra
```

Mỗi tầng ẩn gồm `nn.Linear` rồi đến activation. Tầng đầu ra là `nn.Linear` và **không** gắn activation, để dự đoán trực tiếp một giá trị thực.

Với \(h_0 = [x_1,x_2]\), công thức là:

\[
h_\ell = \phi(W_\ell h_{\ell-1}+b_\ell), \quad \ell=1,\ldots,L,
\]
\[
\hat y = W_{\mathrm{out}}h_L+b_{\mathrm{out}}.
\]

Trong đó \(L\) là số tầng ẩn, \(W\) là ma trận trọng số, \(b\) là vector bias, và \(\phi\) là activation. Baseline của nhóm dùng `tanh`.

Đầu vào theo batch có shape `(n, 2)`, đầu ra có shape `(n, 1)`. Nếu shape đầu vào không có đúng hai đặc trưng, `forward()` báo `ValueError` với shape nhận được.

## 4. Câu trả lời vấn đáp mẫu

### Câu 1: Phần em được giao ngày 29/10 là gì?

**Trả lời:** Ngày 29/10 em được giao viết `src/model.py`, định nghĩa MLP nhận danh sách số nơ-ron mỗi tầng ẩn. Theo đề, mô hình dùng PyTorch `nn.Module`, các tầng tuyến tính dùng `nn.Linear`, có activation ở tầng ẩn và một đầu ra để xấp xỉ hàm Gaussian. `train.py` là việc ngày 30/10, còn `evaluate.py` là ngày 31/10.

### Câu 2: Bài toán đưa gì vào mạng và mạng trả ra gì?

**Trả lời:** Mỗi mẫu có hai đặc trưng \(x_1,x_2\), vì đề cho điểm trong mặt phẳng 2D. Mạng trả về một giá trị \(\hat y\), là ước lượng mật độ Gaussian tại điểm đó. Khi đưa theo batch, input có shape `(n, 2)`, output có shape `(n, 1)`.

### Câu 3: Với `hidden=[16, 16]`, kiến trúc là gì?

**Trả lời:** Hai đầu vào đi vào tầng ẩn thứ nhất có 16 nơ-ron, tiếp tục qua tầng ẩn thứ hai có 16 nơ-ron, rồi đến một đầu ra: `2 → 16 → 16 → 1`. Đây là baseline, không phải yêu cầu mọi cấu hình phải dùng đúng kiến trúc đó.

### Câu 4: Danh sách `hidden` thể hiện điều gì?

**Trả lời:** Mỗi phần tử là số nơ-ron của một tầng ẩn. Số phần tử trong danh sách là số tầng ẩn. Ví dụ `[12]` là một tầng ẩn 12 nơ-ron; `[16, 16, 16]` là ba tầng ẩn, mỗi tầng 16 nơ-ron. Vì vậy cùng một lớp MLP có thể dùng trong thực nghiệm độ rộng và độ sâu.

### Câu 5: `nn.Linear` tính gì?

**Trả lời:** Một lớp tuyến tính tính \(z=Wx+b\), trong đó \(W\) là trọng số được học, \(b\) là bias được học. Một mình nhiều phép tuyến tính liên tiếp vẫn tương đương một phép tuyến tính; activation phi tuyến giữa các tầng giúp mạng biểu diễn hàm phi tuyến.

### Câu 6: Activation dùng ở đâu? Vì sao output không có activation?

**Trả lời:** Activation được đặt sau mỗi `nn.Linear` ở tầng ẩn; mặc định là `tanh`. Tầng cuối không có activation để \(\hat y\) không bị ép vào một khoảng cố định. Giá trị đầu ra là giá trị thực cần hồi quy.

### Câu 7: Tại sao dùng `tanh`?

**Trả lời:** `tanh` tạo tính phi tuyến và cho đầu ra trong khoảng \((-1,1)\). Nhóm chọn nó làm activation baseline. Em không khẳng định nó luôn tốt hơn ReLU hoặc sigmoid nếu chưa có thí nghiệm so sánh.

### Câu 8: Vì sao mô hình nhận danh sách tầng thay vì viết cứng hai tầng?

**Trả lời:** Đề yêu cầu khảo sát số nơ-ron và số tầng ẩn. Nhận danh sách giúp thay kiến trúc bằng cấu hình, ví dụ `[1]`, `[10]`, `[16]` cho B hoặc `[16]`, `[16,16]`, `[16,16,16]` cho C mà không phải viết một lớp riêng cho từng trường hợp.

### Câu 9: Model có tự học khi gọi `forward()` không?

**Trả lời:** Không. `forward()` chỉ thực hiện lan truyền thuận và trả dự đoán. Trọng số được học ở bước huấn luyện thông qua loss, tính gradient và optimizer; đó là phần `train.py`, không phải trách nhiệm riêng của `model.py` ngày 29/10.

### Câu 10: Vì sao dùng PyTorch, không dùng Keras/TensorFlow?

**Trả lời:** Đề yêu cầu triển khai bằng PyTorch và không chấp nhận TensorFlow/Keras thay cho việc cài đặt, huấn luyện MLP. Vì thế lớp mô hình kế thừa `torch.nn.Module` và dùng `torch.nn.Linear`.

### Câu 11: Mô hình có tính trực tiếp công thức Gaussian không?

**Trả lời:** Không. Công thức Gaussian được dùng để tạo nhãn đích \(y\). MLP học ánh xạ từ \((x_1,x_2)\) sang \(\hat y\), sau đó so sánh \(\hat y\) với \(y\). Hàm sinh nhãn thuộc phần dữ liệu.

### Câu 12: Thay đổi `hidden` có ảnh hưởng gì?

**Trả lời:** Tăng số tầng hoặc số nơ-ron làm tăng khả năng biểu diễn nhưng cũng tăng số tham số và thời gian tính toán, có thể làm thay đổi khả năng tổng quát hóa. Muốn kết luận thiếu khớp hay quá khớp phải dựa trên kết quả train/validation của thực nghiệm, không thể kết luận chỉ từ kiến trúc.

## 5. Demo kiểm tra riêng `model.py`

Chạy từ thư mục `BTL01 - Function Approximation` bằng môi trường `.venv`:

```powershell
@'
import torch
from src.model import MLP

for hidden in ([16, 16], [8], [16, 16, 16]):
    model = MLP(hidden=hidden, activation="tanh")
    x = torch.zeros((5, 2), dtype=torch.float32)
    y_hat = model(x)
    print(f"hidden={hidden}, output_shape={tuple(y_hat.shape)}")

try:
    MLP(hidden=[0])
except ValueError as error:
    print("invalid hidden layer rejected:", error)
'@ | .\.venv\Scripts\python.exe -
```

Kết quả cần thấy ba dòng `output_shape=(5, 1)` và trường hợp `[0]` bị từ chối bằng `ValueError`. Đây là kiểm tra shape/đầu vào cấu hình của model, **không phải** kết quả huấn luyện hay kết quả A/B/C.

## 6. Phạm vi không nên nhận nhầm

- **Ngày 29/10 — Khang:** định nghĩa MLP tùy biến trong `src/model.py`.
- **Ngày 30/10 — Khang:** viết vòng huấn luyện trong `src/train.py`.
- **Ngày 31/10 — Khang:** đánh giá và biểu đồ trong `src/evaluate.py`.
- Sinh Gaussian, chia CSV, thống kê và hình dữ liệu ban đầu thuộc Trúc.
- Runner, số lần chạy, bảng mean ± std của A/B thuộc Trung; C và `plots.py` thuộc Nhân.

Ngày deadline theo kế hoạch là lịch nhóm giao việc. Nội dung kỹ thuật căn cứ theo đề gốc trong `Đề và yêu cầu/MLP_Exercise - Yeu Cau.pdf`; khi trả lời vấn đáp nên nói rõ phần cá nhân và không nhận là mình đã chạy các thực nghiệm nếu chưa có log chứng minh.

---

## 7. Ngày 30/10 — Vòng huấn luyện `src/train.py`

### 7.1. Yêu cầu được giao

Theo `Kế hoạch.md`, ngày 30/10 Khang cần viết vòng huấn luyện có `DataLoader`, Adam, `MSELoss`, log loss theo epoch, checkpoint theo validation loss, early stopping và đo thời gian bằng `time.perf_counter()`.

Yêu cầu kỹ thuật tương ứng tại đề gốc, mục 3.2 trang 1:

- Dùng PyTorch tensor và `Dataset/DataLoader` (hoặc `TensorDataset/DataLoader`).
- Dùng `MSELoss` và optimizer như Adam hoặc SGD.
- Tự viết vòng train/validation theo trình tự forward → loss → `zero_grad` → backward → `optimizer.step()`.
- Dùng `model.train()`, `model.eval()` và `torch.no_grad()` đúng ngữ cảnh.
- Ghi loss theo epoch, lưu checkpoint tốt nhất theo validation loss; hỗ trợ CPU.

### 7.2. Cách gọi hàm và giá trị trả về

Hàm chính là:

```python
result = train_model(
    X_train, y_train, X_val, y_val,
    hidden=[16, 16],
    activation="tanh",
    config=train_config,
    seed=0,
    log_path="results/A/logs/A_1000_seed0.csv",
    checkpoint_path="models/A_1000_seed0.pt",
)
```

Hàm nhận train và validation, **không nhận test**. `config` lấy các giá trị liên quan `optimizer`, `lr`, `batch_size`, `max_epochs`, `patience` từ `configs/baseline.yaml`. Các trường khác của cấu hình chung được bỏ qua.

Hàm trả về `TrainingResult`:

| Trường | Ý nghĩa |
|---|---|
| `result.model` | MLP đã được khôi phục về trọng số có validation loss tốt nhất. |
| `result.history` | Danh sách loss train/validation theo epoch. |
| `result.best_epoch` | Epoch đạt validation loss nhỏ nhất. |
| `result.best_val_loss` | MSE validation thấp nhất quan sát được. |
| `result.elapsed_seconds` | Thời gian vòng huấn luyện, không tính ghi file. |
| `result.stopped_early` | `True` nếu dừng do patience trước `max_epochs`. |

Nếu truyền `log_path`, CSV gồm các cột `epoch`, `train_loss`, `val_loss`. Nếu truyền `checkpoint_path`, file `.pt` lưu `model_state_dict`, kiến trúc, activation, epoch/loss validation tốt nhất, seed, thời gian và config train. Có thể nạp lại bằng `load_checkpoint(path)`.

### 7.3. Giải thích một epoch

1. `model.train()` đặt mạng ở chế độ train.
2. `DataLoader` xáo trộn dữ liệu train và đưa từng mini-batch vào.
3. `optimizer.zero_grad(set_to_none=True)` xóa gradient cũ.
4. `predictions = model(batch_x)` thực hiện forward.
5. `loss_fn(predictions, batch_y)` tính MSE.
6. `loss.backward()` tính gradient bằng backpropagation.
7. `optimizer.step()` cập nhật trọng số.
8. Sau khi hết batch, `model.eval()` và `torch.no_grad()` tính validation MSE, không cập nhật trọng số.
9. Ghi loss; nếu validation loss thấp hơn tốt nhất thì sao chép checkpoint. Nếu không cải thiện đủ `patience` epoch liên tiếp thì dừng.

Train loss theo epoch là trung bình có trọng số theo số mẫu trong từng batch. Vì vậy batch cuối nếu nhỏ hơn batch size không bị xem như có trọng số bằng một batch đầy đủ.

### 7.4. Câu hỏi vấn đáp ngày 30/10

**Câu 1: Phần em làm ngày 30/10 là gì?**

**Trả lời:** Em viết vòng train/validation trong `src/train.py`: tạo `DataLoader`, dùng `MSELoss` và Adam theo baseline, tự viết cập nhật gradient, ghi loss từng epoch, chọn checkpoint theo validation loss, early stopping và đo thời gian.

**Câu 2: Tại sao phải gọi `zero_grad()` trước backward?**

**Trả lời:** PyTorch mặc định cộng dồn gradient vào `.grad`. Xóa gradient trước mỗi batch để optimizer cập nhật theo gradient của batch hiện tại, không cộng nhầm gradient batch trước.

**Câu 3: `backward()` và `optimizer.step()` khác nhau thế nào?**

**Trả lời:** `backward()` tính gradient của loss theo các tham số. `optimizer.step()` dùng gradient đó để cập nhật tham số theo thuật toán Adam hoặc SGD.

**Câu 4: Adam làm gì?**

**Trả lời:** Adam là optimizer cập nhật trọng số bằng gradient, đồng thời thích nghi learning rate cho từng tham số dựa trên trung bình động của gradient và bình phương gradient. Trong bài, loại optimizer và learning rate được giữ cố định giữa các cấu hình, trừ khi chính chúng là biến khảo sát.

**Câu 5: Vì sao dùng mini-batch thay vì toàn bộ train một lần?**

**Trả lời:** `DataLoader` chia dữ liệu thành batch để mỗi cập nhật dùng một phần dữ liệu, quản lý bộ nhớ và xáo trộn mẫu thuận tiện. Với bài toán này batch size baseline là 32.

**Câu 6: Validation có tham gia cập nhật trọng số không?**

**Trả lời:** Không. Trọng số chỉ được cập nhật bằng batch train. Validation chỉ dùng để theo dõi loss, early stopping và lưu checkpoint tốt nhất.

**Câu 7: Checkpoint tốt nhất được xác định thế nào?**

**Trả lời:** Sau mỗi epoch, nếu validation MSE nhỏ hơn giá trị tốt nhất trước đó thì lưu một bản sao độc lập của `state_dict`. Kết thúc huấn luyện, nạp lại bản sao này để kết quả trả về đúng mô hình tốt nhất trên validation, không nhất thiết là epoch cuối.

**Câu 8: Early stopping với patience 100 có nghĩa gì?**

**Trả lời:** Nếu validation loss không cải thiện trong 100 epoch liên tiếp thì dừng, dù chưa đạt 1000 epoch. Mục đích là tránh huấn luyện tiếp khi validation không tốt lên và tiết kiệm thời gian. Nếu vẫn cải thiện, có thể chạy tới `max_epochs`.

**Câu 9: Thời gian được đo ở đâu?**

**Trả lời:** `time.perf_counter()` được gọi ngay trước vòng epoch và sau khi vòng lặp kết thúc. Khoảng đo phản ánh thời gian train/validation trong các epoch, không gồm ghi CSV/checkpoint.

**Câu 10: Vì sao `train_model` không nhận test?**

**Trả lời:** Để tách test khỏi quá trình fit và chọn epoch. Nếu dùng test để chọn mô hình, thông tin test rò vào quá trình phát triển và metric test không còn là đánh giá cuối độc lập.

**Câu 11: Vì sao cần cả `model.eval()` lẫn `torch.no_grad()`?**

**Trả lời:** `eval()` chuyển hành vi của module sang suy luận; `no_grad()` ngừng lưu đồ thị gradient. Hai lệnh có chức năng khác nhau nên validation dùng cả hai.

**Câu 12: Vì sao ép nhãn thành shape `(n, 1)`?**

**Trả lời:** Đầu ra `nn.Linear(..., 1)` có shape `(batch, 1)`. Chuẩn hóa shape nhãn để khớp đầu ra, tránh broadcasting ngoài ý muốn khi tính MSE.

**Câu 13: Nếu dữ liệu có NaN/Inf hoặc shape sai thì sao?**

**Trả lời:** Hàm kiểm tra shape, số lượng mẫu và tính hữu hạn trước khi train; dữ liệu không hợp lệ sẽ báo lỗi rõ ràng thay vì chạy tiếp với loss sai.

**Câu 14: Một lần chạy có tái lập tuyệt đối trên mọi máy không?**

**Trả lời:** Seed được đặt cho Python, NumPy và PyTorch, bao gồm xáo trộn batch, giúp tăng khả năng tái lập trong cùng điều kiện. Không cam kết kết quả giống từng bit giữa mọi phiên bản, phần cứng hoặc backend.

### 7.5. Chạy smoke test ngắn cho phần train

Từ PowerShell, chạy trong thư mục `BTL01 - Function Approximation`. Smoke test dùng 32 mẫu và tối đa 3 epoch; nó kiểm tra pipeline chứ **không** tạo kết quả thực nghiệm chính thức:

```powershell
@'
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data import load_train, load_split, load_config
from src.train import train_model, load_checkpoint

X, y = load_train(32)
X_val, y_val = load_split("val")
config = dict(load_config()["train"])
config.update(max_epochs=3, patience=3)

with TemporaryDirectory(prefix="btl01_train_smoke_") as folder:
    output = Path(folder)
    result = train_model(
        X, y, X_val[:32], y_val[:32],
        hidden=[8], activation="tanh", config=config, seed=0,
        log_path=output / "loss.csv",
        checkpoint_path=output / "model.pt",
    )
    model, checkpoint = load_checkpoint(output / "model.pt")
    print("history_rows:", len(result.history))
    print("best_epoch:", result.best_epoch)
    print("best_val_loss:", result.best_val_loss)
    print("elapsed_seconds:", result.elapsed_seconds)
    print("log_columns:", (output / "loss.csv").read_text(encoding="utf-8").splitlines()[0])
    print("checkpoint_epoch:", checkpoint["best_epoch"])
    print("loaded_model:", type(model).__name__)
'@ | .\.venv\Scripts\python.exe -
```

Kết quả cần có từ 1 đến 3 dòng history; `best_epoch` nằm trong số epoch đã chạy; checkpoint nạp được; header CSV là `epoch,train_loss,val_loss`. Loss phải hữu hạn và thời gian không âm. Số loss cụ thể có thể khác nếu thay seed, config hoặc môi trường.

### 7.6. Đối chiếu hoàn thành ngày 30/10

- [x] `DataLoader` + `TensorDataset`, xáo trộn train theo batch.
- [x] `MSELoss`; hỗ trợ Adam và SGD, baseline lấy Adam từ YAML.
- [x] Vòng lặp tự viết: zero grad → forward → loss → backward → optimizer step.
- [x] Chuyển `train()`/`eval()` và `torch.no_grad()` đúng ngữ cảnh.
- [x] Kiểm tra shape `(n,2)` và nhãn `(n,1)`, kiểm tra dữ liệu hữu hạn.
- [x] Log train/validation loss theo epoch ra CSV.
- [x] Early stopping và khôi phục checkpoint theo validation loss nhỏ nhất.
- [x] Đo thời gian bằng `time.perf_counter()`.
- [x] Có thể lưu/nạp checkpoint cùng kiến trúc và metadata.
- [x] Chạy smoke test ngắn; không coi đây là kết quả A/B/C chính thức.

---

## 8. Ngày 31/10 — Đánh giá mô hình với `src/evaluate.py`

### 8.1. Yêu cầu trong kế hoạch và đề

Theo `Kế hoạch.md`, ngày 31/10 Khang phụ trách `evaluate.py`: tính MSE, MAE, RMSE; tạo bề mặt xấp xỉ và đồ thị giá trị thật–dự đoán. Đề gốc mục 3.2 trang 1 yêu cầu đánh giá trên test bằng MSE cùng ít nhất một metric bổ sung, và trực quan hóa \(y\) thật–\(\hat y\) cùng bề mặt xấp xỉ.

### 8.2. API đánh giá

```python
from src.evaluate import evaluate_model, predict

test_scores = evaluate_model(model, X_test, y_test)
y_hat = predict(model, X_test)
```

- `evaluate_model(...)` trả về `{"mse": ..., "mae": ..., "rmse": ...}`.
- `predict(...)` trả NumPy array có shape `(n, 1)`.
- Hàm đánh giá kiểm tra shape, số mẫu, NaN/Inf và dự đoán không hữu hạn; lỗi đầu vào báo rõ.
- Mô hình được chuyển tạm sang chế độ `eval()` và inference nằm trong `torch.no_grad()`. Trạng thái train/eval ban đầu được khôi phục sau prediction.
- Hàm nhận split từ phía gọi. Trong thí nghiệm, dùng validation để phát triển/chọn mô hình; chỉ sau khi chốt mới truyền test để đánh giá cuối.

Metric được tính trên \(y\) gốc:

\[
\mathrm{MSE}=\frac{1}{N}\sum_i(y_i-\hat y_i)^2,\quad
\mathrm{MAE}=\frac{1}{N}\sum_i|y_i-\hat y_i|,\quad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
\]

### 8.3. Hai biểu đồ

```python
from src.evaluate import plot_surface_comparison, plot_y_vs_yhat

scores = plot_y_vs_yhat(y_test, y_hat, "results/final_y_yhat.png")
plot_surface_comparison(
    model, mu, cov, x1_range, x2_range,
    save_path="results/final_surface.png",
)
```

- `plot_y_vs_yhat` vẽ mỗi mẫu test thành một điểm và thêm đường tham chiếu \(y=\hat y\). Hàm cũng trả về MSE/MAE/RMSE của các điểm được vẽ.
- `plot_surface_comparison` tạo lưới đều trên miền, vẽ bề mặt Gaussian thật và bề mặt MLP dự đoán cạnh nhau. `resolution` điều khiển số điểm lưới mỗi chiều.
- Thư mục đích được tạo tự động; biểu đồ được đóng sau khi lưu để tránh giữ figure trong các lần chạy lặp.

### 8.4. Câu hỏi vấn đáp ngày 31/10

**Câu 1: Phần em làm ngày 31/10 là gì?**

**Trả lời:** Em viết `src/evaluate.py`: dự đoán ở chế độ inference, tính MSE/MAE/RMSE, vẽ \(y\)–\(\hat y\) và so sánh bề mặt Gaussian thật với bề mặt MLP.

**Câu 2: Vì sao cần MAE/RMSE ngoài MSE?**

**Trả lời:** MSE nhạy với sai số lớn nhưng có đơn vị bình phương. MAE và RMSE cùng đơn vị với \(y\), dễ diễn giải độ lớn sai số hơn. Đề yêu cầu MSE và ít nhất một metric bổ sung; code cung cấp cả MAE và RMSE.

**Câu 3: Điểm nằm trên đường chéo trong đồ thị có ý nghĩa gì?**

**Trả lời:** Đường là \(y=\hat y\), tức dự đoán đúng bằng giá trị thật. Điểm càng gần đường càng có sai số nhỏ; lệch lên/xuống thể hiện dự đoán cao/thấp hơn thực tế.

**Câu 4: Vì sao vẽ surface trên lưới đều?**

**Trả lời:** Lưới đều cho phép so hình dạng liên tục của hàm thật và MLP trên toàn miền thay vì chỉ nhìn các điểm lấy mẫu ngẫu nhiên.

**Câu 5: Hàm evaluate tự biết dữ liệu nào là test không?**

**Trả lời:** Không. Hàm tính metric trên dữ liệu được truyền vào. Quy tắc chọn mô hình là dùng validation, không dùng test. Trong project này, runner C đã tính metric test cho 12 lượt để báo cáo thực nghiệm C trước khi chốt mô hình; vì vậy phải công bố test đã được truy vấn trước, không gọi đó là test hoàn toàn độc lập.

### 8.5. Kiểm thử ngày 31/10 và số tham chiếu

Đã chạy baseline hiện tại: train 1000 mẫu, hidden `[16, 16]`, `tanh`, Adam, learning rate `0.001`, batch size `32`, tối đa 1000 epoch, patience 100, seed `0`. Huấn luyện chạy đủ 1000 epoch; best epoch là 998. Kết quả một lần chạy:

| Split | MSE | MAE | RMSE |
|---|---:|---:|---:|
| Train | \(3.96226244\times10^{-7}\) | \(4.82903298\times10^{-4}\) | \(6.29465046\times10^{-4}\) |
| Validation | \(4.17493047\times10^{-7}\) | \(5.01333200\times10^{-4}\) | \(6.46137019\times10^{-4}\) |
| Test | \(4.40567471\times10^{-7}\) | \(5.27307225\times10^{-4}\) | \(6.63752568\times10^{-4}\) |

Kiểm thử xác nhận shape dự đoán `(100, 1)`, các metric hữu hạn, CSV/checkpoint được ghi, và hai ảnh được tạo. Các file kiểm thử được tạo trong thư mục tạm rồi tự dọn. Đây là **một lần baseline để kiểm tra pipeline**, không thay kết quả tổng hợp A/B/C hoặc mean ± std của ba seed; thời gian thực tế thay đổi theo máy.

### 8.6. Đối chiếu hoàn thành ngày 31/10

- [x] Hàm metric trả MSE, MAE, RMSE và xử lý shape scalar `(n,)` / `(n, 1)`.
- [x] Prediction hỗ trợ CPU và device của model, dùng `eval()`/`no_grad()`.
- [x] Vẽ y thật–ŷ dự đoán và đường tham chiếu \(y=\hat y\).
- [x] Vẽ bề mặt Gaussian thật cạnh bề mặt MLP trên lưới.
- [x] Chạy baseline end-to-end qua train → evaluate → xuất hình.
- [x] Kiểm tra giá trị không hợp lệ và trường hợp metric có kết quả tính tay đã biết.

## 9. Công việc Khang còn được giao sau ngày 31/10

Theo bảng lịch từng ngày trong `Kế hoạch.md`, các việc tiếp theo vẫn thuộc phần Khang:

| Ngày | Việc được giao | Trạng thái sau khi hoàn thành 4/11 |
|---|---|---|
| **1/11** | Vẽ bề mặt xấp xỉ và đồ thị \(y\)–\(\hat y\); hỗ trợ Trung/Nhân khi gặp lỗi. | Đã tạo hai hình baseline tham khảo trên validation tại `results/diagnostics/figures/`; rà log A/B/C thấy đủ số lần chạy và không phát hiện lỗi đang chặn. Chưa phải mô hình cuối. |
| **2/11** | Viết nháp mục 3 báo cáo: sơ đồ MLP, công thức forward và loss. | Đã tạo `report/section_3_model.md`; đây là bản nháp phương pháp, chưa ghép vào báo cáo cuối. |
| **3/11** | Cùng Trúc chốt mô hình theo validation, lưu `.pt` mô hình được chọn và đánh giá test cuối. | Đã chọn A/2000, `hidden=[16,16]`, seed 1 theo validation; lưu checkpoint và đánh giá test theo quyết định Trúc. Test đã được xem trước trong thực nghiệm C nên phải công bố giới hạn này. |
| **4/11** | Viết phần đánh giá chung: y–ŷ và bề mặt. | Đã tạo hai hình validation/bề mặt từ checkpoint cuối, viết nháp `report/section_5_final_model_evaluation.md` và ghi rõ giới hạn test đã được truy vấn ở C. |
| **5/11** | Đối chiếu số liệu báo cáo với log của Trung (theo lịch ghi chéo phần Trung). | Đã đối chiếu toàn bộ mean/std của 10 cấu hình A và 13 cấu hình B với summary CSV; không thấy sai khác sau làm tròn. |
| **6/11** | Hỏi xoáy cả nhóm về vòng train/validation/test. | Đã chuẩn bị bộ câu hỏi liên phần trong mục 14; bài nói A/B và báo cáo ghép đã có. |
| **7/11** | Chạy lại toàn bộ từ đầu trên máy sạch theo README. | Chưa thể chạy toàn bộ an toàn: các script full A/B/C ghi đè kết quả và run_final bị khóa để tránh chấm lại test; đã kiểm tra môi trường, dựng lại báo cáo và chạy kiểm tra validation-only. |
| **8/11** | Ngày dự phòng, hạn nộp. | Đã chạy preflight checker bằng UTF-8; lệnh trả mã lỗi 1 với 85 lỗi và 306 cảnh báo. Checker xác nhận thiếu DOCX/PDF, đồng thời quét cả `.venv` nên báo nhiều đường dẫn tuyệt đối từ mã của package cài sẵn. Không tạo ZIP; cần Trúc xuất báo cáo và cả nhóm rà checklist/tỉ lệ đóng góp. |

Ngày 1/11 đã có hình baseline tham khảo; ngày 2/11 và 4/11 có thể tiếp tục chuẩn bị nội dung với mô hình hiện có. Các việc chốt mô hình, kiểm tra số liệu toàn bài và chạy lại sạch phụ thuộc tiến độ của cả nhóm. Danh sách này phân biệt **mã lõi đã hoàn thành** với **các deliverable sau đó vẫn còn phải thực hiện**.

---

## 10. Thực hiện công việc ngày 1/11

### 10.1. Tạo hình baseline tham khảo

Vì lịch ngày 3/11 mới chốt mô hình cuối cùng, hình của ngày 1/11 dùng **baseline hiện tại** (`hidden=[16,16]`, `tanh`, seed `0`) để minh họa pipeline. Đồ thị \(y\)–\(\hat y\) dùng **validation 500 mẫu**, không dùng test. Bề mặt xấp xỉ được vẽ trên lưới đều trong miền đề bài. Không gọi đây là kết quả mô hình tốt nhất/cuối cùng.

File đã tạo:

| File | Nội dung |
|---|---|
| `results/diagnostics/figures/baseline_seed0_validation_y_yhat.png` | Scatter y thật–ŷ trên validation và đường chuẩn \(y=\hat y\). |
| `results/diagnostics/figures/baseline_seed0_surface_comparison.png` | Bề mặt Gaussian thật và MLP baseline đặt cạnh nhau. |
| `results/diagnostics/logs/baseline_seed0.csv` | Train loss và validation loss từng epoch của lần chạy tạo hình. |
| `results/diagnostics/baseline_seed0_metrics.csv` | MSE/MAE/RMSE trên train và validation của baseline tham khảo. |

### 10.2. Kết quả lần chạy tạo hình

Baseline dùng 1000 mẫu train, hai tầng ẩn × 16 nơ-ron, `tanh`, Adam, learning rate \(10^{-3}\), batch size 32, tối đa 1000 epoch, patience 100, seed 0. Chạy đủ 1000 epoch; epoch có validation loss thấp nhất là 998.

| Tập | MSE | MAE | RMSE |
|---|---:|---:|---:|
| Train | \(3.96226244\times10^{-7}\) | \(4.82903298\times10^{-4}\) | \(6.29465046\times10^{-4}\) |
| Validation | \(4.17493047\times10^{-7}\) | \(5.01333200\times10^{-4}\) | \(6.46137019\times10^{-4}\) |

Đây là số liệu của một lần chạy baseline để tạo hình, không thay bảng A/B/C, không có ý nghĩa mean ± std và không thay đánh giá test cuối.

### 10.3. Hỗ trợ và rà kết quả A/B/C

Kiểm tra giao diện runner cho thấy A/B gọi `train_model` và `evaluate_model`; C dùng hàm metric riêng trong `run_C.py` để giữ cách tính float32 nhất quán với kết quả C đã lưu. Các kết quả hiện có khớp số cấu hình và seed trong baseline:

- A: 10 kích thước train × 3 seed = 30 log; summary có 10 dòng.
- B: 13 độ rộng × 3 seed = 39 log; summary có 13 dòng.
- C: 4 độ sâu × 3 seed = 12 log; summary mỗi lần chạy có 12 dòng, summary mean ± std có 4 dòng.

Rà soát số file/log và summary không phát hiện thiếu cấu hình, giá trị NaN hoặc lỗi đang chặn. Chưa có yêu cầu lỗi cụ thể từ Trung/Nhân trong lượt này nên không sửa file của hai bạn.

---

## 11. Nháp mục 3 báo cáo — công việc ngày 2/11

Đã tạo [section_3_model.md](report/section_3_model.md) theo phần việc trong [Kế hoạch.md](Kế%20hoạch.md). Bản nháp trình bày kiến trúc baseline, công thức lan truyền thuận, MSE, cấu hình train, cách chọn checkpoint bằng validation và việc không dùng test trong quá trình huấn luyện/chọn mô hình. Đây là nội dung phương pháp; kết quả so sánh A/B/C và kết quả test của mô hình cuối phải lấy từ các bảng/nhật ký tương ứng, không lấy một lần chạy minh họa làm kết luận cuối.

### Câu 14: Mạng biến đổi hai đầu vào thành một dự đoán như thế nào?

**Trả lời:** Mỗi dòng đầu vào có hai giá trị \(x_1,x_2\), nên tensor theo batch có shape \((N,2)\). Mỗi tầng ẩn tính phép tuyến tính \(W h+b\), rồi áp dụng activation `tanh`. Baseline có hai tầng ẩn, mỗi tầng 16 nơ-ron. Tầng cuối ánh xạ sang một giá trị \(\hat y\), nên output có shape \((N,1)\).

### Câu 15: Vì sao dùng MSE làm loss?

**Trả lời:** MSE lấy trung bình bình phương chênh lệch giữa nhãn \(y\) và dự đoán \(\hat y\). Đây là loss hồi quy trực tiếp cho sai số số học; bình phương làm sai số lớn đóng góp nhiều hơn. Đạo hàm của loss được lan truyền ngược để cập nhật trọng số. MSE nhỏ hơn là tốt hơn khi so sánh trên cùng tập dữ liệu và cùng thang đo.

### Câu 16: Validation và test khác nhau thế nào trong quy trình này?

**Trả lời:** Sau mỗi epoch, validation loss được tính để theo dõi quá trình học và chọn trọng số ở epoch tốt nhất; nếu validation không cải thiện đủ lâu thì dừng sớm. Tập test không được đưa vào hàm huấn luyện và không quyết định epoch hay cấu hình. Thực tế runner C đã tính test cho cả 12 lượt theo yêu cầu báo cáo; sau đó nhóm còn tính metric cho checkpoint cuối. Vì test C đã được xem trước khi chốt cuối, phải nêu giới hạn đó.

### Câu 17: Vì sao trả về checkpoint ở epoch tốt nhất thay vì epoch cuối?

**Trả lời:** Loss train có thể tiếp tục giảm trong khi validation loss không còn cải thiện. Lưu và nạp lại `state_dict` tại epoch có validation loss tốt nhất giúp kết quả trả về theo đúng tiêu chí chọn đã định, đồng thời early stopping tránh tiếp tục huấn luyện khi không có cải thiện validation.

### Câu 18: Mục 3 đã chứng minh mô hình nào là tốt nhất chưa?

**Trả lời:** Chưa. Mục 3 mô tả phương pháp và cấu hình. Muốn kết luận phải đối chiếu kết quả validation của các thí nghiệm A/B/C theo tiêu chí nhóm đã thống nhất; sau đó tính metric cho mô hình được chọn trên test. Tuy nhiên, do runner C đã đánh giá các lượt C trên cùng tập test trước đó, kết quả cuối phải được trình bày kèm giới hạn này. Hình baseline tham khảo chỉ chứng minh pipeline chạy và hỗ trợ quan sát, không thay thế phép so sánh cuối.

### Đã kiểm tra và cách tự kiểm tra

- Đối chiếu mô tả kiến trúc, loss, batch size, optimizer, seed, epoch và patience với `src/model.py`, `src/train.py` và `configs/baseline.yaml`.
- Đọc lại mục 3 để đảm bảo tách rõ baseline với các biến thể A/B/C, validation với test, và kết quả minh họa với kết quả chọn cuối.
- Mở file `report/section_3_model.md` để rà sơ đồ, công thức và bảng cấu hình; đây là tài liệu Markdown nên không có lệnh chạy riêng.

---

## 12. Chọn mô hình, lưu checkpoint và đánh giá test — công việc ngày 3/11

### Việc theo kế hoạch và quyết định chọn mô hình

Theo [Kế hoạch.md](Kế%20hoạch.md), Khang phối hợp với Trúc chốt mô hình theo validation, lưu `.pt` và đánh giá test. Trúc xác nhận chọn cấu hình A với 2000 mẫu train, `hidden=[16,16]`, activation `tanh`, seed 1. Tiêu chí chọn chỉ dùng validation:

- Trong nhóm A, B, C, cấu hình A/2000 có validation MSE trung bình thấp nhất: \(1.68128104\times10^{-7}\). So sánh này dùng các kết quả A/B/C đã chạy theo kế hoạch; A dùng 2000 mẫu train, còn B/C dùng 1000.
- Trong ba seed của A/2000, seed 1 có validation MSE thấp nhất trong log A: \(1.04538820\times10^{-7}\).
- Không dùng metric test để chọn cấu hình hay seed.

### Kết quả và file

Chạy `python -m scripts.run_final`. Script huấn luyện lại cấu hình đã chọn bằng train pool và validation, lưu checkpoint rồi mới nạp tập test để tính metric. Kết quả lần chạy này:

| Tập | MSE | MAE | RMSE |
|---|---:|---:|---:|
| Validation | \(1.08195385\times10^{-7}\) | \(2.45049751\times10^{-4}\) | \(3.28930669\times10^{-4}\) |
| Test (100 mẫu) | \(8.65179445\times10^{-8}\) | \(2.16396908\times10^{-4}\) | \(2.94139328\times10^{-4}\) |

Mô hình chọn được có 337 tham số, best epoch 980; chạy đủ 1000 epoch và không dừng sớm. Các file tạo ra:

| File | Nội dung |
|---|---|
| `results/final/selected_model.pt` | Trọng số MLP tốt nhất theo validation loss, kèm kiến trúc, seed và metadata huấn luyện. |
| `results/final/logs/selected_model_seed1.csv` | Train loss và validation loss từng epoch của lần huấn luyện tạo checkpoint. |
| `results/final/final_metrics.json` | Cấu hình chọn, metric validation/test, thông tin checkpoint và ghi chú protocol test. |
| `scripts/run_final.py` | Script tái hiện quy trình; từ chối chạy tiếp nếu `final_metrics.json` đã tồn tại để tránh vô tình chấm lại test. |

### Hạn chế cần nói rõ khi vấn đáp

Thực nghiệm C đã tính metric test cho cả 12 lượt trước khi chốt mô hình. Theo xác nhận của Trúc, cấu hình cuối vẫn được đánh giá trên tập test hiện có, nhưng metric test ở trên phải mô tả là **điểm hậu tuyển chọn trên tập đã được truy vấn trước**, không phải ước lượng độc lập từ một test set hoàn toàn chưa từng xem. Việc dùng test trong C không tham gia quyết định chọn A/2000 seed 1; lựa chọn được làm chỉ từ validation.

Có khác biệt nhỏ giữa seed 1 trong log A và lần tái huấn luyện cuối: log A ghi best epoch 999, validation MSE \(1.04538820\times10^{-7}\), còn lần tạo checkpoint cuối đạt best epoch 980, validation MSE \(1.08195385\times10^{-7}\). Cả hai lần dùng seed 1; nguyên nhân chênh lệch chưa được xác minh, nên không khẳng định chính xác tái lập từng chữ số. Mô hình cuối và số liệu cuối trong bảng trên lấy từ chính checkpoint vừa lưu.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao chọn A/2000 thay vì B/64 hoặc C/4?* Vì tiêu chí chốt là validation MSE. Mean validation MSE A/2000 là \(1.68128104\times10^{-7}\), thấp hơn B/64 (\(2.81076869\times10^{-6}\)) và C/4 (\(4.09843324\times10^{-7}\)). Không nhìn test để chọn.
- *Mô hình cuối cụ thể là gì?* Hai đầu vào → hai tầng ẩn, mỗi tầng 16 nơ-ron, activation `tanh` → một đầu ra. Dùng 2000 mẫu train, seed 1; checkpoint là trọng số tại best epoch 980 của lần huấn luyện tạo file cuối.
- *Có thể nói test là đánh giá độc lập chưa?* Không. Thực nghiệm C đã tính test cho cả bốn độ sâu và ba seed trước đó. Cần công bố giới hạn này; muốn có đánh giá hoàn toàn mới thì nhóm phải thống nhất một tập test mới/chưa từng xem.
- *Vì sao validation MSE của lần tạo checkpoint không khớp hẳn với log A?* Có chênh lệch nhỏ được ghi nhận giữa hai lượt cùng seed. Chưa xác minh được nguyên nhân, nên báo cáo dùng số từ checkpoint cuối và không tuyên bố tái lập bit-for-bit.

### Cách tự kiểm tra lại

Trong Anaconda Prompt tại thư mục BTL01, chạy một lần:

```powershell
conda activate btl01
python -m scripts.run_final
```

Kết quả đúng phải in cấu hình A/2000, seed 1, best epoch 980, các metric ở bảng trên và đường dẫn checkpoint. Lệnh sẽ báo lỗi nếu `results/final/final_metrics.json` đã tồn tại; không xóa file này để chạy lại trên cùng test nếu chưa có quyết định mới của nhóm.

### Còn treo

- [x] Ngày 4/11: viết phần đánh giá chung y–ŷ và bề mặt bằng checkpoint đã chọn; không gọi test hiện tại là hoàn toàn chưa từng được xem.

---

## 13. Đánh giá chung mô hình cuối — công việc ngày 4/11

Theo [Kế hoạch.md](Kế%20hoạch.md), phần Khang ngày 4/11 là đánh giá chung bằng đồ thị \(y\)–\(\hat y\) và bề mặt xấp xỉ. Đã tạo bản nháp [section_5_final_model_evaluation.md](report/section_5_final_model_evaluation.md) để ghép vào báo cáo, dựa trên checkpoint đã chốt ngày 3/11.

### Hình và số liệu

- [selected_model_validation_y_yhat.png](results/final/figures/selected_model_validation_y_yhat.png): scatter 500 điểm validation với đường tham chiếu \(y=\hat y\). Các điểm nằm sát đường chéo; nhiều điểm nhỏ chồng sát gần gốc nên cần đọc cùng metric.
- [selected_model_surface_comparison.png](results/final/figures/selected_model_surface_comparison.png): hai bề mặt Gaussian thật và MLP trên cùng miền \([-2,2]\times[-4,4]\), dùng lưới \(80\times80\).
- Validation của checkpoint: MSE \(1.08195385\times10^{-7}\), MAE \(2.45049751\times10^{-4}\), RMSE \(3.28930669\times10^{-4}\).
- Sai số trên lưới đều so với hàm Gaussian giải tích: MSE \(1.10056953\times10^{-7}\), MAE \(2.49988858\times10^{-4}\), RMSE \(3.31748327\times10^{-4}\). Đây là chẩn đoán trên lưới, không phải một tập kiểm thử độc lập.
- Phân tích vùng validation lưu trong `results/final/validation_error_analysis.json`: vùng \(y<0.01\) có 326 điểm, MAE \(2.04779\times10^{-4}\), đóng góp 43.47% tổng bình phương sai số; vùng \(y\geq0.1\) có 49 điểm, MAE \(4.10412\times10^{-4}\), đóng góp 23.99%.
- Điểm sai tuyệt đối lớn nhất có \(y=0.08321\), \((x_1,x_2)=(0.56321,-2.16518)\), sai \(1.00407\times10^{-3}\). Nó nằm ở vùng trung gian, không phải ngay đỉnh; tỷ lệ so với giá trị đỉnh 0.269 là khoảng 0.37%.

### Cách tạo lại và kiểm tra

Chạy `python -m scripts.plot_final_evaluation` từ thư mục BTL01. Script nạp `results/final/selected_model.pt`, chỉ dùng validation để tạo đồ thị \(y\)–\(\hat y\), và so sánh bề mặt với Gaussian trên lưới 6400 điểm. Script không đọc tập test. Đã chạy thành công, mở xem cả hai hình; scatter bám sát đường chéo và hai bề mặt có hình dạng tổng thể tương tự. `git diff --check` đạt.

Đã sửa câu ở mục 3 để không khẳng định test chỉ được đánh giá sau khi chọn mô hình: thực tế runner C đã tính test cho cả 12 lượt. Báo cáo mục đánh giá cũng nêu rõ giới hạn này.

### Câu hỏi vấn đáp có thể gặp

- *Đồ thị \(y\)–\(\hat y\) gần đường chéo nói lên điều gì?* Dự đoán trên validation nhìn chung gần giá trị thật; nhưng đồ thị cần được xem cùng MSE/MAE/RMSE, vì các điểm có mật độ thấp tập trung sát gốc và có thể chồng lấp.
- *Hai bề mặt giống nhau có chứng minh mô hình đúng mọi nơi không?* Không. Đó là trực quan trên miền và lưới được vẽ. Validation bổ sung đánh giá trên 500 điểm; chưa kết luận ngoài miền hoặc trên phân phối khác.
- *Sai số trên lưới 80×80 có phải test score không?* Không. Lưới được tạo đều trong miền và nhãn tính trực tiếp từ Gaussian đã biết. Đây là chẩn đoán xấp xỉ trên miền, không phải dữ liệu test độc lập.
- *Sai số lớn nhất có nằm quanh đỉnh Gaussian không?* Không theo điểm có sai số tuyệt đối lớn nhất: điểm đó có nhãn thật khoảng 0.0832, không thuộc vùng \(y\geq0.1\). Tuy vậy MAE vùng \(y\geq0.1\) khoảng gấp đôi MAE vùng \(y<0.01\). Vùng mật độ thấp có nhiều điểm nên vẫn chiếm 43.47% tổng bình phương sai số. Nguồn là phép tính trên validation trong `results/final/validation_error_analysis.json`.
- *Vì sao phải nhắc test đã được xem?* Vì runner C đã tính metric test trước khi chốt mô hình cuối; báo cáo cần nói thật hạn chế này thay vì gọi test là hoàn toàn chưa từng được dùng.

### Còn treo

- [x] Ngày 5/11: đối chiếu bảng A/B trong báo cáo với summary CSV của Trung; các mean/std sau làm tròn khớp.
- [x] Ngày 6/11: chuẩn bị câu hỏi hỏi xoáy cho cả nhóm về train/validation/test.
- [ ] Ngày 7/11: chạy lại toàn bộ từ đầu trên máy sạch theo README. Chưa thực hiện đầy đủ vì các runner A/B/C có thể ghi đè kết quả và `run_final` sẽ chấm test lại; chỉ chạy kiểm tra an toàn không dùng test.
- [x] Ngày 8/11: chạy preflight checker ở chế độ không tạo ZIP, có đặt `PYTHONIOENCODING=utf-8`; ghi nhận mã thoát 1, thiếu DOCX/PDF và nhiều false positive đường dẫn tuyệt đối do checker quét `.venv`.
- [ ] Ngày 8/11: Trúc xuất DOCX/PDF; cả nhóm rà lại các cảnh báo/lỗi checklist và xác nhận bảng tỉ lệ đóng góp trước khi nén.

---

## 14. Hỏi xoáy nhóm: dữ liệu, train/validation/test và chọn mô hình

### Câu 19: Vì sao không đưa test vào `train_model()`?

**Trả lời:** `train_model()` nhận riêng train và validation. Train cập nhật trọng số; validation theo dõi loss, early stopping và chọn checkpoint. Nếu test ảnh hưởng cập nhật hoặc chọn mô hình thì sai số cuối bị lạc quan. Tuy nhiên, runner C có lưu test metrics cho 12 lượt theo yêu cầu báo cáo thực nghiệm C; test không được dùng chọn cấu hình và báo cáo phải công khai việc test đã được truy vấn. Tham khảo `src/train.py` và `experiments/run_C.py`.

### Câu 20: Tập train pool, validation và test được tạo ra thế nào?

**Trả lời:** `src/data.py` tính nhãn theo Gaussian đã biết cho các điểm lấy đều trong miền; ba split có kích thước 2000, 500 và 100, được sinh với seed dữ liệu 42 rồi lưu CSV. Train pool cung cấp tiền tố 200–2000 điểm cho A, còn validation cố định 500 điểm cho việc chọn. Test 100 điểm được dùng trong bảng C và lần đánh giá checkpoint cuối.

### Câu 21: Dừng sớm và chọn checkpoint khác nhau thế nào?

**Trả lời:** Mỗi epoch tính validation loss. Nếu không cải thiện trong 100 epoch liên tiếp thì dừng; song song, code giữ bản sao trọng số tại epoch có validation loss tốt nhất. Có thể dừng ở epoch cuối mà checkpoint tốt nhất nằm trước đó. Cấu hình cuối dừng ở giới hạn 1000 epoch, checkpoint được lưu ở best epoch 980 (`results/final/final_metrics.json`).

### Câu 22: Nhóm chọn A/2000 seed 1 dựa trên số liệu nào?

**Trả lời:** Xếp 27 cấu hình A/B/C theo mean validation MSE của ba seed; A/2000 đứng đầu với \(1.681281\times10^{-7}\). Sau đó chọn seed 1 vì trong ba seed của A/2000, nó có validation MSE thấp nhất trong log A. Không dùng test MSE để chọn. So sánh giữa A và B/C cũng cần nhớ rằng A dùng 2000 mẫu, B/C dùng 1000.

### Câu 23: A/2000 có tốt hơn C/4 trong mọi ý nghĩa không?

**Trả lời:** A/2000 có mean validation MSE thấp hơn trong các cấu hình đã thử, nhưng hai cấu hình khác cả số mẫu train lẫn kiến trúc. Vì vậy không thể quy toàn bộ chênh lệch cho một yếu tố hoặc kết luận A/2000 tốt hơn trong mọi môi trường. C/4 còn có độ lệch chuẩn validation lớn hơn qua ba seed.

### Câu 24: Làm sao đọc MSE, MAE và RMSE cùng nhau?

**Trả lời:** MSE bình phương sai số nên nhạy với sai số lớn và có đơn vị bình phương; MAE là sai số tuyệt đối trung bình; RMSE là căn MSE và cùng đơn vị với nhãn. Với checkpoint cuối, validation MSE là \(1.08195\times10^{-7}\), MAE \(2.45050\times10^{-4}\), RMSE \(3.28931\times10^{-4}\). Cần ghi rõ tập và checkpoint, không trộn metric qua tập.

### Câu 25: Sai số lớn nhất có nằm đúng quanh đỉnh Gaussian?

**Trả lời:** Không nhất thiết. Trên validation, điểm sai tuyệt đối lớn nhất có \(y\approx0.0832\), thấp hơn ngưỡng vùng \(y\geq0.1\), với sai số khoảng \(1.004\times10^{-3}\). MAE ở vùng mật độ cao vẫn khoảng gấp đôi vùng \(y<0.01\), nhưng vùng thấp có nhiều điểm hơn và đóng góp 43.47% tổng bình phương sai số. Số được tính từ checkpoint cuối, validation và lưu trong `results/final/validation_error_analysis.json`.

### Câu 26: Nếu ngày 7 chạy README từ đầu thì lệnh nào cần tránh để không làm hỏng kết quả?

**Trả lời:** Không chạy `python -m src.data` vì dữ liệu đã chốt; không chạy đủ runner A/B/C vì chúng có thể ghi đè summary, log và hình; tuyệt đối không chạy `python -m scripts.run_final` lại vì sẽ đánh giá test lần nữa. Khi chỉ kiểm tra báo cáo/hình có thể chạy `python -m scripts.build_report` và `python -m scripts.plot_final_evaluation`, trong đó script sau chỉ đọc validation.
