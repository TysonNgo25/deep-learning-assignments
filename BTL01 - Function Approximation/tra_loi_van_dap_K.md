# Trả lời vấn đáp — Khang: phần việc ngày 29–30/10

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
