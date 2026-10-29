# Tài liệu vấn đáp — Phần Khang: MLP, huấn luyện và đánh giá

Tài liệu ôn vấn đáp cho phần lõi PyTorch Khang phụ trách: `src/model.py`, `src/train.py`, `src/evaluate.py`. Dữ liệu và các script thực nghiệm A/B/C thuộc phần việc của thành viên khác. Các ví dụ dưới đây dùng API của phần lõi; không khẳng định các thực nghiệm A/B/C đã hoàn thành.

## 1. Bài nói mở đầu (khoảng 30 giây)

> Phần em phụ trách là lõi PyTorch để xấp xỉ hàm mật độ Gaussian hai chiều. Mạng nhận hai đầu vào \(x_1,x_2\), qua các tầng fully connected dùng `nn.Linear` và activation `tanh`, rồi cho một đầu ra tuyến tính \(\hat y\). Em tự viết vòng huấn luyện với `DataLoader`, `MSELoss` và Adam. Sau mỗi epoch, mô hình được đánh giá trên validation; early stopping và checkpoint đều dựa trên validation loss. Test không tham gia huấn luyện hay chọn cấu hình, chỉ dùng đánh giá cuối. Module đánh giá tính MSE, MAE, RMSE và tạo đồ thị \(y\)–\(\hat y\), bề mặt thật so với bề mặt MLP.

Nếu được hỏi về số liệu, chỉ trình bày số liệu từ log chính thức của thí nghiệm tương ứng. Smoke test của phần mềm chỉ kiểm tra chức năng, không thay thế kết quả A/B/C.

## 2. Bản đồ phần việc

| Thành phần | Nhiệm vụ của nó | Câu cần nhớ |
|---|---|---|
| [`src/model.py`](../src/model.py) | Định nghĩa MLP tùy biến theo số tầng và số nơ-ron. | Kiến trúc baseline: `2 → 16 → 16 → 1`. |
| [`src/train.py`](../src/train.py) | Kiểm tra shape, đặt seed, tạo batch, cập nhật trọng số, validation, early stopping, log và checkpoint. | Chọn epoch tốt nhất bằng validation loss. |
| [`src/evaluate.py`](../src/evaluate.py) | Dự đoán, tính metric và vẽ đồ thị. | Đánh giá trên tập nào thì metric mô tả tập đó. |
| [`configs/baseline.yaml`](../configs/baseline.yaml) | Cấu hình dùng chung do nhóm chốt. | Không tự đổi các tham số baseline giữa các thí nghiệm. |

API dữ liệu `load_train`/`load_split` trong [`src/data.py`](../src/data.py) do Trúc phụ trách; phần Khang dùng các hàm này nhưng không sở hữu logic sinh/chia dữ liệu.

## 3. Mô hình và công thức

Lớp `MLP` kế thừa `torch.nn.Module`. Tham số `hidden` mô tả độ rộng từng tầng ẩn:

```python
MLP(hidden=[16, 16], activation="tanh")  # baseline: 2 -> 16 -> 16 -> 1
MLP(hidden=[8], activation="relu")       # ví dụ một tầng ẩn: 2 -> 8 -> 1
```

Với \(h_0=x\), các tầng ẩn tính:

\[
h_\ell=\phi(W_\ell h_{\ell-1}+b_\ell),
\qquad
\hat y=W_{\mathrm{out}}h_L+b_{\mathrm{out}}.
\]

Ở baseline, \(\phi=\tanh\). Tầng cuối không có activation, vì \(y\) là giá trị thực và không cần ép vào khoảng \([0,1]\) hay \([-1,1]\). Batch đầu vào shape `(n, 2)` cho đầu ra `(n, 1)`.

### Câu hỏi thường gặp

**Vì sao cần MLP nhiều tầng?**  
Các tầng tuyến tính có activation phi tuyến có thể biểu diễn quan hệ phi tuyến giữa \((x_1,x_2)\) và mật độ Gaussian. Thực nghiệm B/C thay độ rộng hoặc độ sâu để so sánh năng lực mô hình.

**Vì sao dùng `tanh`?**  
Đây là activation đã được nhóm chốt cho baseline. Nó tạo tính phi tuyến và có đầu ra bị chặn trong \([-1,1]\); không nên nói rằng `tanh` bắt buộc cho mọi bài toán hay chắc chắn tốt nhất nếu chưa có kết quả so sánh.

**Mạng có tầng ẩn rỗng thì sao?**  
Trong hiện thực này, `hidden=[]` tạo một phép biến đổi tuyến tính trực tiếp từ 2 đặc trưng đến 1 đầu ra; đó là mô hình tuyến tính, không phải baseline.

## 4. Giải thích vòng huấn luyện theo trình tự

Hàm chính là `train_model(X_train, y_train, X_val, y_val, ...)`.

1. **Kiểm tra dữ liệu:** \(X\) phải có shape `(n, 2)`; \(y\) nhận `(n,)` hoặc `(n, 1)`. Dữ liệu rỗng, sai số lượng hoặc có NaN/Inf sẽ báo lỗi.
2. **Đồng nhất kiểu và shape:** đổi sang `float32`, nhãn 1D được `unsqueeze(1)` thành `(n, 1)`. Việc này khớp đầu ra MLP và ngăn broadcasting làm MSE sai shape.
3. **Đặt seed:** áp dụng cho Python, NumPy, PyTorch và bộ sinh số của `DataLoader`, để khởi tạo và xáo trộn có thể lặp lại trong cùng điều kiện.
4. **Tạo mini-batch:** `TensorDataset` ghép \(X,y\); `DataLoader` chia batch size 32 và xáo trộn dữ liệu train.
5. **Train một epoch:** gọi `model.train()`. Với mỗi batch, mô hình dự đoán, `MSELoss` tính lỗi, `backward()` tính gradient, optimizer cập nhật tham số.
6. **Tính train loss epoch:** loss của từng batch là trung bình batch; code nhân lại với số mẫu rồi chia tổng số mẫu. Nhờ vậy batch cuối nhỏ hơn không bị tính trọng số ngang với batch đầy đủ.
7. **Validation:** gọi `model.eval()` và chạy trong `torch.no_grad()`. Validation loss dùng để theo dõi khả năng tổng quát hóa; không cập nhật trọng số.
8. **Checkpoint và early stopping:** khi validation loss thấp hơn tốt nhất trước đó, sao chép trọng số tốt nhất. Nếu không cải thiện đủ `patience` epoch thì dừng. Sau vòng lặp, nạp lại trọng số tốt nhất thay vì mặc định giữ trọng số epoch cuối.
9. **Ghi kết quả:** log CSV gồm `epoch`, `train_loss`, `val_loss`; checkpoint lưu trọng số và metadata như kiến trúc, activation, seed, epoch tốt nhất, validation loss, thời gian và config.

### Câu hỏi thường gặp

**`model.train()` có phải là lệnh huấn luyện toàn bộ không?**  
Không. Nó đặt module ở chế độ train (quan trọng với các lớp như dropout/batch normalization). Việc cập nhật thực sự do `loss.backward()` và `optimizer.step()`.

**`model.eval()` có tắt gradient không?**  
Không. `eval()` đổi hành vi của một số module; `torch.no_grad()` mới tắt theo dõi gradient trong khối đánh giá. Vì vậy thường dùng cả hai khi validation/inference.

**Tại sao tính validation sau mỗi epoch?**  
Để theo dõi hiệu năng ngoài dữ liệu cập nhật trọng số và chọn epoch/checkpoint phù hợp. Trong thiết kế hiện tại validation loss cũng là tiêu chí early stopping.

**Patience 100 nghĩa là gì?**  
Cho phép tối đa 100 epoch liên tiếp không có validation loss tốt hơn trước khi dừng. Đây là số epoch, không phải 100 batch.

**Vì sao phải clone state dict khi có cải thiện?**  
Nếu chỉ giữ tham chiếu tới tensor tham số, optimizer sẽ tiếp tục cập nhật chúng ở epoch sau. Clone tạo bản độc lập của trọng số tại epoch tốt nhất để có thể khôi phục đúng.

**Dùng test trong train được không?**  
Không. `train_model` chỉ nhận train và validation. Dùng test để chọn epoch/cấu hình làm rò rỉ thông tin và khiến đánh giá cuối không còn độc lập.

**Kết quả có tái lập tuyệt đối trên mọi máy không?**  
Seed giúp giảm ngẫu nhiên và tái lập trong cùng điều kiện, nhưng không nên cam kết tuyệt đối giống từng bit giữa mọi phần cứng, driver hoặc phiên bản PyTorch.

## 5. Loss, metric và cách diễn giải

Hàm huấn luyện dùng Mean Squared Error:

\[
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2.
\]

Hàm `evaluate_model` trả về ba chỉ số:

\[
\mathrm{MAE}=\frac{1}{N}\sum_i|y_i-\hat y_i|,
\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
\]

- **MSE:** phạt mạnh các sai số lớn; đơn vị là bình phương đơn vị \(y\).
- **MAE:** sai lệch tuyệt đối trung bình; cùng đơn vị với \(y\).
- **RMSE:** căn bậc hai MSE; cùng đơn vị với \(y\), thường dễ diễn giải hơn MSE.

Các chỉ số được tính trên thang nhãn gốc, vì baseline không chuẩn hóa \(y\). Do \(y\) của mật độ này nhỏ (đỉnh lý thuyết khoảng 0.269), MSE có thể ở dạng số mũ. Cần trình bày MSE cùng MAE/RMSE; không diễn giải MSE nhỏ mà bỏ qua thang giá trị mục tiêu.

### Đồ thị

- `plot_y_vs_yhat`: mỗi điểm là một cặp \(y\) thật và \(\hat y\); điểm gần đường \(y=\hat y\) biểu thị dự đoán gần thực tế.
- `plot_surface_comparison`: tạo lưới đều trên miền, tính bề mặt Gaussian thật bằng công thức dữ liệu, rồi đặt cạnh bề mặt dự đoán của MLP. Hai bề mặt so sánh hình dạng tổng quát trên cùng miền.

Đồ thị nhìn đẹp không thay thế metric; metric tốt trên validation/test cũng không tự nó chứng minh mọi vùng không gian đều được xấp xỉ tốt. Cần dùng cả số liệu và hình.

## 6. Câu trả lời ngắn về baseline

Theo `configs/baseline.yaml`:

| Tham số | Giá trị |
|---|---|
| Kiến trúc A/baseline | `[16, 16]`, tức hai tầng ẩn, mỗi tầng 16 nơ-ron |
| Activation | `tanh`; đầu ra tuyến tính |
| Số mẫu mặc định cho B/C | 1000 |
| Optimizer / learning rate | Adam / \(10^{-3}\) |
| Batch size | 32 |
| Giới hạn epoch / patience | 1000 / 100 |
| Seed thí nghiệm | 0, 1, 2 |
| Chuẩn hóa nhãn | Không |

Nếu giảng viên hỏi “kết quả cuối là bao nhiêu?”, hãy trích số từ `summary.csv`/log thực nghiệm chính thức, nói rõ cấu hình, seed và tập đánh giá. Không dùng số của một smoke test làm kết quả A/B/C hoặc kết quả báo cáo.

## 7. Thao tác demo code

Chạy từ thư mục `BTL01 - Function Approximation` để import `src` thành công. Ví dụ tối giản để huấn luyện baseline và in metric test sau khi checkpoint đã được chọn qua validation:

```python
from src.data import load_config, load_train, load_split
from src.train import train_model
from src.evaluate import evaluate_model

cfg = load_config()
train_cfg = cfg["train"]
X_train, y_train = load_train(train_cfg["n_train"])
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

result = train_model(
    X_train, y_train, X_val, y_val,
    hidden=train_cfg["hidden"],
    activation=train_cfg["activation"],
    config=train_cfg,
    seed=train_cfg["seeds"][0],
    log_path="results/khang_seed0.csv",
    checkpoint_path="models/khang_seed0.pt",
)

# Test chỉ được chạm tới sau khi train đã chọn checkpoint bằng validation.
print("Best epoch:", result.best_epoch)
print("Best validation loss:", result.best_val_loss)
print("Test metrics:", evaluate_model(result.model, X_test, y_test))
```

Phần này minh họa lõi của Khang; script A/B/C phải tự lặp theo các cấu hình/seed quy định, dùng validation cố định và lưu log theo quy ước chung của nhóm.

## 8. Giới hạn và cách trả lời trung thực

- Lõi code có thể smoke test bằng vài epoch; smoke test chỉ kiểm tra luồng gọi, không thay thế chạy đủ baseline hoặc các thí nghiệm.
- Đoạn demo trên chạy một seed; để báo cáo mean ± std cần tổng hợp các seed 0, 1, 2 như kế hoạch.
- Hiện thực `evaluate_model` nhận tập do bên gọi truyền vào; trách nhiệm không dùng test trong quá trình chọn mô hình cũng cần được giữ ở script thực nghiệm.
- Không tự khẳng định mô hình đã “tốt nhất” nếu chưa so sánh bằng validation theo thiết kế thí nghiệm và chưa có log chứng minh.
- Nếu được hỏi ngoài phạm vi (sinh dữ liệu, biểu đồ dữ liệu ban đầu, triển khai A/B/C), giải thích đó là phần của thành viên phụ trách tương ứng; Khang cung cấp lõi `train_model` và metric để các script ấy dùng.

## 9. Checklist trước buổi vấn đáp

- [ ] Trình bày được sơ đồ `2 → 16 → 16 → 1` và vì sao đầu ra tuyến tính.
- [ ] Giải thích được một batch đi qua forward, MSE, backward, optimizer như thế nào.
- [ ] Phân biệt `train()`, `eval()` và `no_grad()`.
- [ ] Giải thích validation, patience, checkpoint tốt nhất và lý do không dùng test để chọn.
- [ ] Nêu được shape `(n, 2)`, `(n, 1)` và nguy cơ broadcasting.
- [ ] Phân biệt MSE, MAE, RMSE và đọc được số dạng khoa học.
- [ ] Mở được log CSV, checkpoint và hình do lần chạy thực tế sinh ra.
- [ ] Chỉ trình bày số liệu có nguồn log; phân biệt smoke test với kết quả chính thức.

## 10. Ghi chú môi trường và chạy kiểm thử phần Khang

README là tài liệu dùng chung của nhóm nên phần này chỉ ghi lệnh cài/chạy phục vụ demo cá nhân, không yêu cầu thay đổi README.

### Dùng môi trường `.venv` CPU hiện có trên Windows

Mở PowerShell và chạy từ thư mục BTL01:

```powershell
Set-Location "D:\Personal Project\deep-learning-assignments\BTL01 - Function Approximation"
.\.venv\Scripts\python.exe -c "import torch, numpy, pandas, matplotlib, yaml; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

Kết quả hiện tại mong đợi là PyTorch có hậu tố `+cpu`, `CUDA False`. Đây là cấu hình CPU hợp lệ; không cần GPU NVIDIA để kiểm thử lõi.

### Kiểm tra cú pháp và sinh dữ liệu

```powershell
.\.venv\Scripts\python.exe -m py_compile src\model.py src\train.py src\evaluate.py
.\.venv\Scripts\python.exe -m src.data
```

Lệnh thứ nhất không in gì khi cú pháp hợp lệ. Lệnh thứ hai in định thức covariance `0.3500`, giá trị đỉnh xấp xỉ `0.269021`, số mẫu train pool/validation/test lần lượt `2000/500/100` và bảng thống kê. Đây là kiểm tra dữ liệu do module của Trúc phụ trách, không phải kết quả thực nghiệm Khang.

### Kiểm tra luồng train/evaluate

Để demo lõi nhanh mà không chạy 1000 epoch, chạy lệnh sau trong thư mục BTL01. Nó dùng một phần nhỏ của train/validation và tối đa 2 epoch; kết quả chỉ chứng minh các hàm hoạt động, không dùng làm số liệu báo cáo:

```powershell
@'
from src.data import load_train, load_split, load_config
from src.train import train_model
from src.evaluate import evaluate_model

X, y = load_train(16)
X_val, y_val = load_split("val")
config = dict(load_config()["train"])
config.update(max_epochs=2, patience=2)

result = train_model(
    X, y, X_val[:16], y_val[:16],
    hidden=[4], config=config, seed=0,
)
print("best_epoch:", result.best_epoch)
print("epochs_run:", len(result.history))
print("best_val_loss:", result.best_val_loss)
print("validation_metrics:", evaluate_model(result.model, X_val[:16], y_val[:16]))
'@ | .\.venv\Scripts\python.exe -
```

Kết quả phải có `best_epoch` ít nhất là 1, tối đa 2 epoch đã chạy, và `validation_metrics` gồm `mse`, `mae`, `rmse` là các số hữu hạn. Giá trị loss cụ thể có thể thay đổi nếu đổi seed, cấu hình, thư viện hoặc phần cứng.

### Chạy baseline đầy đủ để lấy số tham chiếu

Muốn kiểm tra gần với cấu hình nhóm chốt, dùng ví dụ tại mục 7 nhưng bỏ phần ghi log/checkpoint vào đường dẫn chính thức nếu chỉ đang thử. Baseline là 1000 mẫu, hidden `[16, 16]`, `tanh`, Adam, learning rate `0.001`, batch size `32`, tối đa `1000` epoch, patience `100`. Một lần chạy CPU có thể mất khoảng vài chục giây.

Sau khi chọn epoch bằng validation, mới gọi `evaluate_model` trên test. MSE/MAE/RMSE phải được ghi dạng số mũ khi nhỏ. Kết quả của một seed là số tham khảo; báo cáo A/B/C cần chạy đủ cấu hình và seed theo kế hoạch, tổng hợp mean ± std từ log chính thức. Thời gian chạy thay đổi theo máy nên không cần khớp tuyệt đối với thời gian tham khảo.
