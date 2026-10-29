# Trả lời vấn đáp — Khang, phần được giao ngày 29/10

## 1. Xác định đúng phần việc trong ngày

Theo bảng tiến độ trong [`Kế hoạch.md`](Kế%20hoạch.md), việc Khang được giao **ngày 29/10** là:

> Viết `src/model.py` — MLP nhận danh sách số nơ-ron.

Các việc khác của Khang được giao vào ngày sau, không gộp vào phạm vi ngày 29/10:

- **30/10:** `src/train.py` — vòng lặp huấn luyện, `DataLoader`, Adam, `MSELoss`, log, checkpoint và đo thời gian.
- **31/10:** `src/evaluate.py` — đánh giá MSE/MAE/RMSE và biểu đồ.

Vì vậy tài liệu này tập trung vào thiết kế và giải thích **mô hình MLP** trong [`src/model.py`](src/model.py), không nhận phần việc train/evaluate là yêu cầu của riêng ngày 29/10.

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
