# 3. Mô hình và cài đặt

Nháp ngày 02/11/2026, người viết: Phạm Trần Minh Khang. Mục này mô tả phần lõi PyTorch trong `src/model.py`, `src/train.py` và `src/evaluate.py`. Các tham số lấy từ `configs/baseline.yaml`; kết quả tổng hợp A/B/C được trình bày ở mục 4, không suy ra từ một lần chạy minh họa trong mục này.

## 3.1. Kiến trúc mạng

Bài toán nhận hai đặc trưng \(x_1,x_2\) và dự đoán một giá trị vô hướng \(\hat y\), xấp xỉ mật độ Gaussian \(y=f(x_1,x_2)\). MLP được cài đặt bằng PyTorch: lớp `MLP` kế thừa `torch.nn.Module`, các phép biến đổi tuyến tính dùng `torch.nn.Linear`.

Kiến trúc baseline:

```text
Đầu vào              Tầng ẩn 1       Tầng ẩn 2       Đầu ra
(x₁, x₂)  ─────────►  16 nơ-ron ───►  16 nơ-ron ───►  ŷ
                         tanh             tanh         tuyến tính
```

Tổng quát, danh sách `hidden` quy định số nơ-ron của từng tầng ẩn. Ví dụ `[16, 16]` tạo hai tầng ẩn, còn `[16, 16, 16]` tạo ba tầng ẩn. Nhờ vậy, cùng một lớp mạng dùng được cho baseline, thực nghiệm B (thay độ rộng của một tầng ẩn) và C (thay số tầng ẩn). Tầng ẩn dùng activation theo cấu hình; baseline dùng `tanh`. Tầng đầu ra là tuyến tính, không ép \(\hat y\) vào một khoảng cố định.

Với \(h_0=[x_1,x_2]^T\), phép lan truyền thuận của mạng có \(L\) tầng ẩn là:

\[
h_\ell=\phi(W_\ell h_{\ell-1}+b_\ell), \qquad \ell=1,\ldots,L,
\]
\[
\hat y=W_{\mathrm{out}}h_L+b_{\mathrm{out}},
\]

trong đó \(W_\ell\) và \(b_\ell\) lần lượt là trọng số và bias được học; \(\phi\) là hàm kích hoạt. Với baseline, \(\phi=\tanh\), \(L=2\), mỗi tầng ẩn có 16 nơ-ron. Một batch đầu vào có shape \((N,2)\); đầu ra có shape \((N,1)\).

Hiện thực hóa phần forward:

```python
for size in hidden:
    layers.extend((nn.Linear(input_size, size), activation_layer()))
    input_size = size
layers.append(nn.Linear(input_size, 1))
```

Activation phi tuyến giữa các lớp tuyến tính cho phép mạng biểu diễn ánh xạ phi tuyến. Đầu ra tuyến tính phù hợp với mục tiêu hồi quy giá trị thực; không dùng sigmoid ở đầu ra vì đề không yêu cầu xác suất hay giới hạn đầu ra trong \([0,1]\).

## 3.2. Hàm mất mát

Mạng được tối ưu bằng sai số bình phương trung bình (Mean Squared Error, MSE):

\[
\mathcal{L}_{\mathrm{MSE}}
=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2.
\]

Trong mỗi mini-batch, `torch.nn.MSELoss` tính trung bình bình phương sai số giữa nhãn và dự đoán. Sai số lớn bị phạt mạnh hơn sai số nhỏ do phép bình phương. Nhãn \(y\) không được chuẩn hóa theo baseline (`normalize_y: false`), do đó loss và các chỉ số ở mục đánh giá được tính trên thang nhãn gốc.

Để theo dõi loss train ở cấp epoch, code nhân loss trung bình của mỗi batch với số mẫu trong batch, cộng qua các batch rồi chia tổng số mẫu. Cách tính này xử lý đúng batch cuối nếu batch đó ít mẫu hơn `batch_size`. Validation loss được tính trên toàn bộ tập validation sau mỗi epoch, ở chế độ suy luận và không tạo gradient.

## 3.3. Thiết lập và quy trình huấn luyện

Các giá trị baseline dùng chung cho A, B, C được chốt trong `configs/baseline.yaml`:

| Tham số | Baseline |
|---|---:|
| Số mẫu train mặc định | 1000 (B, C; A thay đổi theo cấu hình) |
| Số tầng ẩn, số nơ-ron mỗi tầng | 2 tầng × 16 |
| Hàm kích hoạt tầng ẩn | `tanh` |
| Hàm kích hoạt đầu ra | Tuyến tính |
| Optimizer | Adam |
| Learning rate | \(10^{-3}\) |
| Batch size | 32 |
| Số epoch tối đa | 1000 |
| Early stopping patience | 100 epoch không cải thiện validation loss |
| Seed huấn luyện | 0, 1, 2 |
| Chuẩn hóa nhãn | Không |

Hàm `train_model` trong `src/train.py` nhận riêng dữ liệu train và validation; test không phải đối số của hàm. Đặc trưng và nhãn được kiểm tra shape, tính hữu hạn rồi chuyển sang `float32`. Nhãn dạng \((N,)\) được đổi thành \((N,1)\) để khớp đầu ra MLP. Dữ liệu train được ghép bằng `TensorDataset` và chia mini-batch bằng `DataLoader`.

Ở mỗi batch, vòng lặp thực hiện:

1. `model.train()` đặt mạng ở chế độ huấn luyện.
2. `optimizer.zero_grad()` xóa gradient còn lại từ batch trước.
3. Forward tạo dự đoán; `MSELoss` tính loss.
4. `loss.backward()` tính gradient qua backpropagation.
5. `optimizer.step()` cập nhật trọng số bằng Adam.

Cuối mỗi epoch, mô hình chuyển sang `model.eval()` và tính validation loss trong `torch.no_grad()`. Nếu validation loss giảm so với tốt nhất trước đó, code lưu bản sao độc lập của `state_dict` và cập nhật `best_epoch`. Khi validation loss không cải thiện trong 100 epoch liên tiếp, early stopping kết thúc huấn luyện. Trước khi trả kết quả, trọng số tại epoch có validation loss thấp nhất được nạp lại; vì vậy mô hình trả về không nhất thiết là trọng số ở epoch cuối.

Thời gian vòng lặp được đo bằng `time.perf_counter()`: bao gồm các epoch huấn luyện, tính validation loss và lưu bản sao trọng số tốt nhất trong bộ nhớ; không tính thời gian ghi CSV, lưu checkpoint ra đĩa hoặc vẽ biểu đồ. Mỗi lần chạy có thể lưu log theo epoch với các cột `epoch`, `train_loss`, `val_loss`; checkpoint `.pt` lưu trọng số tốt nhất cùng cấu hình và metadata cần để nhận diện kiến trúc, seed và epoch được chọn. Các seed giúp so sánh nhiều lần khởi tạo; kết quả nhóm được tổng hợp theo mean ± std ở phần thực nghiệm.

## 3.4. Validation, test và đánh giá

Validation được dùng trong giai đoạn phát triển để theo dõi khả năng tổng quát hóa, dừng sớm và chọn checkpoint. Tập test độc lập không tham gia cập nhật trọng số hoặc chọn epoch/cấu hình; test chỉ được đánh giá sau khi mô hình cuối đã được chọn theo validation.

Module `src/evaluate.py` cung cấp MSE, MAE và RMSE:

\[
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|,
\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
\]

MAE và RMSE cùng đơn vị với nhãn \(y\), trong khi MSE có đơn vị bình phương. Cả ba chỉ số được báo cáo trên thang nhãn gốc. Hàm đánh giá kiểm tra shape và giá trị hữu hạn, chuyển mô hình tạm thời sang `eval()` và thực hiện dự đoán trong `torch.no_grad()`.

Hai biểu đồ hỗ trợ kiểm tra xấp xỉ gồm:

- Đồ thị giá trị thật \(y\) theo giá trị dự đoán \(\hat y\), với đường tham chiếu \(y=\hat y\).
- Hai bề mặt 3D đặt cạnh nhau: hàm Gaussian thật và đầu ra MLP trên một lưới đều bao phủ miền \([-2,2]\times[-4,4]\).

Hình minh họa baseline dùng validation trong thư mục `results/diagnostics/figures/`; đây là minh chứng chạy pipeline, không thay hình/số liệu cuối cùng được chọn sau khi so sánh A/B/C.

## 3.5. Tổ chức mã nguồn

| Module | Trách nhiệm |
|---|---|
| `src/model.py` | Định nghĩa MLP tùy biến theo danh sách kích thước tầng ẩn và activation. |
| `src/train.py` | Kiểm tra dữ liệu, đặt seed, tạo `DataLoader`, huấn luyện, validation, early stopping, checkpoint, log và đo thời gian. |
| `src/evaluate.py` | Dự đoán, tính MSE/MAE/RMSE, vẽ đồ thị \(y\)–\(\hat y\) và so sánh bề mặt. |
| `configs/baseline.yaml` | Lưu baseline và danh sách cấu hình thí nghiệm để các lần chạy dùng nhất quán. |

Các script A/B/C gọi lại lõi huấn luyện với cấu hình tương ứng; việc tổng hợp bảng, biểu đồ và phân tích kết quả từng thực nghiệm được trình bày riêng ở mục 4.
