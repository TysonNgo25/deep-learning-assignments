# 2. Tương ứng API Keras và PyTorch

Người viết: Huỳnh Ngọc Trúc

## 2.1. Bảng tương ứng

Bảng 2.1 liệt kê các thành phần Keras gặp trong 11 notebook và cách viết tương ứng trong PyTorch. Cột "Lưu ý" ghi những chỗ dễ sai khi chuyển.

Bảng 2.1. Thành phần Keras và cách viết PyTorch tương ứng.

| Keras | PyTorch | Lưu ý |
| --- | --- | --- |
| `Input(shape=(H, W, C))` | không có lớp riêng; tensor `(N, C, H, W)` | đổi chiều một lần khi tạo tensor |
| `Dense(units, activation)` | `nn.Linear(in, units)` + hàm kích hoạt trong `forward` | phải tự tính `in` |
| `Conv2D(f, (3, 3), padding='same')` | `nn.Conv2d(c_in, f, 3, padding=1)` | `'valid'` là `padding=0` |
| `MaxPooling2D((2, 2))` | `nn.MaxPool2d(2)` | cùng làm tròn xuống |
| `Dropout(rate)` | `nn.Dropout(p=rate)` | cùng nghĩa: tỷ lệ bị tắt |
| `Flatten()` | `nn.Flatten()` | thứ tự phần tử khác (C, H, W) nhưng số tham số như nhau |
| `softmax` + `categorical_crossentropy` | logits + `nn.CrossEntropyLoss` | không để softmax ở lớp cuối; nhãn là chỉ số lớp |
| `mean_squared_error`, `mean_absolute_error` | `nn.MSELoss`, `nn.L1Loss` | nhãn cùng shape với đầu ra |
| `SGD(lr)`, `Adam(lr)` | `torch.optim.SGD`, `torch.optim.Adam` | SGD Keras không momentum; Adam cùng `betas`, `eps` 1e-8 (Keras 2.0.9) |
| khởi tạo `glorot_uniform`, bias 0 | mặc định Kaiming uniform | gán lại bằng `init_keras_like` |
| `compile` + `fit` + `evaluate` + `predict` | `fit`, `evaluate`, `predict` trong `common.py` | `fit` trả `history` dạng dict như Keras |
| `TimeDistributed(layer)` | gộp `(N·T, ...)` → layer → tách `(N, T, ...)` | mục 3.6 |
| `LSTM(128, return_sequences=True)` | `nn.LSTM(..., batch_first=True)`, lấy `out[:, -1]` khi `return_sequences=False` | PyTorch có 2 bias mỗi cổng |
| `BatchNormalization()` | `nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)` | momentum định nghĩa ngược (mục 3.9) |
| `UpSampling2D()` | `nn.Upsample(scale_factor=2, mode='nearest')` | |
| `Reshape` + `Permute` + softmax theo pixel | `nn.CrossEntropyLoss` trên `(N, 7, H, W)` | mục 3.9 |
| `Lambda(lambda x: x/127.5 - 1)` | một dòng đầu `forward` | |
| `Model(inputs, get_layer('fc2').output)` | `forward(x, return_features=True)` | lấy đầu ra **sau** relu như Keras |
| `ModelCheckpoint`, `EarlyStopping`, `TensorBoard` | lớp cùng tên trong `common.py` | mục 3.7 |
| `ReduceLROnPlateau` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | ngưỡng tuyệt đối 1e-4 như Keras |
| `ImageDataGenerator` + `fit_generator` | `Dataset` tự viết + `DataLoader` | mục 3.8 |
| `model.summary()` | `torchinfo.summary` | |

## 2.2. Thư viện dùng chung `pytorch/common.py`

Keras có sẵn vòng huấn luyện và callback; PyTorch thì không. File `common.py` viết lại phần này một lần cho cả 11 notebook, nhờ đó các notebook PyTorch đọc gần giống notebook Keras. Bảng 2.2 tóm tắt các hàm chính.

Bảng 2.2. Các hàm trong `common.py` và phần Keras mà chúng thay thế.

| Nhóm | Hàm | Thay cho | Ghi chú |
| --- | --- | --- | --- |
| Đường dẫn | `ROOT`, `DATA_DIR`, `RESULTS_DIR`, `EXAMPLES_DIR` | đường dẫn tương đối trong notebook gốc | tính từ vị trí file, chạy được cả máy cá nhân và Kaggle |
| Môi trường | `set_seed`, `get_device` | `device_lib.list_local_devices()` | |
| Dữ liệu | `to_nchw`, `onehot_to_index`, `make_loader` | tham số `batch_size`, `shuffle` của `fit` | `to_nchw` không chuẩn hóa giá trị |
| Dữ liệu | `load_fer2013` | đọc `fer2013_*_onehot.csv` | đọc thẳng `fer2013.csv`, đã kiểm cùng thứ tự dòng với file gốc |
| Mô hình | `init_keras_like` | khởi tạo mặc định của Keras | glorot uniform, bias 0; LSTM: orthogonal, bias cổng quên 1 |
| Mô hình | `count_params`, `count_all`, `summary` | `model.summary()` | `count_all` cộng running mean/var của BatchNorm |
| Chỉ số | `accuracy`, `sign_accuracy`, `mae` | `metrics=[...]` | `accuracy` dùng được cho pixel accuracy |
| Huấn luyện | `fit`, `evaluate`, `predict` | `fit`, `evaluate`, `predict` | đổi `train()`/`eval()`, `no_grad`, lấy trung bình theo số mẫu |
| Callback | `ModelCheckpoint`, `EarlyStopping`, `TensorBoard`, `reduce_lr_on_plateau` | callback cùng tên | không tự khôi phục trọng số tốt nhất, giống Keras |
| Kết quả | `save_history`, `save_metrics`, `plot_history`, `ghi_so_sanh` | `history.history`, `plt.plot(...)` | `ghi_so_sanh` ghi một dòng vào `results/so_sanh.csv` |

Vòng huấn luyện cốt lõi của `fit` như sau (rút gọn). Mỗi batch làm đúng năm bước mà `model.fit` của Keras làm ngầm:

```python
def _run_epoch(model, loader, loss_fn, device, metric=None, optimizer=None):
    training = optimizer is not None
    model.train(training)                          # bật/tắt dropout, BatchNorm
    with torch.set_grad_enabled(training):         # đánh giá thì không tính gradient
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ...                                    # cộng dồn loss, chỉ số theo số mẫu
```

`fit` gọi `_run_epoch` cho tập train rồi tập validation ở mỗi epoch, ghi `loss`, `acc`, `val_loss`, `val_acc`, `lr` vào `history`, gọi các callback và `scheduler.step(val_loss)`. `EarlyStopping` đếm như Keras: dừng sau `patience` epoch liền không giảm `val_loss`. Cách đếm này đã được đối chiếu với hai log gốc: Gender dừng ở epoch 32 khi tốt nhất ở epoch 27, Emotion dừng ở 57 khi tốt nhất ở 52.

## 2.3. Hàm khởi tạo trọng số

Keras khởi tạo `Dense` và `Conv2D` bằng glorot uniform, bias bằng 0. PyTorch mặc định dùng Kaiming uniform cho cả trọng số lẫn bias. Để so sánh công bằng, mọi mô hình gọi `init_keras_like` ngay sau khi tạo:

```python
def init_keras_like(model, forget_bias=True):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            h = m.hidden_size
            for name, p in m.named_parameters():
                if name.startswith("weight_ih"):   nn.init.xavier_uniform_(p)
                elif name.startswith("weight_hh"): nn.init.orthogonal_(p)
                elif name.startswith("bias"):
                    nn.init.zeros_(p)
                    if forget_bias and name.startswith("bias_ih"):
                        p.data[h:2 * h] = 1.0     # cổng quên = 1 như unit_forget_bias của Keras
```

## 2.4. Notebook getting_started

Notebook gốc in phiên bản TensorFlow (2.14.0) và danh sách thiết bị bằng `device_lib.list_local_devices()`, phát hiện một GPU Tesla T4. Bản PyTorch in phiên bản `torch`, `torchvision`, kiểm `torch.cuda.is_available()` và tên GPU, rồi chạy thử một phép nhân ma trận trên thiết bị đã chọn. Trên máy cá nhân (GPU AMD, PyTorch bản CPU) notebook báo `cpu`; các notebook cần GPU chạy trên Kaggle.

Nguồn số liệu: `pytorch/common.py`; `pytorch/0. Getting Started/getting_started_pytorch.ipynb`.
