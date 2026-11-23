---
name: chuyen-doi-btl03
description: Quy trình và công thức chuyển một notebook Keras trong BTL03 sang PyTorch (MLP, CNN, TimeDistributed + LSTM, BatchNorm/UpSampling SegNet, mạng toàn tích chập, callback, ImageDataGenerator, lấy đặc trưng tầng giữa) kèm số tham số phải khớp. Dùng mỗi khi bắt đầu viết hoặc sửa một notebook _pytorch.ipynb hay pytorch/common.py trong BTL03.
---

# Chuyển một notebook Keras sang PyTorch (BTL03)

Quy tắc bắt buộc: `.claude/rules/btl03-code.md`. Bảng số tham số và kết quả gốc: `docs/Kế hoạch.md` mục 3. Ghi chú từng notebook gốc: mục 12.

## Quy trình cho mỗi notebook

1. **Đọc bản gốc trong máy**: `Yêu cầu/examples/<thư mục>/<tên>.ipynb` (không dùng bản GitHub). Ghi lại từ output: `Total params`, log epoch cuối, dòng kết quả test.
2. **Tạo `pytorch/<thư mục>/<tên>_pytorch.ipynb`** với ô đầu là markdown: tên bài, gốc ở đâu, kết quả gốc. Ô code đầu: import, `sys.path` tới `pytorch/`, `from common import *`, `set_seed(0)`, `device = get_device()`.
3. **(1) Data**: tái tạo đúng tiền xử lý gốc, rồi chuyển sang tensor `(N, C, H, W)`. Nhãn one-hot → `argmax` thành chỉ số lớp khi dùng CrossEntropy. Bọc `TensorDataset` + `DataLoader(batch_size=gốc, shuffle=True)` cho train, `shuffle=False` cho val/test.
4. **(2) Model**: ô markdown trích đoạn Keras gốc; ô code là lớp `nn.Module`. Đặt tên lớp con trùng `name=` Keras. `init_keras_like(model)`. In `torchinfo.summary` và `assert count_params(model) == N`.
5. **(3) Train**: loss, optimizer, lr, batch, epoch đúng gốc; `history = fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, metric=..., callbacks=[...])`. In thời gian train.
6. **(4) Evaluate**: `evaluate(...)` trên đúng tập gốc dùng; in giống gốc (`Test loss`, `Test accuracy`). Chuyển luôn phần phụ (dự đoán ảnh ngẫu nhiên, PCA, heatmap).
7. **Lưu**: `save_history`, `save_metrics`, `plot_history` vào `results/<notebook>/`; `ghi_so_sanh(...)`.
8. Restart & Run All (hoặc chạy trên GPU, xem skill `chay-notebook-btl03`), lưu notebook có output. Gọi agent `btl03-doi-chieu-keras` cho notebook vừa xong. Ghi nhật ký (skill `nhat-ky-btl03`).

## Bảng tương ứng nhanh

Đầy đủ ở `docs/Kế hoạch.md` mục 4. Những chỗ hay sai:

| Keras | PyTorch | Bẫy |
|---|---|---|
| `Conv2D(f, (3,3), padding='same')` | `nn.Conv2d(c, f, 3, padding=1)` | `valid` → `padding=0`; Emotion trộn cả hai |
| `Dense(10, activation='softmax')` + `categorical_crossentropy` | `nn.Linear(512, 10)` + `nn.CrossEntropyLoss()` | không thêm softmax; nhãn `LongTensor` chỉ số |
| `Dropout(rate=0.2)` | `nn.Dropout(0.2)` | cùng nghĩa |
| `SGD(lr=0.01)` | `optim.SGD(params, lr=0.01)` | không momentum |
| `Adam(lr=1e-4)` | `optim.Adam(params, lr=1e-4)` | eps tf.keras 1e-7; Keras 2.0.9 và PyTorch 1e-8 |
| `metrics=['accuracy']` | tự tính `(pred.argmax(1) == y).float().mean()` | |
| `model.evaluate` | `evaluate()` với `model.eval()` + `no_grad` | |

## Công thức cho các phần khó

**Khởi tạo giống Keras** (trong `common.py`):

```python
def init_keras_like(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if "weight_ih" in name: nn.init.xavier_uniform_(p)
                elif "weight_hh" in name: nn.init.orthogonal_(p)
                elif "bias" in name: nn.init.zeros_(p)
```

Keras khởi tạo bias cổng quên của LSTM bằng 1 (`unit_forget_bias=True`). Muốn giống hẳn thì gán `bias_ih[hidden:2*hidden] = 1` cho mỗi lớp, ghi chú trong notebook.

**Lấy đặc trưng tầng giữa** (`Model(inputs, get_layer('fc2').output)`):

```python
feats = {}
h = model.fc2.register_forward_hook(lambda m, i, o: feats.__setitem__("fc2", F.relu(o).detach().cpu()))
predict(model, loader); h.remove()
```

Keras `Dense(..., activation='relu', name='fc2')` trả đầu ra **sau** relu; trong PyTorch `fc2` là `nn.Linear` nên hook lấy trước relu, phải tự áp `relu` (hoặc hook vào lớp ReLU). `GlobalAveragePooling2D` sau `pool3` = `x.mean(dim=(2, 3))`.

**TimeDistributed + LSTM** (MNIST_CNN_LSTM):

```python
def forward(self, x):                     # x: (N, T=8, 1, 28, 28)
    n, t = x.shape[:2]
    x = x.reshape(n * t, *x.shape[2:])    # gộp thời gian vào batch = TimeDistributed
    x = self.cnn(x)                       # (N*T, 256, 3, 3)
    x = x.reshape(n, t, -1)               # (N, T, 2304)
    out, _ = self.lstm(x)                 # nn.LSTM(2304, 128, num_layers=2, batch_first=True)
    x = self.dropout(out[:, -1])          # return_sequences=False = bước cuối
    return self.fc(x).squeeze(1)          # Dense(1), hồi quy
```

Loss `nn.L1Loss()` (MAE), nhãn float. Tham số kỳ vọng **1.748.097** (Keras 1.747.073 + 2 lớp × 4 cổng × 128 bias). Lưu ý `num_layers=2` không có dropout giữa hai lớp, đúng như Keras.

**BatchNorm** (SegNet): `nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)`. Keras `momentum=0.99` nghĩa là giữ 99% giá trị cũ; PyTorch `momentum` là phần giá trị mới. Running mean/var là buffer: `count_params` chỉ đếm `requires_grad` → phải ra **29.442.325** (Keras "Trainable params"); in thêm tổng có buffer để đối chiếu 29.458.211.

**SegNet đầu ra theo pixel**: bỏ `Reshape + Permute + softmax`. Mô hình trả `(N, 7, 96, 96)`, nhãn `(N, 96, 96)` kiểu long (argmax của one-hot), `nn.CrossEntropyLoss()`. Pixel accuracy = `(pred.argmax(1) == y).float().mean()`, trung bình trên mọi pixel như Keras.

**Mạng toàn tích chập** (obj_detection): `forward` bắt đầu bằng `x = x / 127.5 - 1.0` (thay `Lambda`), ảnh `(N, 3, 64, 64)`. Đầu ra `tanh` `(N, 1, 1, 1)` → `flatten`. `nn.MSELoss()` với nhãn ±1. Accuracy tự định nghĩa: `(sign(out) == y)`. Khi vẽ heatmap: cùng mô hình, đưa cả ảnh `test.jpg` `(1, 3, H, W)`, không cần dựng lại mạng như Keras (`input_shape=(None, None, 3)`), vì conv PyTorch không cố định kích thước.

**Callback** (Training Utilities, trong `common.py`, `fit` gọi sau mỗi epoch với `logs = {"loss", "acc", "val_loss", "val_acc", "lr"}`):

- `ModelCheckpoint(path_pattern, monitor="val_loss")`: lưu `state_dict` khi `val_loss` giảm; tên file theo mẫu gốc `{epoch:06d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.pt` vào `results/<notebook>/checkpoints/`.
- `EarlyStopping(patience=5)`: dừng khi `val_loss` không giảm 5 epoch; **không** khôi phục trọng số (giống Keras); ô sau nạp checkpoint tốt nhất như gốc.
- `optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5, min_lr=1e-4)`, gọi `scheduler.step(val_loss)`.
- `SummaryWriter("runs/<notebook>")` ghi loss/acc/lr. Gốc hiển thị ảnh `tfboard.PNG`; bản PyTorch hiển thị hình `plot_history` và hướng dẫn `tensorboard --logdir runs`.

**ImageDataGenerator** (Emotion Training Utilities): ảnh float 0–255 một kênh, dùng transform trên tensor:

```python
aug = T.Compose([T.RandomAffine(degrees=45, translate=(0.2, 0.2)), T.RandomHorizontalFlip()])
```

Áp trong `Dataset.__getitem__` của tập train. Keras lấp vùng trống bằng `fill_mode='nearest'`, torchvision lấp 0: ghi chú khác biệt. `steps_per_epoch = len(x)/batch` = một lượt hết dữ liệu.

## Riêng từng notebook

| Notebook | Điểm cần làm đúng |
|---|---|
| `getting_started` | `torch.__version__`, `torch.cuda.is_available()`, tên GPU; thay `device_lib.list_local_devices()` |
| `Func_apxm_MLP` | Cùng bài toán BTL01 (đọc `BTL01 - Function Approximation/src/` để tham khảo, không sửa, không chép cấu hình). Dữ liệu `mgrid` 32j/10j, `multivariate_normal`; `gauss_mlp(B)` 2→B sigmoid→1 sigmoid; MSE; SGD 0.01; batch 128; 3 vòng quét; giữ `range(1, 1001)` như gốc và ghi chú; `evaluate` trả MSE trên toàn tập train và test; lưu `quet_A/B/C.csv` + hình |
| `MNIST_MLP` | 669.706; Flatten trong mô hình (input `(N, 28, 28)`); kiểm `softmax(predict).sum() == 1`; đặc trưng `fc2` 512 chiều |
| `1_MNIST_CNN` | 1.817.610; ba phần PCA (ảnh gốc 784 chiều, `fc2` 512, GAP sau `pool3` 256) vẽ scatter tô theo nhãn |
| `2_Human_Gender_CNN` | 4.063.570; `np.rot90(k=3, axes=(1, 2))`; không chia 255; Adam 1e-4, 30 epoch; nhắc mốc 0.7885 |
| `3_Emotion_Reg_CNN` | 19.028.287; padding same/valid xen kẽ; 218 filter; không chia 255; Adam 1e-4, 15 epoch; validation = PublicTest |
| `MNIST_CNN_LSTM` | sinh dữ liệu với `np.random.seed(2018)` giống hệt; 1.748.097; MAE; SGD 0.01; 30 epoch; dự đoán một mẫu vẽ 8 ảnh |
| `tu_2_Human_Gender_CNN` | như Gender nhưng Adam 1e-3, epoch lớn + 4 callback; nạp checkpoint tốt nhất rồi evaluate |
| `tu_3_Emotion_Reg_CNN` | như Emotion + augmentation + EarlyStopping(5), tối đa 100 epoch |
| `1_segnet` | đọc ảnh `skimage.io.imread`, `resize(order=0, preserve_range=True)` 96×96; nhãn từ mức xám; batch 8; Adam 1e-4; 15 epoch; vẽ ảnh phân đoạn màu `class_to_rgb` (sửa `np.float` → `float`) |
| `2_obj_detection` | 83.367; `train_test_split(test_size=0.3, random_state=0)`; MSE; Adam mặc định (1e-3); 5 epoch; heatmap ngưỡng 0.99 trên `test.jpg` |
