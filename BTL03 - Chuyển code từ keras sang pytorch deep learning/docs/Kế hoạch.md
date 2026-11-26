# BTL03 – Chuyển code từ Keras sang PyTorch – Kế hoạch

Cập nhật: 25/11/2026 (bản 2: dồn lịch vì 24/11 chưa làm; bản 1 ngày 23/11) · Người lập: Huỳnh Ngọc Trúc · Hạn nộp: **Thứ 7 28/11/2026** (trùng hạn BTL02)

> Trạng thái mới nhất và việc làm ngay: [Bàn giao.md](B%C3%A0n%20giao.md) (25/11, chuyển sang Claude Code).

## 1. Đề bài yêu cầu gì

| Hạng mục | Yêu cầu |
|---|---|
| Việc chính | Chuyển code Keras trong `Yêu cầu/examples/` sang PyTorch |
| Nộp | File báo cáo + code đã chuyển + file liên quan |
| Tài liệu đề | `Yêu cầu/01_Introduction-to-Python-TensorFlow-and-Keras.pdf` (cú pháp Keras cơ bản), `Yêu cầu/02_KerasExamples.pdf` (57 trang, đi qua đúng các notebook trong `examples/`: MLP → CNN MNIST/Gender/Emotion → kỹ thuật huấn luyện → CNN+LSTM → Segmentation), `Yêu cầu/README.md` (cài đặt Keras 2.0.9 / TF 1.1) |
| Code gốc | 13 notebook trong `Yêu cầu/examples/`. Bản trong máy **khác** bản GitHub `nhduong/intro_deep` (một số đã chạy lại bằng tf.keras 2.14 trên Colab T4). Luôn đối chiếu với bản trong máy |
| Tên file nộp | Chưa chốt (mục 11). Đề xuất `BTLON03_KerasPyTorch_HuynhNgocTruc_3123411312.zip` |

## 2. Chỗ dễ mất điểm

| Rủi ro | Cách xử lý |
|---|---|
| Kiến trúc chuyển sai mà không biết | So **số tham số** với `model.summary()` gốc (mục 3). Lệch thì phải giải thích được (LSTM, BatchNorm) |
| Để softmax ở lớp cuối rồi dùng `CrossEntropyLoss` | Lớp cuối trả logits; softmax chỉ dùng khi dự đoán/hiển thị. Ghi rõ trong báo cáo |
| Nhãn one-hot đưa thẳng vào `CrossEntropyLoss` | Đổi sang chỉ số lớp (`argmax`) hoặc giữ one-hot có chủ đích, nói rõ |
| Quên `model.train()` / `model.eval()` / `torch.no_grad()` | Gói trong `pytorch/common.py`, mọi notebook dùng chung |
| Thứ tự chiều NHWC (Keras) và NCHW (PyTorch) | Chuyển một lần ở bước tạo Dataset (`permute(0, 3, 1, 2)`), ghi chú tại chỗ |
| So sánh không công bằng | Giữ nguyên siêu tham số gốc (lr, batch, epoch, optimizer, dropout) và tiền xử lý gốc (kể cả chỗ gốc **không** chia 255). Thay đổi thêm thì làm ở ô riêng, ghi "thí nghiệm phụ" |
| Còn sót `import tensorflow/keras` trong code đã chuyển | Hook `btl03_guard.py` chặn trong `pytorch/`; Keras gốc chỉ được trích trong ô markdown để đặt cạnh nhau |
| Số trong báo cáo không khớp kết quả | Mọi số lấy từ `results/so_sanh.csv` và `results/<notebook>/metrics.json` |
| Chép lại lỗi/kỳ quặc của code gốc mà không nói | Ghi vào mục 12 và vào báo cáo (ví dụ `sel_indx` bỏ mẫu 0, "accuracy" của MSE trong obj_detection) |

## 3. Phạm vi và "đáp án" để đối chiếu

Mỗi notebook gốc có một notebook PyTorch cùng tên thêm hậu tố `_pytorch`, cùng thứ tự mục (1) Data → (2) Model → (3) Train → (4) Evaluate.

| # | Notebook gốc | Bài toán | Keras gốc | Tham số Keras | Kết quả gốc (log trong notebook) | Dữ liệu | Chạy ở |
|---|---|---|---|---|---|---|---|
| 0 | `0. Getting Started/getting_started` | kiểm phiên bản, GPU | tf 2.14 | – | Tesla T4 | – | CPU |
| 0 | `0. Getting Started/Iris_Bayesian_decision_making` | Naive Bayes | **không có Keras** | – | 0.9667 | sklearn | không chuyển, ghi chú |
| 1 | `1. MLP/Func_apxm_MLP` | xấp xỉ Gauss 2D, quét A/B/C | tf.keras | – (2→B→1) | chỉ có hình | tự sinh | CPU |
| 1 | `1. MLP/MNIST_MLP` | MNIST 10 lớp | tf.keras | **669.706** | test acc 0.9648 (30 epoch) | torchvision | CPU |
| 2 | `2. CNN + MLP/1_MNIST_CNN` | MNIST CNN + PCA đặc trưng | tf.keras | **1.817.610** | 0.9884 (30 epoch) | torchvision | GPU |
| 2 | `2. CNN + MLP/2_Human_Gender_CNN` | nam/nữ 64×64 | Keras 2.0.9 | **4.063.570** | 0.8264 (30 epoch) | `data/gender/` | GPU |
| 2 | `2. CNN + MLP/3_Emotion_Reg_CNN` | 7 cảm xúc 48×48 | Keras 2.0.9 | **19.028.287** | 0.5954 (15 epoch) | FER2013 (mục 8) | GPU |
| 3 | `3. CNN + LSTM/MNIST_CNN_LSTM` | tổng của ≤8 chữ số (hồi quy) | Keras 2.0.9 | **1.747.073** (PyTorch kỳ vọng 1.748.097, mục 4) | MAE 0.6502 (30 epoch) | torchvision | GPU |
| 4 | `4. Training Utilities/2_Human_Gender_CNN` | Gender + 4 callback | Keras 2.0.9 | 4.063.570 | 0.8773 (dừng epoch 32, tốt nhất epoch 27) | `data/gender/` | GPU |
| 4 | `4. Training Utilities/3_Emotion_Reg_CNN` | Emotion + augmentation + EarlyStopping | Keras 2.0.9 | 19.028.287 | 0.6163 (dừng epoch 57) | FER2013 | GPU |
| 5 | `5. Others/1_segnet` | phân đoạn bàn tay ảnh độ sâu 96×96, 7 lớp | Keras 2.0.9 | **29.458.211** (train được 29.442.325) | pixel acc 0.7732 (15 epoch) | có sẵn trong `Yêu cầu/…/depth_hand` | GPU |
| 5 | `5. Others/2_obj_detection` | xe / không xe 64×64×3, mạng toàn tích chập, heatmap | Keras 2.0.9 | **83.367** | 0.9523 (5 epoch) | có sẵn trong `Yêu cầu/…/car_data` | GPU/CPU |
| – | `colab_getting_started` | hướng dẫn Colab | không có Keras | – | – | – | không chuyển, ghi chú |

Notebook nào có phần phụ sau mục (4) thì chuyển luôn: `MNIST_MLP` (lấy đặc trưng `fc2`), `1_MNIST_CNN` (PCA ảnh gốc, PCA `fc2`, `GlobalAveragePooling2D` sau `pool3`), `2_obj_detection` (nạp trọng số vào mạng nhận ảnh kích thước bất kỳ để vẽ heatmap trên `test.jpg`).

## 4. Quy ước chuyển đổi (chi tiết ở skill `chuyen-doi-btl03`)

| Keras | PyTorch | Lưu ý |
|---|---|---|
| `Input(shape=(H, W, C))` | không có lớp riêng; tensor `(N, C, H, W)` | đổi chiều một lần trong Dataset |
| `Dense(units, activation)` | `nn.Linear(in, units)` + hàm kích hoạt | phải biết `in`; tính tay, ghi comment |
| `Conv2D(f, (3,3), padding='same'/'valid')` | `nn.Conv2d(c_in, f, 3, padding=1/0)` | `same` với kernel 3 = `padding=1` |
| `MaxPooling2D((2,2))` | `nn.MaxPool2d(2)` | cùng làm tròn xuống |
| `Dropout(rate)` | `nn.Dropout(p=rate)` | cùng nghĩa: tỷ lệ **bị tắt** |
| `Flatten()` | `nn.Flatten()` | thứ tự phần tử khác (NCHW), không ảnh hưởng số tham số |
| `softmax` + `categorical_crossentropy` | logits + `nn.CrossEntropyLoss` | nhãn là chỉ số lớp |
| `mean_squared_error` / `mean_absolute_error` | `nn.MSELoss` / `nn.L1Loss` | |
| `SGD(lr)`, `Adam(lr)` | `torch.optim.SGD/Adam` | Keras SGD không momentum; Adam eps: tf.keras 1e-7, Keras 2.0.9 1e-8, PyTorch 1e-8 |
| khởi tạo `glorot_uniform`, bias 0 | mặc định Kaiming uniform | gọi `init_keras_like(model)` trong `common.py` để so công bằng |
| `compile` + `fit` + `evaluate` + `predict` | `fit()`, `evaluate()`, `predict()` tự viết trong `common.py` | trả `history` dạng dict như Keras |
| `TimeDistributed(layer)` | gộp `(N·T, …)` → layer → tách `(N, T, …)` | |
| `LSTM(128, return_sequences=True)` → `LSTM(128)` | `nn.LSTM(…, num_layers=2, batch_first=True)`, lấy `out[:, -1]` | PyTorch có 2 bias mỗi cổng: +4·128 tham số mỗi lớp → 1.747.073 + 1.024 = **1.748.097** |
| `BatchNormalization()` | `nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)` | Keras momentum 0.99 = PyTorch 0.01 (định nghĩa ngược); running mean/var là buffer, không tính vào `parameters()` |
| `UpSampling2D()` | `nn.Upsample(scale_factor=2, mode='nearest')` | |
| `Reshape` + `Permute` + `softmax` theo pixel | `CrossEntropyLoss` trên `(N, 7, H, W)` với nhãn `(N, H, W)` | |
| `Lambda(lambda x: x/127.5 - 1)` | một dòng trong `forward` | |
| `Model(inputs, get_layer('fc2').output)` | `register_forward_hook` hoặc `forward(return_features=True)` | |
| `ModelCheckpoint(save_best_only)` | lưu `state_dict` khi `val_loss` giảm | `.pt` không đưa lên git |
| `EarlyStopping(monitor='val_loss', patience)` | lớp `EarlyStopping` trong `common.py` | Keras không tự khôi phục trọng số tốt nhất; giữ hành vi đó, nạp checkpoint ở ô sau như gốc |
| `ReduceLROnPlateau(factor, patience, min_lr)` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | |
| `TensorBoard(log_dir)` | `torch.utils.tensorboard.SummaryWriter` | log vào `runs/` (git bỏ qua) |
| `ImageDataGenerator(rotation 45, shift 0.2, flip)` + `fit_generator` | `transforms.RandomAffine(45, translate=(0.2, 0.2))` + `RandomHorizontalFlip` trong Dataset | `steps_per_epoch = len/batch` (225 bước) = 1 lượt hết dữ liệu |
| `model.summary()`, `plot_model` | `torchinfo.summary` | |

## 5. Kiểm chứng mỗi notebook

1. **Số tham số** khớp cột "Tham số Keras" (mục 3), hoặc lệch đúng như mục 4 dự đoán.
2. **Shape** đầu ra từng khối khớp bảng summary gốc (in bằng `torchinfo`).
3. **Kết quả** cùng epoch/batch/lr với gốc, ghi vào `results/so_sanh.csv`. Sai khác do khởi tạo ngẫu nhiên là bình thường; chênh nhiều (ví dụ > 3 điểm accuracy) thì tìm nguyên nhân trước khi viết báo cáo.
4. Đồ thị loss/accuracy theo epoch lưu ở `results/<notebook>/`.
5. Seed cố định (`set_seed(0)`), ghi thiết bị và thời gian chạy.

## 6. Cấu trúc thư mục

```
BTL03 - Chuyển code từ keras sang pytorch deep learning/
├── Yêu cầu/                      đề gốc, KHÔNG sửa (hook chặn)
├── docs/Kế hoạch.md              file này
├── data/                         git bỏ qua, chỉ giữ .gitkeep
│   ├── gender/                   4 file .npy (đã có, 23/11)
│   └── emotion/                  fer2013 (mục 8)
├── pytorch/
│   ├── common.py                 set_seed, fit/evaluate/predict, callback, init_keras_like, đếm tham số, đường dẫn
│   ├── 0. Getting Started/       getting_started_pytorch.ipynb
│   ├── 1. MLP/                   Func_apxm_MLP_pytorch.ipynb, MNIST_MLP_pytorch.ipynb
│   ├── 2. CNN + MLP/             1_MNIST_CNN_pytorch, 2_Human_Gender_CNN_pytorch, 3_Emotion_Reg_CNN_pytorch
│   ├── 3. CNN + LSTM/            MNIST_CNN_LSTM_pytorch.ipynb
│   ├── 4. Training Utilities/    2_Human_Gender_CNN_pytorch, 3_Emotion_Reg_CNN_pytorch
│   └── 5. Others/                1_segnet_pytorch, 2_obj_detection_pytorch
├── results/
│   ├── so_sanh.csv               bảng Keras ↔ PyTorch (rule btl03-ket-qua)
│   └── <tên notebook>/           history.csv, metrics.json, hình
├── report/                       báo cáo .md từng mục → .docx + .pdf
├── CodetoTruc.md                 nhật ký ngày (không nộp)
├── README.md
└── requirements.txt
```

Dữ liệu SegNet và xe đọc thẳng từ `Yêu cầu/examples/5. Others/` (không chép). Đường dẫn lấy từ `common.py`, không ghi tuyệt đối.

## 7. Lịch làm việc (23/11 – 28/11, xen với BTL02)

Nguyên tắc: lúc BTL02 đang train trên Kaggle/Colab thì viết code BTL03; notebook nặng của BTL03 gom vào phiên GPU riêng.

| Ngày | Việc | Kết quả để kiểm tra | Trạng thái |
|---|---|---|---|
| T2 23/11 tối | Đọc hết `Yêu cầu/`; dựng thư mục; tải và kiểm dữ liệu Gender; bộ công cụ Claude Code | 4 file `.npy` đúng shape; `docs/Kế hoạch.md` | - [x] |
| T3 24/11 | (không làm BTL03, dồn sang T4) | – | – |
| T4 25/11 chiều–tối | `common.py`; `getting_started`, `Func_apxm_MLP`, `MNIST_MLP` (CPU); viết sẵn 3 notebook thư mục 2 (chạy thử CPU, `CHAY_THU = True`) | MNIST_MLP 669.706 tham số, acc ≈ 0.96; 3 mô hình thư mục 2 đúng số tham số | - [x] (làm 26/11: MNIST_MLP 0.9658; CNN_LSTM và 2 notebook TU cũng đã viết, chờ GPU) |
| T5 26/11 | Chạy GPU thư mục 2 (MNIST_CNN, Gender, Emotion); viết `MNIST_CNN_LSTM` + 2 notebook Training Utilities rồi chạy GPU | 6 dòng `so_sanh.csv`; LSTM 1.748.097 tham số; callback chạy đúng | - [ ] |
| T6 27/11 sáng | `1_segnet`, `2_obj_detection`; chạy GPU; đủ `so_sanh.csv` + hình | `so_sanh.csv` đủ 12 dòng (10 notebook, Func_apxm 3 dòng A/B/C) | - [ ] |
| T6 27/11 chiều–tối | Viết báo cáo (Word + PDF) | Bản đầy đủ | - [ ] |
| T7 28/11 sáng | Rà số liệu (`/btl03-cham-thu`); README; đóng gói (`dong-goi-btl03`); nộp | Đã nộp zip | - [ ] |

Dự phòng khi trễ (theo thứ tự cắt): chạy bản Training Utilities của Emotion với tối đa 30 epoch thay vì 100 (ghi rõ) → bỏ phần heatmap obj_detection → `Func_apxm_MLP` chỉ chạy vòng quét B → gom MNIST_CNN và CNN_LSTM vào cùng phiên GPU với BTL02.

## 8. Dữ liệu

| Bộ | Vị trí | Ghi chú |
|---|---|---|
| MNIST | `torchvision.datasets.MNIST(root=data/mnist)` | tự tải; gốc chia 255 |
| Gender (WIKI) | `data/gender/64_64_*_onehot.npy` | Đã kiểm 23/11: train (11938, 64, 64, 1), val (5968, 64, 64, 1), float64, **giá trị 0–255, gốc không chia 255**. Nhãn one-hot: cột 0 = female, cột 1 = male. **Lệch lớp**: 78,85% male ở cả train và val → đoán toàn "male" đã được 0.7885 (log gốc epoch 1 đúng bằng số này). **662/5968 ảnh val đen hoàn toàn** (612 male, 50 female). Phải `np.rot90(k=3, axes=(1, 2))` như gốc. Không xóa ảnh đen để giữ so sánh công bằng; nêu trong báo cáo |
| Emotion (FER2013) | `data/emotion/` | Gốc đọc `fer2013_training_onehot.csv` và `fer2013_publictest_onehot.csv` (2304 pixel + 7 one-hot, **không chia 255**). Sinh lại từ `fer2013.csv` (cột Usage: Training → train, PublicTest → "test" như gốc). Nguồn: chép hoặc đọc `../BTL02 Emotion Recognition/data/fer2013.csv`. Nhãn 0–6 theo thứ tự gốc FER2013, khớp `get_emotion` của notebook |
| SegNet | `Yêu cầu/examples/5. Others/depth_hand/data/segnet/{train,test}/{depth,labels}` | gốc: 2450 train, 1050 test, resize 96×96 `order=0`, 7 lớp theo mức xám (255, 43, 85, 128, 170, 212, 0) |
| Xe | `Yêu cầu/examples/5. Others/car_data/data/obj_detection/{vehicles,non-vehicles}/*/*.png` | gốc: nhãn +1/−1, `train_test_split(test_size=0.3)` (không seed; ta đặt `random_state=0`, ghi rõ) |

## 9. Bố cục báo cáo

| Mục | Nội dung |
|---|---|
| Bìa + mục lục | Môn Học sâu, BTL03, Huỳnh Ngọc Trúc, 3123411312 |
| 1 | Giới thiệu: Keras và PyTorch khác nhau thế nào (khai báo sẵn và `fit` so với `nn.Module` và vòng lặp tự viết) |
| 2 | Bảng tương ứng API Keras ↔ PyTorch (mục 4) và `common.py` |
| 3 | Từng notebook (3.1 … 3.11): mục đích, đoạn Keras gốc ↔ đoạn PyTorch, điểm khác cần chú ý, số tham số, kết quả |
| 4 | Bảng tổng hợp Keras ↔ PyTorch (`so_sanh.csv`) + đồ thị; phân tích chênh lệch |
| 5 | Khó khăn và cách xử lý; nhận xét về dữ liệu (Gender lệch lớp, ảnh đen; dữ liệu không chuẩn hóa) |
| 6 | Kết luận |
| Phụ lục | Cách chạy lại; danh sách file nộp |

## 10. Checklist nộp bài

- [ ] 11 notebook `_pytorch` chạy hết từ đầu, đã lưu output
- [ ] `pytorch/common.py` có docstring tiếng Việt
- [ ] `results/so_sanh.csv` đủ dòng, số khớp output notebook
- [ ] Không còn `tensorflow`/`keras` trong ô code của `pytorch/`, không có đường dẫn tuyệt đối
- [ ] Báo cáo `.docx` + `.pdf` trong `report/`
- [ ] README có lệnh cài và chạy
- [ ] `requirements.txt`
- [ ] Zip không kèm `data/`, `Yêu cầu/`, `CodetoTruc.md`, `docs/`, `*.pt`, `runs/`

## 11. Còn mở (hỏi thầy hoặc tự chốt)

- [ ] Làm cá nhân hay nhóm 5? (đang giả định: cá nhân, Trúc)
- [ ] Tên file zip nộp?
- [ ] Có cần slide không?
- [ ] Iris và colab_getting_started không có Keras: chỉ ghi chú là đủ?

## 12. Ghi chú từ code gốc (đọc hết `Yêu cầu/` ngày 23/11)

- `Func_apxm_MLP`: **cùng bài toán với BTL01** (xấp xỉ Gauss 2D bằng MLP PyTorch, đã nộp 8/11). Được đọc `BTL01 - Function Approximation/src/model.py`, `train.py` để tham khảo, không sửa BTL01. Bản BTL03 vẫn phải bám đúng notebook gốc (sigmoid, SGD 0.01, batch 128, vòng quét A/B/C), không chép cấu hình BTL01. Lưới huấn luyện 32×32 = 1024 điểm, test 10×10. `np.random.choice(range(1, 1001))` không bao giờ chọn điểm 0 và điểm 1001–1023. Đầu ra `sigmoid`, loss MSE, SGD lr 0.01, batch 128. Ba vòng quét: A (100…1000 mẫu, B=16, C=100), B (8…512 nút, A=800, C=100), C (10…190 epoch, A=800, B=128). Ô (4.2) và (4.3) trong bản trong máy còn output kiểu Keras cũ.
- `MNIST_MLP`: Flatten → 512 relu → Dropout 0.2 → 512 relu → Dropout 0.2 → 10 softmax. SGD 0.01, batch 128, 30 epoch. Có thêm phần kiểm `predict` cộng xác suất = 1 và lấy đặc trưng `fc2` (512 chiều).
- `1_MNIST_CNN`: 3 khối Conv(64/128/256, same) + MaxPool + Dropout 0.2 → FC 512, 512 → 10. 28→14→7→3. Phần phụ: PCA ảnh gốc, PCA đặc trưng `fc2`, PCA sau `GlobalAveragePooling2D(pool3)` (256 chiều). Ô `model.get_layer()` gọi thiếu tham số (lỗi trong gốc).
- `2_Human_Gender_CNN`: 5 khối (16×2, 32×3, 64×3, 128×3, 256×3), mỗi khối MaxPool + Dropout 0.2 → FC 1024, 1024 → 2. 64→2, Flatten 1024. Adam **1e-4**, 30 epoch. Bản Training Utilities: Adam **1e-3**, epoch 100000, ModelCheckpoint + EarlyStopping(5) + ReduceLROnPlateau(0.1, 5, min_lr 1e-4) + TensorBoard.
- `3_Emotion_Reg_CNN`: Conv64 same, Conv64 valid, Pool, Drop 0.25; Conv128 same, Conv128 valid, Conv**218** same, Conv**218** valid, Pool, Drop 0.25; FC 1024, Drop 0.5; FC 7. 48→46→23→21→19→9, Flatten 9·9·218 = 17.658. Adam 1e-4, batch 128, 15 epoch. Bản Training Utilities: augmentation + EarlyStopping(5), 100 epoch, dừng ở 57.
- `MNIST_CNN_LSTM`: `np.random.seed(2018)`, mỗi mẫu 1–8 ảnh MNIST (đệm ảnh 0), nhãn = tổng (0–72, không chuẩn hóa). Conv 64/128/256 same + Pool + Drop 0.2 bọc `TimeDistributed`, Flatten 2304 → LSTM 128 (seq) → LSTM 128 → Dropout 0.22 → Dense 1. MAE, SGD 0.01, batch 128, 30 epoch (~27 phút GTX 1080). Ghi chú markdown gốc nói "48×48" nhưng thực tế 28×28.
- `1_segnet`: encoder VGG-16 kiểu (64,64 / 128,128 / 256×3 / 512×3 / 512×3, mỗi conv có BN + ReLU, 5 lần pool) và decoder đối xứng với UpSampling; cuối Conv 1×1 → 7 + BN → Reshape (7, 9216) → Permute → softmax. Adam 1e-4, batch 8, 15 epoch. Ảnh depth không chuẩn hóa.
- `2_obj_detection`: Lambda chuẩn hóa → Conv10 ×2 same → MaxPool 8×8 → Drop 0.25 → Conv128 8×8 ("dense1") → Drop 0.5 → Conv1 1×1 tanh ("dense2") → Flatten. MSE, Adam mặc định, batch 128, 5 epoch. "accuracy" Keras báo khi dùng MSE với nhãn ±1 không phải accuracy phân loại thật; bản PyTorch tính accuracy theo dấu và ghi rõ. Heatmap: ngưỡng 0.99, ô 8 px, khung 64 px trên `test.jpg`.
- Code gốc dùng `lr=` (đã bỏ trong tf.keras mới, log có cảnh báo). Không ảnh hưởng khi chuyển.
