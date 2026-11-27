# Nhật ký BTL03 – Chuyển code từ Keras sang PyTorch

Nhật ký cá nhân của Trúc (không nộp). Quy tắc ở `CLAUDE.md` mục 1, đối chiếu lịch ở `docs/Kế hoạch.md` mục 7.

---

## T2 23/11/2026 – Đọc đề, dựng thư mục, kiểm dữ liệu Gender, bộ công cụ Claude Code

### 1. Việc theo kế hoạch

Đúng dòng T2 23/11 trong `docs/Kế hoạch.md` mục 7: đọc hết `Yêu cầu/`, dựng thư mục, tải và kiểm dữ liệu Gender, lập kế hoạch. Đã tick.

### 2. Commit trong ngày

- `b885df2f` Up bài tập BTL03 (giải nén car_data, depth_hand): đưa đề gốc `Yêu cầu/` lên git, kèm dữ liệu xe và bàn tay của notebook thư mục 5 (Colab clone về là có).
- `1c08b472` BTL03: đọc đề, dựng thư mục, kế hoạch, bộ công cụ Claude Code: thư mục `pytorch/`, `data/`, `results/`, `report/`, `docs/Kế hoạch.md`, README, requirements, nhật ký; bộ hook/command/rule/skill/agent BTL03 trong `.claude/`; cập nhật `CLAUDE.md`, README gốc, `.gitignore`.

### 3. Kết quả

- Thư mục: `data/gender`, `data/emotion`, `pytorch/` (6 thư mục con trùng tên `Yêu cầu/examples/`), `results/`, `report/`, `docs/`.
- File: `.gitignore`, `README.md`, `requirements.txt`, `docs/Kế hoạch.md`, `CodetoTruc.md`.
- Dữ liệu: 4 file `.npy` Gender chép vào `data/gender/` (git bỏ qua).
- Bộ công cụ Claude Code cho BTL03 trong `.claude/` (hook, command, rule, skill, agent), cập nhật `CLAUDE.md` và `README.md` gốc.

### 5. Đã kiểm tra

- Đề gồm 2 slide (01 cú pháp Keras, 02 đi qua từng notebook), `README.md` (Keras 2.0.9, TF 1.1) và 13 notebook. Notebook trong máy **khác** bản GitHub `nhduong/intro_deep`: một số đã chạy lại bằng tf.keras 2.14, nên lấy số đối chiếu từ bản trong máy.
- Gender: train (11938, 64, 64, 1), val (5968, 64, 64, 1), float64, giá trị 0–255. Nhãn: cột 0 female, cột 1 male.
- Gender lệch lớp: 78,85% male ở cả hai tập. Đoán toàn "male" đã được 0.7885; log gốc epoch 1 có val_acc đúng bằng 0.7885.
- 662/5968 ảnh val đen hoàn toàn (612 male, 50 female).

### 7. Số liệu cho báo cáo

| Notebook | Tham số Keras | Kết quả gốc |
|---|---|---|
| MNIST_MLP | 669.706 | 0.9648 |
| 1_MNIST_CNN | 1.817.610 | 0.9884 |
| 2_Human_Gender_CNN | 4.063.570 | 0.8264 (có callback: 0.8773) |
| 3_Emotion_Reg_CNN | 19.028.287 | 0.5954 (có augmentation: 0.6163) |
| MNIST_CNN_LSTM | 1.747.073 | MAE 0.6502 |
| 1_segnet | 29.458.211 | 0.7732 |
| 2_obj_detection | 83.367 | 0.9523 |

### 8. Câu hỏi vấn đáp có thể gặp

- **Gender đạt 82–88% có tốt không?** Chỉ hơn mức đoán toàn "male" (78,85%) khoảng 4–9 điểm, vì dữ liệu lệch lớp. Nên xem thêm recall lớp female.
- **Vì sao không chia 255 cho ảnh Gender?** Để giữ đúng tiền xử lý của bản Keras, so sánh mới công bằng. Chia 255 là thí nghiệm phụ.
- **Vì sao số tham số LSTM bản PyTorch nhiều hơn 1.024?** PyTorch có hai vector bias (`b_ih`, `b_hh`) cho mỗi cổng, Keras có một. Hai lớp × 4 cổng × 128 = 1.024.

### 9. Còn treo

- [x] Hỏi thầy: cá nhân hay nhóm, tên file zip, có cần slide không (chốt 27/11: nhóm 5; tên theo mẫu BTL02; không cần slide)
- [x] Sinh dữ liệu Emotion từ `fer2013.csv` (26/11: đọc thẳng bằng `load_fer2013`, không cần sinh CSV)
- [x] Viết `pytorch/common.py` (T3 24/11) (xong 26/11)

---

## T5 26/11/2026 – Viết common.py, 9 notebook PyTorch, chạy xong 3 notebook CPU

### 1. Việc theo kế hoạch

Kế hoạch mục 7 (bản 2) dồn việc như sau: T4 25/11 viết `common.py`, 3 notebook CPU và viết sẵn thư mục 2; T5 26/11 chạy GPU thư mục 2, viết CNN_LSTM và 2 notebook Training Utilities. Ngày 25/11 chưa làm BTL03 nên hôm nay gộp cả hai ngày:

1. Viết `pytorch/common.py` (việc T4). Xong, đã chạy thử.
2. `getting_started`, `Func_apxm_MLP`, `MNIST_MLP`: chạy đủ trên CPU, có output, đã ghi `so_sanh.csv` (việc T4). Xong.
3. Viết 3 notebook thư mục 2, chạy thử `CHAY_THU = True` để kiểm số tham số (việc T4). Xong.
4. Viết `MNIST_CNN_LSTM` và 2 notebook Training Utilities, chạy thử (việc T5). Xong.
5. Chạy GPU 6 notebook (thư mục 2, 3, 4) (việc T5). **Chưa**: cần Trúc commit + push để Colab/Kaggle clone về. Xem Còn treo.

Dòng T4 25/11 trong Kế hoạch mục 7 đã tick. Dòng T5 26/11 chưa tick vì chưa chạy GPU.

### 2. Commit trong ngày

- `9b8e06ec` Commit xong bài 2: commit chung với BTL02; phần BTL03 thêm `docs/Bàn giao.md` (bàn giao từ Claude app sang Claude Code), cập nhật `docs/Kế hoạch.md` (lịch bản 2) và `CodetoTruc.md` (điền hash ngày 23/11).
- `87ae2f5c` BTL03: common.py, 9 notebook PyTorch (3 đã chạy CPU), so_sanh.csv, nhật ký 26/11: toàn bộ code và kết quả CPU của ngày.
- `898a2192` BTL03: notebook chạy Kaggle, khóa file khi ghi so_sanh.csv: `chay_kaggle.ipynb`, `_khoa_file` trong `common.py`, `.gitignore`.
- `c1238fe8` BTL03: 1_segnet, 2_obj_detection (chạy CPU), thêm SegNet vào chay_kaggle: 2 notebook thư mục 5, dòng obj_detection trong `so_sanh.csv`, SegNet vào luồng 1 của `chay_kaggle.ipynb`.

### 3. Kết quả

File mới:

- `pytorch/common.py`: phần dùng chung (đường dẫn, seed, `fit`/`evaluate`/`predict`, callback, lưu kết quả, đọc FER2013).
- `pytorch/0. Getting Started/getting_started_pytorch.ipynb`: có output (CPU).
- `pytorch/1. MLP/Func_apxm_MLP_pytorch.ipynb`, `MNIST_MLP_pytorch.ipynb`: có output (CPU).
- `pytorch/2. CNN + MLP/1_MNIST_CNN_pytorch.ipynb`, `2_Human_Gender_CNN_pytorch.ipynb`, `3_Emotion_Reg_CNN_pytorch.ipynb`: đã viết, chạy thử qua, **chưa có output** (chờ GPU).
- `pytorch/3. CNN + LSTM/MNIST_CNN_LSTM_pytorch.ipynb`: như trên.
- `pytorch/4. Training Utilities/2_Human_Gender_CNN_pytorch.ipynb`, `3_Emotion_Reg_CNN_pytorch.ipynb`: như trên.
- `results/so_sanh.csv` (4 dòng), `results/MNIST_MLP/` (history.csv, metrics.json, curves.png), `results/Func_apxm_MLP/` (quet_A/B/C.csv + .png, metrics.json).

Sau commit `87ae2f5c`:

- `chay_kaggle.ipynb` (thư mục gốc BTL03): notebook điều khiển chạy trên Kaggle. Notebook này clone repo, liên kết dữ liệu Gender và FER2013 từ `/kaggle/input`, tải MNIST một lần, rồi chạy 6 notebook GPU bằng `nbconvert` theo 2 luồng song song (mỗi luồng một GPU của T4 x2). Cuối cùng nén notebook có output + `results/` thành `btl03_gpu_*.zip`.
- `common.py`: `ghi_so_sanh` có thêm khóa file `_khoa_file` để hai notebook chạy song song không ghi đè `so_sanh.csv` của nhau.
- `.gitignore`: bỏ qua `_log_kaggle/` (log trong zip Kaggle) và `*.lock`.

Sau commit `898a2192` (làm sớm việc của T6 27/11 sáng):

- `pytorch/5. Others/1_segnet_pytorch.ipynb`: đã viết, chạy thử qua, chờ GPU (đã thêm vào `chay_kaggle.ipynb`, luồng 1).
- `pytorch/5. Others/2_obj_detection_pytorch.ipynb`: chạy đủ 5 epoch trên CPU (mô hình 83.367 tham số), có heatmap trên `test.jpg`.

Môi trường: env `btl02` cài thêm `torchinfo`, `tensorboard`, `scikit-image`, `nbconvert`, `ipykernel`, `notebook`, `seaborn`.

Notebook trong ngày:

| Notebook | Lớp/khối chính | Tham số Keras | Tham số PyTorch | Kết quả Keras | Kết quả PyTorch | Thiết bị |
|---|---|---|---|---|---|---|
| MNIST_MLP | FC 512 → 512 → 10 | 669.706 | 669.706 | test acc 0.9648 | 0.9658 (+0.0010) | CPU, 84 s |
| Func_apxm_MLP quét A | 2 → 16 sigmoid → 1 | 65 | 65 | chỉ có hình, ≈ 0.0067 tại A=1000 | test MSE 0.0086 | CPU |
| Func_apxm_MLP quét B | 2 → B → 1 | 2.049 (B=512) | 2.049 | ≈ 0.0014 tại B=512 | 0.0013 | CPU |
| Func_apxm_MLP quét C | 2 → 128 → 1 | 513 | 513 | ≈ 0.0016 tại C=190 | 0.0015 | CPU |
| 1_MNIST_CNN | 3 khối Conv 64/128/256 + FC 512, 512 | 1.817.610 | 1.817.610 | 0.9884 | chờ GPU | |
| 2_Human_Gender_CNN | 5 khối conv (13 conv) + FC 1024, 1024 | 4.063.570 | 4.063.570 | 0.8264 | chờ GPU | |
| 3_Emotion_Reg_CNN | 6 conv (same/valid xen kẽ, 218 filter) + FC 1024 | 19.028.287 | 19.028.287 | 0.5954 | chờ GPU | |
| MNIST_CNN_LSTM | CNN bọc TimeDistributed + 2 LSTM 128 | 1.747.073 | 1.748.097 (+1.024) | MAE 0.6502 | chờ GPU | |
| tu_2_Human_Gender_CNN | như Gender + 4 callback | 4.063.570 | 4.063.570 | 0.8773 | chờ GPU | |
| tu_3_Emotion_Reg_CNN | như Emotion + augmentation + EarlyStopping | 19.028.287 | 19.028.287 | 0.6163 | chờ GPU | |
| 1_segnet | encoder VGG 13 conv + BN, decoder UpSampling | 29.442.325 train được (tổng 29.458.211) | 29.442.325 (tổng 29.458.211) | pixel acc 0.7732 | chờ GPU | |
| 2_obj_detection | mạng toàn tích chập (Conv 10, 10, 128 8×8, 1 1×1 tanh) | 83.367 | 83.367 | 0.9523 (cách Keras) | 0.9375 (−0.0148); theo dấu 0.9636 | CPU, 106 s |

Số Func_apxm gốc là số đọc từ hình (notebook gốc không in số), nên trong `so_sanh.csv` cột `keras` để trống và số đọc từ hình ghi ở `ghi_chu`.

### 4. Giải thích từng hàm

`pytorch/common.py`:

- `ROOT`, `DATA_DIR`, `RESULTS_DIR`, `RUNS_DIR`, `YEU_CAU_DIR`, `EXAMPLES_DIR`: đường dẫn tính từ `Path(__file__)`, không ghi tuyệt đối, nên clone lên Colab chạy được ngay.
- `set_seed(seed)`: cố định seed của `random`, numpy, torch (CPU và CUDA).
- `get_device()`, `device_name()`: chọn `cuda` nếu có, ngược lại `cpu`; tên thiết bị (ví dụ `cuda (Tesla T4)`) để ghi vào kết quả. Thay `device_lib.list_local_devices()`.
- `to_nchw(x)`: ảnh Keras `(N, H, W, C)` → PyTorch `(N, C, H, W)`, float32, **không chia 255**. Có `np.ascontiguousarray` vì sau `np.rot90` (Gender) mảng có stride âm, torch không nhận.
- `onehot_to_index(y)`: one-hot → chỉ số lớp, vì `nn.CrossEntropyLoss` nhận chỉ số lớp.
- `make_loader(x, y, batch_size, shuffle)`: `TensorDataset` + `DataLoader`; thay tham số `batch_size`, `shuffle` của `model.fit`. Train `shuffle=True` như Keras.
- `fer2013_path()`, `load_fer2013(usage)`: đọc `fer2013.csv` (lấy ở `data/emotion/` hoặc bản của BTL02) ra đúng mảng mà gốc đọc từ `fer2013_training_onehot.csv` / `fer2013_publictest_onehot.csv`.
- `init_keras_like(model)`: khởi tạo như Keras. Dense/Conv dùng glorot uniform và bias 0. LSTM: kernel glorot, kernel hồi quy orthogonal, bias 0, riêng bias cổng quên bằng 1. PyTorch mặc định Kaiming uniform nên phải gán lại để so sánh công bằng.
- `count_params(model)`: số tham số train được, để `assert` với `Total params` của Keras. `count_all` cộng thêm running mean/var của BatchNorm (cho SegNet).
- `summary(model, input_size)`: bảng giống `model.summary()` (dùng torchinfo).
- `accuracy`, `sign_accuracy`, `mae`: thay `metrics=[...]` của `compile`. `accuracy` dùng được cho cả pixel accuracy của SegNet; `sign_accuracy` dùng cho nhãn ±1 của obj_detection.
- `fit(...)`: vòng train thay `model.fit`. Mỗi batch chạy `zero_grad` → forward → loss → `backward` → `step`; cuối epoch tính trên validation với `model.eval()` và `no_grad`; gọi callback và scheduler. Trả `history` dạng dict giống `history.history` của Keras.
- `evaluate(...)`: thay `model.evaluate`, trả `(loss, chỉ số)` lấy trung bình theo số mẫu.
- `predict(...)`: thay `model.predict`, trả logits (cần xác suất thì áp `softmax`).
- `ModelCheckpoint`: lưu `state_dict` khi `val_loss` giảm, tên file theo mẫu gốc `{epoch:06d}-{loss:.4f}-...`.
- `EarlyStopping`: dừng sau `patience` epoch liền không giảm `val_loss`, không tự khôi phục trọng số (giống Keras).
- `TensorBoard`: ghi loss/acc/lr vào `runs/<tên>`.
- `reduce_lr_on_plateau(...)`: trả `ReduceLROnPlateau` của PyTorch, đặt ngưỡng tuyệt đối 1e-4 như Keras.
- `results_dir`, `save_history`, `save_metrics`, `plot_history`: lưu `history.csv`, `metrics.json`, `curves.png` vào `results/<notebook>/`.
- `ghi_so_sanh(...)`: thêm hoặc thay **một dòng** trong `results/so_sanh.csv`, khóa là (notebook, thư mục, chỉ số). Chạy lại notebook thì dòng cũ được thay, không sinh dòng trùng.

Các lớp `nn.Module`: tên lớp con trùng `name=` của Keras (`conv1_1`, `fc2`, `fc3_10ways_softmax`...) để so với summary gốc; lớp cuối trả logits; `forward` có comment shape sau mỗi khối.

- `MnistMLP`: `forward(x, return_features=True)` trả thêm đầu ra `fc2` **sau relu**, thay cho `Model(inputs, get_layer('fc2').output)`.
- `MnistCNN`: trả thêm `pool3` (để tính GlobalAveragePooling bằng `mean(dim=(2, 3))`) và `fc2`, dùng cho 3 hình PCA.
- `GenderCNN`: hàm `khoi_conv` dựng một khối [n × (Conv + ReLU) → MaxPool → Dropout] bằng `nn.Sequential(OrderedDict(...))` để giữ tên lớp như Keras.
- `EmoCNN`: padding `same` = 1, `valid` = 0, xen kẽ như gốc; Flatten 218 × 9 × 9 = 17.658.
- `CnnLstm`: TimeDistributed bằng cách gộp `(N, 8, ...)` thành `(N·8, ...)` cho qua CNN rồi tách lại `(N, 8, 2304)`; hai `nn.LSTM` riêng (`lstm1` trả cả chuỗi, `lstm2` lấy bước cuối `out[:, -1]`).
- `SegNet` (1_segnet): hàm `cbr` tạo bộ [Conv same + BatchNorm + ReLU]; encoder 13 conv + 5 MaxPool, decoder đối xứng có 5 `nn.Upsample(scale_factor=2)`, cuối Conv 1×1 ra 7 kênh + BatchNorm. Trả logits `(N, 7, 96, 96)`, thay `Reshape + Permute + softmax`. `loss_keras` và `pixel_accuracy` xử lý pixel nhãn lạ đúng như Keras (xem "Khác Keras ở đâu").
- `CarConv` (2_obj_detection): mạng toàn tích chập; `forward(x, flatten=True)` khi train (ra `(N, 1)`), `flatten=False` khi vẽ heatmap trên ảnh lớn (ra `(1, 1, 83, 153)` với `test.jpg` 720×1280). Một mô hình dùng cho cả hai, không cần dựng `heatmodel` riêng rồi nạp file như Keras. `keras_accuracy` tính "accuracy" đúng cách Keras (`round(dự đoán) == nhãn`).
- `keras_random_transform` (tu_3_Emotion): viết lại phép biến đổi của `ImageDataGenerator` Keras 2.0.9 bằng `scipy.ndimage.affine_transform`: xoay ±45°, dịch 20%, `order=0`, `mode='nearest'`, lật ngang 50%. `AugmentedFER` là Dataset áp phép biến đổi này mỗi lần lấy ảnh.

### 5. Khác Keras ở đâu

- **Thứ tự chiều**: Keras NHWC, PyTorch NCHW; đổi một lần lúc tạo tensor.
- **Softmax**: Keras để softmax ở lớp cuối + `categorical_crossentropy`; PyTorch trả logits, `nn.CrossEntropyLoss` đã gồm `log_softmax`. Để softmax hai lần thì mô hình học rất chậm.
- **Nhãn**: one-hot (Keras) → chỉ số lớp (PyTorch). Giá trị loss như nhau.
- **Khởi tạo**: Keras glorot uniform + bias 0; PyTorch mặc định Kaiming uniform. Đã gán lại bằng `init_keras_like`.
- **Flatten**: Keras duỗi theo (H, W, C), PyTorch theo (C, H, W). Thứ tự khác nhưng số tham số như nhau.
- **LSTM**: PyTorch có 2 vector bias mỗi cổng → +1.024 tham số (2 lớp × 4 cổng × 128). Keras 2.0.9 dùng `hard_sigmoid` cho cổng, PyTorch dùng sigmoid và không đổi được.
- **EarlyStopping**: đã đối chiếu log gốc. Gender tốt nhất epoch 27 thì dừng 32; Emotion tốt nhất 52 thì dừng 57. Cả hai dừng sau đúng 5 epoch xấu, khớp cách đếm trong `common.py`.
- **ReduceLROnPlateau**: PyTorch giảm lr ở epoch xấu thứ 6 (patience + 1). Log gốc Gender chưa lần nào giảm lr (EarlyStopping dừng trước), nên không ảnh hưởng.
- **Augmentation**: `torchvision.RandomAffine` lấp vùng trống bằng 0, Keras lấp `nearest`; vì vậy viết lại bằng scipy cho giống.
- **Dữ liệu CNN_LSTM**: giữ nguyên vòng lặp numpy và `np.random.seed(2018)` nên dữ liệu sinh ra trùng gốc; chỉ đổi float64 → float32 (bản gốc tốn 3 GB RAM).
- **Ô lỗi trong gốc**: `model.get_layer()` (1_MNIST_CNN) gọi thiếu tham số, đã bỏ và ghi chú.
- **BatchNorm (SegNet)**: Keras `momentum=0.99`, `epsilon=1e-3` = PyTorch `momentum=0.01`, `eps=1e-3` (PyTorch định nghĩa momentum ngược). Running mean/var là buffer nên `count_params` so với "Trainable params" 29.442.325, `count_all` so với tổng 29.458.211.
- **Nhãn SegNet có mức xám lạ**: 51, 102, 153, 204 (0,06% pixel train, 0,37% test, đếm trên toàn bộ ảnh; số ≈ 0,2%/0,4% ghi trước đó là ước lượng trên 1/10 số ảnh) không thuộc 7 lớp. Ở gốc chúng thành one-hot toàn 0, nên góp 0 vào loss nhưng vẫn tính trong mẫu số, và được tính là lớp palm khi đo accuracy. Bản PyTorch gán −100, dùng `cross_entropy(reduction='none').mean()` và `clamp(min=0)` để ra đúng như vậy.
- **Đường dẫn nhãn SegNet**: gốc `fn.replace('depth', 'labels')` thay mọi chữ "depth", hỏng với đường dẫn qua thư mục `depth_hand/`; bản PyTorch ghép đường dẫn riêng.
- **"accuracy" của obj_detection**: loss MSE và đầu ra 1 giá trị nên Keras dùng `binary_accuracy` = `round(dự đoán) == nhãn`. Với nhãn ±1 thì phải |dự đoán| > 0,5 mới tính đúng. Bản PyTorch tính cả cách này (để so) và accuracy theo dấu.
- **Chia train/test obj_detection**: gốc không đặt seed; bản PyTorch `random_state=0`.
- **Heatmap ở mép ảnh**: code vẽ khung gốc sẽ `IndexError` nếu có ô được phát hiện ở hàng/cột cuối (`j*8+64 = 720`); bản PyTorch chặn chỉ số.

### 6. Đã kiểm tra

1. `common.py` (script thử trong thư mục tạm, không ghi vào `results/` thật): MLP 669.706 tham số; CNN_LSTM 1.748.097; bias cổng quên LSTM = 1; `EarlyStopping` tốt nhất 27 → dừng 32; nạp checkpoint tốt nhất rồi `evaluate` ra đúng `val_loss` tốt nhất; softmax của `predict` cộng lại bằng 1; `ghi_so_sanh` gọi lần hai thì thay dòng cũ, không thêm dòng.
2. FER2013: `load_fer2013` cho 28.709 / 3.589 ảnh, giá trị 0–255; 3 dòng đầu (70, 151, 231) và 3 dòng cuối tập train (74, 222, 195) cùng nhãn khớp output `trainingset` của notebook gốc.
3. Gender: dữ liệu **xếp theo nhãn**, male trước, female sau (train vị trí 0–9412 male, 9413–11937 female; val tương tự). Train vẫn đúng vì `shuffle=True`. Chạy thử phải lấy mẫu rải đều cả file (lấy 1.000 ảnh đầu thì toàn male).
4. Chạy thử (`CHAY_THU = True`, CPU) 6 notebook GPU: đều qua, `assert` số tham số đều đúng. Callback của tu_2 lưu checkpoint đúng mẫu tên, dừng sớm, nạp lại file tốt nhất.
5. Quét ô code của 9 notebook: không có `import keras/tensorflow`, `tf.keras` hay đường dẫn tuyệt đối. Keras gốc chỉ nằm trong ô markdown.
6. `_khoa_file`: hai tiến trình cùng ghi 20 dòng vào một `so_sanh.csv` tạm → đủ 41 dòng (1 tiêu đề + 40), không mất dòng, không sót file `.lock`.
7. `chay_kaggle.ipynb`: kiểm cú pháp 13 ô; chạy thử phần điều phối luồng và đóng gói zip trên máy, dùng `getting_started` thay cho notebook nặng → có log BẮT ĐẦU/XONG, zip có đúng đường dẫn `pytorch/...`, `results/...`, `_log_kaggle/...`.
8. Func_apxm: hình dạng 3 đường quét giống hình gốc (B và C giảm dần về ≈ 0,0014–0,0016; A dao động, có chỗ nhảy ở A=500 như gốc).

### 7. Cách tự kiểm tra lại

```bat
conda activate btl02
cd /d "D:\Data\GitHub\deep-learning-assignments\BTL03 - Chuyển code từ keras sang pytorch deep learning\pytorch\1. MLP"
python -m nbconvert --to notebook --execute --inplace MNIST_MLP_pytorch.ipynb
type "..\..\results\so_sanh.csv"
```

Phải thấy: notebook chạy hết không lỗi (khoảng 1–2 phút); ô (2) in `Số tham số PyTorch: 669706`; ô (4) in `Test accuracy` khoảng 0.965–0.967; dòng `MNIST_MLP` trong `so_sanh.csv` được cập nhật theo lần chạy mới.

Lưu ý: gọi `python -m nbconvert` thay cho `jupyter nbconvert` để chắc chắn dùng nbconvert của env `btl02`. Trên máy có một Python 3.12 cài ngoài conda, lệnh `jupyter` có thể chạy nhầm bản đó.

### 8. Số liệu cho báo cáo

- MNIST_MLP: Keras 0.9648 → PyTorch 0.9658 (+0.0010), test loss 0.1159 (Keras 0.1143). 30 epoch trên CPU mất 84 s, trong khi Keras trên T4 mất 1,13 phút. MLP nhỏ nên CPU đủ nhanh.
- Func_apxm_MLP (test MSE ở điểm cuối mỗi vòng quét): A=1000 0.0086; B=512 0.0013; C=190 0.0015.
- Đếm tham số khớp tuyệt đối với Keras ở 8 mô hình (SegNet so "Trainable params", tổng kể cả buffer BN cũng khớp); CNN_LSTM lệch +1.024 đúng như tính trước.
- 2_obj_detection: "accuracy" cách Keras 0.9375 (Keras 0.9523, −1,48 điểm), accuracy theo dấu 0.9636, test MSE 0.1074 (Keras 0.0642). `val_loss` epoch 4 là 0.0629 rồi epoch 5 nhảy lên 0.1074: chỉ 5 epoch và gốc đánh giá ở epoch cuối nên dao động này vào thẳng kết quả. Heatmap trên `test.jpg`: 54 ô > 0.99, khoanh đúng cả 2 xe (xe đen giữa, xe trắng phải), không có vùng nhầm (`results/2_obj_detection/heatmap_test.png`).

### 9. Câu hỏi vấn đáp có thể gặp

- **`fit` của Keras làm gì mà PyTorch phải tự viết?** Chia batch, xáo dữ liệu, forward, tính loss, backward, cập nhật trọng số, rồi cuối epoch đánh giá validation và gọi callback. Trong PyTorch mấy việc này nằm ở `fit()` của `common.py`.
- **Vì sao lớp cuối không có softmax?** `nn.CrossEntropyLoss` đã tính `log_softmax` bên trong. Thêm softmax nữa thì gradient rất nhỏ, mô hình học chậm. Cần xác suất thì gọi `torch.softmax` lúc dự đoán.
- **`model.train()` và `model.eval()` khác gì?** `train()` bật dropout và cho BatchNorm cập nhật thống kê; `eval()` tắt dropout, BatchNorm dùng running mean/var. Đánh giá phải ở `eval()` và trong `torch.no_grad()` để khỏi tính gradient.
- **`TimeDistributed` viết thế nào trong PyTorch?** Gộp chiều thời gian vào batch, `(N, 8, 1, 28, 28)` thành `(N·8, 1, 28, 28)`, cho qua CNN rồi reshape lại `(N, 8, 2304)`. CNN dùng chung trọng số cho cả 8 ảnh.
- **Vì sao phải khởi tạo giống Keras?** Mặc định PyTorch khác Keras. Muốn chênh lệch kết quả chỉ đến từ thư viện và yếu tố ngẫu nhiên thì phải giữ khởi tạo giống nhau.
- **EarlyStopping có lấy lại trọng số tốt nhất không?** Không, giống Keras (`restore_best_weights=False`). Vì vậy bản Gender nạp checkpoint tốt nhất rồi mới đánh giá; bản Emotion đánh giá mô hình ở epoch cuối như gốc.
- **Vì sao "accuracy" obj_detection thấp hơn accuracy theo dấu?** Keras dùng `binary_accuracy` = `round(dự đoán) == nhãn`. Nhãn là ±1 nên dự đoán trong (−0,5; 0,5) làm tròn thành 0 và luôn bị tính sai, dù đúng dấu. Accuracy theo dấu (ngưỡng 0) mới là accuracy phân loại thật.
- **Một mạng sao dùng được cho ảnh 64×64 và ảnh 720×1280?** Mạng không có lớp Dense, chỉ có conv và pooling, nên nhận ảnh kích thước bất kỳ. Ảnh 64×64 cho ra 1×1 điểm số; ảnh lớn cho bản đồ 83×153, mỗi điểm là điểm số của một cửa sổ 64×64 trượt bước 8.
- **BatchNorm momentum 0.99 của Keras sao thành 0.01?** Keras: giá trị mới = 0.99·cũ + 0.01·batch. PyTorch: giá trị mới = (1 − m)·cũ + m·batch. Muốn giống thì m = 0.01.
- **Func_apxm gốc chỉ có hình thì so sánh thế nào?** So hình dạng đường quét và độ lớn đọc từ hình. Trong `so_sanh.csv` cột Keras để trống, số đọc từ hình ghi ở cột ghi chú, không gõ thành số đo.

### 10. Còn treo

- [x] Trúc commit + push code hôm nay (`87ae2f5c`)
- [x] Commit + push `chay_kaggle.ipynb` (`898a2192`)
- [x] Tạo Kaggle Dataset riêng từ 4 file `data/gender/*.npy` (`btl03-gender`, Private)
- [x] Chạy GPU trên Kaggle (`chay_kaggle.ipynb`, T4 x2, Internet On, Save & Run All) 6 notebook (xong 27/11 sau khi sửa lỗi CPU, xem mục 27/11): 1_MNIST_CNN, 2_Human_Gender_CNN, 3_Emotion_Reg_CNN, MNIST_CNN_LSTM, tu_2_Human_Gender_CNN, tu_3_Emotion_Reg_CNN; tải notebook có output + `results/<notebook>/` về
- [x] Sau khi chạy GPU: viết lại câu nhận xét PCA cuối `1_MNIST_CNN` theo hình thật (27/11)
- [x] Viết `1_segnet`, `2_obj_detection` (kế hoạch T6 27/11 sáng, làm sớm 26/11)
- [x] Chạy `1_segnet` trên Kaggle (27/11, cùng lần chạy với 6 notebook kia)
- [x] Hỏi thầy: cá nhân hay nhóm, tên file zip, có cần slide không (chốt 27/11: nhóm 5; tên theo mẫu BTL02; không cần slide)

---

## T6 27/11/2026 – Sửa lỗi Kaggle train bằng CPU, chạy GPU 7 notebook, viết báo cáo nhóm

### 1. Việc theo kế hoạch

Kế hoạch mục 7, T6 27/11 sáng: SegNet, obj_detection, chạy GPU, đủ `so_sanh.csv`. SegNet và obj_detection đã viết từ 26/11. Lần chạy Kaggle tối 26/11 (`chay_kaggle.ipynb`, T4 x2) chạy gần 12 giờ (42.170 s) mà chưa xong, Output 0 B. Sáng nay tìm ra nguyên nhân và sửa; chạy lại trên Kaggle 7 notebook GPU (2 luồng song song, 04:23 → 05:07, khoảng 44 phút), tất cả trên `cuda (Tesla T4)`. `so_sanh.csv` đủ 12 dòng. Dòng T5 26/11 và T6 27/11 sáng trong Kế hoạch mục 7 đã tick.

### 2. Commit trong ngày

- `60565e46` BTL03: sửa summary() kéo mô hình về CPU; chay_kaggle kiểm GPU, timeout, theo dõi % GPU: sửa `summary()` trong `common.py`, thêm ô kiểm GPU + timeout + theo dõi % GPU vào `chay_kaggle.ipynb`, nhật ký.
- Commit kết quả GPU (7 notebook có output, 7 thư mục `results/`, `so_sanh.csv` 12 dòng, ô giải thích SegNet, nhận xét PCA, nhật ký): _(Trúc điền ở mục ngày sau)_

### 3. Kết quả

File đã sửa:

- `pytorch/common.py`:
  - `summary()` truyền cho torchinfo đúng thiết bị đang chứa mô hình (`next(model.parameters()).device`), không mặc định `"cpu"` nữa.
  - `fit()` in dòng `fit: train trên <thiết bị>` trước epoch 1, nhìn output là biết chạy CPU hay GPU.
- `chay_kaggle.ipynb`:
  - Thêm mục "2. Kiểm tra mô hình thật sự train trên GPU". Ô này dựng mô hình nhỏ trên `cuda`, gọi `summary`, train 1 epoch, rồi `assert` mô hình còn ở `cuda`; sai thì dừng ngay.
  - `nbconvert` mỗi ô tối đa 2 giờ (`timeout=7200`, trước là không giới hạn).
  - Vòng theo dõi in thêm % dùng GPU và bộ nhớ GPU mỗi phút.

Kết quả GPU (zip `btl03_gpu_20261127-0507.zip`, giải nén vào thư mục tạm, kiểm rồi mới chép vào repo):

- 7 notebook có output: `2. CNN + MLP/` (3), `3. CNN + LSTM/` (1), `4. Training Utilities/` (2), `5. Others/1_segnet`.
- 7 thư mục `results/<notebook>/` mới; `results/so_sanh.csv` 12 dòng (5 dòng cũ giữ nguyên + 7 dòng GPU).
- Sửa thêm 2 ô markdown: nhận xét PCA cuối `1_MNIST_CNN` (viết theo hình thật); ô "Vì sao bản PyTorch cao hơn Keras khoảng 3,8 điểm?" trong `1_segnet`.

| Notebook | Keras gốc | PyTorch (T4) | Chênh | Epoch Keras / PyTorch | Thời gian PyTorch |
|---|---|---|---|---|---|
| 1_MNIST_CNN | 0.9884 | 0.9891 | +0.0007 | 30 / 30 | 141 s |
| 2_Human_Gender_CNN | 0.8264 | 0.8388 | +0.0124 | 30 / 30 | 183 s |
| 3_Emotion_Reg_CNN | 0.5954 | 0.5890 | −0.0064 | 15 / 15 | 355 s |
| MNIST_CNN_LSTM (MAE, thấp là tốt) | 0.6502 | 0.5588 | −0.0914 | 30 / 30 | 1.150 s |
| tu_2_Human_Gender_CNN | 0.8773 | 0.8809 | +0.0036 | 32 (tốt nhất 27) / 47 (tốt nhất 42) | 282 s |
| tu_3_Emotion_Reg_CNN | 0.6163 | 0.6166 | +0.0003 | 57 (tốt nhất 52) / 54 (tốt nhất 49) | 1.406 s |
| 1_segnet (pixel acc) | 0.7732 | 0.8108 | +0.0376 | 15 / 15 | 480 s |

Báo cáo và làm nhóm (chiều–tối):

- Agent `btl03-doi-chieu-keras` rà 11 notebook so với Keras gốc: không có lỗi nghiêm trọng; 12 dòng `so_sanh.csv` khớp output. Đã xử lý các góp ý: thêm ô giải thích dòng cảnh báo TensorFlow trong notebook Gender bản Training Utilities (cảnh báo nói về TensorFlow do `tensorboard` nạp, PyTorch vẫn chạy cuda); báo cáo LSTM nêu khoảng dao động 9 epoch cuối thay vì một con số; obj_detection nêu rõ hai bản chia train/test khác nhau.
- Sửa số pixel lạ của SegNet thành số đếm trên toàn bộ ảnh: 0,06% train, 0,37% test (số 0,17%/0,42% trước đó là ước lượng trên 1/10 số ảnh).
- Báo cáo `report/`: 7 mục `.md`, `ve_hinh_tong_hop.py` (Hình 4.1 từ `so_sanh.csv`), `ghep_bao_cao.py` (dựa trên script BTL02, cần `pypandoc_binary` và Edge). Ghép ra `BaoCao_BTL03_Nhom05.docx` (2,9 MB) và `.pdf` (3,6 MB, 37 trang).
- **Làm nhóm** (Trúc chốt 27/11): Nhóm 5, 25% mỗi người. Trúc: `common.py`, getting_started, Func_apxm, MNIST_MLP, `chay_kaggle`, mục 1, 2, 3.1, 3.2, 4, 6, phụ lục. Khang: 1_MNIST_CNN, CNN_LSTM, mục 3.3, 3.6. Trung: Gender, Gender + callback, obj_detection, mục 3.4, 3.7, 3.10. Nhân: Emotion, Emotion + augmentation, SegNet, mục 3.5, 3.8, 3.9, 5. Mỗi notebook có dòng `Phụ trách:`, mỗi mục báo cáo có dòng `Người viết:`, phụ lục A có bảng phân công. Cập nhật Kế hoạch mục 11 và `CLAUDE.md` mục 0, 5.
- Env `btl02` cài thêm `pypandoc_binary`, `pymupdf` (xem trang PDF).

Hoàn thiện và đóng gói (tối 27/11, làm sớm việc T7 28/11):

- Trúc chốt không cần slide.
- README viết lại: thành viên + tỷ lệ, bảng kết quả Keras ↔ PyTorch, cấu trúc, cài đặt, dữ liệu (cần đặt `Yêu cầu/` cạnh `pytorch/` khi chạy lại SegNet và phát hiện xe), cách chạy, cách ghép báo cáo.
- Agent `btl03-cham-rubric` chấm thử: ước lượng 91–94/100, không lỗi nặng. Đã sửa: dòng `Người viết:` dưới từng tiểu mục 3.1–3.10; README nói rõ 11 notebook = 10 có mô hình + getting_started, "val acc" là tập gốc dùng làm validation; `chay_kaggle.ipynb` ghi rõ là notebook hỗ trợ (không có output); câu `CHAY_THU` chỉ áp cho notebook có huấn luyện; `check_submission.py` in UTF-8 trên Windows, nhận 3 dòng Func_apxm A/B/C, bỏ cảnh báo count_params cho getting_started.
- Agent `btl03-doi-chieu-keras` đối chiếu số trong báo cáo: các số chính khớp. Đã sửa: SegNet "giữ 3–5 điểm" thành "cao hơn ở cả 15 epoch, chênh 2.8–6.2 điểm train, 1.3–6.0 điểm val" (báo cáo, notebook, nhật ký); Func_apxm vòng quét B gốc ở B = 8 là ≈ 0.0112 (đường test), không phải 0.0105 (đường train); thêm câu dẫn trước Bảng 3.2–3.10.
- `.gitignore` gốc thêm `BTL02_Nhom05_*.zip`, `BTL03_Nhom05_*.zip`, `btl03_gpu_*.zip`.
- `check_submission.py --zip`: 0 lỗi, 0 cảnh báo; nén `BTL03_Nhom05_HuynhNgocTruc_3123411312.zip` ở gốc repo.
- Trúc yêu cầu phần `report/` trong zip giống BTL02: chỉ kèm `BaoCao_BTL03_Nhom05.docx` và `.pdf` (thêm `ZIP_ONLY` vào `check_submission.py` như commit `358d3d09` của BTL02); file `.md`, `ghep_bao_cao.py`, `ve_hinh_tong_hop.py`, `hinh/` ở lại repo. README và phụ lục C ghi rõ. Nén lại: 58 file, 11,1 MB.

### 4. Giải thích lỗi

- Trong torchinfo (`torchinfo.py` dòng 293) có `model = model if device is None else model.to(device)`. `nn.Module.to()` chuyển **chính** mô hình, không tạo bản sao.
- `summary()` cũ đặt `device="cpu"`. Mỗi notebook gọi `model = init_keras_like(...).to(device)` (lên GPU), rồi `summary(model, ...)`, nên mô hình bị kéo về CPU.
- `fit()` lấy thiết bị từ tham số mô hình (`next(model.parameters()).device`), nên batch cũng đưa lên CPU. Không có lỗi "tensor khác thiết bị" nào, chỉ chậm. Train cả 7 notebook bằng CPU thì 12 giờ không đủ (riêng LSTM gốc trên Core i7 mất 13,5 giờ).
- Trên máy không lộ ra vì máy chỉ có CPU. 4 notebook đã chạy ở máy (getting_started, Func_apxm, MNIST_MLP, obj_detection) không bị ảnh hưởng.

### 5. Đã kiểm tra

1. Tái hiện lỗi: mô hình đặt trên CPU, gọi thẳng `torchinfo.summary(..., device='meta')` → mô hình bị chuyển sang `meta`. Gọi `summary()` mới → mô hình vẫn ở CPU.
2. `chay_kaggle.ipynb`: kiểm cú pháp 15 ô; chạy đoạn kiểm GPU trên máy (bỏ 2 dòng `assert cuda`) → in `fit: train trên cpu`, `OK`.
3. 7 notebook từ Kaggle: 0 ô lỗi, 0 ô chưa chạy, đều có dòng `fit: train trên cuda`, đều `CHAY_THU = False`.
4. `so_sanh.csv` trong zip chứa nguyên 5 dòng của máy (Func_apxm A/B/C, MNIST_MLP, obj_detection) + 7 dòng GPU → chép đè an toàn.
5. **SegNet +3,76 điểm** (vượt ngưỡng 3 điểm của rule, nên phải tìm nguyên nhân): so log từng epoch với log Keras gốc → bản PyTorch cao hơn ngay từ epoch 1 (train 0.670 / 0.631, val 0.679 / 0.619) và cao hơn ở cả 15 epoch (chênh 2.8–6.2 điểm train, 1.3–6.0 điểm val; agent đối chiếu sửa lại câu "giữ 3–5 điểm" ban đầu), nên là khác biệt có hệ thống. Nguyên nhân: `Reshape((7, 9216))` + `Permute((2, 1))` của gốc viết cho channels_first nhưng chạy channels_last, nên softmax của mỗi "pixel" lấy điểm số từ 7 pixel khác nhau trong ảnh (xem "Khác Keras ở đâu"). Bản PyTorch tính softmax đúng theo pixel.
6. tu_2 Gender dừng muộn hơn Keras 15 epoch: bản PyTorch kẹt ở mức 0.7885 (đoán toàn male) đến epoch 7, Keras thoát từ epoch 4; sau đó cùng xu hướng. lr giữ 1e-3 suốt (ReduceLROnPlateau không kích hoạt), giống log gốc.
7. LSTM MAE 0.5588 tốt hơn gốc 0.0914 (ngưỡng 0,3): `val_loss` Keras ở 4 epoch cuối dao động 0.60–0.71, nên mức này nằm trong dao động.
8. Gender (thư mục 2): recall female 0.3574 (451/1262), male 0.9679; 662/662 ảnh đen bị đoán male.
9. Xem 3 hình PCA `1_MNIST_CNN`: ảnh gốc chỉ 0 và 1 tách; `fc2` thành cụm rõ; GAP sau `pool3` ở giữa.

### 6. Khác Keras ở đâu

- **Reshape + Permute của SegNet gốc**: viết cho channels_first nhưng chạy channels_last. Dãy bộ nhớ `(96, 96, 7)` là pixel 0 kênh 0…6, pixel 1 kênh 0…6, …; `Reshape((7, 9216))` cắt dãy đó thành 7 hàng liên tiếp, sau `Permute` thì "lớp i của pixel j" là phần tử thứ `i·9216 + j` của dãy, tức pixel `(i·9216 + j) // 7`. Mỗi pixel đầu ra lấy điểm số từ 7 pixel khác nhau. Bản PyTorch không lặp lại lỗi này.

### 7. Câu hỏi vấn đáp có thể gặp

- **Vì sao PyTorch phải tự chuyển mô hình và dữ liệu sang GPU?** Keras/TensorFlow tự đặt phép tính lên GPU. PyTorch để người viết quyết định bằng `.to(device)`, cho cả mô hình lẫn từng batch. Một hàm phụ gọi `.to("cpu")` là cả quá trình train chuyển sang CPU mà không báo lỗi.
- **Vì sao SegNet PyTorch cao hơn Keras gần 4 điểm?** Bản Keras gốc dùng `Reshape((7, 96*96))` trên tensor channels_last `(N, 96, 96, 7)`, nên không tách đúng kênh lớp mà trộn điểm số của 7 pixel khác nhau. Bản PyTorch để đầu ra `(N, 7, 96, 96)` và `CrossEntropyLoss` lấy softmax theo chiều lớp của từng pixel. Kiến trúc, số tham số, siêu tham số giữ nguyên.
- **Gender đạt 0.84 có tốt không?** Đoán toàn male đã được 0.7885. Recall female chỉ 0.36 và mọi ảnh đen đều bị đoán male, nên mô hình nghiêng hẳn về lớp đông.
- **Làm sao biết notebook đang chạy GPU?** Xem `next(model.parameters()).device`, dòng `fit: train trên ...`, hoặc `nvidia-smi` (% GPU > 0 khi đang train).

### 8. Còn treo

- [x] Trúc dừng lần chạy Kaggle cũ, commit + push bản sửa, import lại `chay_kaggle.ipynb`, Save & Run All
- [x] Tải zip, gộp `so_sanh.csv`, kiểm từng notebook
- [ ] Commit kết quả GPU + báo cáo (lệnh ở cuối phiên)
- [x] Viết báo cáo (T6 27/11 chiều–tối): `report/BaoCao_BTL03_Nhom05.docx` + `.pdf`, 37 trang
- [ ] Gửi mỗi bạn notebook + mục báo cáo đứng tên mình để đọc trước vấn đáp
- [x] README, chấm thử, đóng gói `BTL03_Nhom05_HuynhNgocTruc_3123411312.zip` (làm sớm tối 27/11)
- [x] Thêm bìa trường (mẫu `TRƯỜNG ĐẠI HỌC SÀI GÒN.docx` ở gốc repo) vào báo cáo: `report/bia.py` (dùng chung với BTL02), `ghep_bao_cao.py` bỏ trang tiêu đề pandoc, ghép bìa vào `.docx` (docxcompose) và chèn bìa HTML vào `.pdf`. Bìa ghi "BÁO CÁO BÀI TẬP LỚN 03 – CHUYỂN CODE TỪ KERAS SANG PYTORCH", Nhóm 05 + 4 thành viên kèm MSSV, lớp DCT123C3, "TP. HCM, THÁNG 11/2026". PDF 39 trang; DOCX qua `validate.py`; zip nén lại
- [ ] Trúc mở bản `.docx` bằng Word xem bìa (máy không có Word/LibreOffice nên chưa xem ảnh trang Word)
- [ ] Trúc mở PDF xem lần cuối rồi nộp zip
