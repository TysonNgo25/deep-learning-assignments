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

- [ ] Hỏi thầy: cá nhân hay nhóm, tên file zip, có cần slide không
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
- Commit code hôm nay (`common.py`, 9 notebook, `results/`): _(Trúc điền ở mục ngày sau)_

### 3. Kết quả

File mới:

- `pytorch/common.py`: phần dùng chung (đường dẫn, seed, `fit`/`evaluate`/`predict`, callback, lưu kết quả, đọc FER2013).
- `pytorch/0. Getting Started/getting_started_pytorch.ipynb`: có output (CPU).
- `pytorch/1. MLP/Func_apxm_MLP_pytorch.ipynb`, `MNIST_MLP_pytorch.ipynb`: có output (CPU).
- `pytorch/2. CNN + MLP/1_MNIST_CNN_pytorch.ipynb`, `2_Human_Gender_CNN_pytorch.ipynb`, `3_Emotion_Reg_CNN_pytorch.ipynb`: đã viết, chạy thử qua, **chưa có output** (chờ GPU).
- `pytorch/3. CNN + LSTM/MNIST_CNN_LSTM_pytorch.ipynb`: như trên.
- `pytorch/4. Training Utilities/2_Human_Gender_CNN_pytorch.ipynb`, `3_Emotion_Reg_CNN_pytorch.ipynb`: như trên.
- `results/so_sanh.csv` (4 dòng), `results/MNIST_MLP/` (history.csv, metrics.json, curves.png), `results/Func_apxm_MLP/` (quet_A/B/C.csv + .png, metrics.json).

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

### 6. Đã kiểm tra

1. `common.py` (script thử trong thư mục tạm, không ghi vào `results/` thật): MLP 669.706 tham số; CNN_LSTM 1.748.097; bias cổng quên LSTM = 1; `EarlyStopping` tốt nhất 27 → dừng 32; nạp checkpoint tốt nhất rồi `evaluate` ra đúng `val_loss` tốt nhất; softmax của `predict` cộng lại bằng 1; `ghi_so_sanh` gọi lần hai thì thay dòng cũ, không thêm dòng.
2. FER2013: `load_fer2013` cho 28.709 / 3.589 ảnh, giá trị 0–255; 3 dòng đầu (70, 151, 231) và 3 dòng cuối tập train (74, 222, 195) cùng nhãn khớp output `trainingset` của notebook gốc.
3. Gender: dữ liệu **xếp theo nhãn**, male trước, female sau (train vị trí 0–9412 male, 9413–11937 female; val tương tự). Train vẫn đúng vì `shuffle=True`. Chạy thử phải lấy mẫu rải đều cả file (lấy 1.000 ảnh đầu thì toàn male).
4. Chạy thử (`CHAY_THU = True`, CPU) 6 notebook GPU: đều qua, `assert` số tham số đều đúng. Callback của tu_2 lưu checkpoint đúng mẫu tên, dừng sớm, nạp lại file tốt nhất.
5. Quét ô code của 9 notebook: không có `import keras/tensorflow`, `tf.keras` hay đường dẫn tuyệt đối. Keras gốc chỉ nằm trong ô markdown.
6. Func_apxm: hình dạng 3 đường quét giống hình gốc (B và C giảm dần về ≈ 0,0014–0,0016; A dao động, có chỗ nhảy ở A=500 như gốc).

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
- Đếm tham số khớp tuyệt đối với Keras ở 6 mô hình; CNN_LSTM lệch +1.024 đúng như tính trước.

### 9. Câu hỏi vấn đáp có thể gặp

- **`fit` của Keras làm gì mà PyTorch phải tự viết?** Chia batch, xáo dữ liệu, forward, tính loss, backward, cập nhật trọng số, rồi cuối epoch đánh giá validation và gọi callback. Trong PyTorch mấy việc này nằm ở `fit()` của `common.py`.
- **Vì sao lớp cuối không có softmax?** `nn.CrossEntropyLoss` đã tính `log_softmax` bên trong. Thêm softmax nữa thì gradient rất nhỏ, mô hình học chậm. Cần xác suất thì gọi `torch.softmax` lúc dự đoán.
- **`model.train()` và `model.eval()` khác gì?** `train()` bật dropout và cho BatchNorm cập nhật thống kê; `eval()` tắt dropout, BatchNorm dùng running mean/var. Đánh giá phải ở `eval()` và trong `torch.no_grad()` để khỏi tính gradient.
- **`TimeDistributed` viết thế nào trong PyTorch?** Gộp chiều thời gian vào batch, `(N, 8, 1, 28, 28)` thành `(N·8, 1, 28, 28)`, cho qua CNN rồi reshape lại `(N, 8, 2304)`. CNN dùng chung trọng số cho cả 8 ảnh.
- **Vì sao phải khởi tạo giống Keras?** Mặc định PyTorch khác Keras. Muốn chênh lệch kết quả chỉ đến từ thư viện và yếu tố ngẫu nhiên thì phải giữ khởi tạo giống nhau.
- **EarlyStopping có lấy lại trọng số tốt nhất không?** Không, giống Keras (`restore_best_weights=False`). Vì vậy bản Gender nạp checkpoint tốt nhất rồi mới đánh giá; bản Emotion đánh giá mô hình ở epoch cuối như gốc.
- **Func_apxm gốc chỉ có hình thì so sánh thế nào?** So hình dạng đường quét và độ lớn đọc từ hình. Trong `so_sanh.csv` cột Keras để trống, số đọc từ hình ghi ở cột ghi chú, không gõ thành số đo.

### 10. Còn treo

- [ ] Trúc commit + push code hôm nay (lệnh ở cuối phiên)
- [ ] Chạy GPU (Colab/Kaggle) 6 notebook: 1_MNIST_CNN, 2_Human_Gender_CNN, 3_Emotion_Reg_CNN, MNIST_CNN_LSTM, tu_2_Human_Gender_CNN, tu_3_Emotion_Reg_CNN; tải notebook có output + `results/<notebook>/` về
- [ ] Sau khi chạy GPU: viết lại câu nhận xét PCA cuối `1_MNIST_CNN` theo hình thật
- [ ] `1_segnet`, `2_obj_detection` (T6 27/11 sáng)
- [ ] Hỏi thầy: cá nhân hay nhóm, tên file zip, có cần slide không
