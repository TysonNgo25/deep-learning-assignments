# 1. Giới thiệu

Người viết: Huỳnh Ngọc Trúc

## 1.1. Đề bài và phạm vi

Đề yêu cầu chuyển các ví dụ Keras trong thư mục `Yêu cầu/examples/` sang PyTorch, chạy lại và so sánh với kết quả gốc. Thư mục gồm 13 notebook, đi theo đúng trình tự của slide `02_KerasExamples.pdf`: mạng nhiều lớp (MLP), mạng tích chập (CNN), các kỹ thuật hỗ trợ huấn luyện, CNN kết hợp LSTM và phân đoạn ảnh.

Bài chuyển **11 notebook có dùng Keras** (Bảng 1.1). Hai notebook không chuyển vì không có Keras:

- `Iris_Bayesian_decision_making.ipynb`: phân loại Iris bằng Naive Bayes của scikit-learn (accuracy 0.9667), không có mạng nơ-ron.
- `colab_getting_started.ipynb`: hướng dẫn dùng Google Colab, không có mô hình.

Mọi số "Keras gốc" trong báo cáo lấy từ output đã lưu trong các notebook gốc **trong thư mục đề** (không lấy từ slide hay bản trên GitHub). Một số notebook gốc chạy bằng Keras 2.0.9 trên GTX 1080, một số đã được chạy lại bằng tf.keras 2.14 trên Colab T4; cột "Bản Keras" của Bảng 1.1 ghi rõ.

Bảng 1.1. Mười một notebook được chuyển và kết quả gốc dùng để so sánh.

| Mục | Notebook gốc | Bài toán | Bản Keras | Tham số | Kết quả gốc |
| --- | --- | --- | --- | ---: | --- |
| 2.4 | `0. Getting Started/getting_started` | kiểm phiên bản, GPU | tf.keras 2.14 | – | phát hiện GPU Tesla T4 |
| 3.1 | `1. MLP/Func_apxm_MLP` | xấp xỉ hàm Gauss 2 chiều | tf.keras 2.14 / Keras cũ | 65–2.049 | chỉ có hình |
| 3.2 | `1. MLP/MNIST_MLP` | MNIST 10 lớp | tf.keras 2.14 | 669.706 | test acc 96.48% |
| 3.3 | `2. CNN + MLP/1_MNIST_CNN` | MNIST bằng CNN, PCA đặc trưng | tf.keras 2.14 | 1.817.610 | 98.84% |
| 3.4 | `2. CNN + MLP/2_Human_Gender_CNN` | giới tính nam/nữ, ảnh 64×64 | Keras 2.0.9 | 4.063.570 | 82.64% |
| 3.5 | `2. CNN + MLP/3_Emotion_Reg_CNN` | 7 cảm xúc, ảnh 48×48 | Keras 2.0.9 | 19.028.287 | 59.54% |
| 3.6 | `3. CNN + LSTM/MNIST_CNN_LSTM` | tổng của 1–8 chữ số viết tay | Keras 2.0.9 | 1.747.073 | MAE 0.6502 |
| 3.7 | `4. Training Utilities/2_Human_Gender_CNN` | Gender + 4 callback | Keras 2.0.9 | 4.063.570 | 87.73% |
| 3.8 | `4. Training Utilities/3_Emotion_Reg_CNN` | Emotion + augmentation + EarlyStopping | Keras 2.0.9 | 19.028.287 | 61.63% |
| 3.9 | `5. Others/1_segnet` | phân đoạn bàn tay trên ảnh độ sâu | Keras 2.0.9 | 29.442.325 (train được) | pixel acc 77.32% |
| 3.10 | `5. Others/2_obj_detection` | phát hiện xe, mạng toàn tích chập | Keras 2.0.9 | 83.367 | 95.23% |

## 1.2. Keras và PyTorch khác nhau ở đâu

Hai thư viện cùng dựng được các mô hình trên, nhưng cách viết khác nhau ở bốn điểm. Bốn điểm này quyết định phần lớn công việc chuyển đổi.

**Khai báo mô hình.** Keras ghép các lớp bằng API hàm (`x = Dense(...)(x)`) hoặc `Sequential`, tự suy ra kích thước đầu vào của mỗi lớp. PyTorch khai báo một lớp con của `nn.Module`: các lớp con tạo trong `__init__`, đường đi của dữ liệu viết trong `forward`. Mỗi `nn.Linear`, `nn.Conv2d` phải ghi rõ số đầu vào, nên phải tự tính kích thước sau mỗi khối.

**Huấn luyện.** Keras có sẵn `compile` (chọn loss, optimizer, chỉ số) và `fit` (lặp epoch, chia batch, đánh giá validation, gọi callback). PyTorch không có `fit`: người viết tự lặp qua `DataLoader`, gọi `zero_grad`, `backward`, `step`, và tự chuyển giữa chế độ huấn luyện (`model.train()`) và đánh giá (`model.eval()` cùng `torch.no_grad()`). Bài gom phần này vào `pytorch/common.py` (mục 2.2) để 11 notebook dùng chung một vòng lặp.

**Thứ tự chiều của ảnh.** Keras (TensorFlow) mặc định để kênh ở cuối, `(N, H, W, C)`; PyTorch để kênh sau chiều batch, `(N, C, H, W)`. Ảnh được đổi chiều một lần khi tạo tensor.

**Thiết bị.** TensorFlow tự đặt phép tính lên GPU nếu có. PyTorch để người viết chọn và chuyển cả mô hình lẫn từng batch dữ liệu bằng `.to(device)`. Mục 5.1 kể một lỗi thật gặp phải vì điểm này.

## 1.3. Nguyên tắc chuyển đổi

- **Giữ nguyên bản gốc:** cùng kiến trúc, số filter, padding, dropout, loss, optimizer, learning rate, batch, số epoch và tiền xử lý. Gốc không chia ảnh cho 255 (Gender, Emotion, SegNet) thì bản PyTorch cũng không chia.
- **Kiểm số tham số:** mỗi notebook có `assert` so số tham số với `model.summary()` của Keras. Hai chỗ lệch có lý do được nói rõ: LSTM (mục 3.6) và BatchNorm (mục 3.9).
- **Khởi tạo như Keras:** dùng `init_keras_like` (mục 2.2) để trọng số ban đầu theo cùng phân phối với Keras.
- **Một notebook gốc ứng với một notebook PyTorch** `pytorch/<thư mục>/<tên>_pytorch.ipynb`, cùng thứ tự mục (1) Data, (2) Declare model, (3) Train, (4) Evaluate. Đoạn Keras gốc được trích trong ô markdown ngay trên ô PyTorch tương ứng; ô code chỉ có PyTorch.
- **Seed cố định** `set_seed(0)`; riêng phần sinh dữ liệu CNN+LSTM giữ `np.random.seed(2018)` như gốc.
- **Phân công:** bài do Nhóm 5 thực hiện, mỗi thành viên phụ trách một nhóm notebook và phần báo cáo tương ứng (Bảng A.1, phụ lục).
- **Thiết bị chạy:** 4 notebook nhẹ chạy trên CPU cá nhân; 7 notebook còn lại chạy trên GPU Tesla T4 của Kaggle ngày 27/11/2026.
