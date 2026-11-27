# BTL03 – Chuyển code từ Keras sang PyTorch

Học phần Học sâu · Nhóm 05 · Nhóm trưởng: Huỳnh Ngọc Trúc (3123411312)

Chuyển 11 notebook Keras trong `Yêu cầu/examples/` (MLP, CNN, CNN+LSTM, các kỹ thuật hỗ trợ huấn luyện, SegNet, phát hiện xe) sang PyTorch. Kiến trúc, siêu tham số và tiền xử lý giữ nguyên như bản gốc; kết quả hai bản đặt cạnh nhau trong `results/so_sanh.csv` và báo cáo `report/BaoCao_BTL03_Nhom05.pdf`.

## Thành viên

| Thành viên | MSV | Phụ trách | Tỷ lệ |
|---|---|---|---|
| Huỳnh Ngọc Trúc (nhóm trưởng) | 3123411312 | `common.py`, getting_started, Func_apxm_MLP, MNIST_MLP, `chay_kaggle.ipynb`; báo cáo mục 1, 2, 3.1, 3.2, 4, 6, phụ lục | 25% |
| Phạm Trần Minh Khang | 3123411143 | 1_MNIST_CNN, MNIST_CNN_LSTM; mục 3.3, 3.6 | 25% |
| Tạ Quang Trung | 3123411318 | 2_Human_Gender_CNN, Gender + callback, 2_obj_detection; mục 3.4, 3.7, 3.10 | 25% |
| Ngô Nguyễn Thành Nhân | 3123411205 | 3_Emotion_Reg_CNN, Emotion + augmentation, 1_segnet; mục 3.5, 3.8, 3.9, 5 | 25% |

## Kết quả

| Notebook PyTorch | Chỉ số | Keras gốc | PyTorch | Thiết bị |
|---|---|---:|---:|---|
| `0. Getting Started/getting_started_pytorch` | kiểm phiên bản, GPU | – | – | CPU |
| `1. MLP/Func_apxm_MLP_pytorch` | test MSE (quét A/B/C) | chỉ có hình | 0.0086 / 0.0013 / 0.0015 | CPU |
| `1. MLP/MNIST_MLP_pytorch` | test acc | 96.48% | 96.58% | CPU |
| `2. CNN + MLP/1_MNIST_CNN_pytorch` | test acc | 98.84% | 98.91% | T4 |
| `2. CNN + MLP/2_Human_Gender_CNN_pytorch` | val acc | 82.64% | 83.88% | T4 |
| `2. CNN + MLP/3_Emotion_Reg_CNN_pytorch` | val acc | 59.54% | 58.90% | T4 |
| `3. CNN + LSTM/MNIST_CNN_LSTM_pytorch` | MAE | 0.6502 | 0.5588 | T4 |
| `4. Training Utilities/2_Human_Gender_CNN_pytorch` | val acc | 87.73% | 88.09% | T4 |
| `4. Training Utilities/3_Emotion_Reg_CNN_pytorch` | val acc | 61.63% | 61.66% | T4 |
| `5. Others/1_segnet_pytorch` | pixel acc | 77.32% | 81.08% | T4 |
| `5. Others/2_obj_detection_pytorch` | test acc (cách Keras) | 95.23% | 93.75% | CPU |

11 notebook gồm 10 notebook có mô hình và `getting_started` (chỉ kiểm phiên bản, GPU). "val acc" là accuracy trên tập mà notebook gốc dùng làm `validation_data` (tập val của Gender, PublicTest của FER2013); `so_sanh.csv` ghi chung là `test_acc` như gốc gọi. Số tham số của cả 10 mô hình khớp `model.summary()` gốc (CNN+LSTM +1.024 do LSTM PyTorch có 2 bias; SegNet so với "Trainable params"). Giải thích từng chênh lệch ở báo cáo mục 3 và 4. `Iris_Bayesian_decision_making` (scikit-learn) và `colab_getting_started` không dùng Keras nên không chuyển.

## Cấu trúc

| Thư mục / file | Nội dung |
|---|---|
| `pytorch/common.py` | Phần dùng chung: seed, thiết bị, `fit`/`evaluate`/`predict`, callback (ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau), khởi tạo giống Keras, đếm tham số, đọc FER2013, lưu kết quả |
| `pytorch/<thư mục>/<tên>_pytorch.ipynb` | Bản PyTorch của từng notebook gốc, cùng tên thư mục và thứ tự mục; đoạn Keras gốc trích trong ô markdown ngay trên ô PyTorch |
| `chay_kaggle.ipynb` | Chạy 7 notebook cần GPU trên Kaggle (T4 x2, 2 luồng song song) và nén kết quả |
| `results/so_sanh.csv` | Bảng so sánh Keras ↔ PyTorch: số tham số, kết quả, chênh lệch, epoch, thời gian, thiết bị |
| `results/<notebook>/` | `history.csv`, `metrics.json`, hình (đường học, PCA, heatmap, ảnh phân đoạn, vòng quét) |
| `report/` | Báo cáo `BaoCao_BTL03_Nhom05.docx` / `.pdf` (file nguồn `.md`, script vẽ hình và ghép báo cáo chỉ có trong kho mã GitHub; file nộp chỉ kèm bản `.docx`/`.pdf`) |
| `requirements.txt` | Thư viện cần cài |

## Cài đặt

Python 3.11:

```
pip install -r requirements.txt
```

## Dữ liệu

Dữ liệu không nộp kèm (nặng). Notebook SegNet và phát hiện xe đọc dữ liệu có sẵn trong đề tại `Yêu cầu/examples/5. Others/`, nên khi chạy lại cần đặt thư mục `Yêu cầu/` của đề (gồm `examples/`) vào trong thư mục bài này, cạnh `pytorch/`. Các notebook đều đã có output nên không cần chạy lại để xem kết quả. Đặt dữ liệu như sau trước khi chạy:

| Bộ | Lấy ở đâu | Đặt ở |
|---|---|---|
| MNIST | `torchvision` tự tải | `data/mnist/` |
| Gender (WIKI) | link Google Drive trong notebook gốc `2_Human_Gender_CNN` | `data/gender/` (4 file `64_64_*_onehot.npy`) |
| FER2013 | `fer2013.csv` (Kaggle) | `data/emotion/fer2013.csv` |
| Bàn tay (SegNet), xe, `test.jpg` | có sẵn trong đề | `Yêu cầu/examples/5. Others/` |

## Cách chạy

Chạy một notebook và lưu output vào chính file (đứng ở thư mục chứa notebook):

```
cd "pytorch/1. MLP"
python -m nbconvert --to notebook --execute --inplace MNIST_MLP_pytorch.ipynb
```

Hoặc mở bằng Jupyter rồi Kernel → Restart & Run All. Mỗi notebook có huấn luyện có biến `CHAY_THU` ở ô đầu: `True` chạy thử với ít dữ liệu để kiểm code, `False` (mặc định) chạy đầy đủ và ghi kết quả vào `results/`.

- Chạy được trên CPU: getting_started, Func_apxm_MLP (khoảng 30 s), MNIST_MLP (khoảng 1,5 phút), 2_obj_detection (khoảng 2 phút).
- Cần GPU: 1_MNIST_CNN, Gender, Emotion, CNN_LSTM, hai notebook Training Utilities, SegNet (tổng khoảng 45 phút trên Kaggle T4 x2). Dùng `chay_kaggle.ipynb`: Import vào Kaggle, bật GPU T4 x2 và Internet, Add Input dữ liệu Gender và FER2013, Save & Run All, rồi tải `btl03_gpu_*.zip` ở tab Output và giải nén đè vào thư mục này.

Ghép lại báo cáo sau khi sửa các file `.md` (script và file nguồn chỉ có trong kho mã GitHub; cần `pip install pypandoc_binary` và Microsoft Edge hoặc Chrome):

```
python report/ve_hinh_tong_hop.py
python report/ghep_bao_cao.py
```
