# BTL03 – Chuyển code từ Keras sang PyTorch

Chuyển toàn bộ notebook Keras trong `Yêu cầu/examples/` (MLP, CNN, CNN+LSTM, SegNet, phát hiện xe) sang PyTorch, giữ nguyên kiến trúc, siêu tham số và tiền xử lý để so kết quả với bản gốc.

Người làm: Huỳnh Ngọc Trúc · 3123411312 · Hạn nộp T7 28/11/2026.

## Cài đặt

Dùng chung env `btl02` (đã có torch CPU), cài thêm vài gói:

```
conda activate btl02
pip install -r requirements.txt
```

Notebook nặng (CNN, LSTM, SegNet) chạy trên GPU Colab/Kaggle. Xem skill `chay-notebook-btl03` trong `.claude/skills/`.

## Dữ liệu

| Bộ | Lấy ở đâu | Đặt ở |
|---|---|---|
| MNIST | tự tải bằng torchvision | `data/mnist/` |
| Gender (WIKI) | link Google Drive trong notebook gốc `2_Human_Gender_CNN` | `data/gender/` (4 file `.npy`) |
| FER2013 | Kaggle, dùng chung với BTL02 | `data/emotion/` hoặc `../BTL02 Emotion Recognition/data/fer2013.csv` |
| Bàn tay (SegNet), xe | có sẵn trong `Yêu cầu/examples/5. Others/` | đọc tại chỗ |

`data/` không đưa lên git.

## Cấu trúc

| Thư mục / file | Nội dung |
|---|---|
| `Yêu cầu/` | Đề gốc của thầy: 2 slide PDF, README cài Keras, 13 notebook Keras và dữ liệu mẫu. Không sửa |
| `pytorch/common.py` | Dùng chung: seed, `fit`/`evaluate`/`predict`, callback (checkpoint, early stopping, giảm lr), khởi tạo giống Keras, đếm tham số, đường dẫn |
| `pytorch/<thư mục>/<tên>_pytorch.ipynb` | Bản PyTorch của từng notebook gốc, cùng tên thư mục và thứ tự mục |
| `results/so_sanh.csv` | Bảng so sánh Keras ↔ PyTorch: số tham số, kết quả, thời gian |
| `results/<notebook>/` | `history.csv`, `metrics.json`, hình |
| `report/` | Báo cáo |
| `docs/Kế hoạch.md` | Kế hoạch, bảng đối chiếu, lịch |

## Cách chạy

Mở Jupyter ở thư mục `pytorch/` rồi chạy từng notebook từ trên xuống (Kernel → Restart & Run All):

```
cd /d "<repo>\BTL03 - Chuyển code từ keras sang pytorch deep learning\pytorch"
conda activate btl02
jupyter notebook
```

| Notebook PyTorch | Gốc Keras | Kết quả gốc |
|---|---|---|
| `0. Getting Started/getting_started_pytorch` | `getting_started` | kiểm GPU |
| `1. MLP/Func_apxm_MLP_pytorch` | `Func_apxm_MLP` | đồ thị lỗi theo A, B, C |
| `1. MLP/MNIST_MLP_pytorch` | `MNIST_MLP` | acc 0.9648 |
| `2. CNN + MLP/1_MNIST_CNN_pytorch` | `1_MNIST_CNN` | acc 0.9884 |
| `2. CNN + MLP/2_Human_Gender_CNN_pytorch` | `2_Human_Gender_CNN` | acc 0.8264 |
| `2. CNN + MLP/3_Emotion_Reg_CNN_pytorch` | `3_Emotion_Reg_CNN` | acc 0.5954 |
| `3. CNN + LSTM/MNIST_CNN_LSTM_pytorch` | `MNIST_CNN_LSTM` | MAE 0.6502 |
| `4. Training Utilities/2_Human_Gender_CNN_pytorch` | cùng tên | acc 0.8773 |
| `4. Training Utilities/3_Emotion_Reg_CNN_pytorch` | cùng tên | acc 0.6163 |
| `5. Others/1_segnet_pytorch` | `1_segnet` | pixel acc 0.7732 |
| `5. Others/2_obj_detection_pytorch` | `2_obj_detection` | 0.9523 |

`Iris_Bayesian_decision_making` (sklearn) và `colab_getting_started` không dùng Keras nên không chuyển.
