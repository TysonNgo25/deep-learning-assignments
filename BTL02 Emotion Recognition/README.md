# BTL02 – Emotion Recognition (FER2013, CNN PyTorch) – Nhóm 5

Nhận dạng 7 cảm xúc khuôn mặt (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) trên ảnh xám 48×48 của FER2013 bằng CNN viết bằng PyTorch.

## Cài đặt

```
conda create -n btl02 python=3.11 -y
conda activate btl02
pip install -r requirements.txt
```

## Dữ liệu

Tải `fer2013.csv` từ Kaggle (Challenges in Representation Learning: Facial Expression Recognition Challenge) và đặt vào `data/fer2013.csv`.
Cột `Usage`: Training (28.709) dùng để train, PublicTest (3.589) làm validation, PrivateTest (3.589) làm test.

## Cấu trúc

| Thư mục / file | Nội dung |
|---|---|
| `configs/baseline.yaml` | Cấu hình baseline |
| `src/` | `paths.py`, `data.py`, `models.py`, `train.py`, `evaluate.py`, `visualize.py` |
| `experiments/` | Script chạy từng nhóm thực nghiệm E1–E6 và mô hình cuối |
| `results/` | Log CSV, `summary.csv`, hình, trọng số mô hình cuối |
| `NhatKyThucNghiem.md` | Nhật ký các lần chạy |
| `report/`, `slides/` | Báo cáo và slide |
| `docs/Kế hoạch.md` | Kế hoạch và phân công |

## Cách chạy

Đứng ở thư mục BTL02 và chạy dạng module:

| Lệnh | Làm gì | Kết quả |
|---|---|---|
| `python -m src.data` | Đọc `fer2013.csv`, kiểm số ảnh 28.709 / 3.589 / 3.589, vẽ EDA | `results/eda/` |
| `python -m src.models` | In số tham số của 4 kiến trúc (có/không BatchNorm) | màn hình |
| `python -m src.train --run BASE --thu` | Chạy thử pipeline 2 epoch × 20 batch | `results/_thu/` |
| `python -m src.train --run BASE` | Train baseline (`configs/baseline.yaml`) | `results/BASE/` |
| `python -m src.train --run <mã> --set train.lr=0.001 --change "..."` | Một run = baseline + ghi đè khóa config | `results/<nhóm>/` |
| `python -m experiments.run_E1` … `run_E6` | Chạy cả nhóm thực nghiệm, bảng tóm tắt, đồ thị | `results/E1/` … `results/E6/` |

Máy không có GPU thì thêm `--workers 0` khi chạy thử; thực nghiệm đầy đủ chạy trên GPU bằng `experiments/chay_colab_kaggle.ipynb` (Colab hoặc Kaggle).

Mỗi run ghi `logs/<mã>.csv` (theo epoch: `epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time_s`), `checkpoints/<mã>.pt` (mô hình có val_loss nhỏ nhất), `configs/<mã>.json` (config đầy đủ và máy chạy) và một dòng trong `summary.csv`. Validation là PublicTest; PrivateTest chỉ dùng một lần cho mô hình cuối.

## Phân công

Xem `docs/Kế hoạch.md`, mục 8.
