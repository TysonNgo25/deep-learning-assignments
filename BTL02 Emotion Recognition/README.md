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

(Bổ sung các lệnh train, thực nghiệm khi viết xong code.)

## Phân công

Xem `docs/Kế hoạch.md`, mục 8.
