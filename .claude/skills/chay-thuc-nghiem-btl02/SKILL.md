---
name: chay-thuc-nghiem-btl02
description: Cách cài môi trường và chạy code BTL02 Emotion Recognition an toàn - trên máy Trúc (Windows, conda env btl02, chỉ CPU) hoặc GPU Colab/Kaggle - gồm chạy thử baseline, chạy một nhóm thực nghiệm E1–E7, mô hình cuối, mà không ghi đè log đã ghi nhật ký và không đánh giá PrivateTest quá một lần. Dùng trước khi chạy bất kỳ lệnh python nào trong BTL02 Emotion Recognition.
---

# Chạy code BTL02

## 1. Môi trường trên máy Trúc

**Máy Trúc không có GPU NVIDIA** (không có `nvidia-smi`; GPU là AMD), torch chạy CPU. Env `btl02` đã tạo ngày 23/11: Python 3.11, torch 2.15 CPU, torchvision 0.29.2, đủ `requirements.txt`.

Cài lại từ đầu (Anaconda Prompt):

```bat
conda create -n btl02 python=3.11 -y
conda activate btl02
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r "BTL02 Emotion Recognition\requirements.txt"
```

`conda create` báo `PackagesNotFoundError: python=3.11` thì gần như chắc là máy thiếu bộ nhớ lúc giải phụ thuộc (kênh `defaults` có 3.11), tắt bớt ứng dụng rồi chạy lại.

Python của env: `C:\Users\hntruc\anaconda3\envs\btl02\python.exe` (Git Bash: `/c/Users/hntruc/anaconda3/envs/btl02/python.exe`). Đặt `PYTHONIOENCODING=utf-8` để in tiếng Việt.

**Máy thường thiếu RAM** (AMDChat ~3,7 GB, WSL ~3 GB, VS Code ~3 GB). Trên máy luôn thêm `--workers 0`; báo out of memory thì nhờ Trúc tắt AMDChat, `wsl --shutdown`, rồi chạy lại. Đo được: CPU mất ~0,6 s/batch train, một epoch đầy đủ ~5 phút, nên trên máy **chỉ chạy thử**.

Luôn đứng ở thư mục `BTL02 Emotion Recognition` và chạy dạng module:

```powershell
Set-Location "D:\Data\GitHub\deep-learning-assignments\BTL02 Emotion Recognition"
$py = "C:\Users\hntruc\anaconda3\envs\btl02\python.exe"
& $py -m src.train --run BASE --thu --workers 0
```

## 2. Chạy trên GPU Colab/Kaggle

Notebook `experiments/chay_colab_kaggle.ipynb` làm hết: `git clone` repo công khai `TysonNgo25/deep-learning-assignments`, lấy dữ liệu, chạy BASE và E1–E6, nén `results_*.zip` (không kèm `checkpoints/`, `_thu/`).

1. Trúc **commit + push** code mới nhất trước (notebook clone từ GitHub).
2. Kaggle (khuyên dùng): Add Input dataset `nicolejyt/facialexpressionrecognition`, Accelerator GPU, bật Internet, **Save Version → Save & Run All** để chạy nền ≤ 12 giờ; zip nằm ở tab Output.
   Colab: đặt `fer2013.csv` ở `MyDrive/fer2013.csv`; notebook nối `results/` vào `MyDrive/BTL02_results` nên ngắt kết nối không mất log.
3. Run đã có log thì `run_E*` tự bỏ qua, chạy lại ô là tiếp tục run còn thiếu.
4. Tải zip, giải nén đè vào thư mục BTL02, `git status --short results/` để xem file mới.
5. Ghi nhật ký từng run (`/btl02-ghi-run`), cột thời gian kèm loại GPU (lấy từ `results/<nhóm>/configs/<run>.json`, khóa `device`).

## 3. Lệnh

| Lệnh | Làm gì | Ghi vào |
| --- | --- | --- |
| `-m src.data` | Đọc `fer2013.csv`, kiểm 28.709/3.589/3.589, vẽ EDA | `results/eda/` |
| `-m src.models` | Bảng số tham số 4 kiến trúc × có/không BN | (in ra màn hình) |
| `-m src.train --run BASE --thu [--epochs 2 --max-batches 20]` | Chạy thử pipeline (mặc định 2 epoch × 20 batch); chỉ để kiểm code, **không** dùng để so lr/kiến trúc | `results/_thu/<run>_<giờ>/` (git bỏ qua) |
| `-m src.train --run BASE --change "baseline"` | Baseline đầy đủ | `results/BASE/` |
| `-m src.train --run <mã> --set nhóm.khóa=giá_trị --change "<mô tả>"` | Một run bất kỳ = baseline + override (`--set` lặp được, khóa sai báo lỗi) | `results/<nhóm>/` |
| `-m experiments.run_E1 [--runs E1-1 E1-3] [--thu] [--workers 0]` | Nhóm E1; tương tự `run_E3` … `run_E6` | `results/E1/` … `summary.csv`, `figures/curves.png` |
| `-m experiments.run_E2 [--arch medium_01]` | E2-1: kiến trúc tốt nhất E1 (tự đọc `results/E1/summary.csv`) + BN | `results/E2/` |
| `-m src.evaluate --ckpt <file .pt> [--out …]` | Đánh giá checkpoint **chỉ trên validation** (dòng lệnh không đọc được PrivateTest): `metrics.json`, `classification_report.csv`, `predictions_val.csv`, confusion matrix, ca đúng/sai | `results/<nhóm>/eval/<mã>/` |
| `-m experiments.run_final` | (chưa viết) Train F-0/1/2 rồi gọi `src.evaluate.evaluate_final`: chọn seed theo val_loss, đánh giá PrivateTest **một lần**, ghi `final_metrics.json` trước khi vẽ hình; gọi lần hai báo lỗi | `results/final/` |
| `-m src.visualize [--ckpt <file .pt>] [--out …] [--steps 150]` | Chỉ dùng validation. Mặc định `results/final/best_model.pt`. Ghi `last_conv_filters.png` (activation maximization), `last_conv_top_images.png`, `last_conv_weights.png`, `feature_maps.png`, `gradcam.png` (conv đầu/giữa/cuối × 7 cảm xúc). CPU ~3–4 phút, GPU < 1 phút | `results/final/figures/` (hoặc `<nhóm>/figures/`) |

Mỗi run ghi: `logs/<run>.csv` (theo epoch), `checkpoints/<run>.pt` (val_loss nhỏ nhất, git bỏ qua), `configs/<run>.json` (config đầy đủ, máy, mean/std train), thêm 1 dòng `summary.csv`.

## 4. Không được làm nếu Trúc chưa đồng ý rõ ràng

- **Không chạy `run_final` khi `results/final/final_metrics.json` đã có** (hook chặn). Đó là đánh giá test lần hai.
- Không đọc PrivateTest trong bất kỳ script nào khác ngoài `run_final`/`evaluate` mô hình cuối (`get_dataloaders(..., include_test=True)`).
- Không ghi đè log: `train.py` báo `FileExistsError` nếu log đã có; chạy lại thì đặt mã `<run>_r2`.
- Không sửa `configs/baseline.yaml` từ 25/11 (hook hỏi lại).
- Không chạy run đầy đủ trên máy Trúc (CPU, ~5 phút/epoch); chạy thử bằng `--thu`.

## 5. Sau khi chạy

`git status --short .` để báo Trúc file nào trong `results/` đổi. Ghi nhật ký thực nghiệm. Đổi ngoài ý muốn thì đưa lệnh `git restore <file>` để Trúc tự chạy.
