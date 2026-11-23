---
name: chay-thuc-nghiem-btl02
description: Cách cài môi trường và chạy code BTL02 Emotion Recognition an toàn - trên máy Trúc (Windows, conda env btl02) hoặc GPU Colab/Kaggle - gồm chạy thử baseline, chạy một nhóm thực nghiệm E1–E7, mô hình cuối, mà không ghi đè log đã ghi nhật ký và không đánh giá PrivateTest quá một lần. Dùng trước khi chạy bất kỳ lệnh python nào trong BTL02 Emotion Recognition.
---

# Chạy code BTL02

## 1. Môi trường trên máy Trúc

Chỉ cài một lần (Anaconda Prompt):

```bat
conda create -n btl02 python=3.11 -y
conda activate btl02
nvidia-smi
```

- Có GPU NVIDIA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- Không có GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

Sau đó `pip install -r "BTL02 Emotion Recognition\requirements.txt"`.

Python của env: `C:\Users\hntruc\anaconda3\envs\btl02\python.exe` (Git Bash: `/c/Users/hntruc/anaconda3/envs/btl02/python.exe`). PowerShell lỗi font thì đặt `$env:PYTHONIOENCODING = "utf-8"`.

Luôn đứng ở thư mục `BTL02 Emotion Recognition` và chạy dạng module:

```powershell
Set-Location "D:\Data\GitHub\deep-learning-assignments\BTL02 Emotion Recognition"
$py = "C:\Users\hntruc\anaconda3\envs\btl02\python.exe"
& $py -m src.data          # kiểm tra dữ liệu, vẽ EDA
```

## 2. Chạy trên GPU Colab/Kaggle

CPU chạy một run CNN trên 28.709 ảnh mất rất lâu, nên các nhóm E1–E7 và mô hình cuối chạy trên GPU:

1. Nén thư mục BTL02 (không kèm `results/`, không kèm `Yêu cầu/`) cùng `data/fer2013.csv`, tải lên Colab (Runtime → T4 GPU) hoặc Kaggle (Accelerator GPU).
2. Giải nén, `cd` vào thư mục, `pip install -r requirements.txt` (torch có sẵn).
3. Chạy đúng lệnh như trên máy: `python -m experiments.run_E1`.
4. Nén `results/<nhóm>/` (bỏ `checkpoints/` nếu không cần) tải về, giải nén đè đúng chỗ trong repo.
5. Ghi nhật ký từng run (`/btl02-ghi-run`), cột thời gian kèm loại máy.

## 3. Lệnh (cập nhật khi viết xong code)

| Lệnh | Làm gì | Ghi vào |
| --- | --- | --- |
| `-m src.data` | Đọc `fer2013.csv`, kiểm số ảnh, vẽ histogram lớp và ảnh mẫu | `results/eda/` |
| `-m src.train --run BASE --epochs 2` | Chạy thử baseline 2 epoch để kiểm pipeline | `results/BASE/` |
| `-m src.train --run BASE` | Baseline đầy đủ | `results/BASE/` |
| `-m experiments.run_E1 [--runs E1-1 E1-3]` | Nhóm E1 (hoặc vài run) | `results/E1/` |
| … `run_E2` đến `run_E7` | Các nhóm còn lại | `results/E2/` … |
| `-m experiments.run_final` | Mô hình cuối 3 seed + đánh giá PrivateTest **một lần** | `results/final/` |
| `-m src.visualize` | Filter conv cuối, feature map, Grad-CAM từ `best_model.pt` | `results/final/figures/` |

## 4. Không được làm nếu Trúc chưa đồng ý rõ ràng

- **Không chạy `run_final` khi `results/final/final_metrics.json` đã có** (hook chặn). Đó là đánh giá test lần hai.
- Không đọc PrivateTest trong bất kỳ script nào khác ngoài `run_final`/`evaluate` mô hình cuối.
- Không chạy lại một run đã ghi nhật ký mà ghi đè log; chạy lại thì đặt mã `<run>_r2`.
- Không sửa `configs/baseline.yaml` sau 24/11 (hook hỏi lại).
- Chạy thử luôn dùng `--epochs 1` hoặc `2` trước khi chạy đầy đủ.

## 5. Sau khi chạy

`git status --short .` để báo Trúc file nào trong `results/` đổi. Ghi nhật ký thực nghiệm. Đổi ngoài ý muốn thì đưa lệnh `git restore <file>` để Trúc tự chạy.
