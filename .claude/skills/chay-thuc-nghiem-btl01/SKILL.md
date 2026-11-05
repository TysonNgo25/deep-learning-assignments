---
name: chay-thuc-nghiem-btl01
description: Cách chạy code BTL01 an toàn trên máy Trúc (Windows, conda env btl01) - sinh dữ liệu, vẽ hình, chạy thử baseline, chạy lại một phần thực nghiệm A/B/C - mà không ghi đè kết quả đã nộp và không đánh giá test thêm lần nào. Dùng trước khi chạy bất kỳ lệnh python nào trong BTL01 - Function Approximation.
---

# Chạy code BTL01

## Môi trường

- Python của env: `C:\Users\hntruc\anaconda3\envs\btl01\python.exe` (Git Bash: `/c/Users/hntruc/anaconda3/envs/btl01/python.exe`).
- Luôn đứng ở thư mục `BTL01 - Function Approximation` và chạy dạng module (`python -m src.data`) để import được `src`.
- Script tạm ngoài repo thì đặt `PYTHONPATH` trỏ vào thư mục BTL01 trước khi chạy.
- PowerShell in tiếng Việt lỗi font thì đặt `$env:PYTHONIOENCODING = "utf-8"`.

Mẫu PowerShell:

```powershell
Set-Location "D:\Data\GitHub\deep-learning-assignments\BTL01 - Function Approximation"
$py = "C:\Users\hntruc\anaconda3\envs\btl01\python.exe"
& $py -m scripts.run_baseline
```

## Lệnh có sẵn

| Lệnh | Làm gì | Ghi vào |
| --- | --- | --- |
| `-m src.data` | Sinh lại dữ liệu (seed 42) | `data/` |
| `-m scripts.plot_data` | Hình dữ liệu cho mục 2 | `report/figures/` |
| `-m scripts.run_baseline` | Train thử baseline 2×16, seed 0, không dùng test | `results/baseline/` |
| `-m experiments.run_A --values 200 400 --seeds 0` | Chạy lại một phần A | `results/A/summary_partial.csv` |
| `-m experiments.run_B --values 1 4 --seeds 0` | Chạy lại một phần B | `results/B/summary_partial.csv` |
| `-m experiments.run_C --values 1 --seeds 0` | Chạy lại một phần C | `results/C/summary_partial.csv` |

Chạy một phần vẫn ghi đè file log của đúng giá trị/seed đó (`results/<nhóm>/logs/<nhóm>_<giá trị>_seed<k>.csv`). Cùng máy thì số liệu trùng, chỉ cột thời gian (`elapsed_seconds`, `train_time_s`) đổi. Chạy xong mà không cần giữ thì khôi phục bằng `git restore`.

## Không được làm nếu Trúc chưa đồng ý rõ ràng

- **Không chạy `scripts/run_final.py`.** Mô hình cuối đã chọn và đã đánh giá test một lần; chạy lại là đánh giá test lần hai.
- Không chạy runner A/B/C đầy đủ (không có `--values`): sẽ ghi đè `summary.csv`, log và hình đã dùng trong báo cáo.
- Không chạy lại `src.data`: dữ liệu đã chốt, sinh lại có thể làm mọi log không còn khớp.
- Không sửa `configs/baseline.yaml`.

## Sau khi chạy

Chạy `git status --short .` và báo Trúc file nào trong `results/` hoặc `data/` bị đổi. Nếu đổi ngoài ý muốn, đưa lệnh `git restore <file>` để Trúc tự chạy.
