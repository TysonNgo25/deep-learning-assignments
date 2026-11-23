---
name: dong-goi-btl03
description: Kiểm tra checklist nộp bài và nén file zip cho BTL03 (Chuyển code Keras sang PyTorch, hạn T7 28/11/2026) - đủ 11 notebook có output, không còn keras trong ô code, so_sanh.csv đủ, báo cáo docx/pdf. Dùng khi nói "đóng gói BTL03", "nén file nộp BTL03", "kiểm tra trước khi nộp BTL03", "BTL03 còn thiếu gì".
---

# Đóng gói và nộp BTL03

## 1. Kiểm tự động

Từ thư mục `BTL03 - Chuyển code từ keras sang pytorch deep learning`:

```powershell
& "C:\Users\hntruc\anaconda3\envs\btl02\python.exe" "..\.claude\skills\dong-goi-btl03\check_submission.py"
```

[check_submission.py](check_submission.py) kiểm:

- Đủ file: README, requirements, `pytorch/common.py`, 11 notebook `_pytorch.ipynb`, `results/so_sanh.csv`, báo cáo `.docx` + `.pdf`.
- Mỗi notebook: có output (đã chạy), không có ô lỗi, ô code không import tensorflow/keras, không còn `CHAY_THU = True`.
- `so_sanh.csv` có dòng cho mỗi notebook.
- Không có đường dẫn tuyệt đối trong `.py`/`.ipynb`.

Mã thoát 1 nếu có LỖI.

## 2. Kiểm bằng tay

- [ ] Đã hỏi/chốt các câu ở `docs/Kế hoạch.md` mục 11 (cá nhân hay nhóm, tên zip, slide).
- [ ] Báo cáo có bảng tương ứng API, đủ 10 mục notebook, bảng tổng hợp, phân tích chênh lệch.
- [ ] Số trong báo cáo khớp `so_sanh.csv` (agent `btl03-doi-chieu-keras` với "và báo cáo").
- [ ] Gọi agent `btl03-cham-rubric` chấm thử lần cuối.

## 3. Nén

```powershell
& "C:\Users\hntruc\anaconda3\envs\btl02\python.exe" "..\.claude\skills\dong-goi-btl03\check_submission.py" --zip "BTLON03_KerasPyTorch_HuynhNgocTruc_3123411312.zip"
```

Đổi tên zip nếu thầy quy định khác. Zip ghi ra gốc repo (đã có trong `.gitignore` gốc: `BTLON03_*.zip`). Script bỏ `data/`, `Yêu cầu/`, `docs/`, `CodetoTruc.md`, `checkpoints/`, `runs/`, `*.pt`, `__pycache__`, `.ipynb_checkpoints`. In danh sách file đã nén để Trúc xem. Không commit file zip.
