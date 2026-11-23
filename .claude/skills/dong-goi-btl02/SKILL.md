---
name: dong-goi-btl02
description: Kiểm tra checklist nộp bài và nén file BTLON02_Emotion_HuynhNgocTruc_3123411312.zip cho BTL02 Emotion Recognition (hạn T7 28/11/2026). Dùng khi nói "đóng gói BTL02", "nén file nộp", "kiểm tra trước khi nộp", "còn thiếu gì".
---

# Đóng gói và nộp BTL02

## 1. Kiểm tự động

Từ thư mục `BTL02 Emotion Recognition`:

```powershell
& "C:\Users\hntruc\anaconda3\envs\btl02\python.exe" "..\.claude\skills\dong-goi-btl02\check_submission.py"
```

[check_submission.py](check_submission.py) kiểm:

- Đủ file bắt buộc: README, requirements, baseline.yaml, các file `src/`, `experiments/`, `NhatKyThucNghiem.md`, `results/final/best_model.pt`, `final_metrics.json`, hình filter conv cuối và confusion matrix, báo cáo `.docx` + `.pdf`, slide `.pptx`.
- Mỗi nhóm E1–E6 có `summary.csv`; nhật ký có ít nhất 10 run đã ghi.
- Không còn `tensorflow`/`keras`, không có đường dẫn tuyệt đối.
- Mỗi file `.py` có dòng `Phụ trách:` (đề yêu cầu ghi rõ phân công trong code).
- Liệt kê các dòng nhắc `PrivateTest` ngoài `run_final.py`/`evaluate.py` để đọc lại (có thể rò rỉ test).
- `requirements.txt` chưa ghim phiên bản thì cảnh báo.

Mã thoát 1 nếu có LỖI.

## 2. Kiểm bằng tay

- [ ] Báo cáo có đủ 4 yêu cầu của đề, đặc biệt hình **filter lớp conv cuối**.
- [ ] Bảng phân công + tỷ lệ cộng đủ 100% có trong báo cáo, slide và khớp `docs/Kế hoạch.md` mục 8.
- [ ] Mỗi thành viên đã nhận và đọc phần đứng tên mình.
- [ ] README có lệnh chạy lại từ đầu.
- [ ] Gọi agent `btl02-cham-rubric` chấm thử lần cuối.

## 3. Nén

```powershell
& "C:\Users\hntruc\anaconda3\envs\btl02\python.exe" "..\.claude\skills\dong-goi-btl02\check_submission.py" --zip "BTLON02_Emotion_HuynhNgocTruc_3123411312.zip"
```

Zip ghi ra gốc repo. Script bỏ `data/` (fer2013.csv), `Yêu cầu/`, `CodetoTruc.md`, `docs/Kế hoạch.md`, `Kế hoạch.md`, `checkpoints/`, `__pycache__`. In danh sách file đã nén để Trúc xem. Không commit file zip.
