---
description: Báo tiến độ BTL03 so với kế hoạch và checklist nộp, còn mấy ngày, notebook nào có nguy cơ trễ
---

Báo tiến độ BTL03 (Chuyển code Keras sang PyTorch):

1. Đối chiếu `docs/Kế hoạch.md` mục 7 và mục 10 với repo: `pytorch/common.py`, 11 notebook `_pytorch` (có output chưa), `results/so_sanh.csv`, `results/<notebook>/`, `report/`.
2. Chạy kiểm tự động từ thư mục BTL03 (env btl02):
   `python ../.claude/skills/dong-goi-btl03/check_submission.py`
3. Trả lời một bảng: Notebook | Đã viết | Đã chạy (thiết bị) | Tham số khớp Keras | Kết quả Keras ↔ PyTorch | Ghi chú. Sau bảng: việc có nguy cơ trễ và phương án dự phòng theo mục 7.
4. Tick ô trạng thái trong mục 7 cho ngày đã xong (hỏi Trúc trước khi sửa).
