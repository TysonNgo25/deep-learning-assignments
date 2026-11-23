---
paths:
  - "BTL02 Emotion Recognition/results/**"
  - "BTL02 Emotion Recognition/NhatKyThucNghiem.md"
  - "BTL02 Emotion Recognition/experiments/**"
---

# Quy tắc thực nghiệm BTL02

- Mỗi run **chỉ đổi một biến** so với `configs/baseline.yaml`. Mã run theo `docs/Kế hoạch.md` mục 4: `BASE`, `E1-1` … `E6-3`, `E7-1`, `F-0/1/2`.
- Validation = PublicTest. Chọn cấu hình theo `val_acc` tại epoch có `val_loss` nhỏ nhất (ghi rõ trong báo cáo). PrivateTest chỉ chạy ở mô hình cuối, **một lần** (hook chặn chạy lại `run_final`).
- Chạy xong mỗi run: ghi ngay 1 dòng vào `NhatKyThucNghiem.md` (lệnh `/btl02-ghi-run <mã run>`). Số liệu lấy từ log CSV, không gõ tay.
- Không xóa, không ghi đè log đã ghi vào nhật ký. Chạy lại một run thì đặt hậu tố `_r2` và ghi dòng mới, nêu lý do.
- Run chạy trên Colab/Kaggle: tải `results/<nhóm>/` về đúng chỗ trong repo, ghi cột "Thời gian" kèm loại máy (ví dụ `412 s (T4)`).
- E1–E6: 1 seed (0). Mô hình cuối: 3 seed (0, 1, 2), báo `mean ± std` (độ lệch chuẩn mẫu).
