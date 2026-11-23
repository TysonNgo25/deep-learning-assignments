---
paths:
  - "BTL02 Emotion Recognition/results/**"
  - "BTL02 Emotion Recognition/NhatKyThucNghiem.md"
  - "BTL02 Emotion Recognition/experiments/**"
---

# Quy tắc thực nghiệm BTL02

- Mỗi run **chỉ đổi một biến** so với `configs/baseline.yaml`. Mã run theo `docs/Kế hoạch.md` mục 4: `BASE`, `E1-1` … `E6-3`, `E7-1`, `E7-2`, `F-0/1/2`. Một "biến" có thể là vài khóa config cùng nghĩa (E4-1 tắt cả lật/xoay/dịch, E6-1 bỏ cả hai dropout, E3-1 SGD đi kèm lr 1e-3); ghi đủ các khóa vào `--change`.
- **E7 là nhánh riêng** (làm nếu có GPU): VGG16, ảnh phóng 128×128×3, cấu hình theo `Yêu cầu/code/train.html` (SGD lr 4e-4 momentum 0,9 nesterov, FC 2048, dropout 0,1, xoay ±30°). E7-1 (train từ đầu) và E7-2 (pretrained ImageNet) chỉ khác nhau ở trọng số khởi tạo. So E7 với baseline chỉ để tham khảo, không tính là thí nghiệm một biến.
- Baseline (`configs/baseline.yaml`): Medium, ReLU, không BN, Adam **lr 1e-4**, batch 64, tối đa **60** epoch, patience **10**. Chốt 24/11, từ 25/11 hook hỏi lại khi sửa.
- Validation = PublicTest. Chọn cấu hình theo `val_acc` tại epoch có `val_loss` nhỏ nhất (ghi rõ trong báo cáo). PrivateTest chỉ chạy ở mô hình cuối, **một lần** (hook chặn chạy lại `run_final`).
- Chạy xong mỗi run: ghi ngay 1 dòng vào `NhatKyThucNghiem.md` (lệnh `/btl02-ghi-run <mã run>`). Số liệu lấy từ log CSV, không gõ tay.
- Không xóa, không ghi đè log đã ghi vào nhật ký. Chạy lại một run thì đặt hậu tố `_r2` và ghi dòng mới, nêu lý do.
- Run chạy trên Colab/Kaggle: tải `results/<nhóm>/` về đúng chỗ trong repo, ghi cột "Thời gian" kèm loại máy (ví dụ `412 s (T4)`).
- E1–E6: 1 seed (0). Mô hình cuối: 3 seed (0, 1, 2), báo `mean ± std` (độ lệch chuẩn mẫu).
