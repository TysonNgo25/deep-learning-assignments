---
paths:
  - "BTL02 Emotion Recognition/report/**"
  - "BTL02 Emotion Recognition/slides/**"
---

# Quy tắc báo cáo và slide BTL02

- Báo cáo phải có đủ 4 yêu cầu của đề: tóm tắt code; kết quả nhiều thực nghiệm; **hình filter lớp conv cuối**; phân tích bằng lời của nhóm. Bố cục theo `docs/Kế hoạch.md` mục 9.
- Accuracy ghi phần trăm 2 chữ số thập phân (`64.83%`); loss 4 chữ số (`0.9712`). Nhiều seed ghi `mean ± std`.
- Mỗi con số truy được về `results/` hoặc `NhatKyThucNghiem.md`; cuối mỗi mục có đoạn "Nguồn số liệu".
- Không câu nào ngụ ý PrivateTest được dùng để chọn mô hình.
- Hình/bảng đánh số theo mục (Hình 3.1, Bảng 3.2); chú thích bảng trên bảng, hình dưới hình; nhắc trong văn trước khi xuất hiện; trục có nhãn.
- Báo accuracy kèm F1 từng lớp vì dữ liệu lệch lớp (Disgust rất ít).
- Văn phong: tiếng Việt, câu ngắn, mỗi nhận xét kèm con số. Không câu sáo, không gạch ngang dài nối ý. Thuật ngữ Anh để trong ngoặc ở lần đầu.
- Bảng phân công (thành viên, MSV, nhiệm vụ, tỷ lệ cộng đủ 100%) đặt ở phụ lục báo cáo và slide cuối, khớp `docs/Kế hoạch.md` mục 8.
