---
paths:
  - "BTL03 - Chuyển code từ keras sang pytorch deep learning/report/**"
---

# Quy tắc báo cáo BTL03

- Bố cục theo `docs/Kế hoạch.md` mục 9: giới thiệu → bảng tương ứng API → từng notebook (3.1–3.11) → bảng tổng hợp → khó khăn → kết luận → phụ lục cách chạy.
- **Mỗi notebook (mục 3.x)** có đủ: bài toán và dữ liệu; đoạn Keras gốc và đoạn PyTorch tương ứng đặt cạnh nhau (chỉ đoạn quan trọng: khai báo mô hình, compile/fit hoặc vòng train, callback); điểm khác cần chú ý; số tham số Keras ↔ PyTorch; kết quả Keras ↔ PyTorch; 1 hình đường học.
- Số liệu lấy từ `results/so_sanh.csv` và `results/<notebook>/metrics.json`. Cuối mỗi mục có dòng "Nguồn số liệu".
- Accuracy ghi phần trăm 2 chữ số thập phân (`96.48%`); loss và MAE 4 chữ số (`0.6502`). Số tham số có dấu chấm ngăn nghìn (`1.817.610`).
- Chênh lệch giữa hai bản phải được giải thích (khởi tạo ngẫu nhiên, eps Adam, bias LSTM, momentum BatchNorm, dừng sớm), không viết "kết quả tương đương" khi không có số.
- Nhắc các điểm về dữ liệu đã kiểm: Gender lệch lớp 78,85% male (mốc đoán toàn male = 0.7885) và 662 ảnh val đen; Gender/Emotion/SegNet không chuẩn hóa như gốc.
- Ghi rõ notebook không chuyển (Iris dùng sklearn, colab_getting_started) và lý do.
- Hình/bảng đánh số theo mục (Hình 3.2, Bảng 4.1); chú thích bảng trên bảng, hình dưới hình; nhắc trong văn trước khi xuất hiện; trục có nhãn.
- Code trong báo cáo dùng font đơn cách, không chụp màn hình code.
- Văn phong: tiếng Việt, câu ngắn, mỗi nhận xét kèm con số. Không câu sáo, không gạch ngang dài nối ý. Thuật ngữ Anh để trong ngoặc ở lần đầu.
- Bìa: môn Học sâu, BTL03, Huỳnh Ngọc Trúc, 3123411312 (thêm Nhóm 5 nếu thầy yêu cầu làm nhóm, xem `docs/Kế hoạch.md` mục 11).
