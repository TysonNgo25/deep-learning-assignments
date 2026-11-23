---
name: viet-bao-cao-btl03
description: Viết, sửa, ghép báo cáo BTL03 (Chuyển code Keras sang PyTorch) trong report/ - bảng tương ứng API, từng notebook Keras ↔ PyTorch, bảng tổng hợp từ results/so_sanh.csv, phân tích chênh lệch - rồi chuyển sang Word/PDF. Dùng khi viết nháp một mục, ghép báo cáo, xuất .docx/.pdf.
---

# Báo cáo BTL03

Quy ước số liệu, đánh số, văn phong: `.claude/rules/btl03-bao-cao.md`.

## Bố cục (`report/section_*.md`, ghép thành `.docx` + `.pdf`)

| Mục | File | Nội dung |
| --- | --- | --- |
| Bìa + mục lục | — | Môn Học sâu, BTL03 "Chuyển code từ Keras sang PyTorch", Huỳnh Ngọc Trúc, 3123411312 |
| 1 | `section_1_gioi_thieu.md` | Đề, phạm vi 11 notebook (bảng mục 3 Kế hoạch), 2 notebook không chuyển; Keras khác PyTorch về triết lý (khai báo + `fit` vs `nn.Module` + vòng lặp tự viết; NHWC vs NCHW) |
| 2 | `section_2_tuong_ung_api.md` | Bảng tương ứng (Kế hoạch mục 4); `common.py`: các hàm và vai trò (thay `compile/fit/evaluate/predict`, callback) |
| 3 | `section_3_notebook.md` | 3.1 Func_apxm_MLP · 3.2 MNIST_MLP · 3.3 MNIST_CNN · 3.4 Gender · 3.5 Emotion · 3.6 CNN+LSTM · 3.7 Gender + callback · 3.8 Emotion + augmentation · 3.9 SegNet · 3.10 phát hiện xe (· 3.0 getting_started gộp vào mục 2) |
| 4 | `section_4_tong_hop.md` | Bảng tổng hợp từ `so_sanh.csv` (tham số + kết quả hai bản + chênh lệch); hình cột so sánh; phân tích chênh lệch |
| 5 | `section_5_kho_khan.md` | Khó khăn và cách xử lý (lấy từ mục "Khác Keras ở đâu" trong `CodetoTruc.md`); nhận xét dữ liệu (Gender lệch lớp 78,85%, 662 ảnh đen; dữ liệu không chuẩn hóa) |
| 6 | `section_6_ket_luan.md` | Kết luận, hạn chế |
| Phụ lục | `phu_luc.md` | Cách chạy lại (README), danh sách file nộp, tài liệu tham khảo (slide 01, 02 của thầy, tài liệu PyTorch) |

## Mỗi mục 3.x có

1. Bài toán, dữ liệu, kết quả gốc (1–2 câu).
2. Đoạn Keras gốc và đoạn PyTorch đặt cạnh nhau (bảng 2 cột hoặc 2 khối code liền nhau): khai báo mô hình; phần huấn luyện (compile/fit ↔ loss/optimizer/fit của `common.py`); callback hoặc phần đặc biệt (TimeDistributed, BN, heatmap).
3. Điểm khác cần chú ý (gạch đầu dòng ngắn, mỗi dòng có lý do).
4. Bảng nhỏ: tham số Keras | PyTorch; kết quả Keras | PyTorch; epoch; thời gian; thiết bị.
5. Hình đường học `results/<notebook>/curves.png` (và hình riêng: PCA, ảnh phân đoạn, heatmap, vòng quét A/B/C).
6. Nhận xét 2–4 câu có số. Dòng "Nguồn số liệu".

## Xuất Word/PDF

Ghép các `section_*.md` theo thứ tự. Đọc skill docx có sẵn (anthropic-skills:docx) trước khi dựng `.docx`, hoặc dùng `pandoc` nếu máy có. Tên file: `report/BTL03_KerasPyTorch_HuynhNgocTruc_3123411312.docx` và `.pdf` (đổi nếu thầy quy định khác, Kế hoạch mục 11). Code trong báo cáo dùng font đơn cách cỡ 9–10.

## Sau khi viết

Gọi agent `btl03-doi-chieu-keras` với "và báo cáo" để đối chiếu số; sửa chỗ lệch trước khi báo xong.
