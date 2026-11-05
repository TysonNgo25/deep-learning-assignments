---
name: viet-bao-cao-btl01
description: Viết, sửa hoặc ghép các mục báo cáo BTL01 trong BTL01 - Function Approximation/report/ (section_*.md) theo quy ước số liệu, đánh số hình/bảng và yêu cầu của đề. Dùng khi viết nháp một mục, sửa câu chữ, ghép báo cáo bản 1, đánh số lại hình/bảng, hoặc chuẩn bị chuyển sang Word.
---

# Viết và ghép báo cáo BTL01

## Bố cục

Đề yêu cầu 6 mục, 10–20 trang, xuất Word và PDF. File hiện có trong `report/`:

| Mục | File | Người viết |
| --- | --- | --- |
| 1. Giới thiệu | `section_1_intro.md` | Trúc |
| 2. Dữ liệu | `section_2_data.md` | Trúc |
| 3. Mô hình và cài đặt | `section_3_model.md` | Khang |
| 4.2 Kết quả A, B | `section_4_2_experiments_AB.md` | Trung |
| 4.3 Kết quả C | `section_4_3_experiment_C.md` | Nhân |
| 5. Thảo luận, 6. Kết luận | chưa có | Trung, Nhân (5), Trúc (6) |

Khi sửa mục của người khác, chỉ sửa lỗi số liệu, đánh số, chính tả; nội dung lớn thì ghi lại để Trúc báo người viết.

## Quy ước số liệu

- MSE ghi dạng số mũ, ví dụ `1.68e-07`; trong một bảng giữ cùng số chữ số có nghĩa. Báo kèm RMSE và so với đỉnh hàm (≈ 0.269) để người đọc hình dung độ lớn.
- Kết quả nhiều seed ghi `mean ± std` (độ lệch chuẩn mẫu, 3 seed 0, 1, 2).
- Mọi con số phải truy được về một file trong `results/` hoặc `data/`. Cuối mỗi mục có đoạn "Nguồn số liệu" liệt kê file.
- Không viết câu nào ngụ ý tập test được dùng để chọn cấu hình, chọn seed hay dừng sớm. Test chỉ xuất hiện ở hai chỗ: từng lượt C (đề yêu cầu) và đánh giá mô hình cuối (`results/final/final_metrics.json`).
- Số của mô hình cuối lấy đúng Bảng 1.3 ở mục 1.5, không tự tính lại.

## Đánh số hình và bảng

- Đánh theo mục: Hình 2.1, Hình 2.2, Bảng 4.1... Chú thích bảng đặt trên bảng, chú thích hình đặt dưới hình.
- Mỗi hình/bảng phải được nhắc trong văn trước khi xuất hiện ("Bảng 4.2 cho thấy...").
- Khi ghép, rà lại toàn bộ: không trùng số, không nhảy số, tham chiếu chéo trỏ đúng.
- Trục đồ thị phải có nhãn và đơn vị; trục MSE dùng thang log nếu trải nhiều bậc.

## Văn phong

- Tiếng Việt, câu ngắn, nói thẳng kết quả. Mỗi nhận xét đi kèm con số làm căn cứ.
- Không dùng câu sáo kiểu "đóng vai trò quan trọng", "không thể phủ nhận", "nhìn chung có thể thấy". Không dùng gạch ngang dài để nối ý.
- Thuật ngữ tiếng Anh để trong ngoặc ở lần đầu: quá khớp (overfitting), dừng sớm (early stopping).
- Kết luận chỉ trong phạm vi thực nghiệm (miền, số mẫu, kiến trúc đã thử); nêu hạn chế.

## Sau khi viết

Gọi agent `btl01-doi-chieu-so-lieu` để đối chiếu số trong mục vừa viết với log, sửa các chỗ "Lệch" hoặc "Không truy được" trước khi báo xong.
