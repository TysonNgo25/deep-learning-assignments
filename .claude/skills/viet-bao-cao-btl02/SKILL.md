---
name: viet-bao-cao-btl02
description: Viết, sửa, ghép báo cáo và slide BTL02 Emotion Recognition trong report/ và slides/ theo 4 yêu cầu của đề (tóm tắt code, nhiều thực nghiệm, filter conv cuối, phân tích), quy ước số liệu và bảng phân công. Dùng khi viết nháp một mục, ghép báo cáo, chuyển sang Word/PDF, hoặc làm slide.
---

# Báo cáo và slide BTL02

## Bố cục báo cáo (`report/section_*.md`, ghép thành `.docx` + `.pdf`)

| Mục | File | Nội dung | Đứng tên |
| --- | --- | --- | --- |
| Bìa + mục lục | — | Tên môn, BTL02, Nhóm 5, 4 thành viên + MSV | Trúc |
| 1 | `section_1_intro_data.md` | Bài toán FER, FER2013, EDA (histogram lớp, ảnh mẫu) | Trúc |
| 2 | `section_2_code.md` | **Tóm tắt code**: data, models, train, evaluate, visualize; sơ đồ pipeline | Trúc |
| 3 | `section_3_experiments.md` | Baseline + E1–E6 (E7 nếu có): bảng, đồ thị loss/acc, nhận xét | Khang (3.1–3.2), Trung (3.3–3.4), Nhân (3.5–3.6) |
| 4 | `section_4_visualize.md` | **Filter conv cuối**, feature map, **Grad-CAM (bắt buộc)**, confusion matrix valid + test, ca đúng/ca sai | Khang (4.1), Trung (4.2), Nhân (Grad-CAM) |
| 5 | `section_5_final.md` | Mô hình cuối 3 seed, kết quả PrivateTest, so với mức 69–71%, hạn chế, kết luận | Trúc |
| Phụ lục | `appendix.md` | Bảng phân công + tỷ lệ, tài liệu tham khảo | Trúc |

Quy ước số liệu, đánh số hình/bảng, văn phong: xem `.claude/rules/btl02-bao-cao.md`.

## Hình bắt buộc ở mục 4 (theo `docs/Kế hoạch.md` mục 5 và `Yêu cầu/code/valid_vgg16.html`)

| Hình | File | Ghi chú |
| --- | --- | --- |
| Filter lớp conv cuối | `results/final/figures/last_conv_filters.png` | Đề bắt buộc |
| Grad-CAM | `results/final/figures/gradcam.png` | **Bắt buộc**: 3 hàng (conv đầu, giữa, cuối) × 7 cột (mỗi cảm xúc 1 ảnh **đoán đúng**) |
| Confusion matrix | `results/final/figures/confusion_matrix.png` | **Valid và test đặt cạnh nhau**; nhãn theo thứ tự gốc FER2013 0–6, không theo thứ tự chữ cái như code mẫu |
| Ca đúng / ca sai | `true_cases.png`, `wrong_cases.png` | Ảnh lấy từ đúng tập đang nói (không lặp lỗi `x_train[valid_idx]` của code mẫu) |

E7 (VGG16 128×128×3), nếu chạy, đặt ở mục 3.7 với ghi chú "khác baseline nhiều biến, chỉ để tham khảo"; so E7-1 với E7-2 để nói về tác dụng của pretrained.

## Mỗi mục thực nghiệm (3.x) có

1. Mục đích: đổi biến gì, giữ nguyên gì.
2. Bảng: mã run, cấu hình, best epoch, val_loss, val_acc, train_acc, thời gian.
3. Đồ thị val_acc/val_loss theo epoch của các run trong nhóm (một hình).
4. Nhận xét 3–5 câu có số: run nào tốt nhất, vì sao (giải thích bằng lý thuyết: gradient biến mất với sigmoid, BN ổn định phân phối, augmentation giảm quá khớp…).

## Slide (`slides/BTL02_Nhom05.pptx`, 12–15 trang)

Đề → dữ liệu → pipeline code → baseline → E1…E6 (mỗi nhóm 1 trang) → mô hình cuối → filter/Grad-CAM → confusion matrix → kết luận → phân công. Trang nào cũng ghi người trình bày theo bảng phân công.

## Sau khi viết

Gọi agent `btl02-doi-chieu-so-lieu` để đối chiếu số với log, sửa chỗ "Lệch"/"Không truy được" trước khi báo xong.
