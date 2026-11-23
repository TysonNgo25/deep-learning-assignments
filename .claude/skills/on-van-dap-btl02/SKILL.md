---
name: on-van-dap-btl02
description: Hỏi đáp tương tác để ôn vấn đáp BTL02 Emotion Recognition - hỏi từng câu về data.py, models.py, train.py, các thực nghiệm E1–E6, filter conv cuối, Grad-CAM, mô hình cuối, chấm và gợi ý. Dùng khi nói "ôn vấn đáp BTL02", "hỏi bài tôi", hoặc khi soạn tài liệu cho từng thành viên đọc phần đứng tên mình.
---

# Ôn vấn đáp BTL02

## Chuẩn bị

1. Hỏi ôn phần nào (một lựa chọn): lõi PyTorch (`data.py`, `models.py`, `train.py`), một nhóm thực nghiệm, visualize (filter, feature map, Grad-CAM), mô hình cuối, hoặc phần đứng tên của một thành viên (theo `docs/Kế hoạch.md` mục 8).
2. Đọc code thật, `NhatKyThucNghiem.md`, `results/*/summary.csv`, câu hỏi đã có trong `CodetoTruc.md`. Có thể gọi agent `btl02-giam-khao-van-dap` soạn sẵn bộ câu.

## Cách hỏi

- Mỗi lượt **một câu**, đợi trả lời rồi chấm: đạt / gần đạt / chưa đạt (đúng không, giải thích được vì sao không, dẫn được code/số không).
- Xen kẽ câu "làm gì", "vì sao", câu bẫy. Ví dụ bẫy: "dùng PrivateTest để chọn epoch được không", "accuracy 65% có tốt không khi Happy chiếm 25%", "vì sao val_acc cao hơn train_acc ở đầu", "sigmoid sao học chậm", "filter conv cuối nhìn giống nhiễu có phải mô hình hỏng".
- Sai: gợi ý dần (hỏi lại → chỉ dòng code → ví dụ nhỏ → giải thích). Đúng: hỏi xoáy thêm.
- Số liệu trong đáp án lấy từ `results/`, không đoán.

## Kết thúc

Khoảng 10 câu: tóm tắt phần vững, phần cần ôn kèm file/dòng code. Muốn lưu thì thêm câu chưa vững vào mục "Câu hỏi vấn đáp có thể gặp" của ngày hôm nay trong `CodetoTruc.md`.
