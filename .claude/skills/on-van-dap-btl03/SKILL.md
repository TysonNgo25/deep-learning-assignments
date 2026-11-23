---
name: on-van-dap-btl03
description: Hỏi đáp tương tác để ôn vấn đáp BTL03 (Chuyển code Keras sang PyTorch) - đưa một đoạn Keras gốc, hỏi bên PyTorch viết thế nào và vì sao, về vòng train, callback, LSTM/TimeDistributed, BatchNorm, SegNet, số tham số, chênh lệch kết quả; chấm và gợi ý. Dùng khi nói "ôn vấn đáp BTL03", "hỏi bài tôi phần Keras/PyTorch".
---

# Ôn vấn đáp BTL03

## Chuẩn bị

1. Hỏi ôn phần nào (một lựa chọn): `common.py` (vòng train, callback), một notebook, một chủ đề (LSTM, BatchNorm, augmentation, lấy đặc trưng tầng giữa), hoặc toàn bài.
2. Đọc notebook gốc, bản `_pytorch`, `common.py`, `results/so_sanh.csv`, câu hỏi đã có trong `CodetoTruc.md` của BTL03. Có thể gọi agent `btl03-giam-khao-van-dap` soạn sẵn bộ câu.

## Cách hỏi

- Mỗi lượt **một câu**, đợi trả lời rồi chấm: đạt / gần đạt / chưa đạt (đúng không, giải thích được vì sao không, chỉ được ô code/số không).
- Kiểu câu đặc trưng của bài này: dán một đoạn Keras gốc và hỏi "viết lại bằng PyTorch", "bên mình đặt ở ô nào", "khác gì".
- Xen câu bẫy: "sao không để softmax ở lớp cuối", "Gender 87% có tốt không" (mốc 78,85%), "EarlyStopping của Keras có khôi phục trọng số tốt nhất không", "sao LSTM bên mình nhiều hơn 1.024 tham số", "momentum BN 0.99 bên Keras là bao nhiêu bên PyTorch", "accuracy của obj_detection dùng MSE nghĩa là gì", "quên `model.eval()` thì Dropout/BN ra sao".
- Sai: gợi ý dần (hỏi lại → chỉ ô code → ví dụ nhỏ → giải thích). Đúng: hỏi xoáy thêm.
- Số liệu trong đáp án lấy từ `results/`, không đoán.

## Kết thúc

Khoảng 10 câu: tóm tắt phần vững, phần cần ôn kèm notebook/ô code. Muốn lưu thì thêm câu chưa vững vào mục "Câu hỏi vấn đáp có thể gặp" của ngày hôm nay trong `CodetoTruc.md`.
