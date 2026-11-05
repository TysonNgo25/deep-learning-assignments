---
name: on-van-dap-btl01
description: Hỏi đáp tương tác để Trúc ôn vấn đáp BTL01 - hỏi từng câu về model.py, train.py, dữ liệu, thực nghiệm A/B/C và cách chọn mô hình, chấm câu trả lời và gợi ý. Dùng khi Trúc nói "ôn vấn đáp", "hỏi bài tôi", "kiểm tra tôi hiểu train.py chưa", hoặc trước buổi tập 6/11.
---

# Ôn vấn đáp BTL01

## Chuẩn bị

1. Hỏi Trúc muốn ôn phần nào (một lựa chọn): phần của Trúc (dữ liệu, mục 1–2, README), lõi PyTorch (`model.py`, `train.py`, `evaluate.py`), thực nghiệm A/B/C, chọn mô hình cuối, hay trộn tất cả.
2. Đọc code thật của phần đó và câu hỏi đã có trong `CodetoTruc.md` (mục "Câu hỏi vấn đáp có thể gặp") và `tra_loi_van_dap_K.md`. Có thể gọi agent `btl01-giam-khao-van-dap` để soạn sẵn bộ câu.

## Cách hỏi

- Mỗi lượt chỉ hỏi **một câu**, đợi Trúc trả lời rồi mới chấm.
- Xen kẽ câu "làm gì", câu "vì sao", và câu bẫy. Cứ 3–4 câu thì có một câu nối hai phần (ví dụ dữ liệu và early stopping).
- Chấm mỗi câu theo 3 ý: đúng không, giải thích được vì sao không, có dẫn được code hoặc số liệu không. Báo ngắn: đạt / gần đạt / chưa đạt.
- Trả lời sai: không đưa đáp án ngay. Gợi ý theo thứ tự: hỏi lại cách khác → chỉ dòng code cần xem → đưa ví dụ nhỏ → cuối cùng mới giải thích.
- Trả lời đúng: hỏi tiếp một câu khó hơn cùng chủ đề, như thầy hỏi xoáy.
- Số liệu trong đáp án phải lấy từ `results/`, không đoán.

## Kết thúc

Sau khoảng 10 câu hoặc khi Trúc dừng: tóm tắt câu nào vững, câu nào cần ôn lại, kèm file/dòng code cần đọc lại. Nếu Trúc muốn lưu, thêm các câu chưa vững vào mục "Câu hỏi vấn đáp có thể gặp" của ngày hôm nay trong `CodetoTruc.md`.
