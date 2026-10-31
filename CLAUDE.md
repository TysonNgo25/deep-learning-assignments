# Quy tắc của Trúc

## 1. Nhật ký `CodetoTruc.md`

File: [BTL01 - Function Approximation/CodetoTruc.md](BTL01%20-%20Function%20Approximation/CodetoTruc.md)

- **Ngày nào có commit trong BTL01 thì ngày đó phải có một mục trong `CodetoTruc.md`.** Không có ngoại lệ, kể cả commit nhỏ hay commit revert.
- Mỗi ngày một mục. Ngày có nhiều commit thì gộp chung vào mục của ngày đó.
- Mục mới thêm ở **cuối file**, ngăn với mục trước bằng `---`.
- Tiêu đề theo mẫu: `## <Thứ> dd/mm/yyyy – <việc chính trong ngày>`, ví dụ `## T5 29/10/2026 – Viết src/data.py, sinh dữ liệu`.
- Viết bằng tiếng Việt, giải thích kỹ (để Trúc ôn vấn đáp và viết báo cáo).

### Nội dung mỗi mục

Theo bố cục của mục ngày 29/10, bỏ phần nào không có nội dung:

1. **Việc theo kế hoạch**: đối chiếu với [Kế hoạch.md](BTL01%20-%20Function%20Approximation/K%E1%BA%BF%20ho%E1%BA%A1ch.md).
2. **Commit trong ngày**: hash ngắn + message của từng commit, kèm một câu nói commit đó làm gì.
3. **Kết quả**: file sinh ra / file đã sửa, lệnh để chạy.
4. **Giải thích từng hàm**: hàm làm gì và vì sao viết như vậy.
5. **Đã kiểm tra**: kiểm tra gì, ra số liệu nào.
6. **Cách tự kiểm tra lại**: lệnh chạy trong Anaconda Prompt và kết quả đúng phải thấy.
7. **Số liệu cho báo cáo** (nếu có).
8. **Câu hỏi vấn đáp có thể gặp**.
9. **Còn treo**: checkbox `- [ ]`. Việc treo của ngày trước làm xong thì quay lại tick `- [x]`.

### Khi nào cập nhật

- Làm xong việc trong ngày thì viết mục nhật ký **trước khi Trúc commit**, để nhật ký đi cùng commit.
- Đầu mỗi phiên làm BTL01, chạy `git log --date=short --format="%ad %h %s"` và so với các tiêu đề trong `CodetoTruc.md`. Ngày nào có commit mà chưa có mục thì báo Trúc và bổ sung.

## 2. Git

- Trúc tự commit và push. Claude chỉ sửa file trong máy và đưa lệnh git cho Trúc chạy.
- Không bao giờ thêm dòng `Co-Authored-By: Claude` vào commit. Đây là repo nhóm, có chấm điểm.
- Nếu Trúc nhờ commit: không thêm trailer, và hỏi lại trước khi push.
