---
name: nhat-ky-btl03
description: Viết mục nhật ký ngày trong BTL03 (Chuyển code Keras sang PyTorch)/CodetoTruc.md theo CLAUDE.md - đối chiếu lịch, commit, notebook đã chuyển, số tham số và kết quả Keras ↔ PyTorch, câu hỏi vấn đáp. Dùng khi Trúc làm xong việc trong ngày ở BTL03 và sắp commit, khi nói "viết nhật ký BTL03", "ghi lại hôm nay", hoặc đầu phiên khi hook báo ngày thiếu nhật ký.
---

# Nhật ký BTL03

`CodetoTruc.md` của BTL03 là nhật ký cá nhân, **không nộp**. Bảng so sánh nộp kèm bài là `results/so_sanh.csv` (rule `btl03-ket-qua`).

## Viết mục ngày

Quy tắc gốc ở `CLAUDE.md` mục 1; bố cục theo [template.md](../nhat-ky-btl01/template.md) và mục T2 23/11 đã có trong file. Riêng BTL03:

1. Tìm ngày thiếu: `git log --date=short --format="%ad %h %an %s" -- "BTL03 - Chuyển code từ keras sang pytorch deep learning"` so với `grep -n "^## " ".../CodetoTruc.md"`. Hook SessionStart cũng in danh sách này.
2. "Việc theo kế hoạch" đối chiếu `docs/Kế hoạch.md` mục 7, tick ô ngày đó nếu xong.
3. Có notebook mới thì thêm bảng **"Notebook trong ngày"**: Notebook | Lớp/khối chính | Tham số Keras | Tham số PyTorch | Kết quả Keras | Kết quả PyTorch | Thiết bị. Số lấy từ `results/so_sanh.csv`.
4. "Giải thích từng hàm" áp cho `common.py` và các lớp `nn.Module`: hàm làm gì, tương ứng gì bên Keras, vì sao viết vậy.
5. Thêm mục **"Khác Keras ở đâu"**: liệt kê điểm khác gặp trong ngày (khởi tạo, eps, bias LSTM, momentum BN, fill_mode...), mỗi điểm một câu. Đây là nguyên liệu cho báo cáo mục 5.
6. Lệnh tự kiểm tra viết cho Anaconda Prompt: `cd /d "D:\Data\GitHub\deep-learning-assignments\BTL03 - Chuyển code từ keras sang pytorch deep learning"`, `conda activate btl02`, rồi `jupyter nbconvert --execute ...` hoặc mở notebook; nêu số tham số và kết quả đúng phải thấy.
7. Thêm mục ở **cuối file**, ngăn bằng `---`, tiêu đề `## <Thứ> dd/mm/yyyy – <việc chính>`. Commit chưa có hash thì ghi `_(Trúc điền ở mục ngày sau)_`. Việc treo của ngày trước làm xong thì quay lại tick.

Không tự commit. Đưa Trúc lệnh `git add ...` và `git commit -m "..."`, không có dòng `Co-Authored-By`.
