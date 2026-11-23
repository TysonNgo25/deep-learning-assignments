---
name: nhat-ky-btl02
description: Viết mục nhật ký ngày trong BTL02 Emotion Recognition/CodetoTruc.md và ghi dòng run vào NhatKyThucNghiem.md theo CLAUDE.md. Dùng khi Trúc làm xong việc trong ngày và sắp commit, khi nói "viết nhật ký", "cập nhật CodetoTruc", "ghi lại hôm nay", "ghi run", hoặc đầu phiên BTL02 khi hook báo ngày thiếu nhật ký.
---

# Nhật ký BTL02

Có hai loại nhật ký, đừng nhầm:

| File | Dùng cho | Nộp cho thầy? |
| --- | --- | --- |
| `CodetoTruc.md` | Nhật ký ngày của Trúc: làm gì, giải thích hàm, câu hỏi vấn đáp | Không |
| `NhatKyThucNghiem.md` | Nhật ký thực nghiệm: mỗi run 1 dòng | **Có** (đề yêu cầu) |

## A. Mục ngày trong CodetoTruc.md

Quy tắc gốc ở `CLAUDE.md` mục 1, bố cục theo [template.md](../nhat-ky-btl01/template.md) của BTL01 với các thay đổi:

1. Tìm ngày thiếu: `git log --date=short --format="%ad %h %an %s" -- "BTL02 Emotion Recognition"` so với `grep -n "^## " "BTL02 Emotion Recognition/CodetoTruc.md"`. Hook SessionStart cũng in danh sách này.
2. "Việc theo kế hoạch" đối chiếu `docs/Kế hoạch.md` mục 7 (lịch ngày) và tick ô trạng thái ngày đó nếu xong.
3. Lệnh tự kiểm tra viết cho Anaconda Prompt: `cd /d "D:\Data\GitHub\deep-learning-assignments\BTL02 Emotion Recognition"`, `conda activate btl02`, rồi `python -m ...`.
4. Có run mới trong ngày thì thêm bảng "Run trong ngày" trỏ về các dòng trong `NhatKyThucNghiem.md`.
5. Thêm mục ở **cuối file**, ngăn bằng `---`, tiêu đề `## <Thứ> dd/mm/yyyy – <việc chính>`. Commit chưa có hash thì ghi `_(Trúc điền ở mục ngày sau)_`.

## B. Dòng run trong NhatKyThucNghiem.md

1. Đọc `results/<nhóm>/logs/<mã run>.csv`, lấy dòng có `val_loss` nhỏ nhất: `epoch`, `val_loss`, `val_acc`. Thời gian = tổng `epoch_time_s`.
2. "Đổi gì so với baseline" lấy từ `docs/Kế hoạch.md` mục 4 (đúng một thay đổi). "Người phụ trách" lấy cột tương ứng.
3. Nhận xét 1 câu có số: so với baseline (`+1.8 điểm val_acc`), dấu hiệu quá khớp (khoảng cách train_acc − val_acc), số epoch đến early stopping.
4. Thêm dòng theo thứ tự mã run, không sửa dòng cũ. Chạy lại thì thêm dòng `<run>_r2`.

Không tự commit. Đưa Trúc lệnh `git add ...` và `git commit -m "..."`, không có dòng `Co-Authored-By`.
