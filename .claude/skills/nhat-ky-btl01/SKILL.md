---
name: nhat-ky-btl01
description: Viết hoặc bổ sung mục nhật ký ngày trong BTL01 - Function Approximation/CodetoTruc.md theo quy tắc ở CLAUDE.md. Dùng khi Trúc làm xong việc trong ngày và sắp commit, khi nói "viết nhật ký", "cập nhật CodetoTruc", "ghi lại hôm nay", hoặc đầu phiên làm BTL01 để tìm ngày có commit mà chưa có mục.
---

# Viết nhật ký CodetoTruc.md

Quy tắc gốc nằm ở `CLAUDE.md` mục 1. Skill này là các bước làm cụ thể.

## 1. Tìm ngày còn thiếu

Chạy từ gốc repo:

```bash
git log --date=short --format="%ad %h %an %s" -- "BTL01 - Function Approximation"
grep -n "^## " "BTL01 - Function Approximation/CodetoTruc.md"
```

So từng ngày có commit với các tiêu đề `## <Thứ> dd/mm/yyyy`. Ngày nào có commit mà chưa có mục thì báo Trúc trước khi làm việc khác.

## 2. Thu thập nội dung

- Việc theo kế hoạch: đọc bảng lịch ngày ở `Kế hoạch.md` mục 4, cột của Trúc (và cột người khác nếu Trúc làm thay).
- Commit trong ngày: `git log --since="<yyyy-mm-dd> 00:00" --until="<yyyy-mm-dd> 23:59" --format="%h %an %s"`. Ghi tên người theo tài khoản: `NgocTruc` là Trúc, `MinKhang` là Khang, `trungboro116` là Trung, `TysonNgo25` là Nhân.
- File đã đổi: `git diff --stat` (chưa commit) hoặc `git show --stat <hash>`.
- Mục "Còn treo" của các ngày trước: việc nào đã xong thì đổi `- [ ]` thành `- [x]` ngay tại mục cũ.

## 3. Viết mục mới

- Thêm ở **cuối file**, ngăn mục trước bằng một dòng `---`.
- Tiêu đề: `## <Thứ> dd/mm/yyyy – <việc chính>`. Thứ viết tắt: T2, T3, T4, T5, T6, T7, CN. Ngày luôn 2 chữ số, ví dụ `## T5 05/11/2026 – Ghép báo cáo bản 1`.
- Bố cục theo [template.md](template.md). Bỏ phần nào không có nội dung, không để tiêu đề rỗng.
- Commit chưa có hash (Trúc sắp commit) thì ghi `_(Trúc điền ở mục ngày sau)_` như mục 04/11.
- Lệnh tự kiểm tra viết cho Anaconda Prompt (`cd /d`, `conda activate btl01`, `set PYTHONPATH=%CD%`), kèm kết quả đúng phải thấy.
- Số liệu nào ghi vào nhật ký cũng phải có cột nguồn (file trong `results/` hoặc `data/`).
- Viết tiếng Việt, giải thích kỹ để Trúc ôn vấn đáp: hàm làm gì và vì sao viết như vậy, không chỉ liệt kê.

## 4. Sau khi viết

Không tự commit. Đưa Trúc lệnh để tự chạy, ví dụ:

```bash
git add "BTL01 - Function Approximation/CodetoTruc.md" <các file khác của ngày>
git commit -m "<mô tả ngắn bằng tiếng Việt>"
git push
```

Không thêm dòng `Co-Authored-By` vào message (repo nhóm, có chấm điểm).
