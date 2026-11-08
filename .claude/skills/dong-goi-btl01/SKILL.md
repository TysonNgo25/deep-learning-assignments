---
name: dong-goi-btl01
description: Kiểm tra checklist nộp bài và nén file zip nộp BTL01 (mốc 7/11, hạn 8/11/2026). Dùng khi nói "đóng gói", "nén file nộp", "kiểm tra trước khi nộp", "checklist nộp", hoặc muốn biết còn thiếu file nào.
---

# Đóng gói và nộp BTL01

## 1. Kiểm tự động

Từ thư mục `BTL01 - Function Approximation`:

```powershell
& "C:\Users\hntruc\anaconda3\envs\btl01\python.exe" "..\.claude\skills\dong-goi-btl01\check_submission.py"
```

Script [check_submission.py](check_submission.py) kiểm:

- Có đủ file bắt buộc: README, requirements, config, 3 file dữ liệu, code `src/` và `experiments/`, summary A/B/C, checkpoint và `final_metrics.json`.
- Có báo cáo `.docx` trong `report/`, slide `.pptx` trong `slides/`, và hai bản PDF trong `docs/`.
- Đủ số log: A 30, B 39, C 12.
- Không còn `import tensorflow`/`keras`, không còn đường dẫn tuyệt đối trong `.py`, `.yaml`, `.ipynb`.
- Dòng code nào vừa nhắc `test` vừa có `min`/`argmin`/`sort`/`best` thì in ra để người kiểm đọc lại (có thể là rò rỉ test).
- README nhắc `models/best_model.pt` mà file không có thì cảnh báo (checkpoint thật đang ở `results/final/selected_model.pt`).
- `requirements.txt` chưa ghim phiên bản thì cảnh báo.

Mã thoát 1 nếu có LỖI. CẢNH BÁO thì đọc và quyết định.

## 2. Kiểm bằng tay (script không làm được)

- [ ] Code chạy lại từ đầu trên máy khác chỉ bằng lệnh trong README (Khang làm ngày 7/11).
- [ ] Báo cáo đủ 6 mục, 10–20 trang, hình/bảng có số và chú thích.
- [ ] Báo cáo có bảng baseline và bảng A, B, C ghi mean ± std.
- [ ] Slide 10–15 trang, trục đồ thị có nhãn.
- [ ] Bảng thành viên, nhiệm vụ, tỷ lệ đóng góp cộng đủ 100%.
- [ ] Gọi agent `btl01-cham-rubric` để chấm thử lần cuối.

## 3. Nén file

Khi không còn LỖI:

```powershell
& "C:\Users\hntruc\anaconda3\envs\btl01\python.exe" "..\.claude\skills\dong-goi-btl01\check_submission.py" --zip "BTL01_Nhom05_HuynhNgocTruc_3123411312.zip"
```

Tên theo mẫu của đề `BTL01_Nhom_HoTen_MSSV.zip`; hỏi lại Trúc nếu lớp quy định dùng tên người khác. File zip được ghi ra thư mục cha (gốc repo) để không tự nén chính nó.

Script bỏ qua `__pycache__`, `.ipynb_checkpoints`, file `.pyc`, và các file nội bộ trong `EXCLUDE` ở đầu script (đề của thầy, kế hoạch, nhật ký cá nhân). Muốn đưa file nào vào thì sửa danh sách đó. Script in danh sách file đã nén để Trúc xem lại.

Không commit file zip lên repo.
