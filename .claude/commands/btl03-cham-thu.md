---
description: Chấm thử BTL03 bằng agent btl03-cham-rubric và đối chiếu từng notebook PyTorch với bản Keras gốc
---

1. Gọi agent `btl03-cham-rubric` chấm toàn bộ BTL03.
2. Song song, gọi agent `btl03-doi-chieu-keras` cho từng notebook `_pytorch.ipynb` đã có (có thể gom 3–4 notebook một lần gọi).
3. Nếu đã có `report/`, nhờ `btl03-doi-chieu-keras` đối chiếu thêm số trong báo cáo với `results/so_sanh.csv`.
4. Gộp kết quả: điểm dự kiến, lỗi nghiêm trọng (sai kiến trúc, còn Keras, số không khớp), danh sách việc cần sửa xếp theo điểm bị đe dọa, kèm ngày phải xong theo `docs/Kế hoạch.md` mục 7.
