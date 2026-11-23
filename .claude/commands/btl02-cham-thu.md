---
description: Chấm thử BTL02 bằng agent btl02-cham-rubric và đối chiếu số liệu báo cáo
---

1. Gọi agent `btl02-cham-rubric` chấm toàn bộ BTL02 Emotion Recognition.
2. Nếu đã có `report/section_*.md`, gọi song song agent `btl02-doi-chieu-so-lieu` cho toàn bộ báo cáo.
3. Gộp kết quả: điểm dự kiến, lỗi nghiêm trọng, danh sách việc cần sửa xếp theo điểm bị đe dọa, kèm ngày phải xong theo `docs/Kế hoạch.md` mục 7.
