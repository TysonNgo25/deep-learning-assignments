# Deep Learning Assignments – Nhóm 5

Bài tập học phần Học sâu. Code viết bằng **PyTorch** (code mẫu của thầy là Keras/TensorFlow).

| STT | MSV | Họ và tên |
| --- | --- | --- |
| 19 | 3123411143 | Phạm Trần Minh Khang |
| 29 | 3123411205 | Ngô Nguyễn Thành Nhân |
| 43 | 3123411312 | Huỳnh Ngọc Trúc (nhóm trưởng) |
| 44 | 3123411318 | Tạ Quang Trung |

## Các bài

| Thư mục | Nội dung | Trạng thái |
| --- | --- | --- |
| [`lab1/`](lab1/) | Bài thực hành 1 (bai1–bai3) | đã làm |
| [`lab03_house_price/`](lab03_house_price/) | Lab 03: dự đoán giá nhà Ames (scikit-learn, MLP PyTorch) | đã làm |
| [`BTL01 - Function Approximation/`](BTL01%20-%20Function%20Approximation/) | Xấp xỉ hàm Gauss 2 chiều bằng MLP | đã nộp 8/11/2026 |
| [`BTL02 Emotion Recognition/`](BTL02%20Emotion%20Recognition/) | Nhận dạng 7 cảm xúc FER2013 bằng CNN, thực nghiệm E1–E7 | đang làm, hạn 28/11/2026 |
| [`BTL03 - Chuyển code từ keras sang pytorch deep learning/`](BTL03%20-%20Chuy%E1%BB%83n%20code%20t%E1%BB%AB%20keras%20sang%20pytorch%20deep%20learning/) | Chuyển 11 notebook Keras (MLP, CNN, CNN+LSTM, SegNet, phát hiện xe) sang PyTorch và so kết quả | đang làm, hạn 28/11/2026 |

Mỗi bài có `README.md` riêng hướng dẫn cài môi trường và chạy. Thư mục `Yêu cầu/` (hoặc `Đề và yêu cầu/`) trong mỗi bài là đề gốc của thầy, giữ nguyên.

## Dữ liệu

Dữ liệu lớn không đưa lên git (FER2013, Gender `.npy`...). Cách tải và chỗ đặt ghi trong README của từng bài.

## Làm việc với Claude Code

Thư mục `.claude/` chứa công cụ hỗ trợ cho từng bài (quy tắc chung ở [`CLAUDE.md`](CLAUDE.md)):

| Loại | BTL01 | BTL02 | BTL03 |
| --- | --- | --- | --- |
| Lệnh (`commands/`) | – | `/btl02-bat-dau`, `/btl02-ghi-run`, `/btl02-tien-do`, `/btl02-cham-thu` | `/btl03-bat-dau`, `/btl03-ghi-ket-qua`, `/btl03-tien-do`, `/btl03-cham-thu` |
| Skill (`skills/`) | chạy thực nghiệm, nhật ký, báo cáo, ôn vấn đáp, đóng gói | chạy thực nghiệm, nhật ký, báo cáo, ôn vấn đáp, đóng gói | chuyển đổi, chạy notebook, nhật ký, báo cáo, ôn vấn đáp, đóng gói |
| Agent (`agents/`) | chấm rubric, đối chiếu số liệu, giám khảo vấn đáp | chấm rubric, đối chiếu số liệu, giám khảo vấn đáp | chấm rubric, đối chiếu với Keras, giám khảo vấn đáp |
| Rule (`rules/`) | – | code, thực nghiệm, báo cáo | code, kết quả, báo cáo |
| Hook (`hooks/`) | – | `btl02_guard.py` | `btl03_guard.py` |

Hook chặn `git push` và commit có `Co-Authored-By` (repo nhóm, có chấm điểm), chặn sửa đề gốc trong `Yêu cầu/` và chặn `tensorflow`/`keras` trong code PyTorch. Hook chạy bằng `C:/Users/hntruc/anaconda3/python.exe` (máy Trúc); máy khác cần sửa đường dẫn này trong `.claude/settings.json`.
