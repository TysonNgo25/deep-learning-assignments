# Phụ lục

Người viết: Huỳnh Ngọc Trúc

## A. Bảng phân công và tỷ lệ đóng góp

Nhóm 5. Nhóm trưởng: Huỳnh Ngọc Trúc. Mỗi file mã nguồn có dòng `Phụ trách:` ở đầu, khớp với Bảng A.1.

Bảng A.1. Phân công thành viên và tỷ lệ đóng góp.

| Thành viên | STT | MSV | Mã nguồn và thực nghiệm | Phần báo cáo | Slide, thuyết trình | Tỷ lệ |
| --- | ---: | --- | --- | --- | --- | ---: |
| Huỳnh Ngọc Trúc (nhóm trưởng) | 43 | 3123411312 | `src/paths.py`, `src/data.py`, `src/train.py`, `src/evaluate.py`; `experiments/common.py`, `tong_hop.py`, `run_final.py`, notebook chạy trên Kaggle; `configs/`; BASE, mô hình cuối F-0/1/2; đóng gói | Mục 1, mục 2, mục 3.0 và 3.7, mục 5, phụ lục | Mở đầu, dữ liệu, tóm tắt mã nguồn, mô hình cuối và kết luận | 25% |
| Phạm Trần Minh Khang | 19 | 3123411143 | `src/models.py`; `src/visualize.py` (filter lớp conv cuối, feature map); `experiments/run_E1.py`, `run_E2.py`; thực nghiệm E1, E2 | Mục 3.1, 3.2, 4.1 | Kiến trúc, BatchNorm, filter lớp conv cuối | 25% |
| Tạ Quang Trung | 44 | 3123411318 | `src/evaluate.py` (confusion matrix, ca đúng/sai); `experiments/run_E3.py`, `run_E4.py`; thực nghiệm E3, E4 | Mục 3.3, 3.4, 4.2 | Optimizer, augmentation, confusion matrix và phân tích lỗi | 25% |
| Ngô Nguyễn Thành Nhân | 29 | 3123411205 | `src/visualize.py` (Grad-CAM); `experiments/run_E5.py`, `run_E6.py`; thực nghiệm E5, E6 | Mục 3.5, 3.6, 4.3 | Hàm kích hoạt, regularization, Grad-CAM, khung slide | 25% |
| **Tổng** | | | | | | **100%** |

Thực nghiệm E7 (VGG16 trên ảnh 128 × 128) có trong kế hoạch ở dạng tùy chọn; nhóm không thực hiện, theo phương án dự phòng khi thiếu thời gian (mục 3.7, 5.6).

## B. Danh sách file nộp

Bảng B.1. Nội dung file nén nộp bài.

| Thành phần | Đường dẫn | Ghi chú |
| --- | --- | --- |
| Báo cáo | `report/` (`.docx`, `.pdf`) | 5 mục và phụ lục |
| Slide | `slides/` (`.pptx`, `.pdf`) | |
| Nhật ký thực nghiệm | `NhatKyThucNghiem.md` | 21 dòng: BASE, E1-1 đến E6-3, F-0 đến F-2 |
| Mã nguồn | `src/`, `experiments/`, `configs/`, `README.md`, `requirements.txt` | Lệnh chạy ở `README.md` |
| Kết quả | `results/` | Log CSV mỗi epoch, `summary.csv`, cấu hình mỗi run, hình, bảng tổng hợp |
| Trọng số mô hình cuối | `results/final/best_model.pt` | 155.6 MB; nạp bằng `src.evaluate.load_checkpoint` |

Dữ liệu `fer2013.csv` không nằm trong file nộp; tải từ Kaggle (mục 1.2) và đặt vào `data/fer2013.csv`.

## C. Tái lập kết quả

Đứng ở thư mục bài, sau khi cài môi trường theo `README.md`:

| Bước | Lệnh | Tạo ra |
| --- | --- | --- |
| Kiểm tra dữ liệu, EDA | `python -m src.data` | `results/eda/` (mục 1) |
| Số tham số, kiểm tra khởi tạo | `python -m src.models` | `results/models/` (mục 2.3) |
| Baseline | `python -m src.train --run BASE --change "baseline"` | `results/BASE/` |
| Các nhóm thực nghiệm | `python -m experiments.run_E1` … `run_E6` | `results/E1/` … `results/E6/` (mục 3) |
| Bảng tổng hợp | `python -m experiments.tong_hop` | `results/tong_hop.csv`, `results/figures/` |
| Mô hình cuối, test một lần | `python -m experiments.run_final` | `results/final/` (mục 4, 5); chỉ chạy được một lần |
| Ảnh trùng train/test | `python -m src.evaluate --trung-lap results/final` | `results/final/duplicate_effect.json` (mục 5.5) |

Các bước huấn luyện cần GPU; nhóm chạy trên Kaggle bằng `experiments/chay_colab_kaggle.ipynb`. Cùng cấu hình và seed, kết quả lặp lại đến từng epoch (E1-2 trùng BASE, F-0 trùng E1-1).

## D. Tài liệu tham khảo

[1] I. J. Goodfellow, D. Erhan, P. L. Carrier và cộng sự. "Challenges in Representation Learning: A Report on Three Machine Learning Contests." ICONIP 2013. Dữ liệu: Kaggle, "Challenges in Representation Learning: Facial Expression Recognition Challenge".

[2] A. Paszke, S. Gross, F. Massa và cộng sự. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019.

[3] K. Simonyan, A. Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015.

[4] S. Ioffe, C. Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML 2015.

[5] K. He, X. Zhang, S. Ren, J. Sun. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." ICCV 2015.

[6] D. P. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization." ICLR 2015.

[7] J. Yosinski, J. Clune, A. Nguyen, T. Fuchs, H. Lipson. "Understanding Neural Networks Through Deep Visualization." Deep Learning Workshop, ICML 2015.

[8] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.

[9] X. Li, S. Chen, X. Hu, J. Yang. "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift." CVPR 2019.

Tài liệu của môn học: đề bài `Yêu cầu/Emotion_Recognition.pdf`, báo cáo mẫu `Yêu cầu/demo_report.pdf`, mã nguồn mẫu `Yêu cầu/demo_code.html` và `Yêu cầu/code/*.html` (Keras), chỉ dùng để tham khảo cách tổ chức.
