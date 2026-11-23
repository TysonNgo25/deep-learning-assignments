# BTL02 – Emotion Recognition (FER2013, CNN PyTorch) – Kế hoạch

Cập nhật: 23/11/2026 · Người lập: Huỳnh Ngọc Trúc (nhóm trưởng) · Nhóm 5 · Hạn nộp: **Thứ 7 28/11/2026**

## 1. Đề bài yêu cầu gì

| Hạng mục | Yêu cầu |
|---|---|
| Dữ liệu | Kaggle FER2013, ảnh xám 48×48, 7 lớp: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral |
| Chia tập | Training 28.709 · PublicTest 3.589 (validation) · PrivateTest 3.589 (test, chỉ đánh giá 1 lần ở mô hình cuối) |
| Mô hình | CNN, input 48×48, output vector 7 chiều |
| Hướng cải thiện | Thiết kế mạng (Conv, MaxPool, Dropout, FC); kỹ thuật train (optimizer, augmentation); hàm kích hoạt (ReLU, sigmoid…) |
| Mức tham chiếu | Tốt nhất thế giới khoảng 69% public / 71% private; báo cáo mẫu đạt val_acc 0,64–0,67 |
| Báo cáo | (1) Tóm tắt code · (2) Kết quả nhiều thực nghiệm với tham số khác nhau · (3) **Visualize filter lớp conv cuối** · (4) Phân tích, giải thích theo cách của nhóm |
| Code | Python, **PyTorch** |
| Nộp | Report, slide, nhật ký thực nghiệm, code (ghi rõ phân công thành viên) |
| Tên file | `BTLON02_Emotion_HuynhNgocTruc_3123411312.zip`, nhóm trưởng nộp |

## 2. Chỗ dễ mất điểm

| Rủi ro | Cách xử lý |
|---|---|
| Code mẫu và báo cáo mẫu viết bằng Keras/TF 1.14 | Chỉ mượn cách tổ chức (loader, augmentation, callbacks, 4 kiến trúc Small → Large); mô hình và vòng train viết lại bằng PyTorch |
| Dùng test để chọn mô hình | PublicTest = validation để chọn; PrivateTest chạy **một lần** ở mô hình cuối |
| Lớp Disgust rất ít ảnh, accuracy che mất lớp nhỏ | Báo thêm confusion matrix, precision/recall/F1 từng lớp; thử CE có trọng số lớp (E6) |
| So sánh không công bằng | Mỗi thực nghiệm chỉ đổi **một biến** so với `configs/baseline.yaml` |
| Quên visualize filter conv cuối | Có hẳn bước trong ngày 27/11 và hình riêng trong báo cáo |
| Chạy CPU quá chậm | Chạy thực nghiệm trên GPU (Colab/Kaggle); code dùng đường dẫn tương đối `pathlib`, tự chọn `cuda`/`cpu` |
| Số liệu báo cáo không khớp log | Mọi con số trong report lấy từ `results/*/summary.csv` và `NhatKyThucNghiem.md` |

## 3. Baseline

Chốt ngày 24/11, ghi vào `configs/baseline.yaml`.

| Tham số | Giá trị |
|---|---|
| Kiến trúc | Medium: Block 1 2×Conv64, Block 2 3×Conv128, Block 3 3×Conv256; mỗi block MaxPool 2×2 + Dropout 0,2; FC 1024 + Dropout 0,5; FC 7 |
| Kích hoạt | ReLU |
| BatchNorm | Không |
| Optimizer | Adam, lr 1e-3 |
| Batch size | 64 |
| Augmentation | Lật ngang + xoay ±10° + dịch 10% |
| Loss | CrossEntropy |
| Epoch | Tối đa 40, early stopping patience 8 theo val_loss |
| Checkpoint | Lưu model có val_loss nhỏ nhất |
| Chuẩn hóa ảnh | Chia 255, chuẩn hóa theo mean/std của tập train |
| Seed | 0 |

## 4. Danh sách thực nghiệm

| Mã run | Nhóm | Biến thay đổi | Giá trị | Người phụ trách |
|---|---|---|---|---|
| E1-1 | Kiến trúc | Độ sâu/rộng | Small (2 block, FC 1024) | Khang |
| E1-2 | Kiến trúc | Độ sâu/rộng | Medium (= baseline) | Khang |
| E1-3 | Kiến trúc | Độ sâu/rộng | Medium_01 (3/4/4 conv mỗi block) | Khang |
| E1-4 | Kiến trúc | Độ sâu/rộng | Large (4 block 64-128-256-512, FC 2048) | Khang |
| E2-1 | BatchNorm | Có BN | Kiến trúc tốt nhất E1 + BN sau mỗi Conv | Khang |
| E3-1 | Optimizer | Optimizer | SGD momentum 0,9, lr 1e-2 | Trung |
| E3-2 | Optimizer | Learning rate | Adam lr 1e-4 | Trung |
| E3-3 | Optimizer | Scheduler | Adam lr 1e-3 + ReduceLROnPlateau | Trung |
| E4-1 | Augmentation | Mức tăng cường | Không augmentation | Trung |
| E4-2 | Augmentation | Mức tăng cường | Chỉ lật ngang | Trung |
| E4-3 | Augmentation | Mức tăng cường | Lật + xoay + dịch + zoom 10% | Trung |
| E5-1 | Kích hoạt | Activation | LeakyReLU | Nhân |
| E5-2 | Kích hoạt | Activation | ELU | Nhân |
| E5-3 | Kích hoạt | Activation | Sigmoid | Nhân |
| E6-1 | Regularization | Dropout | Không dropout | Nhân |
| E6-2 | Regularization | Dropout | Dropout block 0,4 | Nhân |
| E6-3 | Regularization | Loss | CE có trọng số lớp | Nhân |
| E7-1 | Transfer (tùy chọn) | Mô hình | ResNet18 pretrained ImageNet | Nhân |
| F-0/1/2 | Mô hình cuối | Seed | Cấu hình tốt nhất, seed 0, 1, 2 | Trúc |

Ghi chú: E1–E6 chạy 1 seed. Mô hình cuối chạy 3 seed, báo cáo trung bình ± độ lệch chuẩn. Chỉ mô hình cuối được đánh giá trên PrivateTest.

## 5. Đánh giá mô hình cuối

| Nội dung | File kết quả |
|---|---|
| Đường loss, accuracy theo epoch | `results/final/figures/curves.png` |
| Accuracy val (trung bình ± std 3 seed) và test (1 lần) | `results/final/final_metrics.json` |
| Confusion matrix | `results/final/figures/confusion_matrix.png` |
| Precision / Recall / F1 từng lớp | `results/final/classification_report.csv` |
| Ảnh đoán đúng, đoán sai | `results/final/figures/true_cases.png`, `wrong_cases.png` |
| **Filter lớp conv cuối** | `results/final/figures/last_conv_filters.png` |
| Feature map của 1 ảnh qua các lớp conv | `results/final/figures/feature_maps.png` |
| Grad-CAM (nếu kịp) | `results/final/figures/gradcam.png` |

## 6. Cấu trúc thư mục

```
BTL02 Emotion Recognition/
  configs/baseline.yaml
  data/                      (fer2013.csv – không đưa lên git)
  src/  data.py  models.py  train.py  evaluate.py  visualize.py  paths.py
  experiments/  run_E1.py … run_E6.py  run_final.py
  results/<E>/logs/*.csv  figures/  summary.csv
  results/final/  best_model.pt  final_metrics.json  figures/
  NhatKyThucNghiem.md
  docs/  Kế hoạch.md
  report/  slides/  README.md  requirements.txt
```

## 7. Lịch làm việc (23/11 – 28/11)

| Ngày | Việc | Kết quả để kiểm tra | Trạng thái |
|---|---|---|---|
| T2 23/11 tối | Tải `fer2013.csv`; dựng thư mục; `paths.py`, `data.py`; EDA (histogram lớp, ảnh mẫu) | Đếm đúng 28.709 / 3.589 / 3.589, đủ 7 lớp | - [x] |
| T3 24/11 | `models.py` (4 kiến trúc, tùy BN/activation/dropout); `train.py` (log CSV, checkpoint, early stopping, đo thời gian); chạy thử baseline; chốt `baseline.yaml` | Baseline chạy hết, val_acc khoảng 55–62% | - [ ] |
| T4 25/11 | Chạy E1, E2 trên GPU; ghi nhật ký; viết `evaluate.py` | `results/E1`, `results/E2` có `summary.csv` | - [ ] |
| T5 26/11 | Chạy E3, E4, E5, E6; ghi nhật ký; viết `visualize.py` | Đủ `summary.csv` E3–E6 | - [ ] |
| T6 27/11 sáng | Mô hình cuối 3 seed; test 1 lần; xuất toàn bộ hình mục 5 | Có `final_metrics.json` và đủ hình | - [ ] |
| T6 27/11 chiều–tối | Viết report (Word + PDF); làm slide; gửi mỗi thành viên phần ghi tên mình | Bản đầy đủ report + slide | - [ ] |
| T7 28/11 sáng | Rà số liệu với log; điền bảng phân công + tỷ lệ; README; đóng gói zip; nộp | Đã nộp zip | - [ ] |

Dự phòng khi trễ: bỏ E7 trước → rút E5 còn ReLU vs Sigmoid → rút E6 còn E6-1 và E6-3.

## 8. Phân công thành viên

| Thành viên | STT | MSV | Code / thực nghiệm | Phần báo cáo | Slide / vấn đáp | Tỷ lệ |
|---|---|---|---|---|---|---|
| Huỳnh Ngọc Trúc (nhóm trưởng) | 43 | 3123411312 | `paths.py`, `data.py`, `train.py`, `evaluate.py`, mô hình cuối (F), đóng gói | 1. Giới thiệu & dữ liệu · 2. Tóm tắt code · 5. Mô hình cuối & kết luận | Mở đầu, pipeline, demo code | ...% |
| Phạm Trần Minh Khang | 19 | 3123411143 | `models.py`, E1, E2, hình filter conv cuối + feature map | 3.1 Kiến trúc · 3.2 BatchNorm · 4.1 Visualize filter | E1–E2, filter | ...% |
| Tạ Quang Trung | 44 | 3123411318 | E3, E4, confusion matrix, phân tích ca sai | 3.3 Optimizer · 3.4 Augmentation · 4.2 Phân tích lỗi | E3–E4, confusion matrix | ...% |
| Ngô Nguyễn Thành Nhân | 29 | 3123411205 | E5, E6, E7, Grad-CAM | 3.5 Kích hoạt · 3.6 Regularization | E5–E6, khung slide | ...% |

Tỷ lệ đóng góp điền ở bước đóng gói, cộng đủ 100%. Trước T6 27/11 gửi mỗi bạn code và mục báo cáo ghi tên mình để đọc, tập trả lời vấn đáp.

## 9. Bố cục báo cáo

| Mục | Nội dung | Người viết |
|---|---|---|
| 1 | Giới thiệu bài toán, dữ liệu FER2013, EDA | Trúc |
| 2 | Tóm tắt code: data, model, train, evaluate | Trúc |
| 3 | Thực nghiệm E1–E6: bảng kết quả, đồ thị, nhận xét | Khang, Trung, Nhân |
| 4 | Visualize filter, feature map, Grad-CAM; phân tích lỗi | Khang, Trung, Nhân |
| 5 | Mô hình cuối, kết quả test, hạn chế, kết luận | Trúc |
| Phụ lục | Bảng phân công, tài liệu tham khảo | Trúc |

## 10. Nhật ký thực nghiệm (`NhatKyThucNghiem.md`)

Mỗi lần chạy ghi 1 dòng ngay sau khi chạy xong:

| Mã run | Ngày | Người phụ trách | Đổi gì so với baseline | Epoch tốt nhất | val_loss | val_acc | Thời gian | Nhận xét |
|---|---|---|---|---|---|---|---|---|
| E1-1 | 25/11 | Khang | Kiến trúc Small | | | | | |

## 11. Checklist nộp bài

- [ ] Report `.docx` + `.pdf`
- [ ] Slide `.pptx` + `.pdf`
- [ ] `NhatKyThucNghiem.md`
- [ ] Code `src/`, `experiments/`, `configs/`, `README.md`, `requirements.txt` (không kèm `fer2013.csv`)
- [ ] `results/` gồm log CSV, `summary.csv`, hình
- [ ] Trọng số mô hình cuối `best_model.pt`
- [ ] Bảng phân công + tỷ lệ đóng góp cộng đủ 100%
- [ ] Zip `BTLON02_Emotion_HuynhNgocTruc_3123411312.zip`
