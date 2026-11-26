# BTL02 – Emotion Recognition (FER2013, CNN PyTorch) – Kế hoạch

Cập nhật: 23/11/2026 (bản 2: sau khi đọc hết code mẫu) · Người lập: Huỳnh Ngọc Trúc (nhóm trưởng) · Nhóm 5 · Hạn nộp: **Thứ 7 28/11/2026**

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
| Thứ tự nhãn khác code mẫu | Code mẫu dùng LabelEncoder (thứ tự chữ cái: …, Neutral, Sad, Surprise). Ta giữ thứ tự gốc FER2013 0–6 (`src/paths.py`). So confusion matrix với mẫu phải để ý |
| Chép lỗi của code mẫu | Mẫu vẽ ảnh validation bằng `x_train[valid_idx]` và có mật khẩu email trong code: không chép |

## 3. Baseline

Chốt ngày 24/11, ghi vào `configs/baseline.yaml`.

| Tham số | Giá trị |
|---|---|
| Kiến trúc | Medium: Block 1 2×Conv64, Block 2 3×Conv128, Block 3 3×Conv256; mỗi block MaxPool 2×2 + Dropout 0,2; FC 1024 + Dropout 0,5; FC 7 |
| Kích hoạt | ReLU |
| BatchNorm | Không |
| Optimizer | Adam, lr **1e-4** (theo code mẫu; lr 1e-3 chạy thử bị kẹt val_acc 0,235) |
| Batch size | 64 |
| Augmentation | Lật ngang + xoay ±10° + dịch 10% |
| Loss | CrossEntropy |
| Epoch | Tối đa **60**, early stopping patience **10** theo val_loss (mẫu đạt tốt nhất ở epoch 31–32) |
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
| E3-1 | Optimizer | Optimizer | SGD momentum 0,9, lr 1e-3 | Trung |
| E3-2 | Optimizer | Learning rate | Adam lr 1e-3 | Trung |
| E3-3 | Optimizer | Scheduler | Adam lr 1e-4 + ReduceLROnPlateau | Trung |
| E4-1 | Augmentation | Mức tăng cường | Không augmentation | Trung |
| E4-2 | Augmentation | Mức tăng cường | Chỉ lật ngang | Trung |
| E4-3 | Augmentation | Mức tăng cường | Lật + xoay + dịch + zoom 10% | Trung |
| E5-1 | Kích hoạt | Activation | LeakyReLU | Nhân |
| E5-2 | Kích hoạt | Activation | ELU | Nhân |
| E5-3 | Kích hoạt | Activation | Sigmoid | Nhân |
| E6-1 | Regularization | Dropout | Không dropout | Nhân |
| E6-2 | Regularization | Dropout | Dropout block 0,4 | Nhân |
| E6-3 | Regularization | Loss | CE có trọng số lớp | Nhân |
| E7-1 | Transfer (làm nếu có GPU) | Trọng số khởi tạo | VGG16 train từ đầu, ảnh phóng 128×128×3 | Nhân |
| E7-2 | Transfer (làm nếu có GPU) | Trọng số khởi tạo | VGG16 pretrained ImageNet, ảnh 128×128×3 (chỉ khác E7-1 ở trọng số) | Nhân |
| F-0/1/2 | Mô hình cuối | Seed | Cấu hình tốt nhất, seed 0, 1, 2 | Trúc |

Ghi chú: E7 là nhánh riêng (ảnh 128×128×3, cấu hình theo `code/train.html`: SGD lr 4e-4 momentum 0,9 nesterov, FC 2048, dropout 0,1, xoay ±30°). So E7-1 với E7-2 để thấy tác dụng của pretrained; so E7 với baseline chỉ để tham khảo vì khác nhiều biến. Code mẫu của thầy đạt val_acc 0,699 (VGG16 từ đầu) và 0,707 (pretrained), trong khi báo cáo mẫu ghi VGG16 là "fail case".

E1–E6 chạy 1 seed. Mô hình cuối chạy 3 seed, báo cáo trung bình ± độ lệch chuẩn. Chỉ mô hình cuối được đánh giá trên PrivateTest.

## 5. Đánh giá mô hình cuối

| Nội dung | File kết quả |
|---|---|
| Đường loss, accuracy theo epoch | `results/final/figures/curves.png` |
| Accuracy val (trung bình ± std 3 seed) và test (1 lần) | `results/final/final_metrics.json` |
| Confusion matrix trên valid và test, đặt cạnh nhau (như `valid_vgg16.html`) | `results/final/figures/confusion_matrix.png` |
| Precision / Recall / F1 từng lớp | `results/final/classification_report.csv` |
| Ảnh đoán đúng, đoán sai | `results/final/figures/true_cases.png`, `wrong_cases.png` |
| **Filter lớp conv cuối** | `results/final/figures/last_conv_filters.png` |
| Feature map của 1 ảnh qua các lớp conv | `results/final/figures/feature_maps.png` |
| **Grad-CAM (bắt buộc)** tại conv đầu, giữa, cuối; mỗi cảm xúc 1 ảnh đoán đúng (7 cột × 3 hàng), như `valid_vgg16.html` | `results/final/figures/gradcam.png` |

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
| T3 24/11 | `models.py` (4 kiến trúc, tùy BN/activation/dropout); `train.py` (log CSV, checkpoint, early stopping, đo thời gian); chạy thử baseline; chốt `baseline.yaml` | Baseline chạy hết, val_acc khoảng 60–66% (mẫu: 0,66) | - [x] |
| T4 25/11 | Chạy E1, E2 trên GPU; ghi nhật ký; viết `evaluate.py` | `results/E1`, `results/E2` có `summary.csv` | - [x] |
| T5 26/11 | Chạy E3, E4, E5, E6; E7 nếu có GPU; ghi nhật ký; viết `visualize.py` (filter, feature map, Grad-CAM) | Đủ `summary.csv` E3–E6 | - [x] |
| T6 27/11 sáng | Mô hình cuối 3 seed; test 1 lần; xuất toàn bộ hình mục 5 | Có `final_metrics.json` và đủ hình | - [x] |
| T6 27/11 chiều–tối | Viết report (Word + PDF); làm slide; gửi mỗi thành viên phần ghi tên mình | Bản đầy đủ report + slide | - [ ] |
| T7 28/11 sáng | Rà số liệu với log; điền bảng phân công + tỷ lệ; README; đóng gói zip; nộp | Đã nộp zip | - [ ] |

Dự phòng khi trễ: bỏ E7 trước → rút E5 còn ReLU vs Sigmoid → rút E6 còn E6-1 và E6-3.

## 8. Phân công thành viên

| Thành viên | STT | MSV | Code / thực nghiệm | Phần báo cáo | Slide / vấn đáp | Tỷ lệ |
|---|---|---|---|---|---|---|
| Huỳnh Ngọc Trúc (nhóm trưởng) | 43 | 3123411312 | `paths.py`, `data.py`, `train.py`, `evaluate.py`, mô hình cuối (F), đóng gói | 1. Giới thiệu & dữ liệu · 2. Tóm tắt code · 3.0, 3.7 Thiết lập chung, tổng kết thực nghiệm · 5. Mô hình cuối & kết luận | Mở đầu, pipeline, demo code | 25% |
| Phạm Trần Minh Khang | 19 | 3123411143 | `models.py`, E1, E2, hình filter conv cuối + feature map | 3.1 Kiến trúc · 3.2 BatchNorm · 4.1 Visualize filter | E1–E2, filter | 25% |
| Tạ Quang Trung | 44 | 3123411318 | E3, E4, confusion matrix, phân tích ca sai | 3.3 Optimizer · 3.4 Augmentation · 4.2 Phân tích lỗi | E3–E4, confusion matrix | 25% |
| Ngô Nguyễn Thành Nhân | 29 | 3123411205 | E5, E6, Grad-CAM (E7 không làm, bỏ theo dự phòng) | 3.5 Kích hoạt · 3.6 Regularization · 4.3 Grad-CAM | E5–E6, Grad-CAM, khung slide | 25% |

Tỷ lệ đóng góp chốt ngày 26/11: mỗi thành viên 25%, cộng đủ 100%. Trước T6 27/11 gửi mỗi bạn code và mục báo cáo ghi tên mình để đọc, tập trả lời vấn đáp.

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

- [x] Report `.docx` + `.pdf`
- [x] Slide `.pptx` + `.pdf`
- [x] `NhatKyThucNghiem.md`
- [ ] Code `src/`, `experiments/`, `configs/`, `README.md`, `requirements.txt` (không kèm `fer2013.csv`)
- [ ] `results/` gồm log CSV, `summary.csv`, hình
- [ ] Trọng số mô hình cuối `best_model.pt`
- [x] Bảng phân công + tỷ lệ đóng góp cộng đủ 100%
- [ ] Zip `BTLON02_Emotion_HuynhNgocTruc_3123411312.zip`

## 12. Ghi chú từ code mẫu (đọc hết `Yêu cầu/` ngày 23/11)

| File | Điều rút ra |
|---|---|
| `demo_code.html` (Keras, 2018) | 4 kiến trúc Small/Medium/Medium_01/Large; Adam lr 1e-4, batch 128, patience 10, tối đa 120 epoch; aug xoay 10°, dịch 10%, zoom 10%, lật ngang; Medium_01 tốt nhất val_loss 0,94 / val_acc 0,66 ở epoch 32; có phần true cases / wrong cases |
| Số tham số (mẫu) | Small **38,9 triệu** (chỉ 2 lần pool, Flatten 36.864 chiều) > Medium_01 12,1 triệu: mạng nông hơn nhưng nhiều tham số hơn, đưa vào phân tích E1 |
| `code/train.html` (2019) | VGG16 ImageNet ở 128×128×3, SGD lr 4e-4 momentum 0,9 nesterov, FC 2048, dropout 0,1, xoay 30°, early stopping 30 theo val_accuracy; có tùy chọn Cyclic LR |
| `code/valid_vgg16.html`, `valid_imagenet_vgg16.html` | VGG16 từ đầu val_acc 0,699; pretrained 0,707. Confusion matrix valid và test cạnh nhau; heatmap tại conv đầu/giữa/cuối, mỗi cảm xúc 1 ảnh đoán đúng |
| `demo_report.pdf` | Bố cục: xử lý dữ liệu → train (aug, mô hình, callbacks) → kết quả → đánh giá (histogram, confusion, ca sai, ca đúng) → so sánh 4 mô hình → fail case |
