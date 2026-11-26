# Nhật ký thực nghiệm – BTL02 Emotion Recognition

Mỗi lần chạy ghi 1 dòng ngay sau khi chạy xong. Số liệu lấy từ `results/<E>/logs/<mã run>.csv`.
Baseline: `configs/baseline.yaml`. Validation = PublicTest; PrivateTest chỉ dùng cho mô hình cuối.

| Mã run | Ngày | Người phụ trách | Đổi gì so với baseline | Epoch tốt nhất | val_loss | val_acc | Thời gian | Nhận xét |
|---|---|---|---|---|---|---|---|---|
| BASE | 25/11 | Trúc | (baseline) | 47 / 57 | 0.9429 | 66.43% | 1127 s (T4) | Ngang báo cáo mẫu (0.64–0.67); train_acc 70.54% cao hơn val 4.11 điểm, quá khớp nhẹ; early stopping ở epoch 57. |
| E1-1 | 25/11 | Khang | Kiến trúc Small (2 block, FC 1024) | 41 / 51 | 0.9552 | 67.01% | 1476 s (T4) | +0.59 điểm so với BASE, cao nhất E1, nhưng val_loss cao hơn (0.9552 so với 0.9429); 38.9 triệu tham số, gấp 3.4 lần Medium. |
| E1-2 | 25/11 | Khang | Kiến trúc Medium (= baseline) | 47 / 57 | 0.9429 | 66.43% | 1127 s (T4) | Trùng khít BASE ở mọi epoch (cùng seed, cùng loại GPU): xác nhận kết quả tái lập được. |
| E1-3 | 25/11 | Khang | Kiến trúc Medium_01 (3/4/4 conv mỗi block) | 45 / 55 | 0.9495 | 65.03% | 1536 s (T4) | −1.39 điểm; train_acc − val_acc = 6.00 điểm, quá khớp rõ hơn BASE (4.11). |
| E1-4 | 25/11 | Khang | Kiến trúc Large (4 block 64-128-256-512, FC 2048) | 41 / 51 | 0.9464 | 65.87% | 2124 s (T4) | −0.56 điểm; chậm nhất nhóm (2124 s); khoảng cách train − val 5.14 điểm. |
| E2-1 | 25/11 | Khang | Kiến trúc tốt nhất E1 (Small) + BatchNorm sau mỗi Conv; so với E1-1 | 60 / 60 | 1.0104 | 62.36% | 1869 s (T4) | −4.65 điểm so với E1-1; chạm trần 60 epoch, val_loss thấp nhất ở epoch cuối và train_acc (56.12%) còn thấp hơn val ở cả 60 epoch: chưa hội tụ. |
| E3-1 | 25/11 | Trung | SGD momentum 0.9, lr 1e-3 | 60 / 60 | 1.0060 | 61.83% | 1146 s (T4) | −4.60 điểm; chạm trần 60 epoch, val_loss vẫn giảm: SGD hội tụ chậm hơn Adam trong cùng số epoch. |
| E3-2 | 25/11 | Trung | Adam lr 1e-3 | 28 / 38 | 1.0442 | 61.16% | 735 s (T4) | −5.27 điểm; lr gấp 10 lần làm val_loss nhỏ nhất cao hơn (1.0442) và dừng sớm ở epoch 38. |
| E3-3 | 25/11 | Trung | Adam lr 1e-4 + ReduceLROnPlateau | 33 / 43 | 0.9723 | 65.20% | 851 s (T4) | −1.23 điểm; lr giảm còn 1e-5 ở epoch 26 (val_acc lúc đó 64.59%), còn 1e-7 ở epoch 37, mạng gần như ngừng học: patience 2 quá ngắn. |
| E4-1 | 25/11 | Trung | Không augmentation | 11 / 21 | 1.1096 | 59.88% | 416 s (T4) | −6.55 điểm; quá khớp nhanh nhất: train − val 8.62 điểm, val_loss tăng lên 1.4957 ở epoch 21. |
| E4-2 | 25/11 | Trung | Chỉ lật ngang | 16 / 26 | 1.0688 | 61.52% | 515 s (T4) | −4.90 điểm; train − val 7.14 điểm, dừng ở epoch 26: chỉ lật ngang chưa đủ chống quá khớp. |
| E4-3 | 25/11 | Trung | Lật + xoay + dịch + zoom 10% | 41 / 51 | 0.9506 | 66.26% | 1010 s (T4) | −0.17 điểm, ngang BASE; train − val chỉ 1.42 điểm so với 4.11 của BASE: zoom giảm quá khớp. |
| E5-1 | 25/11 | Nhân | LeakyReLU (độ dốc âm 0.01) | 44 / 54 | 0.9486 | 66.29% | 1087 s (T4) | −0.14 điểm, ngang BASE; train − val 3.75 điểm. |
| E5-2 | 25/11 | Nhân | ELU | 31 / 41 | 0.9643 | 64.89% | 827 s (T4) | −1.53 điểm; dừng ở epoch 41; train_acc ≈ val_acc (−0.11 điểm), không quá khớp. |
| E5-3 | 25/11 | Nhân | Sigmoid | 60 / 60 | 1.1531 | 56.59% | 1205 s (T4) | −9.84 điểm, thấp nhất mọi run; chạm trần 60 epoch, val_loss vẫn giảm: sigmoid bão hòa, gradient nhỏ, học rất chậm. |
| E6-1 | 25/11 | Nhân | Không dropout (block 0, FC 0) | 17 / 27 | 1.0102 | 62.55% | 533 s (T4) | −3.87 điểm; val_loss tăng từ 1.0102 lên 1.0722 sau epoch 17, dừng ở epoch 27: quá khớp khi bỏ dropout. |
| E6-2 | 25/11 | Nhân | Dropout block 0.4 | 60 / 60 | 0.9466 | 65.78% | 1185 s (T4) | −0.64 điểm; chạm trần 60 epoch, val_loss dao động quanh 0.96 rồi thấp nhất ở epoch 60; train − val 1.30 điểm: dropout mạnh giảm quá khớp nhưng học chậm hơn. |
| E6-3 | 25/11 | Nhân | CE có trọng số lớp | 35 / 45 | 1.0618 | 62.19% | 892 s (T4) | −4.24 điểm accuracy; trọng số lớp nhắm vào lớp hiếm nên cần F1 từng lớp mới đánh giá đủ (log chỉ có accuracy). |
