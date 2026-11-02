# Bảng kết quả C — bàn giao ngày 02/11/2026

Nguồn: `summary.csv`, 12 log `logs/C_<tầng>_seed<seed>.csv`, baseline `configs/baseline.yaml`.
Mỗi giá trị là trung bình ± độ lệch chuẩn mẫu (ddof=1) của 3 seed 0, 1, 2.
Cố định 1000 mẫu train; 16 nơ-ron/tầng; tanh; Adam; lr=1e-3; batch=32; tối đa 1000 epoch; patience=100; y gốc.

## train

| Số tầng | MSE | MAE | RMSE |
| ---: | ---: | ---: | ---: |
| 1 | 9.281447e-06 ± 2.091271e-06 | 2.396883e-03 ± 3.044031e-04 | 3.032622e-03 ± 3.563352e-04 |
| 2 | 4.135533e-07 ± 4.025497e-08 | 5.034436e-04 ± 2.551394e-05 | 6.425849e-04 ± 3.093426e-05 |
| 3 | 5.520759e-07 ± 1.436262e-07 | 5.885155e-04 ± 9.310941e-05 | 7.384310e-04 ± 1.009623e-04 |
| 4 | 3.983979e-07 ± 1.559558e-07 | 4.780881e-04 ± 1.011019e-04 | 6.225607e-04 ± 1.273739e-04 |

## val

| Số tầng | MSE | MAE | RMSE |
| ---: | ---: | ---: | ---: |
| 1 | 1.193230e-05 ± 4.160528e-06 | 2.601272e-03 ± 4.188874e-04 | 3.414008e-03 ± 6.444211e-04 |
| 2 | 4.350545e-07 ± 2.451342e-08 | 5.186514e-04 ± 1.900518e-05 | 6.594145e-04 ± 1.845216e-05 |
| 3 | 5.502185e-07 ± 1.332506e-07 | 5.814734e-04 ± 7.089621e-05 | 7.378664e-04 ± 9.304647e-05 |
| 4 | 4.098433e-07 ± 1.852259e-07 | 4.857772e-04 ± 1.091899e-04 | 6.281402e-04 ± 1.514093e-04 |

## test

| Số tầng | MSE | MAE | RMSE |
| ---: | ---: | ---: | ---: |
| 1 | 1.013326e-05 ± 1.566264e-06 | 2.459928e-03 ± 3.591795e-04 | 3.176605e-03 ± 2.523098e-04 |
| 2 | 4.585057e-07 ± 2.212564e-08 | 5.408598e-04 ± 1.305674e-05 | 6.770003e-04 ± 1.626063e-05 |
| 3 | 6.837284e-07 ± 1.103983e-07 | 6.790123e-04 ± 7.207988e-05 | 8.250776e-04 ± 6.680604e-05 |
| 4 | 4.708029e-07 ± 2.427884e-07 | 5.200208e-04 ± 1.470775e-04 | 6.687152e-04 ± 1.882400e-04 |

## Thời gian và epoch tốt nhất

| Số tầng | Thời gian train (giây) | Best epoch |
| ---: | ---: | ---: |
| 1 | 15.24 ± 1.60 | 994.33 ± 8.08 |
| 2 | 21.55 ± 1.98 | 988.00 ± 18.19 |
| 3 | 21.90 ± 6.17 | 745.33 ± 268.34 |
| 4 | 25.72 ± 4.08 | 755.67 ± 131.18 |

## Đối chiếu log

| Log | Epoch thực chạy | Best epoch theo validation |
| --- | ---: | ---: |
| [C_1_seed0.csv](logs/C_1_seed0.csv) | 1000 | 985 |
| [C_1_seed1.csv](logs/C_1_seed1.csv) | 1000 | 999 |
| [C_1_seed2.csv](logs/C_1_seed2.csv) | 1000 | 999 |
| [C_2_seed0.csv](logs/C_2_seed0.csv) | 1000 | 998 |
| [C_2_seed1.csv](logs/C_2_seed1.csv) | 1000 | 967 |
| [C_2_seed2.csv](logs/C_2_seed2.csv) | 1000 | 999 |
| [C_3_seed0.csv](logs/C_3_seed0.csv) | 537 | 437 |
| [C_3_seed1.csv](logs/C_3_seed1.csv) | 1000 | 926 |
| [C_3_seed2.csv](logs/C_3_seed2.csv) | 973 | 873 |
| [C_4_seed0.csv](logs/C_4_seed0.csv) | 1000 | 905 |
| [C_4_seed1.csv](logs/C_4_seed1.csv) | 759 | 659 |
| [C_4_seed2.csv](logs/C_4_seed2.csv) | 803 | 703 |

## Hình và giới hạn diễn giải

- [Loss train/validation](figures/C_loss_curves.png): mean theo các seed còn chạy ở từng epoch; early stopping làm số seed ở đoạn cuối thay đổi.
- [Test MSE](figures/C_test_mse.png), [test RMSE](figures/C_test_rmse.png), [thời gian](figures/C_train_time.png), [RMSE và thời gian](figures/C_error_and_time.png).
- Best epoch và min val loss đã đối chiếu với log; config/seed/độ sâu/số mẫu khớp baseline; bảng mean/std đã tính lại từ summary.
- Metric train/val/test và thời gian lấy từ summary của lượt chạy đã lưu. Log không chứa dự đoán hoặc checkpoint từng lượt C, nên chưa thể tính lại độc lập các metric từ trọng số.
- Test đã được đánh giá cho 12 lượt để báo cáo C. Không dùng bảng test để chốt cấu hình. Quyết định mô hình cuối ngày 03/11 thuộc bước phối hợp với nhóm.
- Thời gian là đo vòng train trên cùng máy của lượt chạy trước, không gồm xuất file/vẽ hình; không đại diện cho mọi máy.
