---
paths:
  - "BTL03 - Chuyển code từ keras sang pytorch deep learning/results/**"
---

# Quy tắc kết quả BTL03

- **`results/so_sanh.csv`** là nguồn duy nhất cho bảng so sánh trong báo cáo. Mỗi notebook một dòng (Func_apxm_MLP: một dòng mỗi vòng quét A/B/C), các cột:
  `notebook, thu_muc, chi_so, keras, pytorch, chenh_lech, keras_params, pytorch_params, epochs_keras, epochs_pytorch, optimizer, lr, batch, device, train_time_s, seed, ghi_chu`
  - `chi_so`: `test_acc`, `val_acc`, `mae`, `pixel_acc` hoặc `test_mse`, đúng thứ gốc báo.
  - `keras`: số in trong output của notebook gốc trong máy (`docs/Kế hoạch.md` mục 3), không lấy từ slide hay GitHub.
  - `chenh_lech = pytorch − keras`, 4 chữ số thập phân.
  - `ghi_chu`: điểm khác biệt có chủ đích (ví dụ "LSTM PyTorch có 2 bias → +1024 tham số", "dừng sớm ở epoch 29").
- Thêm/cập nhật dòng bằng `ghi_so_sanh()` trong `common.py` hoặc lệnh `/btl03-ghi-ket-qua`. Không gõ số bằng tay. Không ghi đè cả file (hook hỏi lại).
- **`results/<tên notebook>/`** (tên notebook không kèm `_pytorch`; hai notebook trùng tên ở thư mục 2 và 4 thì thêm tiền tố `tu_` cho bản Training Utilities: `tu_2_Human_Gender_CNN`):
  - `history.csv`: `epoch, train_loss, train_metric, val_loss, val_metric, lr, epoch_time_s`.
  - `metrics.json`: `test_loss, test_metric, n_params, epochs_ran, best_epoch, train_time_s, device, seed, torch_version`.
  - Hình: `curves.png` (loss và metric theo epoch), cùng hình riêng của notebook (PCA, heatmap, ảnh phân đoạn, vòng quét A/B/C).
- Chạy trên Colab/Kaggle: tải thư mục `results/<notebook>/` và notebook đã có output về đúng chỗ trong repo; cột `device` ghi loại GPU (ví dụ `cuda (T4)`).
- Chạy lại một notebook: ghi đè `results/<notebook>/` được, nhưng dòng trong `so_sanh.csv` phải cập nhật theo lần chạy được lưu trong notebook nộp. Output notebook và `so_sanh.csv` phải khớp.
- Chênh lệch lớn (accuracy lệch > 3 điểm, MAE lệch > 0,3, số tham số lệch ngoài dự đoán) thì dừng lại tìm nguyên nhân trước khi ghi báo cáo. Hay gặp: quên `model.eval()`, để softmax trước `CrossEntropyLoss`, sai padding, chia 255 khi gốc không chia, khởi tạo khác.
