---
paths:
  - "BTL02 Emotion Recognition/src/**"
  - "BTL02 Emotion Recognition/experiments/**"
  - "BTL02 Emotion Recognition/configs/**"
---

# Quy tắc code BTL02

- **Chỉ PyTorch.** Không `import tensorflow`/`keras` (hook chặn). Code mẫu Keras trong `Yêu cầu/` chỉ để tham khảo cách tổ chức.
- **Đường dẫn** lấy từ `src/paths.py` (`ROOT`, `DATA_DIR`, `RESULTS_DIR`, `FER_CSV`, `EMOTIONS`). Không ghi đường dẫn tuyệt đối (hook chặn).
- Chạy dạng module từ thư mục BTL02: `python -m src.train ...`, `python -m experiments.run_E1`.
- **Thiết bị:** `device = "cuda" if torch.cuda.is_available() else "cpu"`. Code phải chạy được cả trên máy Trúc (CPU) lẫn Colab/Kaggle (GPU).
- **Seed:** đặt `random`, `numpy`, `torch` (và `torch.cuda`) qua một hàm `set_seed(seed)` trong `src/train.py`.
- **Cấu hình:** đọc `configs/baseline.yaml`, thực nghiệm chỉ ghi đè đúng một khóa (`override={"model.activation": "elu"}`). Không sửa baseline sau 24/11.
- **Chia tập:** cột `Usage` của `fer2013.csv`: `Training` → train, `PublicTest` → validation, `PrivateTest` → test. Chuẩn hóa bằng mean/std của **tập train**.
- **PrivateTest** chỉ được đọc trong `experiments/run_final.py` và `src/evaluate.py` khi đánh giá mô hình cuối. Không dùng để chọn cấu hình, chọn epoch hay early stopping.
- **Vòng train** tự viết: `model.train()` khi train, `model.eval()` + `torch.no_grad()` khi đánh giá; checkpoint theo `val_loss` nhỏ nhất; early stopping theo `val_loss`.
- **Log theo epoch** ghi CSV `results/<nhóm>/logs/<mã run>.csv`, các cột: `epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time_s`.
- **Checkpoint** `results/<nhóm>/checkpoints/<mã run>.pt` (không đưa lên git), trừ `results/final/best_model.pt`.
- **summary.csv** của mỗi nhóm, mỗi run 1 dòng: `run_id, change, seed, best_epoch, best_val_loss, val_acc_at_best, train_acc_at_best, epochs_ran, train_time_s, n_params`.
- **Ghi phân công trong code:** đầu mỗi file `.py` có docstring nêu file làm gì và dòng `Phụ trách: <tên>` theo `docs/Kế hoạch.md` mục 8 (Trúc: paths, data, train, evaluate, run_final; Khang: models, run_E1, run_E2, phần filter trong visualize; Trung: run_E3, run_E4, confusion matrix; Nhân: run_E5, run_E6, run_E7, Grad-CAM).
- Comment và docstring bằng tiếng Việt, ngắn, nói vì sao chứ không nhắc lại code.
