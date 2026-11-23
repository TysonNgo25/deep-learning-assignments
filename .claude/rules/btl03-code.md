---
paths:
  - "BTL03 - Chuyển code từ keras sang pytorch deep learning/pytorch/**"
  - "BTL03 - Chuyển code từ keras sang pytorch deep learning/requirements.txt"
---

# Quy tắc code BTL03 (Keras → PyTorch)

- **Chỉ PyTorch trong ô code.** Không `import tensorflow`/`keras` (hook chặn). Đoạn Keras gốc đặt trong **ô markdown** ngay trên ô PyTorch tương ứng, trích nguyên văn từ `Yêu cầu/examples/...` để người chấm so được.
- **Trung thành với bản gốc** (bản trong máy, không phải GitHub): cùng kiến trúc, số filter, padding, dropout, optimizer, lr, batch, số epoch, loss, tiền xử lý. Gốc không chia 255 (Gender, Emotion, SegNet) thì cũng không chia. Muốn thử khác thì làm ở ô riêng tiêu đề "Thí nghiệm phụ", không đè lên kết quả chính.
- **Tên và vị trí:** `pytorch/<thư mục gốc>/<tên gốc>_pytorch.ipynb`. Giữ thứ tự mục của gốc: (1) Data → (2) Declare model → (3) Train → (4) Evaluate → phần phụ (PCA, dự đoán ảnh ngẫu nhiên, heatmap...).
- **`pytorch/common.py`** chứa phần dùng chung, notebook nào cũng import (`sys.path.insert(0, str(Path.cwd().parent))`):
  `ROOT`, `DATA_DIR`, `RESULTS_DIR`, `YEU_CAU_DIR` (pathlib, tương đối theo `__file__`); `set_seed`; `get_device`; `init_keras_like` (glorot uniform, bias 0); `count_params`; `fit` (trả `history` dict giống Keras, nhận callback); `evaluate`; `predict`; `ModelCheckpoint`, `EarlyStopping` (theo `val_loss`, không tự khôi phục trọng số, như Keras); `save_history`, `save_metrics`, `plot_history`; `ghi_so_sanh` (thêm/cập nhật một dòng `results/so_sanh.csv`).
- **Đường dẫn** chỉ lấy từ `common.py`. Không ghi đường dẫn tuyệt đối (hook chặn). Dữ liệu SegNet/xe đọc từ `YEU_CAU_DIR / "examples/5. Others/..."`, không chép.
- **Thiết bị:** `device = get_device()`; code chạy được cả CPU (máy Trúc, GPU AMD) và CUDA (Colab/Kaggle).
- **Seed:** `set_seed(0)` đầu mỗi notebook. Chỗ gốc có seed riêng (`np.random.seed(2018)` trong CNN_LSTM) thì giữ seed đó cho phần sinh dữ liệu.
- **Mô hình** là lớp `nn.Module` đặt tên theo `name=` của Keras (`mnist_mlp`, `emo_cnn`...). Mỗi lớp trong `__init__` đặt tên trùng `name=` của lớp Keras (`self.fc1`, `self.conv1_2`) để đối chiếu summary. Comment shape sau mỗi khối (`# (N, 64, 14, 14)`).
- **Lớp cuối trả logits**, không softmax, khi loss là `CrossEntropyLoss`. Ghi một comment nói vì sao. Softmax chỉ dùng khi `predict` cần xác suất.
- **Kiểm số tham số ngay sau khi khai báo mô hình:** in `torchinfo.summary(...)` và `assert count_params(model) == <số Keras>` (hoặc số kỳ vọng đã giải thích, ví dụ LSTM 1.748.097). Bảng số ở `docs/Kế hoạch.md` mục 3.
- **Vòng train:** `model.train()` khi train; `model.eval()` + `torch.no_grad()` khi đánh giá/dự đoán. Validation dùng đúng tập gốc dùng làm `validation_data`.
- **Kết quả** ghi vào `results/<tên notebook>/`: `history.csv`, `metrics.json`, hình `.png`; checkpoint `.pt` (git bỏ qua); log TensorBoard vào `runs/` (git bỏ qua). Rồi gọi `ghi_so_sanh(...)` (quy tắc trong `btl03-ket-qua.md`).
- **Notebook nộp phải có output**: chạy Restart & Run All, lưu lại. Không để ô lỗi (ô lỗi trong gốc như `model.get_layer()` thì viết bản đúng và ghi chú).
- Comment và markdown bằng tiếng Việt, ngắn, nói **vì sao** và **khác Keras ở đâu**, không đọc lại code.
