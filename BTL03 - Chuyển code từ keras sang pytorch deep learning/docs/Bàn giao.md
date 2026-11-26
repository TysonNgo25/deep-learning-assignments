# BTL03 – Bàn giao sang Claude Code (25/11/2026)

Tóm tắt phiên lập kế hoạch 23–25/11 (làm trên Claude app) để Claude Code tiếp tục. Đọc file này trước, rồi `docs/Kế hoạch.md`.

## 1. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Đọc đề | Xong: 2 slide PDF, `Yêu cầu/README.md`, 13 notebook. Ghi chú từng notebook ở Kế hoạch mục 12 |
| Thư mục, README, requirements, `.gitignore` | Xong (commit `1c08b472`, 23/11) |
| Dữ liệu Gender | Xong: 4 file `.npy` trong `data/gender/`, đã kiểm (Kế hoạch mục 8) |
| Dữ liệu xe, bàn tay | Có sẵn trong `Yêu cầu/examples/5. Others/`, **đã lên git** (commit `b885df2f`), clone trên Colab là có |
| Dữ liệu Emotion | Chưa: sinh từ `../BTL02 Emotion Recognition/data/fer2013.csv` |
| Bộ công cụ Claude Code BTL03 | Đã cài trong `.claude/` (hook, 4 lệnh, 3 rule, 6 skill, 3 agent); `CLAUDE.md` mục 0, 5, 6 |
| Code | **Chưa có gì**: `pytorch/common.py` và 11 notebook đều chưa viết |
| Lịch | Trễ 1 ngày (24/11 không làm). Lịch mới ở Kế hoạch mục 7 bản 2 |

## 2. Việc làm ngay (T4 25/11 chiều–tối)

1. `/btl03-bat-dau`.
2. Nạp skill `chuyen-doi-btl03`, viết `pytorch/common.py` (danh sách hàm trong rule `btl03-code`).
3. `getting_started_pytorch`, `Func_apxm_MLP_pytorch`, `MNIST_MLP_pytorch`: chạy đủ trên CPU, ghi `so_sanh.csv`.
4. Viết sẵn 3 notebook thư mục 2, chạy thử với `CHAY_THU = True` để kiểm số tham số; chạy thật trên GPU ngày T5.
5. Nhật ký (`nhat-ky-btl03`), đưa lệnh commit cho Trúc.

Câu mở đầu gợi ý trong Claude Code:

> Đọc `BTL03 - Chuyển code từ keras sang pytorch deep learning/docs/Bàn giao.md` và `docs/Kế hoạch.md`, rồi làm việc của T4 25/11: viết `pytorch/common.py` và 3 notebook thư mục 0–1 theo skill `chuyen-doi-btl03`.

## 3. Đã chốt

- Phạm vi 11 notebook có Keras, **gồm cả thư mục 5** (SegNet, obj_detection). Iris và `colab_getting_started` không chuyển, chỉ ghi chú.
- Đối chiếu với notebook **trong máy**, không dùng bản GitHub `nhduong/intro_deep` (khác nhau; nhiều bản trong máy đã chạy lại bằng tf.keras 2.14).
- Không cần cài TensorFlow: số tham số và kết quả Keras lấy từ output có sẵn trong notebook gốc (Kế hoạch mục 3).
- Giữ nguyên tiền xử lý gốc, kể cả chỗ không chia 255 (Gender, Emotion, SegNet). Thay đổi thì để ở ô "Thí nghiệm phụ".
- Tên notebook `<tên gốc>_pytorch.ipynb`; kết quả bản Training Utilities đặt tiền tố `tu_`.
- Dùng chung env `btl02` + `requirements.txt` của BTL03.
- `Func_apxm_MLP` cùng bài toán với BTL01: được đọc `BTL01 - Function Approximation/src/` để tham khảo, nhưng bám cấu hình notebook gốc, không sửa BTL01.

## 4. Sự thật đã kiểm, dùng cho báo cáo

- Gender: train (11938, 64, 64, 1), val (5968, 64, 64, 1), float64, 0–255; cột 0 female, cột 1 male. 78,85% male ở cả hai tập (đoán toàn male = 0.7885, đúng bằng val_acc epoch 1 của log gốc). 662/5968 ảnh val đen hoàn toàn (612 male, 50 female). Phải `np.rot90(k=3, axes=(1, 2))`.
- Số tham số kỳ vọng: MNIST_MLP 669.706 · MNIST_CNN 1.817.610 · Gender 4.063.570 · Emotion 19.028.287 · CNN_LSTM 1.748.097 (Keras 1.747.073 + 1.024 bias) · SegNet 29.442.325 trainable (Keras tổng 29.458.211) · obj_detection 83.367. MNIST_CNN và Gender đã tính tay khớp với summary gốc.

## 5. Lưu ý môi trường

- Máy Trúc: GPU AMD (không CUDA), hay thiếu RAM. Notebook nặng chạy Colab/Kaggle (skill `chay-notebook-btl03`).
- Trên Claude app (Cowork), shell trên máy từng không khởi động được vì Windows thiếu bộ nhớ, và `git status` cả repo chạy rất chậm (quá 2 phút). Trong Claude Code chạy trực tiếp thì không bị giới hạn này.
- Công cụ từ xa của Claude app không ghi được vào `.claude/`. Sửa hook/skill thì làm trong Claude Code.
- Hook `btl03_guard.py` đã thử 10 tình huống. Sau khi Claude Code mở phiên đầu, kiểm tra có dòng `[BTL03 Keras → PyTorch] … Notebook đã chuyển: 0/11`.
- Skill `chay-notebook-btl03` mục 3 còn câu "nếu thư mục dữ liệu đã được commit": đã commit rồi (`b885df2f`), có thể sửa lại câu đó cho gọn.

## 6. Còn mở

- [ ] Hỏi thầy: cá nhân hay nhóm 5; tên file zip (đề xuất `BTLON03_KerasPyTorch_HuynhNgocTruc_3123411312.zip`); có cần slide không.
- [ ] Sinh dữ liệu Emotion.
