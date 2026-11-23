# Nhật ký BTL03 – Chuyển code từ Keras sang PyTorch

Nhật ký cá nhân của Trúc (không nộp). Quy tắc ở `CLAUDE.md` mục 1, đối chiếu lịch ở `docs/Kế hoạch.md` mục 7.

---

## T2 23/11/2026 – Đọc đề, dựng thư mục, kiểm dữ liệu Gender, bộ công cụ Claude Code

### 1. Việc theo kế hoạch

Đúng dòng T2 23/11 trong `docs/Kế hoạch.md` mục 7: đọc hết `Yêu cầu/`, dựng thư mục, tải và kiểm dữ liệu Gender, lập kế hoạch. Đã tick.

### 2. Commit trong ngày

_(Trúc điền ở mục ngày sau)_

### 3. Kết quả

- Thư mục: `data/gender`, `data/emotion`, `pytorch/` (6 thư mục con trùng tên `Yêu cầu/examples/`), `results/`, `report/`, `docs/`.
- File: `.gitignore`, `README.md`, `requirements.txt`, `docs/Kế hoạch.md`, `CodetoTruc.md`.
- Dữ liệu: 4 file `.npy` Gender chép vào `data/gender/` (git bỏ qua).
- Bộ công cụ Claude Code cho BTL03 trong `.claude/` (hook, command, rule, skill, agent), cập nhật `CLAUDE.md` và `README.md` gốc.

### 5. Đã kiểm tra

- Đề gồm 2 slide (01 cú pháp Keras, 02 đi qua từng notebook), `README.md` (Keras 2.0.9, TF 1.1) và 13 notebook. Notebook trong máy **khác** bản GitHub `nhduong/intro_deep`: một số đã chạy lại bằng tf.keras 2.14, nên lấy số đối chiếu từ bản trong máy.
- Gender: train (11938, 64, 64, 1), val (5968, 64, 64, 1), float64, giá trị 0–255. Nhãn: cột 0 female, cột 1 male.
- Gender lệch lớp: 78,85% male ở cả hai tập. Đoán toàn "male" đã được 0.7885; log gốc epoch 1 có val_acc đúng bằng 0.7885.
- 662/5968 ảnh val đen hoàn toàn (612 male, 50 female).

### 7. Số liệu cho báo cáo

| Notebook | Tham số Keras | Kết quả gốc |
|---|---|---|
| MNIST_MLP | 669.706 | 0.9648 |
| 1_MNIST_CNN | 1.817.610 | 0.9884 |
| 2_Human_Gender_CNN | 4.063.570 | 0.8264 (có callback: 0.8773) |
| 3_Emotion_Reg_CNN | 19.028.287 | 0.5954 (có augmentation: 0.6163) |
| MNIST_CNN_LSTM | 1.747.073 | MAE 0.6502 |
| 1_segnet | 29.458.211 | 0.7732 |
| 2_obj_detection | 83.367 | 0.9523 |

### 8. Câu hỏi vấn đáp có thể gặp

- **Gender đạt 82–88% có tốt không?** Chỉ hơn mức đoán toàn "male" (78,85%) khoảng 4–9 điểm, vì dữ liệu lệch lớp. Nên xem thêm recall lớp female.
- **Vì sao không chia 255 cho ảnh Gender?** Để giữ đúng tiền xử lý của bản Keras, so sánh mới công bằng. Chia 255 là thí nghiệm phụ.
- **Vì sao số tham số LSTM bản PyTorch nhiều hơn 1.024?** PyTorch có hai vector bias (`b_ih`, `b_hh`) cho mỗi cổng, Keras có một. Hai lớp × 4 cổng × 128 = 1.024.

### 9. Còn treo

- [ ] Hỏi thầy: cá nhân hay nhóm, tên file zip, có cần slide không
- [ ] Sinh dữ liệu Emotion từ `fer2013.csv`
- [ ] Viết `pytorch/common.py` (T3 24/11)
