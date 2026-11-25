# Bàn giao phiên chat 23/11/2026 (bản 2) → Claude Code

Phiên chat trên claude.ai ngày T2 23/11/2026, kết thúc lúc 18:20. Đây là bản 2, viết sau khi đã đọc hết thư mục `Yêu cầu/`. Claude Code đọc file này đầu phiên rồi làm tiếp BTL02.

## 1. Bối cảnh

- Bài: **BTL02 Emotion Recognition**, môn Học sâu. Dữ liệu FER2013, CNN viết bằng **PyTorch**. Hạn nộp **T7 28/11/2026**.
- Trúc làm một mình. Bảng phân công vẫn ghi đủ 4 thành viên Nhóm 5 (`docs/Kế hoạch.md` mục 8). Trước T6 27/11 phải gửi cho mỗi bạn phần đứng tên mình để đọc trước.
- Trúc là nhóm trưởng, nộp file `BTLON02_Emotion_HuynhNgocTruc_3123411312.zip`.
- Đề yêu cầu: report, slide, nhật ký thực nghiệm, code ghi rõ phân công. Báo cáo phải có tóm tắt code, nhiều thực nghiệm, **filter lớp conv cuối**, và phần phân tích.
- Máy Trúc **không có GPU** (không có `nvidia-smi`). E1–E7 và mô hình cuối chạy trên Colab/Kaggle.

## 2. Tình trạng repo lúc 18:20

| Đã xong | Ghi chú |
|---|---|
| `src/paths.py`, `src/data.py` + EDA (`results/eda/`) | Số ảnh đúng 28.709 / 3.589 / 3.589. Lệch lớp 16,55 lần (Happy / Disgust). `class_weights` nằm trong `stats.json` |
| `src/models.py` | 4 kiến trúc, tùy chọn BN / activation / dropout, có `last_conv()` để vẽ filter |
| `src/train.py` | Có seed, checkpoint, early stopping, log CSV, summary, `--set`, `--thu`, `--workers`. Chặn ghi đè log đã có |
| Chạy thử BASE (`results/_thu/`) | Với lr 1e-3, 2 epoch × 20 batch: val_acc đứng yên ở 0,235 |
| `.claude/` (skill, agent, rule, lệnh, hook), `CLAUDE.md` | Đã cài |

## 3. Thay đổi trong bản 2 (sau khi đọc hết code mẫu)

| Thay đổi | File đã sửa | Lý do |
|---|---|---|
| Baseline: lr **1e-3 → 1e-4**, max_epochs **40 → 60**, patience **8 → 10** | `configs/baseline.yaml`, `docs/Kế hoạch.md` mục 3 | Code mẫu (`demo_code.html`) dùng Adam 1e-4, patience 10, đạt tốt nhất ở epoch 31–32. Lần chạy thử với lr 1e-3 bị kẹt |
| E3 đổi giá trị: E3-1 SGD lr 1e-3, E3-2 Adam lr 1e-3, E3-3 Adam 1e-4 + plateau | Kế hoạch mục 4 | Vì baseline đã chuyển sang lr 1e-4 |
| E7 tách thành E7-1 (VGG16 train từ đầu) và E7-2 (VGG16 pretrained), cùng ảnh 128×128×3; nâng lên "làm nếu có GPU" | Kế hoạch mục 4, 7, 8 | `code/train.html` + `valid_*.html` của thầy đạt val_acc 0,699 / 0,707, cao hơn các CNN nhỏ (0,64–0,67) |
| **Grad-CAM bắt buộc** tại conv đầu, giữa, cuối × 7 cảm xúc; confusion matrix cho cả valid và test | Kế hoạch mục 5 | `valid_vgg16.html` làm đúng như vậy |
| Thêm mục 12 "Ghi chú từ code mẫu" và 2 rủi ro mới (thứ tự nhãn, lỗi trong code mẫu) | Kế hoạch mục 2, 12 | |

## 4. Việc Claude Code phải làm trước

- [ ] Ghi thay đổi baseline vào `CodetoTruc.md` (mục ngày 23/11 hoặc 24/11). Dòng 34 trong "Kết quả" vẫn ghi Adam 1e-3, 40 epoch, patience 8.
- [ ] Sửa các file trong `.claude/` (công cụ chat không ghi được vào `.claude`):
  - `rules/btl02-thuc-nghiem.md`: thêm mã run `E7-2`. Ghi rõ E7 là nhánh riêng, ảnh 128×128×3.
  - `skills/chay-thuc-nghiem-btl02/SKILL.md`: cập nhật bảng lệnh theo `train.py` thật (`--thu`, `--set`, `--change`, `--workers 0`). Ghi chú máy không có GPU.
  - `skills/viet-bao-cao-btl02/SKILL.md` và `agents/btl02-cham-rubric.md`: Grad-CAM là bắt buộc; confusion matrix cho valid và test.
  - `hooks/btl02_guard.py`: hook hỏi lại khi sửa `baseline.yaml` từ 25/11. Chốt baseline trước ngày đó.
- [ ] Chạy thử lại `python -m src.train --run BASE --thu --workers 0` với lr 1e-4. val_acc phải tăng qua 2 epoch, không còn đứng ở 0,235.
- [ ] Xem lại 2 việc còn treo về `train.py` trong `CodetoTruc.md` (thứ tự `set_seed` / `get_dataloaders`, `--workers`). Có vẻ cả hai đã xong trong code, kiểm tra rồi tick.

## 5. Việc tiếp theo (theo lịch, `docs/Kế hoạch.md` mục 7)

| Ngày | Việc |
|---|---|
| T3 24/11 | Chạy BASE đầy đủ trên Colab/Kaggle, ghi nhật ký, chốt baseline. Viết `experiments/run_E1.py` … `run_E6.py` (gọi `src.train.run(run_id, override, change)`) và notebook/hướng dẫn chạy trên Colab |
| T4 25/11 | Chạy E1, E2. Viết `src/evaluate.py` (accuracy, confusion matrix valid + test, classification report, ca đúng / ca sai) |
| T5 26/11 | Chạy E3–E6, E7 nếu kịp (cần thêm VGG16 128×128×3 vào `models.py` / `data.py`, `torchvision.models.vgg16`). Viết `src/visualize.py`: filter conv cuối, feature map, Grad-CAM (conv đầu / giữa / cuối × 7 cảm xúc) |
| T6 27/11 | `experiments/run_final.py`: 3 seed + đánh giá PrivateTest 1 lần. Viết báo cáo, làm slide, gửi phần đứng tên cho từng bạn |
| T7 28/11 | Đối chiếu số liệu, chạy `check_submission.py`, nén zip, nộp |

Dự phòng khi trễ: bỏ E7 → rút E5 còn ReLU vs Sigmoid → rút E6 còn E6-1 và E6-3.

## 6. Lưu ý kỹ thuật

- Thứ tự nhãn: dùng thứ tự gốc FER2013 0–6 (`src/paths.py`). Code mẫu dùng LabelEncoder theo chữ cái (Neutral đứng trước Sad và Surprise).
- Trong mẫu, Small có 38,9 triệu tham số, nhiều hơn Medium_01 (12,1 triệu), vì chỉ pool 2 lần. `python -m src.models` in bảng số tham số để đối chiếu, đưa vào phân tích E1.
- Không chép hai chỗ lỗi của code mẫu: vẽ `x_train[valid_idx]` thay cho ảnh validation, và mật khẩu email ghi trong code.
- Git: Trúc tự commit và push, không thêm `Co-Authored-By`.

## 7. Câu mở đầu gợi ý trong Claude Code

> Đọc `BTL02 Emotion Recognition/docs/BanGiao.md` (bản 2), làm hết mục 4, rồi tiếp tục lịch ngày 24/11.
