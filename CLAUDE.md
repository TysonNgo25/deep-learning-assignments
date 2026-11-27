# Quy tắc của Trúc

## 0. Bài đang làm

Hai bài cùng hạn **T7 28/11/2026**. Đầu phiên hỏi Trúc đang làm bài nào nếu chưa rõ; công việc xen kẽ: lúc BTL02 train trên GPU thì viết code BTL03.

| Bài | Nội dung | Ai làm | Kế hoạch |
| --- | --- | --- | --- |
| **BTL02 Emotion Recognition** | FER2013, CNN PyTorch, thực nghiệm E1–E7 | Trúc làm, bảng phân công đủ 4 thành viên Nhóm 5 | [BTL02 Emotion Recognition/docs/Kế hoạch.md](BTL02%20Emotion%20Recognition/docs/K%E1%BA%BF%20ho%E1%BA%A1ch.md) |
| **BTL03 Chuyển code Keras → PyTorch** | 11 notebook Keras trong `Yêu cầu/examples/` (MLP, CNN, CNN+LSTM, SegNet, phát hiện xe) chuyển sang PyTorch, so với kết quả gốc | Trúc làm, bảng phân công đủ 4 thành viên Nhóm 5 (chốt 27/11) | [BTL03 - Chuyển code từ keras sang pytorch deep learning/docs/Kế hoạch.md](BTL03%20-%20Chuy%E1%BB%83n%20code%20t%E1%BB%AB%20keras%20sang%20pytorch%20deep%20learning/docs/K%E1%BA%BF%20ho%E1%BA%A1ch.md) |

- BTL01 đã nộp (8/11/2026): chỉ đọc, không chạy lại, không sửa kết quả.

## 1. Nhật ký `CodetoTruc.md`

Mỗi bài có file nhật ký riêng:

| Bài | Nhật ký | Kế hoạch để đối chiếu |
| --- | --- | --- |
| BTL01 | [BTL01 - Function Approximation/CodetoTruc.md](BTL01%20-%20Function%20Approximation/CodetoTruc.md) | [Kế hoạch.md](BTL01%20-%20Function%20Approximation/K%E1%BA%BF%20ho%E1%BA%A1ch.md) |
| BTL02 | [BTL02 Emotion Recognition/CodetoTruc.md](BTL02%20Emotion%20Recognition/CodetoTruc.md) | [docs/Kế hoạch.md](BTL02%20Emotion%20Recognition/docs/K%E1%BA%BF%20ho%E1%BA%A1ch.md) mục 7 |
| BTL03 | [BTL03 - Chuyển code từ keras sang pytorch deep learning/CodetoTruc.md](BTL03%20-%20Chuy%E1%BB%83n%20code%20t%E1%BB%AB%20keras%20sang%20pytorch%20deep%20learning/CodetoTruc.md) | [docs/Kế hoạch.md](BTL03%20-%20Chuy%E1%BB%83n%20code%20t%E1%BB%AB%20keras%20sang%20pytorch%20deep%20learning/docs/K%E1%BA%BF%20ho%E1%BA%A1ch.md) mục 7 |

- **Ngày nào có commit trong một bài thì ngày đó phải có một mục trong `CodetoTruc.md` của bài đó.** Không có ngoại lệ, kể cả commit nhỏ hay commit revert.
- Mỗi ngày một mục. Ngày có nhiều commit thì gộp chung vào mục của ngày đó.
- Mục mới thêm ở **cuối file**, ngăn với mục trước bằng `---`.
- Tiêu đề theo mẫu: `## <Thứ> dd/mm/yyyy – <việc chính trong ngày>`, ví dụ `## T5 29/10/2026 – Viết src/data.py, sinh dữ liệu`.
- Viết bằng tiếng Việt, giải thích kỹ (để Trúc ôn vấn đáp và viết báo cáo).

### Nội dung mỗi mục

Theo bố cục của mục ngày 29/10 (BTL01), bỏ phần nào không có nội dung:

1. **Việc theo kế hoạch**: đối chiếu với kế hoạch của bài.
2. **Commit trong ngày**: hash ngắn + message của từng commit, kèm một câu nói commit đó làm gì.
3. **Kết quả**: file sinh ra / file đã sửa, lệnh để chạy.
4. **Giải thích từng hàm**: hàm làm gì và vì sao viết như vậy.
5. **Đã kiểm tra**: kiểm tra gì, ra số liệu nào.
6. **Cách tự kiểm tra lại**: lệnh chạy trong Anaconda Prompt và kết quả đúng phải thấy.
7. **Số liệu cho báo cáo** (nếu có).
8. **Câu hỏi vấn đáp có thể gặp**.
9. **Còn treo**: checkbox `- [ ]`. Việc treo của ngày trước làm xong thì quay lại tick `- [x]`.

### Khi nào cập nhật

- Làm xong việc trong ngày thì viết mục nhật ký **trước khi Trúc commit**, để nhật ký đi cùng commit.
- Đầu mỗi phiên, chạy `git log --date=short --format="%ad %h %s" -- "<thư mục bài>"` và so với các tiêu đề trong `CodetoTruc.md`. Ngày nào có commit mà chưa có mục thì báo Trúc và bổ sung. Với BTL02 và BTL03, hook SessionStart tự in danh sách ngày thiếu.

## 2. Git

- Trúc tự commit và push. Claude chỉ sửa file trong máy và đưa lệnh git cho Trúc chạy.
- Không bao giờ thêm dòng `Co-Authored-By: Claude` vào commit. Đây là repo nhóm, có chấm điểm.
- Nếu Trúc nhờ commit: không thêm trailer, và hỏi lại trước khi push.
- Hook `.claude/hooks/btl02_guard.py` chặn `git push` và commit có `Co-Authored-By`.

## 3. BTL02 – quy tắc riêng

- **Chỉ PyTorch.** Code mẫu và báo cáo mẫu trong `Yêu cầu/` là Keras/TF 1.14, chỉ mượn cách tổ chức. Không sửa thư mục `Yêu cầu/`.
- **Chia tập:** Training → train, PublicTest → validation (chọn cấu hình, early stopping), PrivateTest → test, **chỉ đánh giá một lần** ở mô hình cuối (`experiments/run_final.py`).
- **Mỗi run chỉ đổi một biến** so với `configs/baseline.yaml`. Baseline chốt 24/11, sau đó không sửa.
- **Nhật ký thực nghiệm** `NhatKyThucNghiem.md` là file nộp cho thầy: chạy xong run nào ghi ngay dòng đó (`/btl02-ghi-run`). Khác với `CodetoTruc.md` (nhật ký cá nhân, không nộp).
- **Phân công:** mỗi file `.py` có dòng `Phụ trách: <tên>` đúng với `docs/Kế hoạch.md` mục 8; báo cáo và slide có bảng phân công + tỷ lệ cộng đủ 100%.
- Đề bắt buộc có hình **filter lớp conv cuối** trong báo cáo.
- Môi trường: conda env `btl02` trên máy; chạy thực nghiệm trên GPU Colab/Kaggle. Chi tiết ở skill `chay-thuc-nghiem-btl02`.
- Tên file nộp: `BTLON02_Emotion_HuynhNgocTruc_3123411312.zip` (Trúc là nhóm trưởng, nộp).

## 4. Công cụ Claude cho BTL02

| Loại | Tên | Dùng khi |
| --- | --- | --- |
| Lệnh | `/btl02-bat-dau` | Đầu phiên: việc hôm nay, ngày thiếu nhật ký, run còn thiếu |
| Lệnh | `/btl02-ghi-run <mã run>` | Ghi run vào `NhatKyThucNghiem.md` từ log CSV |
| Lệnh | `/btl02-tien-do` | Tiến độ so với kế hoạch và checklist |
| Lệnh | `/btl02-cham-thu` | Chấm thử + đối chiếu số liệu |
| Skill | `chay-thuc-nghiem-btl02` | Trước khi chạy bất kỳ lệnh python nào |
| Skill | `nhat-ky-btl02` | Viết `CodetoTruc.md`, ghi nhật ký thực nghiệm |
| Skill | `viet-bao-cao-btl02` | Viết/ghép báo cáo, làm slide |
| Skill | `on-van-dap-btl02` | Ôn vấn đáp, soạn tài liệu cho từng thành viên |
| Skill | `dong-goi-btl02` | Checklist và nén zip nộp |
| Agent | `btl02-cham-rubric` | Chấm thử theo đề, tìm chỗ mất điểm (chỉ đọc) |
| Agent | `btl02-doi-chieu-so-lieu` | Đối chiếu số báo cáo/nhật ký với log (chỉ đọc) |
| Agent | `btl02-giam-khao-van-dap` | Soạn câu hỏi vấn đáp theo thành viên/file (chỉ đọc) |
| Rule | `.claude/rules/btl02-code.md`, `btl02-thuc-nghiem.md`, `btl02-bao-cao.md` | Tự nạp khi làm việc với file trong thư mục tương ứng |
| Hook | `.claude/hooks/btl02_guard.py` | SessionStart: nhắc hạn, ngày thiếu nhật ký, số run. PreToolUse: chặn push, Co-Authored-By, tensorflow/keras, đường dẫn tuyệt đối, sửa `Yêu cầu/`, chạy lại `run_final`; hỏi lại khi sửa baseline sau 24/11, sửa `results/final/`, xóa `results/` hoặc `data/` |

## 5. BTL03 – quy tắc riêng

Thư mục: `BTL03 - Chuyển code từ keras sang pytorch deep learning/` (gọi tắt BTL03).

- **Phạm vi:** 11 notebook có Keras trong `Yêu cầu/examples/` (bảng ở `docs/Kế hoạch.md` mục 3). Iris (sklearn) và `colab_getting_started` không có Keras: không chuyển, ghi chú trong báo cáo. Đối chiếu với notebook **trong máy**, không dùng bản GitHub `nhduong/intro_deep` (khác nhau).
- **Mỗi notebook gốc → một `pytorch/<thư mục>/<tên>_pytorch.ipynb`**, cùng thứ tự mục. Keras gốc chỉ được trích trong ô markdown; ô code chỉ PyTorch (hook chặn). Phần dùng chung để trong `pytorch/common.py`.
- **Trung thành với gốc:** cùng kiến trúc, siêu tham số, tiền xử lý (Gender, Emotion, SegNet gốc **không chia 255** thì cũng không chia). Khác đi thì để ở ô "Thí nghiệm phụ".
- **Kiểm số tham số** mỗi mô hình bằng `assert` so với `model.summary()` gốc. Lệch hợp lệ duy nhất: LSTM +1.024 (PyTorch 2 bias), SegNet so với "Trainable params".
- **Kết quả** ghi `results/<notebook>/` và `results/so_sanh.csv` (bảng Keras ↔ PyTorch dùng cho báo cáo, không gõ số tay).
- **Dữ liệu** trong `data/` (git bỏ qua). Gender đã kiểm 23/11: lệch lớp 78,85% male, 662/5968 ảnh val đen; nêu trong báo cáo, không xóa.
- Không sửa, xóa, di chuyển gì trong `Yêu cầu/` (hook chặn).
- Môi trường: dùng chung env `btl02` + `requirements.txt` của BTL03; notebook nặng chạy GPU Colab/Kaggle. Chi tiết ở skill `chay-notebook-btl03`.
- **Phân công:** Nhóm 5, 25% mỗi người (báo cáo phụ lục A). Mỗi notebook có dòng `Phụ trách: <tên>` ở ô đầu, mỗi mục báo cáo có dòng `Người viết:`; giữ khớp với bảng.
- Tên file nộp: báo cáo `report/BaoCao_BTL03_Nhom05.docx/.pdf`, zip `BTL03_Nhom05_HuynhNgocTruc_3123411312.zip` (chốt 27/11).

## 6. Công cụ Claude cho BTL03

| Loại | Tên | Dùng khi |
| --- | --- | --- |
| Lệnh | `/btl03-bat-dau` | Đầu phiên: notebook phải làm hôm nay, ngày thiếu nhật ký, notebook đã xong |
| Lệnh | `/btl03-ghi-ket-qua <notebook>` | Ghi kết quả PyTorch cạnh số Keras gốc vào `results/so_sanh.csv` |
| Lệnh | `/btl03-tien-do` | Tiến độ so với kế hoạch và checklist |
| Lệnh | `/btl03-cham-thu` | Chấm thử + đối chiếu từng notebook với Keras gốc |
| Skill | `chuyen-doi-btl03` | Trước khi viết/sửa bất kỳ notebook `_pytorch` hay `common.py`: quy trình, công thức các phần khó, số tham số |
| Skill | `chay-notebook-btl03` | Trước khi chạy notebook/lệnh python nào; CPU hay GPU, đưa dữ liệu lên Colab/Kaggle |
| Skill | `nhat-ky-btl03` | Viết `CodetoTruc.md` của BTL03 |
| Skill | `viet-bao-cao-btl03` | Viết/ghép báo cáo, xuất Word/PDF |
| Skill | `on-van-dap-btl03` | Ôn vấn đáp kiểu "đoạn Keras này viết PyTorch thế nào" |
| Skill | `dong-goi-btl03` | Checklist và nén zip nộp |
| Agent | `btl03-doi-chieu-keras` | So từng lớp, siêu tham số, số tham số, kết quả của notebook PyTorch với bản Keras gốc; đối chiếu số báo cáo (chỉ đọc) |
| Agent | `btl03-cham-rubric` | Chấm thử theo đề, tìm chỗ mất điểm (chỉ đọc) |
| Agent | `btl03-giam-khao-van-dap` | Soạn câu hỏi vấn đáp theo notebook/chủ đề (chỉ đọc) |
| Rule | `.claude/rules/btl03-code.md`, `btl03-ket-qua.md`, `btl03-bao-cao.md` | Tự nạp khi làm việc với `pytorch/`, `results/`, `report/` của BTL03 |
| Hook | `.claude/hooks/btl03_guard.py` | SessionStart: nhắc hạn, ngày thiếu nhật ký, số notebook đã chuyển, số dòng `so_sanh.csv`. PreToolUse: chặn tensorflow/keras trong ô code `pytorch/`, đường dẫn tuyệt đối, sửa/xóa/di chuyển `Yêu cầu/`; hỏi lại khi ghi đè cả `so_sanh.csv`, xóa `results/` hoặc `data/` |
