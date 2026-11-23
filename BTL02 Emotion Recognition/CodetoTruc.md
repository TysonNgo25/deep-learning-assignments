# Nhật ký BTL02 – Emotion Recognition (Trúc)

Nhật ký cá nhân của Trúc, không nộp. Quy tắc ở `CLAUDE.md` mục 1. Kế hoạch: [docs/Kế hoạch.md](docs/Kế%20hoạch.md). Nhật ký thực nghiệm nộp cho thầy nằm ở [NhatKyThucNghiem.md](NhatKyThucNghiem.md).

---

## T2 23/11/2026 – Lập kế hoạch, tạo khung thư mục, cấu hình Claude cho BTL02

### Việc theo kế hoạch

Theo [docs/Kế hoạch.md](docs/Kế%20hoạch.md) mục 7, việc tối 23/11 là "tải fer2013.csv, dựng thư mục, paths.py, data.py, EDA". Hôm nay đã làm:

1. Đọc đề `Yêu cầu/Emotion_Recognition.pdf`, báo cáo mẫu `demo_report.pdf` và 3 notebook code mẫu (Keras/TF 1.14).
2. Viết kế hoạch đầy đủ: `docs/Kế hoạch.md` (yêu cầu, rủi ro, baseline, 19 run, lịch ngày, phân công 4 thành viên, checklist nộp).
3. Tạo khung thư mục: `configs/`, `data/`, `src/`, `experiments/`, `results/`, `report/`, `slides/`, cùng `README.md`, `requirements.txt`, `.gitignore`, `NhatKyThucNghiem.md`, `configs/baseline.yaml`, `src/paths.py`.
4. Thêm bộ công cụ Claude cho BTL02: skill, agent, rule, lệnh, hook; cập nhật `CLAUDE.md`.

Chưa làm: tải `fer2013.csv`, `data.py`, EDA (xem Còn treo).

### Commit trong ngày

_(Trúc điền ở mục ngày sau)_

### Kết quả

- [docs/Kế hoạch.md](docs/Kế%20hoạch.md): kế hoạch và phân công.
- [configs/baseline.yaml](configs/baseline.yaml): baseline Medium CNN, ReLU, Adam 1e-3, batch 64, tối đa 40 epoch, patience 8.
- [src/paths.py](src/paths.py): hằng số đường dẫn và danh sách 7 cảm xúc.
- `.claude/` ở gốc repo: 5 skill `*-btl02`, 3 agent `btl02-*`, 3 rule, 4 lệnh `/btl02-*`, hook `btl02_guard.py` trong `.claude/settings.json`.

### Giải thích

- `src/paths.py`: mọi đường dẫn tính từ `Path(__file__).resolve().parent.parent` (thư mục BTL02), nên code chạy được trên máy Trúc, máy thầy và Colab mà không sửa. `EMOTIONS` theo đúng thứ tự nhãn 0–6 của FER2013.
- `baseline.yaml`: mỗi thực nghiệm chỉ ghi đè một khóa trong file này, nhờ vậy so sánh giữa các run là công bằng.
- Hook `btl02_guard.py`: chặn những lỗi làm mất điểm hoặc vi phạm quy tắc (dùng Keras, đường dẫn tuyệt đối, đánh giá PrivateTest lần hai, push thay Trúc).

### Cách tự kiểm tra lại

Mở Claude Code ở gốc repo: đầu phiên phải thấy dòng `[BTL02 Emotion Recognition] Hôm nay ...` do hook in ra. Gõ `/btl02-bat-dau` phải ra việc hôm nay theo kế hoạch.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao dùng PublicTest làm validation mà không tách từ Training?* FER2013 chia sẵn 3 phần; dùng PublicTest làm validation và PrivateTest làm test giống cách chấm của cuộc thi Kaggle, và giữ PrivateTest hoàn toàn chưa nhìn thấy đến lúc đánh giá cuối.
- *Vì sao mỗi lần chỉ đổi một tham số?* Đổi nhiều tham số cùng lúc thì không biết thay đổi accuracy là do tham số nào.

### Còn treo

- [ ] Tải `fer2013.csv` từ Kaggle vào `data/`
- [ ] Viết `src/data.py` + EDA
- [ ] Tạo conda env `btl02`, kiểm tra có GPU không (`nvidia-smi`)
- [ ] Xóa file `Kế hoạch.md` cũ ở gốc thư mục BTL02 (bản đúng ở `docs/`)
