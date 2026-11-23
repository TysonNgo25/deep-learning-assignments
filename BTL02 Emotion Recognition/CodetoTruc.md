# Nhật ký BTL02 – Emotion Recognition (Trúc)

Nhật ký cá nhân của Trúc, không nộp. Quy tắc ở `CLAUDE.md` mục 1. Kế hoạch: [docs/Kế hoạch.md](docs/Kế%20hoạch.md). Nhật ký thực nghiệm nộp cho thầy nằm ở [NhatKyThucNghiem.md](NhatKyThucNghiem.md).

---

## T2 23/11/2026 – Lập kế hoạch, khung thư mục, công cụ Claude; viết src/data.py + EDA

### Việc theo kế hoạch

Theo [docs/Kế hoạch.md](docs/Kế%20hoạch.md) mục 7, việc tối 23/11 là "tải fer2013.csv, dựng thư mục, paths.py, data.py, EDA". Hôm nay đã làm:

1. Đọc đề `Yêu cầu/Emotion_Recognition.pdf`, báo cáo mẫu `demo_report.pdf` và 3 notebook code mẫu (Keras/TF 1.14).
2. Viết kế hoạch đầy đủ: `docs/Kế hoạch.md` (yêu cầu, rủi ro, baseline, 19 run, lịch ngày, phân công 4 thành viên, checklist nộp).
3. Tạo khung thư mục: `configs/`, `data/`, `src/`, `experiments/`, `results/`, `report/`, `slides/`, cùng `README.md`, `requirements.txt`, `.gitignore`, `NhatKyThucNghiem.md`, `configs/baseline.yaml`, `src/paths.py`.
4. Thêm bộ công cụ Claude cho BTL02: skill, agent, rule, lệnh, hook; cập nhật `CLAUDE.md`.
5. Viết `src/data.py`: đọc CSV, chia tập theo `Usage`, chuẩn hóa theo mean/std của train, Dataset/DataLoader có augmentation, trọng số lớp, EDA. Chạy thử bằng CSV giả trước, sau đó chạy trên dữ liệu thật.
6. Tải `fer2013.csv` (Kaggle `nicolejyt/facialexpressionrecognition`, 301 MB, 3 cột `emotion,pixels,Usage`) vào `data/`.
7. Tạo conda env `btl02` (Python 3.11, torch 2.15 CPU, torchvision 0.29.2). Máy không có GPU NVIDIA, nên thực nghiệm chạy trên Colab/Kaggle.
8. Chạy `python -m src.data` trên dữ liệu thật: **đúng 28.709 / 3.589 / 3.589, đủ 7 lớp**. Đã tick ô 23/11 trong kế hoạch mục 7.

### Commit trong ngày

- `b885df2f` Up bài tập BTL03 (giải nén car_data, depth_hand): commit này cũng đưa thư mục `Yêu cầu/` của BTL02 (đề, báo cáo mẫu, code mẫu) lên repo.
- `396bc7a5` BTL02: thêm docs/Kế hoạch.md: bản kế hoạch đầu tiên.
- `7fc928bb` BTL02: tạo khung thư mục, baseline.yaml, README, nhật ký thực nghiệm: dựng thư mục và các file khung.
- `3b97b309` BTL02: kế hoạch, khung thư mục, bộ skill/agent/rule/lệnh/hook, cập nhật CLAUDE.md: cài bộ công cụ Claude vào `.claude/`, sửa `CLAUDE.md`.
- `ad7c5e17` BTL02: viết src/data.py (đọc FER2013, DataLoader, augmentation, EDA): bản đầu của `data.py` (còn dùng `pd.read_csv`), thêm `Phụ trách` vào `paths.py`, lệnh chạy trong README.
- _(commit đọc CSV từng dòng + `results/eda/` + số liệu EDA: Trúc điền hash ở mục ngày sau)_

### Kết quả

- [docs/Kế hoạch.md](docs/Kế%20hoạch.md): kế hoạch và phân công.
- [configs/baseline.yaml](configs/baseline.yaml): baseline Medium CNN, ReLU, Adam 1e-3, batch 64, tối đa 40 epoch, patience 8.
- [src/paths.py](src/paths.py): hằng số đường dẫn và danh sách 7 cảm xúc. Hôm nay thêm dòng `Phụ trách: Huỳnh Ngọc Trúc`.
- [src/data.py](src/data.py): **mới**, đọc dữ liệu + DataLoader + EDA.
- [README.md](README.md): thêm lệnh `python -m src.data` vào mục "Cách chạy".
- `.claude/` ở gốc repo: 5 skill `*-btl02`, 3 agent `btl02-*`, 3 rule, 4 lệnh `/btl02-*`, hook `btl02_guard.py` trong `.claude/settings.json`.

Lệnh chạy (khi đã có `data/fer2013.csv`): `python -m src.data`. Lệnh ghi vào `results/eda/`:

| File | Nội dung | Dùng trong báo cáo |
|---|---|---|
| `class_counts.csv` | Số ảnh mỗi lớp của train/val/test, % mỗi lớp trong train | Bảng mục 1 |
| `stats.json` | Tổng số ảnh, khớp chuẩn hay không, mean/std pixel train, lớp nhiều/ít nhất, tỷ lệ chênh, trọng số lớp | Số liệu mục 1, 2 |
| `class_distribution.png` | Cột nhóm: số ảnh từng lớp ở 3 tập | Hình mục 1 |
| `samples_per_class.png` | 7 hàng × 8 ảnh train ngẫu nhiên (seed 0) | Hình mục 1 |
| `mean_face_per_class.png` | Ảnh trung bình của từng lớp | Hình mục 1 (phân tích) |
| `augmentation_examples.png` | 1 ảnh gốc + 7 bản augmentation baseline | Hình mục 2 |

### Giải thích từng hàm

Phần `src/paths.py`, `baseline.yaml`, hook:

- `src/paths.py`: mọi đường dẫn tính từ `Path(__file__).resolve().parent.parent` (thư mục BTL02), nên code chạy được trên máy Trúc, máy thầy và Colab mà không sửa. `EMOTIONS` theo đúng thứ tự nhãn 0–6 của FER2013.
- `baseline.yaml`: mỗi thực nghiệm chỉ ghi đè một khóa trong file này, nhờ vậy so sánh giữa các run là công bằng.
- Hook `btl02_guard.py`: chặn những lỗi làm mất điểm hoặc vi phạm quy tắc (dùng Keras, đường dẫn tuyệt đối, đánh giá PrivateTest lần hai, push thay Trúc).

Phần `src/data.py`:

- **Hằng số `SPLITS`, `EXPECTED_COUNTS`**: ánh xạ tên trong code (`train/val/test`) sang giá trị cột `Usage` (`Training/PublicTest/PrivateTest`) và số ảnh chuẩn để tự kiểm.
- **`load_fer2013(csv_path, splits=("train", "val"))`**: đọc từng dòng bằng `csv.DictReader`, đổi ngay chuỗi pixel thành ảnh uint8, trả `{tập: (ảnh uint8 N×48×48, nhãn int64)}`. **Mặc định không đọc PrivateTest**: muốn có tập test phải viết rõ `splits=(..., "test")`, các dòng của tập không cần thì bỏ qua luôn. Nhờ vậy khó lỡ tay dùng test để chọn mô hình.
  - Vì sao không dùng `pd.read_csv`: bản đầu dùng pandas và **bị out of memory trên máy Trúc**. Pandas phải giữ cả khoảng 300 MB chuỗi pixel cùng lúc. Đọc từng dòng thì chỉ giữ ảnh đã đổi sang uint8, khoảng 83 MB cho cả 35.887 ảnh.
- **`_parse_pixels(text)`**: tách chuỗi 2304 số thành ảnh 48×48. Báo lỗi nếu số pixel sai, để phát hiện ngay file CSV hỏng.
- **`compute_mean_std(images)`**: mean/std của pixel sau khi chia 255, **chỉ trên tập train**. Cộng dồn tổng và tổng bình phương theo từng khối 4096 ảnh, rồi dùng std = √(E[x²] − E[x]²). Lý do: đổi cả 28.709×2304 pixel sang float64 cùng lúc tốn khoảng 0,5 GB, trong khi máy Trúc đang thiếu RAM.
- **`class_counts(labels)`**: `np.bincount` với `minlength=7`, để lớp không có ảnh vẫn hiện số 0.
- **`class_weights(labels)`**: w_c = N / (K·n_c), dùng cho E6-3 (CE có trọng số lớp). Lớp có đúng N/K ảnh được trọng số 1; Disgust ít ảnh nên trọng số lớn nhất (với FER2013 thật dự kiến khoảng 9,4, còn Happy khoảng 0,57).
- **`build_train_transform(aug, fill)`**: dựng augmentation từ khối `augmentation` của config, tương ứng `ImageDataGenerator` trong code mẫu Keras: `hflip` → `RandomHorizontalFlip(0.5)`; `rotation_deg`, `shift`, `zoom` → một `RandomAffine`. Tắt hết thì trả `None` (dùng cho E4-1). Tham số `fill` là giá trị điền vào viền trống sau khi xoay/dịch, xem mục "Đã kiểm tra" để biết vì sao truyền mean của train.
- **`FERDataset`**: giữ ảnh ở uint8 (train khoảng 66 MB) và chỉ đổi sang float lúc lấy từng ảnh. Thứ tự trong `__getitem__`: chia 255 → augmentation → chuẩn hóa `(x − mean)/std`. Augmentation làm trên ảnh [0, 1], nên `fill` cũng ở thang [0, 1].
- **`get_dataloaders(cfg, include_test=False, seed=None)`**: hàm mà `train.py` sẽ gọi. Trả về:
  - `loaders`: chỉ tập train được xáo và augmentation; val/test giữ nguyên ảnh gốc (đánh giá phải trên ảnh thật).
  - `info`: `mean`, `std`, số ảnh mỗi lớp, `class_weights`.
  - `generator` cố định theo seed, để thứ tự xáo trộn lặp lại được. `pin_memory` chỉ bật khi có CUDA.
- **`check_counts(data)`**: in `[OK ]`/`[SAI]` cho từng tập: số ảnh so với chuẩn, đủ 7 lớp, shape.
- **`run_eda(...)`**: kiểm số ảnh, ghi `class_counts.csv` và `stats.json`, vẽ 4 hình. PrivateTest chỉ dùng để **đếm** số ảnh mỗi lớp (mô tả dữ liệu), không vẽ ảnh, không tính thống kê nào đưa vào mô hình. Dùng backend `Agg` để chạy được cả khi không có màn hình (Colab/Kaggle).
- **`main()`**: tham số `--csv` và `--out` (mặc định `data/fer2013.csv`, `results/eda`). Số ảnh không khớp chuẩn thì in cảnh báo và thoát với mã 1.

### Đã kiểm tra

Trước khi có `fer2013.csv`, mình tạo một CSV giả trong thư mục tạm: 700/140/140 ảnh, tỷ lệ lớp giống FER2013 (Disgust ít nhất, Happy nhiều nhất). Môi trường chạy thử là env `ai_prj` (torch 2.14 CPU), cộng torchvision 0.29.2 cài tạm ngoài repo; env của Trúc không bị sửa.

| Kiểm tra | Kết quả |
|---|---|
| `python -m src.data` trên CSV giả | In `[SAI]` cho 3 tập (đúng, vì không phải 28.709/3.589/3.589), đủ 7/7 lớp, shape (48, 48), thoát mã 1; ghi đủ 6 file EDA |
| CSV không tồn tại | `FileNotFoundError` kèm hướng dẫn tải từ Kaggle |
| Batch train | `(64, 1, 48, 48)` float32, nhãn int64 |
| `get_dataloaders(cfg)` mặc định | Chỉ có `train`, `val`; `include_test=True` mới có `test` |
| Val sau chuẩn hóa | mean 0,007, std 0,997 (≈ 0 và 1, đúng vì val không augmentation) |
| Train có augmentation, **viền điền 0** | mean −0,57, std 2,09: lệch hẳn |
| Train có augmentation, **viền điền mean** | mean 0,05, std 0,96: hết lệch |
| Tái lập | Cùng seed (có `torch.manual_seed` như `set_seed` sẽ làm) → batch đầu giống hệt; khác seed → khác |
| `num_workers = 2` trên Windows | Chạy được khi đặt `OPENBLAS_NUM_THREADS=1`; không đặt thì worker chết vì OpenBLAS không xin được bộ nhớ (máy lúc đó chỉ còn 1,0/15,8 GB RAM trống) |
| Không augmentation (E4-1) | `build_train_transform` trả `None` |
| Augmentation E4-3 | Lật 0,5 + xoay ±10° + dịch 0,1 + scale (0,9; 1,1) |

Chạy trên **dữ liệu thật** (env `btl02`, CPU, 28 s):

| Kiểm tra | Kết quả |
|---|---|
| Số ảnh | `[OK ]` train 28.709, val 3.589, test 3.589; cả 3 tập đủ 7/7 lớp; `counts_match_fer2013: true` |
| Bản `pd.read_csv` đầu tiên | `ParserError: out of memory`. Commit memory của máy chỉ còn 0,4 GB trống (AMDChat 3,7 GB, WSL 3,0 GB, VS Code 2,9 GB). Đã đổi sang đọc từng dòng |
| `conda create -n btl02 python=3.11` | Lần Trúc chạy báo `PackagesNotFoundError: python=3.11`, dù kênh `defaults` có Python 3.11.0–3.11.18. Khả năng cao do thiếu bộ nhớ lúc giải phụ thuộc; chạy lại khi máy trống hơn thì tạo được |
| Hình EDA | Đã xem cả 4 hình: ảnh mẫu đúng lớp, ảnh trung bình rõ miệng há (Surprise) và miệng cười (Happy), viền augmentation màu xám trung bình |

Về chuyện điền viền: lúc đầu viền trống sau xoay/dịch được điền 0 (đen). Sau chuẩn hóa, giá trị đó thành (0 − mean)/std, rất âm; với FER2013 thật vào khoảng −2. Mạng có thể học nhầm "có viền đen" như một đặc trưng. Điền bằng mean của train thì viền thành đúng 0 sau chuẩn hóa, tức là "không có thông tin". Cách này gần với `fill_mode='nearest'` mặc định của Keras trong code mẫu.

### Cách tự kiểm tra lại

Trong Anaconda Prompt (sau khi tạo env `btl02` và tải `fer2013.csv`):

```bat
cd /d "D:\Data\GitHub\deep-learning-assignments\BTL02 Emotion Recognition"
conda activate btl02
python -m src.data
```

Kết quả đúng phải thấy:

```
[OK ] train (Training   ):  28709 ảnh (chuẩn 28709), 7/7 lớp, shape (48, 48)
[OK ] val   (PublicTest ):   3589 ảnh (chuẩn 3589), 7/7 lớp, shape (48, 48)
[OK ] test  (PrivateTest):   3589 ảnh (chuẩn 3589), 7/7 lớp, shape (48, 48)
```

Sau đó là bảng số ảnh mỗi lớp (train: Angry 3995, Disgust 436, Fear 4097, Happy 7215, Sad 4830, Surprise 3171, Neutral 4965), dòng `Mean/std pixel tập train (sau khi chia 255): 0.5077 / 0.2550`, dòng "Lớp nhiều nhất Happy, ít nhất Disgust, chênh 16.55 lần", và `results/eda/` có 6 file.

Máy báo out of memory thì tắt AMDChat, chạy `wsl --shutdown`, đóng bớt Zalo/Discord rồi chạy lại.

### Số liệu cho báo cáo

Nguồn: `results/eda/class_counts.csv` và `results/eda/stats.json`.

| Lớp | Train | % train | Val (PublicTest) | Test (PrivateTest) | Trọng số lớp |
|---|---|---|---|---|---|
| Angry | 3995 | 13,92 | 467 | 491 | 1,0266 |
| Disgust | 436 | 1,52 | 56 | 55 | 9,4066 |
| Fear | 4097 | 14,27 | 496 | 528 | 1,0010 |
| Happy | 7215 | 25,13 | 895 | 879 | 0,5684 |
| Sad | 4830 | 16,82 | 653 | 594 | 0,8491 |
| Surprise | 3171 | 11,05 | 415 | 416 | 1,2934 |
| Neutral | 4965 | 17,29 | 607 | 626 | 0,8260 |
| **Tổng** | **28709** | 100 | **3589** | **3589** | |

- Mean / std pixel tập train (sau khi chia 255): **0,5077 / 0,2550**.
- Happy nhiều gấp **16,55 lần** Disgust. Đoán toàn bộ là Happy đã được 895/3589 ≈ 24,9% trên val; đó là mức sàn để so với mô hình.
- Nhận xét cho mục 1 báo cáo:
  - Ảnh trung bình của Surprise và Happy khác biệt rõ ở vùng miệng (há tròn, cười rộng).
  - Angry, Fear, Sad, Neutral có ảnh trung bình gần giống nhau, nên dự kiến hay bị nhầm lẫn nhau. Kiểm lại bằng confusion matrix ở mô hình cuối.
  - Ảnh mẫu có cả ảnh dính watermark, tay che mặt, mặt nghiêng: dữ liệu khá nhiễu.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao dùng PublicTest làm validation mà không tách từ Training?* FER2013 chia sẵn 3 phần; dùng PublicTest làm validation và PrivateTest làm test giống cách chấm của cuộc thi Kaggle, và giữ PrivateTest hoàn toàn chưa nhìn thấy đến lúc đánh giá cuối.
- *Vì sao mỗi lần chỉ đổi một tham số?* Đổi nhiều tham số cùng lúc thì không biết thay đổi accuracy là do tham số nào.
- *Vì sao chuẩn hóa bằng mean/std của tập train mà không phải toàn bộ dữ liệu?* Dùng thống kê của val/test là để thông tin của tập đánh giá lọt vào lúc train (rò rỉ dữ liệu). Val/test được chuẩn hóa bằng đúng mean/std của train, giống lúc dùng mô hình với ảnh mới.
- *Vì sao chỉ augmentation tập train?* Augmentation tạo thêm biến thể để mạng học bất biến với lật/xoay/dịch, chống quá khớp. Đánh giá phải trên ảnh gốc thì số liệu mới phản ánh đúng dữ liệu thật và so sánh được giữa các run.
- *Vì sao lật ngang mà không lật dọc?* Mặt người đối xứng trái–phải nên lật ngang vẫn là cùng cảm xúc; lật dọc tạo mặt lộn ngược, không có trong thực tế.
- *Viền trống sau khi xoay điền giá trị gì, vì sao?* Điền mean của train, để sau chuẩn hóa viền bằng 0, mạng không học nhầm viền đen như một đặc trưng (đã thử: điền 0 thì phân bố train lệch, mean −0,57, std 2,09).
- *Trọng số lớp tính thế nào?* w_c = N/(K·n_c). Lớp hiếm (Disgust) có trọng số lớn, nên lỗi trên lớp đó bị phạt nặng hơn trong CrossEntropy.
- *Tập test có được dùng trong EDA không?* Chỉ đếm số ảnh mỗi lớp để mô tả dữ liệu. Không xem ảnh test, không tính thống kê nào cho mô hình từ test.
- *Dữ liệu lệch lớp thế nào, ảnh hưởng gì?* Happy chiếm 25,1% train, Disgust chỉ 1,5% (chênh 16,55 lần). Mô hình dễ thiên về lớp đông và gần như bỏ Disgust, nên ngoài accuracy phải báo F1 từng lớp, và thử CE có trọng số lớp (E6-3, trọng số Disgust 9,41).
- *Vì sao đọc CSV từng dòng mà không dùng pandas?* `pd.read_csv` giữ cả khoảng 300 MB chuỗi pixel và bị out of memory trên máy nhóm. Đọc từng dòng và đổi ngay sang uint8 chỉ cần khoảng 83 MB.

### Còn treo

- [x] Tải `fer2013.csv` từ Kaggle vào `data/`
- [x] Chạy `python -m src.data` trên dữ liệu thật: `[OK ]` cả 3 tập, đã xem 4 hình, đã tick ô 23/11 trong kế hoạch
- [x] Viết `src/data.py` + EDA
- [x] Tạo conda env `btl02`. Máy không có GPU NVIDIA (không có `nvidia-smi`), nên E1–E7 và mô hình cuối chạy trên Colab/Kaggle
- [ ] (tùy chọn) Đếm ảnh trùng trong FER2013 (Kaggle báo 34.034 chuỗi pixel khác nhau trên 35.887 ảnh), kiểm có ảnh nào nằm ở cả train lẫn val/test không, đưa vào mục 1 báo cáo
- [x] Xóa file `Kế hoạch.md` cũ ở gốc thư mục BTL02 (đã không còn)
- [ ] `train.py` (24/11): `set_seed()` phải gọi **trước** `get_dataloaders()`, nếu không augmentation không lặp lại được khi `num_workers = 0`
- [ ] `train.py` (24/11): cho phép ghi đè `num_workers` từ dòng lệnh (`--workers 0`) khi chạy thử trên máy Trúc thiếu RAM; baseline giữ 2 cho Colab/Kaggle
