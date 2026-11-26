# Nhật ký BTL02 – Emotion Recognition (Trúc)

Nhật ký cá nhân của Trúc, không nộp. Quy tắc ở `CLAUDE.md` mục 1. Kế hoạch: [docs/Kế hoạch.md](docs/Kế%20hoạch.md). Nhật ký thực nghiệm nộp cho thầy nằm ở [NhatKyThucNghiem.md](NhatKyThucNghiem.md).

---

## T2 23/11/2026 – Lập kế hoạch, khung thư mục; data.py + EDA; models.py, train.py, script thực nghiệm

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

Làm sớm việc của ngày T3 24/11 (kế hoạch mục 7):

9. `src/models.py`: 4 kiến trúc Small / Medium / Medium_01 / Large theo đúng code mẫu, tùy chọn BN / activation / dropout, khởi tạo He.
10. `src/train.py`: seed, vòng train/eval tự viết, checkpoint và early stopping theo val_loss, log CSV, `summary.csv`, ghi đè config bằng `--set`, chạy thử `--thu`, `--workers`, chặn ghi đè log.
11. `experiments/common.py`, `run_E1.py` … `run_E6.py` và notebook `experiments/chay_colab_kaggle.ipynb` để chạy trên GPU.
12. **Đổi baseline** (bàn giao bản 2, sau khi đọc hết code mẫu): Adam lr 1e-3 → **1e-4**, tối đa 40 → **60** epoch, patience 8 → **10**, `fc_units: null` (theo kiến trúc, để Large có FC 2048). Kế hoạch mục 3 và 4 (giá trị E3) đổi theo.
13. Cập nhật `.claude/`: rule thực nghiệm (E7-1, E7-2, baseline mới), skill chạy thực nghiệm (lệnh thật, máy không GPU, notebook), skill báo cáo + agent chấm (Grad-CAM bắt buộc, confusion matrix valid + test), hook (đếm cả run `_r2`).

Chưa xong của 24/11: **chạy BASE đầy đủ trên GPU** và chốt baseline (xem Còn treo).

### Commit trong ngày

- `b885df2f` Up bài tập BTL03 (giải nén car_data, depth_hand): commit này cũng đưa thư mục `Yêu cầu/` của BTL02 (đề, báo cáo mẫu, code mẫu) lên repo.
- `396bc7a5` BTL02: thêm docs/Kế hoạch.md: bản kế hoạch đầu tiên.
- `7fc928bb` BTL02: tạo khung thư mục, baseline.yaml, README, nhật ký thực nghiệm: dựng thư mục và các file khung.
- `3b97b309` BTL02: kế hoạch, khung thư mục, bộ skill/agent/rule/lệnh/hook, cập nhật CLAUDE.md: cài bộ công cụ Claude vào `.claude/`, sửa `CLAUDE.md`.
- `ad7c5e17` BTL02: viết src/data.py (đọc FER2013, DataLoader, augmentation, EDA): bản đầu của `data.py` (còn dùng `pd.read_csv`), thêm `Phụ trách` vào `paths.py`, lệnh chạy trong README.
- `b9164c36` BTL02: đọc CSV từng dòng (tránh hết RAM), chạy EDA trên FER2013 thật: `load_fer2013` đọc bằng module `csv`, thêm `results/eda/`, số liệu EDA vào nhật ký.
- _(commit models.py, train.py, experiments, đổi baseline: Trúc điền hash ở mục ngày sau)_

### Kết quả

- [docs/Kế hoạch.md](docs/Kế%20hoạch.md): kế hoạch và phân công.
- [configs/baseline.yaml](configs/baseline.yaml): baseline Medium CNN, ReLU, không BN, **Adam lr 1e-4**, batch 64, **tối đa 60 epoch, patience 10**, `fc_units: null`. (Bản đầu ngày: Adam 1e-3, 40 epoch, patience 8; đổi theo code mẫu, xem "Đã kiểm tra".)
- [src/models.py](src/models.py), [src/train.py](src/train.py): **mới**.
- [experiments/common.py](experiments/common.py), `run_E1.py` … `run_E6.py`, [experiments/chay_colab_kaggle.ipynb](experiments/chay_colab_kaggle.ipynb): **mới**.
- [.gitignore](.gitignore): thêm `results/_thu/` (kết quả chạy thử).
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

Phần `src/models.py` (đứng tên Khang):

- **`ARCHS`**: mỗi kiến trúc là danh sách khối (số filter của từng conv) cộng số nơ-ron FC, chép đúng cấu trúc `SMALL_CNN` … `LARGE_CNN` trong `demo_code.html`:

  | Kiến trúc | Khối conv | FC | Tham số |
  |---|---|---|---|
  | small | [64,64] [128,128,256,256] | 1024 | 38.901.191 |
  | medium (baseline) | [64×2] [128×3] [256×3] | 1024 | 11.327.303 |
  | medium_01 | [64×3] [128×4] [256×4] | 1024 | 12.101.895 |
  | large | [64×4] [128×4] [256×4] [512×4] | 2048 | 20.406.599 |

- **`EmotionCNN`**: mỗi khối là [Conv 3×3 padding 1 (+BN) + activation] × k → MaxPool 2×2 → Dropout; sau cùng Flatten → FC → activation → Dropout → FC 7.
  - padding 1 tương đương `padding='same'` của Keras: chỉ MaxPool làm ảnh nhỏ đi (48 → 24 → 12 → 6 → 3).
  - Có BN thì conv bỏ bias, vì tham số β của BN đã làm việc dịch.
  - Lớp cuối trả **logits**, không softmax, vì `nn.CrossEntropyLoss` đã gồm log-softmax. Keras dùng softmax + `categorical_crossentropy`, hai cách tương đương về toán.
- **`_init_weights`**: khởi tạo He (Kaiming normal) cho mọi Conv/Linear, bias 0. Lý do đo được ở "Đã kiểm tra". Dùng chung cho mọi run, nên không phải biến thực nghiệm.
- **`build_model(cfg["model"])`**, **`count_params`**, **`conv_layers`**, **`last_conv`**: `last_conv` trả lớp Conv cuối (Medium: weight 256×256×3×3) để Khang vẽ filter conv cuối (đề bắt buộc).
- `python -m src.models` in bảng tham số 4 kiến trúc × có/không BN.

Phần `src/train.py` (đứng tên Trúc):

- **`load_config(override)`**: đọc `baseline.yaml` rồi ghi đè khóa dạng `"train.lr"`. Khóa không có trong baseline thì báo `KeyError`. Lý do: gõ sai tên khóa mà không báo lỗi thì run đó âm thầm thành bản sao của baseline.
- **`set_seed(seed)`**: đặt seed cho `random`, `numpy`, `torch`, `torch.cuda`, và tắt `cudnn.benchmark`. Phải gọi **trước** `get_dataloaders` và `build_model`, để khởi tạo trọng số, thứ tự xáo và augmentation đều lặp lại được.
- **`group_of(run_id)`**: `E1-2` → `E1`, `E3-2_r2` → `E3`, `F-0` → `final`, `BASE` → `BASE`; quyết định thư mục `results/<nhóm>/`.
- **`build_optimizer`** (adam | sgd), **`build_scheduler`** (none | plateau): ReduceLROnPlateau dùng `factor=0.1, patience=2` giống code mẫu.
- **`run_epoch(model, loader, criterion, device, optimizer=None)`**: có optimizer thì `model.train()` + backward; không có thì `model.eval()` + `torch.no_grad()`. Loss trung bình theo ảnh (nhân `loss.item()` với số ảnh trong batch, vì batch cuối nhỏ hơn). `train_acc` tính lúc đang train (còn dropout + augmentation), giống `acc` của Keras.
- **`train(run_id, cfg, ...)`**: vòng epoch.
  - Mỗi epoch ghi một dòng log và `flush` ngay, để Colab có ngắt thì log vẫn còn.
  - val_loss giảm (so sánh `<`) thì lưu checkpoint; không giảm thì tăng bộ đếm, đủ `patience` thì dừng (early stopping).
  - Scheduler `step(val_loss)` sau mỗi epoch.
  - Hết run thì thêm 1 dòng `summary.csv`, số lấy ở epoch có val_loss nhỏ nhất.
  - Log đã có thì báo `FileExistsError`: không ghi đè run đã chạy.
  - `configs/<run>.json` lưu config đầy đủ, tên GPU, mean/std train, để biết chính xác run chạy với gì.
- **`run(run_id, override, change, workers)`**: điểm vào cho `run_E*.py`. `workers` không phải biến thực nghiệm, chỉ để máy yếu chạy được.
- **CLI**: `--run`, `--set khóa=giá_trị` (giá trị đọc bằng `yaml.safe_load`: `0.001` thành số, `true` thành bool), `--change`, `--workers`, `--thu` (2 epoch × 20 batch, ghi vào `results/_thu/<run>_<giờ>/`).

Phần `experiments/` (đứng tên theo kế hoạch mục 8: E1–E2 Khang, E3–E4 Trung, E5–E6 Nhân, `common.py` Trúc):

- Mỗi `run_E*.py` chỉ khai báo `RUNS = {mã: (override, mô tả)}`.
- `common.run_group` **bỏ qua run đã có log**, nên chạy lại lệnh sau khi bị ngắt là tiếp tục từ run còn thiếu.
- `print_summary` in bảng nhóm; `plot_group` vẽ `results/<nhóm>/figures/curves.png` (val_loss, val_acc theo epoch, kèm đường BASE nét đứt để so).
- `run_E2` tự chọn kiến trúc tốt nhất từ `results/E1/summary.csv` (val_acc tại epoch val_loss nhỏ nhất), hoặc nhận `--arch`.
- Run đổi nhiều khóa nhưng một ý: E3-1 (SGD đi cùng lr 1e-3), E4-1 (tắt cả lật/xoay/dịch), E4-2 (chỉ giữ lật), E6-1 (bỏ cả 2 dropout).
- `chay_colab_kaggle.ipynb`:
  - `git clone` repo công khai, lấy `fer2013.csv` từ Kaggle Input hoặc `MyDrive`.
  - Colab: nối `results/` vào Drive để không mất log khi ngắt.
  - Chạy BASE → E1 … E6, rồi nén `results_*.zip` (không kèm checkpoint).

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

**Kiểm tra `models.py`, `train.py`** (env `btl02`, CPU):

| Kiểm tra | Kết quả |
|---|---|
| `python -m src.models` | Cả 4 kiến trúc, có/không BN, output `(2, 7)`. Số tham số như bảng ở trên; Small nhiều nhất vì chỉ pool 2 lần, flatten 12×12×256 = 36.864 chiều |
| 13 kiểu override của E1–E6, mỗi run 1 epoch × 2 batch | Đều chạy, ghi đủ log / checkpoint / config / summary |
| Khóa sai `model.activaton` | `KeyError: Không có khóa 'model.activaton' trong baseline.yaml` |
| Chạy lại run đã có log | `FileExistsError … chạy lại thì đặt mã E1-1_r2` |
| Tái lập: 2 lần cùng seed | Log giống hệt (`train_loss 3.3448, val_loss 1.89543`) |
| Tốc độ CPU | ~0,6 s/batch train, một epoch đầy đủ ~5 phút: run thật phải chạy GPU |

**Chuyện baseline "kẹt" ở val_acc 0,235** (đáng ghi nhớ cho vấn đáp và phần phân tích):

1. Chạy thử `--thu` (2 epoch × 20 batch) với lr 1e-3 rồi 1e-4 đều ra val_acc 0,2352. Nhưng `--thu` chỉ cho mạng thấy 2.560 ảnh, và đánh giá trên 1.280 ảnh val đầu tiên; 0,2352 là tỷ lệ Happy trong 1.280 ảnh đó. Tức là mạng đang đoán toàn Happy, chuyện rất bình thường lúc mới bắt đầu. **Phép thử này không nói được lr nào tốt hơn**; ghi chú trong `baseline.yaml` coi đó là lý do đổi lr là chưa chính xác. Lý do đúng để dùng 1e-4 là code mẫu dùng 1e-4.
2. Thử dài hơn (3 epoch × 60 batch, val đầy đủ): lr 1e-4 ra val_acc 0,2494 → 0,2494 → 0,2327; lr 1e-3 ra 0,2527 → 0,2494 → 0,2494. 0,2494 = 895/3589 = tỷ lệ Happy trong val. Cả hai vẫn đoán toàn Happy.
3. Đo độ lớn tín hiệu qua Medium trên 1 batch thật:

   | Khởi tạo | std sau ReLU (lớp 1 → 9) | Gradient trung bình ở conv1 |
   |---|---|---|
   | PyTorch mặc định | 0,37 → 0,0047 | 6×10⁻⁶ |
   | Glorot (Keras) | 0,089 → 0,009 | 2×10⁻³ |
   | He / Kaiming | ~0,8–1,0 mọi lớp | 6,7×10⁻² |

   Mặc định của PyTorch đặt Var(w) = 1/(3·fan_in). Qua mỗi lớp ReLU, phương sai nhân thêm khoảng fan_in·Var(w)/2 = 1/6; qua 9 lớp thì tín hiệu gần như mất, gradient ở lớp đầu nhỏ hơn He khoảng 10.000 lần. He đặt Var(w) = 2/fan_in nên phương sai giữ nguyên qua ReLU.
4. Thêm khởi tạo He, chạy lại 3 epoch × 60 batch với lr 1e-4: val_acc **0,2622 → 0,2739 → 0,2633**, đã vượt mức đoán toàn Happy. Mới chỉ ~11.500 ảnh, nên chưa nói được độ chính xác cuối; BASE đầy đủ trên GPU mới là bằng chứng.

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

Mô hình và vòng train (trên máy, chỉ chạy thử):

```bat
python -m src.models
python -m src.train --run BASE --thu --workers 0
python -m experiments.run_E3 --thu --workers 0
```

- `src.models`: phải thấy bảng 8 dòng, medium 11,327,303 tham số.
- `src.train --thu`: phải thấy 2 dòng `epoch` và dòng `[BASE] tốt nhất epoch …`, cùng thư mục mới trong `results/_thu/` có `logs/BASE.csv`, `checkpoints/BASE.pt`, `configs/BASE.json`, `summary.csv`.
- `run_E3 --thu`: phải chạy đủ E3-1, E3-2, E3-3, mỗi run 1 epoch.

Chạy thật trên GPU: mở `experiments/chay_colab_kaggle.ipynb` trên Kaggle hoặc Colab (hướng dẫn ở ô đầu notebook). Với BASE, sau khoảng 5 epoch đầy đủ val_acc phải rõ ràng trên 0,25. Nếu đứng yên ở 0,2494 (đoán toàn Happy) thì dừng lại và báo.

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

- *Vì sao Small ít lớp hơn mà nhiều tham số nhất?* Small chỉ MaxPool 2 lần, nên ảnh vào FC còn 12×12×256 = 36.864 chiều, nhân FC 1024 ra 37,7 triệu tham số chỉ riêng lớp đó. Medium pool 3 lần, flatten còn 6×6×256 = 9.216 chiều. Số tham số chủ yếu nằm ở lớp FC đầu tiên, không phải ở conv.
- *Vì sao mô hình không có softmax ở cuối?* `nn.CrossEntropyLoss` nhận logits và tự làm log-softmax, ổn định số học hơn tự softmax rồi lấy log. Lúc dự đoán chỉ cần `argmax` của logits (softmax không đổi thứ tự).
- *Vì sao dùng khởi tạo He?* Đã đo: khởi tạo mặc định làm std tín hiệu teo 0,37 → 0,005 qua 9 lớp ReLU, gradient conv1 chỉ 6×10⁻⁶, mạng kẹt ở mức đoán toàn Happy (val_acc 0,2494). He giữ phương sai qua ReLU (Var = 2/fan_in), gradient conv1 6,7×10⁻². Khởi tạo dùng chung cho mọi run nên không phá nguyên tắc đổi một biến.
- *Checkpoint và early stopping theo gì?* Theo val_loss nhỏ nhất (giống `ModelCheckpoint(monitor='val_loss')` và `EarlyStopping` của code mẫu); patience 10. Báo cáo val_acc tại đúng epoch đó, không lấy val_acc lớn nhất (lấy max là chọn epoch theo chính số đang báo cáo).
- *`train_acc` thấp hơn val_acc có lạ không?* Không: train_acc tính lúc còn dropout và augmentation; val không có hai thứ đó.
- *Vì sao E3-1 đổi cả optimizer lẫn lr?* SGD với lr 1e-4 gần như không học trong 60 epoch, so như vậy không công bằng với SGD. "Biến" ở E3-1 là cặp (optimizer, lr thường dùng với nó).

### Còn treo

- [x] Tải `fer2013.csv` từ Kaggle vào `data/`
- [x] Chạy `python -m src.data` trên dữ liệu thật: `[OK ]` cả 3 tập, đã xem 4 hình, đã tick ô 23/11 trong kế hoạch
- [x] Viết `src/data.py` + EDA
- [x] Tạo conda env `btl02`. Máy không có GPU NVIDIA (không có `nvidia-smi`), nên E1–E7 và mô hình cuối chạy trên Colab/Kaggle
- [x] (tùy chọn) Đếm ảnh trùng trong FER2013: làm ngày 25/11 (`duplicate_stats`), 34.034 ảnh khác nhau, 280 ảnh val và 288 ảnh test trùng train; đã đưa vào báo cáo mục 1.5
- [x] Xóa file `Kế hoạch.md` cũ ở gốc thư mục BTL02 (đã không còn)
- [x] `train.py`: `set_seed()` gọi **trước** `get_dataloaders()` (trong `train()`); đã kiểm tái lập, 2 lần cùng seed ra log giống hệt
- [x] `train.py`: `--workers 0` ghi đè `num_workers` khi chạy thử trên máy thiếu RAM; baseline giữ 2 cho Colab/Kaggle
- [x] **Chạy BASE đầy đủ trên GPU** (xong 25/11 trên Kaggle T4: val_acc 66.43% ở epoch 47/57, đã ghi `NhatKyThucNghiem.md`)
- [ ] Sửa ghi chú đầu `configs/baseline.yaml`: lý do đổi lr là "theo code mẫu", không phải "lr 1e-3 bị kẹt" (phép thử `--thu` không phân biệt được hai lr)
- [x] Trúc commit + push trước khi chạy notebook (commit `e5b06e03`, 25/11)

---

## T4 25/11/2026 – Đưa lên Kaggle (2 × T4), chạy BASE + E1–E6; evaluate.py, visualize.py, run_final.py; báo cáo mục 1, 2

### Việc theo kế hoạch

Theo [docs/Kế hoạch.md](docs/Kế%20hoạch.md) mục 7:
- Việc 24/11 còn lại: chạy BASE đầy đủ trên GPU, chốt baseline.
- Việc 25/11: chạy E1, E2 trên GPU, ghi nhật ký, viết `evaluate.py`.

Code cho các việc này đã viết sẵn từ 23/11. Hôm nay:

1. Commit + push `models.py`, `train.py`, `experiments/`, notebook (`e5b06e03`).
2. Dựng phiên Kaggle:
   - Import `experiments/chay_colab_kaggle.ipynb`.
   - Add Input dataset FER2013 (`nicolejyt/facialexpressionrecognition`). Tìm theo tên không ra, phải dán link dataset vào ô tìm kiếm.
   - **Xác minh số điện thoại tài khoản Kaggle.** Chưa xác minh thì Session options không có Accelerator và công tắc Internet bị khóa.
   - Bật GPU **T4 x2** + Internet.
3. Chạy các ô trên Kaggle: clone đúng `e5b06e03`, `fer2013.csv` 287 MB, `--thu` trên T4 chạy 40 batch trong khoảng 1,1 s (CPU máy Trúc khoảng 19 s).
4. Sửa notebook: gộp 6 ô `run_E1` … `run_E6` thành **một ô chạy 2 luồng song song**, mỗi luồng một GPU (xem "Giải thích").
5. BASE trên Kaggle (phiên nháp, xem 7 epoch rồi dừng): val_acc 0,301 → 0,381 → 0,421 → 0,470 → 0,500 → 0,510 → **0,535**, val_loss giảm đều, khoảng 19 s/epoch trên T4. **Hết kẹt ở 0,2494**: khởi tạo He có tác dụng.
6. Bấm Save & Run All: Version #1 (GPU T4 x2) chạy nền BASE + E1–E6.
7. Viết **`src/evaluate.py`** (việc 25/11 trong kế hoạch) trong lúc GPU chạy.
8. Viết luôn **`src/visualize.py`** (việc 26/11): filter lớp conv cuối, feature map, Grad-CAM.
9. Viết luôn **`experiments/run_final.py`** + `configs/final.yaml` (việc 27/11), thêm phần "3. Mô hình cuối" vào notebook. Chỉ còn chờ kết quả E1–E6 để điền `final.yaml`.
10. Viết **báo cáo mục 1** (`report/section_1_intro_data.md`) và **mục 2** (`report/section_2_code.md`), cùng `report/tai_lieu_tham_khao.md`. Để mọi số trong báo cáo truy được về `results/`:
    - EDA thêm thống kê ảnh trùng (`duplicate_stats` trong `data.py` → `stats.json`).
    - `python -m src.models` ghi `results/models/params.csv` (số tham số) và `init_check.csv` (khởi tạo mặc định so với He).

### Commit trong ngày

- `e5b06e03` BTL02: models.py (4 kiến trúc, He init), train.py, run_E1–E6, notebook Colab/Kaggle; baseline lr 1e-4, 60 epoch, patience 10: toàn bộ code train và chạy thực nghiệm viết ngày 23/11 (giải thích ở mục 23/11).
- `597f4201` BTL02: notebook chạy E1–E6 song song trên 2 GPU Kaggle, nhật ký 25/11: ô 2 luồng thay 6 ô chạy tuần tự.
- `fcdf6eea` BTL02: evaluate.py (metrics, confusion valid+test, ca đúng/sai, evaluate_final đánh giá test 1 lần): thêm `src/evaluate.py`.
- `ed57e54e` BTL02: visualize.py (filter conv cuối, ảnh kích hoạt mạnh nhất, feature map, Grad-CAM): thêm `src/visualize.py`.
- `d10dd58b` BTL02: run_final.py + configs/final.yaml (3 seed, test 1 lần), notebook thêm phần mô hình cuối.
- `3170e6e7` BTL02: báo cáo mục 1, 2; EDA thêm thống kê ảnh trùng; results/models (số tham số, khởi tạo He).

### Kết quả

- [experiments/chay_colab_kaggle.ipynb](experiments/chay_colab_kaggle.ipynb): ô chạy E1–E6 mới; log từng luồng ở `/kaggle/working/log_gpu0.txt`, `log_gpu1.txt`.
- [src/evaluate.py](src/evaluate.py): **mới**. Hai chế độ:

  | Cách gọi | Đọc tập | Ghi vào |
  |---|---|---|
  | `python -m src.evaluate --ckpt <file .pt>` | chỉ validation | `results/<nhóm>/eval/<mã run>/`: `metrics.json`, `classification_report.csv`, `predictions_val.csv`, `figures/confusion_matrix.png`, `true_cases.png`, `wrong_cases.png` |
  | `evaluate_final(results/final, ["F-0","F-1","F-2"])` (gọi từ `run_final.py`, 27/11) | validation + **PrivateTest, một lần** | `results/final/`: `best_model.pt`, `final_metrics.json`, `classification_report.csv`, `predictions_val.csv`, `predictions_test.csv`, `figures/confusion_matrix.png` (valid + test cạnh nhau), `true_cases.png`, `wrong_cases.png` (ảnh test), `curves.png` (3 seed) |

- [src/visualize.py](src/visualize.py): **mới**. `python -m src.visualize [--ckpt …]`, mặc định mô hình cuối `results/final/best_model.pt`, chỉ dùng validation. Ghi 5 hình vào `results/final/figures/` (hoặc `<nhóm>/figures/`):

  | Hình | Nội dung | Mục báo cáo |
  |---|---|---|
  | `last_conv_filters.png` | **Filter lớp conv cuối** (đề bắt buộc): ảnh tổng hợp làm 64 filter hoạt động mạnh nhất kích hoạt cực đại | 4.1 (Khang) |
  | `last_conv_top_images.png` | 10 filter × 6 ảnh validation thật kích hoạt mạnh nhất | 4.1 (Khang) |
  | `last_conv_weights.png` | Trọng số 3×3 của 64 filter (trung bình theo 256 kênh vào) | 4.1 (Khang) |
  | `feature_maps.png` | Một ảnh Happy qua 3 khối (48×48 → 24×24 → 12×12), 8 kênh mạnh nhất mỗi khối | 4.1 (Khang) |
  | `gradcam.png` | Grad-CAM tại conv đầu / giữa / cuối × 7 cảm xúc, ảnh đoán đúng tự tin nhất mỗi lớp | 4 (Nhân) |

- [configs/final.yaml](configs/final.yaml): **mới**, cấu hình mô hình cuối = `override` so với baseline + cờ `chot`. Hiện `chot: false`, `override: {}`; điền sau khi có E1–E6.
- [experiments/run_final.py](experiments/run_final.py): **mới**.
  - `python -m experiments.run_final --thu`: chạy thử, không đọc test.
  - `python -m experiments.run_final`: chạy thật.
  - `--seeds 0 2 --chi-train`: chia GPU.
- Notebook: thêm phần **"3. Mô hình cuối"** (đóng gói thành phần 4). Ô này tự dừng khi `final.yaml` chưa chốt.
- README và skill `chay-thuc-nghiem-btl02`: thêm lệnh `src.evaluate`, `src.visualize`, `run_final`.
- [src/data.py](src/data.py): thêm `duplicate_stats`; [results/eda/stats.json](results/eda/stats.json) có thêm khóa `duplicates`. Bốn hình EDA chạy lại không đổi.
- [src/models.py](src/models.py): `main()` ghi [results/models/params.csv](results/models/params.csv) và [results/models/init_check.csv](results/models/init_check.csv); thêm hàm `init_check`.
- [report/section_1_intro_data.md](report/section_1_intro_data.md), [report/section_2_code.md](report/section_2_code.md), [report/tai_lieu_tham_khao.md](report/tai_lieu_tham_khao.md): **mới**.

### Giải thích

Ô chạy E1–E6 trong notebook:

- `streams = [["E1", "E2", "E3"], ["E4", "E5", "E6"]]`: luồng 0 chạy trên GPU 0, luồng 1 trên GPU 1, cùng lúc. E2 lấy kiến trúc tốt nhất từ `results/E1/summary.csv`, nên phải nằm sau E1 trong cùng luồng. Chia như vậy mỗi luồng có 8–9 run.
- Mỗi luồng là một tiến trình `bash -c "python -m experiments.run_E1; python -m experiments.run_E2; …"`, đặt `CUDA_VISIBLE_DEVICES=<i>`. Trong tiến trình đó chỉ thấy một GPU, nên `train.py` (vốn dùng `cuda`) không phải sửa gì. Các lệnh nối bằng `;` chứ không phải `&&`, để một nhóm lỗi không chặn các nhóm sau.
- Hai luồng ghi vào hai thư mục `results/<nhóm>/` khác nhau, mỗi nhóm có `summary.csv` riêng, nên không tranh nhau ghi file.
- Vòng `while` in dòng cuối của mỗi log 2 phút một lần để theo dõi; hết thì in 4.000 ký tự cuối mỗi log kèm mã thoát.
- Chỉ có 1 GPU (Colab) thì gộp 6 nhóm thành một luồng, chạy tuần tự.

`src/evaluate.py` (đứng tên Trúc; phần confusion matrix và ca đúng/ca sai là của Trung theo kế hoạch mục 8):

- **`load_checkpoint(path)`**: dựng lại mô hình từ config lưu trong checkpoint, nạp trọng số, `model.eval()`. Lấy luôn `train_mean`/`train_std` trong checkpoint để chuẩn hóa ảnh đúng như lúc train.
- **`predict(...)`**: chạy mô hình trên cả tập, không augmentation, trong `torch.no_grad()`. Trả nhãn đoán (`argmax`) và xác suất softmax.
- **`split_metrics(y_true, y_pred)`**: accuracy, macro F1, weighted F1, precision/recall/F1/support từng lớp (sklearn `classification_report`). **Macro F1** lấy trung bình đều 7 lớp, nên Disgust nặng ngang Happy; accuracy thì gần như bỏ qua lớp nhỏ. Báo cả hai để thấy mô hình có bỏ lớp hiếm không.
- **`write_report_csv`**, **`write_predictions`**: báo cáo từng lớp, và dự đoán từng ảnh (kèm 7 xác suất) để phân tích lỗi và để agent đối chiếu số liệu kiểm lại.
- **`plot_confusion`**:
  - Valid và test đặt cạnh nhau, như `valid_vgg16.html` của thầy.
  - Màu theo **tỷ lệ trên mỗi hàng** (tức recall của lớp thật), nên Disgust (56 ảnh) vẫn đọc được; mỗi ô ghi cả số ảnh lẫn %.
  - Nhãn theo thứ tự gốc FER2013 0–6, không theo chữ cái như code mẫu.
- **`plot_cases(correct=True/False)`**: mỗi hàng là một lớp thật, 5 ảnh.
  - Ca đúng: chọn ngẫu nhiên theo seed.
  - Ca sai: chọn những ảnh **mô hình tự tin nhất mà vẫn sai**, là các ca đáng phân tích nhất.
  - Ảnh và nhãn lấy từ cùng một tập. Lỗi `x_train[valid_idx]` của code mẫu nằm ở hình "Validating Images" phần khám phá dữ liệu (ảnh train kèm nhãn `y_valid`), không phải ở hình ca đúng/sai: ca đúng/sai của mẫu dùng đúng `x_test` (agent đối chiếu phát hiện ngày 25/11, đã sửa báo cáo mục 2).
- **`plot_curves`**: loss/accuracy theo epoch, train nét đứt, val nét liền, mỗi seed một màu.
- **`evaluate_checkpoint`** (chế độ dòng lệnh): **chỉ đọc validation**. Dòng lệnh không có tùy chọn nào để đọc PrivateTest.
- **`evaluate_final(final_dir, run_ids)`**:
  1. `final_metrics.json` đã có thì báo lỗi, vì đó sẽ là đánh giá test lần hai.
  2. Đọc `summary.csv` của F-0/1/2, tính val_acc trung bình ± **độ lệch chuẩn mẫu** (chia n−1).
  3. Chọn seed có `best_val_loss` nhỏ nhất, **chỉ dựa trên validation**, rồi chép checkpoint thành `best_model.pt`.
  4. Đánh giá mô hình đó trên validation và PrivateTest. Ghi `final_metrics.json` **ngay khi có số test**, để dù vẽ hình có lỗi thì test vẫn được đánh dấu là đã dùng. Sau đó mới vẽ hình.

`src/visualize.py` (đứng tên Khang cho filter + feature map, Nhân cho Grad-CAM):

- **`capture(model, module, x)`**: gắn forward hook tạm thời để lấy đầu ra của một lớp bất kỳ trong lúc chạy mô hình, rồi gỡ hook ngay (`try/finally`), không sửa code mô hình.
- **`activation_after(model, conv)`**: lớp kích hoạt ngay sau một Conv (bỏ qua BN nếu có). Feature map và Grad-CAM lấy ở đây, vì đó là thứ lớp sau thực sự nhận được.
- **`mean_activations`**: mức kích hoạt trung bình của từng kênh trên mỗi ảnh validation (N×256). Dùng để chọn 64 filter **hoạt động mạnh nhất** (filter "chết", luôn bằng 0, không có gì để xem) và để tìm ảnh kích hoạt mạnh nhất. Script in số filter chết.
- **`filter_patterns`** (activation maximization):
  - Bắt đầu từ nhiễu nhỏ, cập nhật **ảnh đầu vào** (không đổi trọng số) bằng Adam để tăng trung bình đầu ra của filter.
  - Lấy đầu ra conv **trước** ReLU: sau ReLU, filter đang bằng 0 thì gradient cũng bằng 0, ảnh không đổi.
  - Ba cách chống nhiễu: dịch ngẫu nhiên ±2 pixel mỗi bước, phạt L2, và cứ 4 bước làm mờ Gaussian 3×3.
  - Tối ưu 64 ảnh cùng lúc trong một batch, chỉ lấy gradient theo ảnh (`torch.autograd.grad`).
- **`plot_filter_weights`**: vẽ thẳng trọng số. Lớp cuối có 256 kênh vào nên phải lấy trung bình mới vẽ được 3×3; khó đọc bằng mắt. Có hình này để giải thích vì sao phải dùng hai cách kia.
- **`plot_top_images`**: cách dễ giải thích nhất khi vấn đáp, ví dụ "filter X bắt ảnh tối / miệng mở".
- **`plot_feature_maps`**: lấy đầu ra conv cuối mỗi khối (conv đứng ngay trước MaxPool) cho ảnh Happy đoán đúng tự tin nhất.
- **`grad_cam(model, x1, target, module)`**: A = feature map tại lớp đó; α_k = trung bình không gian của ∂y_target/∂A_k; bản đồ = ReLU(Σ α_k·A_k), phóng về 48×48, chuẩn hóa [0, 1].
- **`plot_gradcam`**: hàng đầu là ảnh gốc, 3 hàng sau là Grad-CAM. Mỗi cột một cảm xúc, chọn ảnh validation **đoán đúng** có xác suất cao nhất. Lớp không có ca đúng thì để trống và ghi chú, không lỗi.

`experiments/run_final.py` (đứng tên Trúc) và `configs/final.yaml`:

- **Vì sao tách `final.yaml`**: mô hình cuối gộp nhiều thay đổi tốt nhất, ví dụ kiến trúc của E1, BN của E2, scheduler của E3. Đây là chỗ duy nhất được đổi nhiều biến cùng lúc so với baseline, nên phải ghi riêng, mỗi khóa kèm run làm căn cứ, không sửa `baseline.yaml` đã chốt.
- **Cờ `chot`**: `false` thì chạy thật bị từ chối (`sys.exit`), chỉ cho `--thu`. Notebook có ô mô hình cuối nên Save & Run All lúc chưa chốt vẫn an toàn, không vô tình đánh giá PrivateTest bằng cấu hình dở dang.
- **`load_final()`**: đọc `final.yaml` và gọi thử `load_config(override)` ngay, để khóa sai tên báo lỗi **trước** khi tốn giờ train.
- **`train_seeds(override, seeds, out_dir)`**: train `F-<seed>` với `override + {"seed": seed}`, seed đã có log thì bỏ qua. Nhờ vậy chia được 2 GPU: hai tiến trình `--chi-train` train seed {0, 2} và {1} song song, sau đó lệnh không có `--chi-train` chỉ còn bước đánh giá.
- **`main()`**: thứ tự kiểm tra là
  1. `final_metrics.json` đã có thì dừng (test đã dùng).
  2. Chưa chốt thì dừng.
  3. Train.
  4. Đủ 3 seed mới gọi `evaluate_final` (test một lần), rồi `visualize(best_model.pt)`.
- **`--thu`**: train 3 seed × 1 epoch × 5 batch vào `results/_thu/final_<giờ>/`, đánh giá **validation** cho F-0 và vẽ hình với ít bước. Không bao giờ đọc PrivateTest.

`data.py` (thêm) và `models.py` (thêm):

- **`duplicate_stats(data)`**:
  - Băm MD5 2304 byte của từng ảnh, so trong và giữa các tập.
  - Đếm: ảnh lặp trong mỗi tập, số ảnh khác nhau của cả bộ, ảnh val/test trùng ảnh train, ảnh test trùng ảnh val, nhóm ảnh trùng mà khác nhãn, ảnh một màu (min = max).
  - Dùng mã băm 16 byte thay vì giữ nguyên ảnh làm khóa, để không tốn thêm khoảng 80 MB bộ nhớ.
  - PrivateTest chỉ được dùng để đếm (mô tả dữ liệu), không đưa gì vào mô hình.
- **`init_check()`** trong `models.py`:
  - Đưa 64 ảnh nhiễu N(0, 1) (giống ảnh đã chuẩn hóa) qua Medium hai lần: một lần với khởi tạo mặc định (`reset_parameters()`), một lần với He.
  - Ghi std sau từng ReLU, std logits, và gradient trung bình ở conv1 với nhãn ngẫu nhiên.
  - Thay cho số đo hôm 23/11 (đo bằng ảnh thật, nằm ngoài `results/`), để báo cáo có nguồn.

### Đã kiểm tra

| Kiểm tra | Kết quả |
|---|---|
| Mọi ô code của notebook parse được (`ast.parse`, bỏ dòng `!`) | OK |
| Giả lập ô 2 luồng bằng `echo`/`sleep` (Git Bash trên máy) | Luồng 0 chạy E1 → E2 → E3 với `GPU=0`, luồng 1 chạy E4 → E5 → E6 với `GPU=1`, song song, mã thoát 0; dòng tiến độ in đúng |
| `nvidia-smi` trên Kaggle | 2 × Tesla T4, 15.360 MiB mỗi card |
| `--run BASE --thu` trên T4 | `[BASE] baseline \| Tesla T4 \| 11,327,303 tham số`; epoch 2 chạy 40 batch trong 1,1 s |
| BASE đầy đủ, 7 epoch đầu (phiên nháp) | val_acc 0,301 / 0,381 / 0,421 / 0,470 / 0,500 / 0,510 / 0,535; val_loss 1,765 → 1,219; ~19 s/epoch; GPU 0 99%, GPU 1 0% (nên mới sửa notebook chạy song song) |
| `evaluate --ckpt` trên checkpoint chạy thử (He, 3 epoch × 60 batch), dữ liệu thật, chỉ val | accuracy **0,2633 = đúng val_acc epoch 3 trong log train**: `evaluate.py` và `train.py` tính cùng một số. Macro F1 0,1649, Disgust F1 0 |
| `evaluate_final` trên **CSV giả** (F-0/1/2 train 2 epoch × 4 batch) | Chọn F-0 (val_loss 1,9989 nhỏ nhất); val_acc 0,1928 ± 0,0515, khớp tính tay (std mẫu); val_acc do evaluate tính = log train; ghi đủ file, confusion valid + test cạnh nhau |
| Gọi `evaluate_final` lần hai | `RuntimeError: … final_metrics.json đã có: PrivateTest đã được đánh giá. Không đánh giá lần hai.` |
| Xem hình | Confusion matrix nhãn đúng thứ tự 0–6. `wrong_cases.png` (val thật, hàng Surprise) có **một ảnh biển cảnh báo, không phải mặt người**: FER2013 có lẫn ảnh rác, đưa vào phân tích lỗi |

PrivateTest của dữ liệu thật **chưa bị đọc lần nào**: phần test của `evaluate_final` chỉ thử trên CSV giả.

**Kiểm tra `visualize.py`** trên checkpoint chạy thử (val_acc 0,2633), dữ liệu thật, chỉ validation, `--steps 60`, CPU 2 phút 32 giây:

| Hình | Thấy gì |
|---|---|
| `gradcam.png` | Đủ 4 hàng × 7 cột. Cột Disgust ghi "(không có ca đúng)" vì mô hình yếu chưa đoán đúng ảnh Disgust nào. Ở conv cuối, vùng đỏ đã rơi vào miệng (Angry) và mặt (Happy); ở conv đầu là cạnh/viền |
| `last_conv_top_images.png` | Filter lớp cuối của mô hình yếu này phản ứng chủ yếu theo **độ sáng** (ảnh rất tối / rất sáng). Lộ ra thêm ảnh rác trong FER2013: **biển cảnh báo** và **biểu tượng máy ảnh** |
| `feature_maps.png` | Khối 1 (48×48) còn rõ hình mặt, viền, kính; khối 3 (12×12) chỉ còn các vùng sáng rời rạc |
| `last_conv_filters.png` | Bản đầu (chỉ dịch + L2) ra nhiễu hạt. **Thêm làm mờ Gaussian** thì ra mẫu hình rõ: sọc chéo (filter 88, 34), lưới chấm đều (21, 250, 17) |
| Số filter chết ở lớp cuối | 0/256 |

Hình thật cho báo cáo sẽ vẽ lại từ mô hình cuối (`results/final/best_model.pt`) ngày 27/11.

**Kiểm tra `run_final.py`** (không lần nào đọc PrivateTest thật):

| Kiểm tra | Kết quả |
|---|---|
| `python -m experiments.run_final` khi `chot: false` | `configs/final.yaml chưa chốt (chot: false)…`, thoát mã 1, **không tạo `results/final/`** |
| `train_seeds` trên CSV giả, chạy 2 lần | Lần 1 train F-0, F-1; lần 2 in `[F-0] đã có log, bỏ qua.`, `summary.csv` không bị ghi trùng |
| `--thu --workers 0` trên dữ liệu thật, CPU 3 phút 17 giây | Train F-0/1/2 → evaluate validation (accuracy 0,2385) → 5 hình. F-0 (seed 0, cấu hình baseline) val_loss **2,2088, trùng E1-2 và E3-3** chạy thử hôm 23/11: tái lập đúng |
| val_acc trong log F-0 (0,2594) khác evaluate (0,2385) | Đúng như thiết kế: `--thu` chỉ chấm 5 batch val, evaluate chấm đủ 3.589 ảnh. Chạy thật thì hai số trùng nhau (đã kiểm 0,2633 = 0,2633) |
| Notebook sau khi thêm phần mô hình cuối | Mọi ô code parse được; thứ tự BASE → E1–E6 → mô hình cuối → đóng gói |

**Thống kê ảnh trùng** (`python -m src.data`, khóa `duplicates` trong `stats.json`):

| Chỉ số | Giá trị |
|---|---|
| Ảnh khác nhau trong cả bộ | **34.034** / 35.887, khớp đúng con số Kaggle hiển thị |
| Ảnh lặp trong train / val / test | 1.236 / 26 / 17 |
| Ảnh val trùng một ảnh train | **280 (7,80% val)** |
| Ảnh test trùng một ảnh train | **288 (8,02% test)** |
| Ảnh test trùng một ảnh val | 44 |
| Nhóm ảnh trùng pixel nhưng khác nhãn | 57 |
| Ảnh một màu | 11 train, 1 val, 0 test |

Chạy lại EDA thì 4 hình không đổi (git chỉ báo `stats.json` thay đổi): EDA tái lập đúng.

**`init_check`** (`results/models/init_check.csv`, ảnh nhiễu N(0, 1), batch 64):

| Khởi tạo | std sau ReLU 1 → 9 | Gradient conv1 |
|---|---|---|
| Mặc định PyTorch | 0,3437 → 0,0051 | 4,97×10⁻⁶ |
| He | 0,8181 → 1,5366 | 2,76×10⁻² (lớn hơn khoảng 5.500 lần) |

Cùng xu hướng với số đo bằng ảnh thật hôm 23/11 (0,37 → 0,0047; 6,1×10⁻⁶ so với 6,7×10⁻²).

**Báo cáo mục 1, 2:** đối chiếu với code mẫu thì phát hiện mình viết sai một dòng. Code mẫu **cũng** dùng PublicTest làm validation và PrivateTest làm test (`public_files` → `x_valid`), không phải tự tách validation. Đã sửa Bảng 2.6. Đã kiểm thêm: code mẫu có `zoom_range = 0.1`, `fill_mode='nearest'`, batch 128, patience 10; báo cáo mẫu ghi "Best at Epoch 31/53/31", val_acc 0,64–0,67.

### Câu hỏi vấn đáp có thể gặp

- *Chạy 2 run cùng lúc trên 2 GPU có làm kết quả khác đi không?* Không. Mỗi run là một tiến trình riêng, có seed riêng, dùng một GPU riêng. Kết quả chỉ phụ thuộc config và seed. Thời gian mỗi epoch có thể chậm hơn một chút vì hai tiến trình dùng chung CPU để nạp dữ liệu, nên cột "Thời gian" trong nhật ký ghi rõ "T4, chạy song song".
- *Vì sao không dùng `DataParallel` để một run chạy trên cả 2 GPU?* Mạng nhỏ (11 triệu tham số), ảnh 48×48: chia batch 64 ra 2 card tốn chi phí đồng bộ gần bằng phần tiết kiệm. Chạy 2 run độc lập trên 2 card đơn giản hơn và nhanh gần gấp đôi.
- *Mô hình cuối chọn thế nào, có dùng test để chọn không?* Không. Chọn seed có val_loss nhỏ nhất trong 3 seed, chỉ dựa trên PublicTest, rồi mới đánh giá PrivateTest một lần. `final_metrics.json` ghi ngay sau khi có số test, và code từ chối đánh giá lần hai.
- *Vì sao báo cả accuracy và macro F1?* Accuracy tính trên mọi ảnh nên bị Happy (25%) chi phối; Disgust chỉ 1,5% nên đoán sai hết Disgust cũng chỉ mất khoảng 1,5 điểm accuracy. Macro F1 lấy trung bình đều 7 lớp, nên Disgust F1 = 0 kéo macro F1 xuống rõ rệt.
- *Confusion matrix tô màu theo gì?* Theo tỷ lệ trên mỗi hàng (recall của lớp thật), không theo số ảnh. Nếu tô theo số ảnh thì hàng Happy luôn đậm nhất, còn hàng Disgust gần như trắng dù sai hết.
- *Độ lệch chuẩn 3 seed tính thế nào?* Độ lệch chuẩn mẫu, chia n−1 = 2, vì 3 seed là mẫu nhỏ từ mọi seed có thể.
- *"Visualize filter lớp conv cuối" làm thế nào?* Trọng số lớp cuối là 256 filter × 256 kênh vào × 3×3, không vẽ thẳng thành ảnh nhìn được. Nhóm dùng activation maximization: giữ nguyên trọng số, tối ưu **ảnh đầu vào** bằng gradient để filter kích hoạt mạnh nhất; ảnh thu được là "mẫu hình filter đó tìm". Thêm hình ảnh validation thật kích hoạt mạnh nhất để đối chiếu.
- *Vì sao phải làm mờ ảnh khi tối ưu?* Không làm mờ thì gradient đẩy ảnh thành nhiễu tần số cao (đã thử: ra nhiễu hạt). Làm mờ, dịch ngẫu nhiên và phạt L2 là ép ảnh thành mẫu hình trơn, giống ảnh tự nhiên hơn.
- *Grad-CAM là gì, khác feature map thế nào?* Feature map chỉ cho biết kênh nào bật ở đâu. Grad-CAM nhân feature map với gradient của điểm lớp đúng, nên cho biết vùng nào **làm tăng điểm của lớp đó**. Ở conv đầu, Grad-CAM ra cạnh/viền (đặc trưng cấp thấp); ở conv cuối, ra vùng lớn như miệng, mắt.
- *Vì sao visualize chỉ dùng ảnh validation?* PrivateTest chỉ dùng một lần trong `evaluate_final`. Chọn ảnh minh họa cũng là "nhìn" dữ liệu, nên giữ test hoàn toàn chưa thấy.
- *Mô hình cuối đổi nhiều biến so với baseline, có trái nguyên tắc "mỗi lần đổi một biến" không?* Không. Nguyên tắc một biến là cho E1–E6, để biết từng thay đổi tác động thế nào. Mô hình cuối là bước **gộp** các thay đổi đã được chứng minh tốt, ghi rõ trong `configs/final.yaml` kèm run làm căn cứ. Chạy 3 seed để biết kết quả không phải may mắn của một seed.
- *FER2013 có ảnh trùng không, ảnh hưởng gì?* Có: 34.034 ảnh khác nhau trên 35.887. 280 ảnh val (7,80%) và 288 ảnh test (8,02%) trùng y hệt một ảnh train, nên accuracy đo trên FER2013 hơi cao hơn so với ảnh hoàn toàn mới. Nhóm không xóa ảnh trùng, để so được với báo cáo mẫu và các công bố dùng cùng cách chia, nhưng nêu trong phần hạn chế. 57 nhóm ảnh trùng mà khác nhãn cho thấy nhãn có nhiễu.
- *Vì sao chạy 3 seed mà chỉ test một mô hình?* 3 seed cho biết độ dao động của val_acc (trung bình ± độ lệch chuẩn). Chọn seed theo val_loss rồi đánh giá test một lần, nên số test không bị "chọn lọc" theo chính test.

### Còn treo

- [x] Xem 5 epoch đầu của BASE trên Kaggle: val_acc 0,535 ở epoch 7, đã vượt xa 0,2494
- [x] Commit + push notebook mới (`597f4201`), Import lại vào Kaggle, **Save & Run All** (Version #1)
- [x] Kiểm tra notebook trong Version #1: nhiều khả năng là bản 6 ô cũ, chạy tuần tự (xem mục 26/11, suy từ giờ tạo zip). Kết quả không đổi
- [x] Tải `results_*.zip` của Version #1, giải nén, ghi `NhatKyThucNghiem.md` cho BASE và các run E1–E6 (26/11)
- [x] Viết `src/evaluate.py` (accuracy, macro F1, confusion matrix valid + test, classification report, ca đúng/sai, đường học)
- [x] `src/visualize.py`: filter conv cuối (3 cách), feature map, Grad-CAM conv đầu/giữa/cuối × 7 cảm xúc
- [x] Đưa 2 loại ảnh rác (biển cảnh báo, biểu tượng máy ảnh) và thống kê ảnh trùng vào báo cáo mục 1.5
- [x] Báo cáo mục 1 và mục 2
- [ ] Báo cáo mục 3 (khi có E1–E6), mục 4 và 5 (khi có mô hình cuối), phụ lục (bảng phân công, ghép `tai_lieu_tham_khao.md`)
- [ ] Phần hạn chế ở mục 5: khoảng 8% ảnh val/test trùng ảnh train, 57 nhóm nhãn mâu thuẫn (số ở `stats.json`)
- [x] `experiments/run_final.py` + `configs/final.yaml` + phần mô hình cuối trong notebook
- [x] Có kết quả E1–E6: điền `configs/final.yaml` (`model.arch: small`, căn cứ E1-1), `chot: true` (26/11)
- [ ] Commit + push `results/BASE`, `results/E1` … `E6` và `final.yaml`; Import lại notebook; Save & Run All để chạy mô hình cuối

---

## T5 26/11/2026 – Kết quả BASE + E1–E6, mô hình cuối và PrivateTest, báo cáo đủ 5 mục, Word/PDF, slide

### Việc theo kế hoạch

Theo [docs/Kế hoạch.md](docs/Kế%20hoạch.md) mục 7:
- T3 24/11: baseline chạy hết, val_acc trong khoảng 60–66%. Kết quả 66,43%, đạt.
- T4 25/11: có `summary.csv` của E1, E2, có `evaluate.py`. Đạt.
- T5 26/11: có `summary.csv` E3–E6, có `visualize.py` (viết từ 25/11). Đạt.

Đã tick cả ba ô. **E7 (VGG16 128×128) chưa chạy**: kế hoạch ghi "làm nếu có GPU", và phương án dự phòng là bỏ E7 trước. Hôm nay:

1. Tải `results_20261125-1359.zip` (Version #1 trên Kaggle), giải nén vào thư mục tạm, chỉ chép `results/BASE` và `results/E1` … `E6` vào repo. **Không lấy `results/eda/` trong zip**, vì đó là bản cũ Kaggle clone từ commit `e5b06e03`, chưa có thống kê ảnh trùng; giải nén đè sẽ mất `stats.json` mới mà báo cáo mục 1 đang dùng.
2. Ghi 18 dòng (BASE, E1-1 … E6-3) vào [NhatKyThucNghiem.md](NhatKyThucNghiem.md); mọi số đọc từ `results/<nhóm>/logs/<run>.csv`.
3. Trúc chọn cấu hình mô hình cuối: **Small (E1-1)**. Đã điền [configs/final.yaml](configs/final.yaml): `model.arch: small`, `chot: true`.
4. Viết `experiments/tong_hop.py` (bảng tổng hợp + 2 hình) và **báo cáo mục 3** (`report/section_3_experiments.md`): 3.0 thiết lập chung, 3.1–3.6 từng nhóm (bảng, hình, nhận xét có số), 3.7 tổng kết.
5. Chạy mô hình cuối trên Kaggle (notebook `notebook0e986dcbeb`, Save & Run All): F-0, F-1, F-2 (Small, seed 0/1/2), chọn F-1, đánh giá PrivateTest **một lần** lúc 03:48:49 UTC 26/11. Chép `results/final/` vào repo, ghi F-0/1/2 vào nhật ký thực nghiệm. Đã tick ô T6 27/11 sáng trong kế hoạch (làm sớm).
6. Viết **báo cáo mục 4** (`report/section_4_visualize.md`: filter conv cuối, feature map, confusion matrix, ca đúng/sai, Grad-CAM) và **mục 5** (`report/section_5_final.md`: 3 seed, test, so với mức tham chiếu, ảnh trùng, hạn chế, kết luận). Thêm 2 phân tích: `duplicate_effect` (evaluate.py) và số filter chết lớp conv cuối (`last_conv_stats.json`).
7. Chốt tỷ lệ đóng góp (Trúc 25%, Khang 25%, Trung 25%, Nhân 25%), viết **phụ lục** (`report/appendix.md`: bảng phân công, danh sách file nộp, bảng lệnh tái lập, tài liệu tham khảo [1]–[9]). Cập nhật kế hoạch mục 8 theo việc đã làm thật: E7 không làm, Trúc viết thêm 3.0 và 3.7, Nhân viết 4.3. Bảng 2.1 báo cáo thêm `tong_hop.py`. Đã kiểm dòng `Phụ trách:` của 15 file `.py` và notebook khớp Bảng A.1.
8. **Ghép báo cáo** thành `report/BaoCao_BTL02_Nhom05.docx` (3,2 MB) và `.pdf` (4,6 MB, 52 trang) bằng script mới `report/ghep_bao_cao.py`. Máy không có Word/LibreOffice/pandoc: dùng pandoc 3.9 trong gói `pypandoc_binary` (cài tạm ngoài repo), PDF in bằng Edge chạy ẩn từ HTML.
9. **Slide** 15 trang: `slides/BTL02_Nhom05.pptx` (1,6 MB) và `.pdf` (1,8 MB), script `slides/lam_slide.js`. Mỗi slide ghi người trình bày theo bảng phân công và có ghi chú lời nói (speaker notes) để từng bạn tập. Đã tick checklist nộp: báo cáo, slide, nhật ký thực nghiệm, bảng phân công.

### Commit trong ngày

- `ad79a0af` BTL02: kết quả BASE + E1–E6, nhật ký thực nghiệm 18 run, chốt mô hình cuối (Small): thêm `results/BASE`, `results/E1` … `E6`, 18 dòng nhật ký thực nghiệm, `final.yaml` chốt.
- `88c6f596` BTL02: báo cáo mục 3, tong_hop.py; sửa làm tròn accuracy (BASE 66.43%): mục 3 báo cáo, `tong_hop.py`, sửa 66.42 → 66.43 ở các file.
- `f271546a` BTL02: kết quả mô hình cuối (3 seed, PrivateTest 1 lần), F-0/1/2 vào nhật ký; bỏ best_model.pt khỏi git (155.6 MB): thêm `results/final/` (trừ `best_model.pt`), 3 dòng F trong nhật ký thực nghiệm, `.gitignore` bỏ `best_model.pt`.
- _(commit báo cáo mục 4, 5, `duplicate_effect`, `last_conv_stats`: Trúc điền hash ở mục ngày sau)_

### Kết quả

- `results/BASE/`, `results/E1/` … `results/E6/`: mỗi nhóm có `logs/`, `configs/`, `summary.csv`, `figures/curves.png` (không có `checkpoints/`, đúng thiết kế notebook).
- [NhatKyThucNghiem.md](NhatKyThucNghiem.md): 18 dòng.
- [configs/final.yaml](configs/final.yaml): chốt Small.
- [docs/Kế hoạch.md](docs/Kế%20hoạch.md): tick ô 24/11, 25/11, 26/11.
- [experiments/tong_hop.py](experiments/tong_hop.py): **mới**. `python -m experiments.tong_hop` ghi `results/tong_hop.csv` (mỗi run một dòng, có cột so với BASE, train − val, chạm trần), `results/figures/tong_hop_val_acc.png`, `results/BASE/figures/curves.png`.
- [report/section_3_experiments.md](report/section_3_experiments.md): **mới**, 8 bảng, 8 hình. [report/tai_lieu_tham_khao.md](report/tai_lieu_tham_khao.md): thêm [9] (Li và cộng sự 2019, dropout và BatchNorm).

### Run trong ngày

Chi tiết và nhận xét từng run ở [NhatKyThucNghiem.md](NhatKyThucNghiem.md). Tất cả chạy trên Tesla T4, seed 0, ngày 25/11.

| Nhóm | Run tốt nhất trong nhóm (val_acc tại epoch val_loss nhỏ nhất) | So với BASE 66,43% |
|---|---|---|
| E1 kiến trúc | E1-1 Small 67,01% | +0,59 |
| E2 BatchNorm | E2-1 62,36% (so với E1-1: −4,65) | −4,07 |
| E3 optimizer | E3-3 plateau 65,20% | −1,23 |
| E4 augmentation | E4-3 thêm zoom 66,26% | −0,17 |
| E5 kích hoạt | E5-1 LeakyReLU 66,29% | −0,14 |
| E6 regularization | E6-2 dropout 0,4: 65,78% | −0,64 |

### Đã kiểm tra

| Kiểm tra | Kết quả |
|---|---|
| Nội dung zip | 85 file, không có đường dẫn lạ; đủ `summary.csv` của BASE và E1–E6 |
| Chép vào repo | Chỉ chép BASE, E1–E6 (các thư mục này chưa có trong repo, không ghi đè gì); `results/eda/` giữ nguyên |
| E1-2 so với BASE | `diff` 6 cột đầu của hai log (bỏ cột thời gian): **giống hệt ở cả 57 epoch**. Cùng seed, khác phiên GPU vẫn tái lập đúng |
| Tổng thời gian E1-2 = BASE = 1127 s | Thời gian từng epoch khác nhau (19,59 so với 19,91 s ở epoch 1…), tổng bằng nhau chỉ là trùng hợp |
| Version #1 chạy song song hay tuần tự? | Bắt đầu khoảng 08:06 UTC, zip tạo lúc 13:59 UTC (khoảng 5 giờ 53 phút). Tổng thời gian 18 run khoảng 5 giờ 9 phút cộng BASE: khớp chạy **tuần tự**. Nếu song song, luồng dài nhất chỉ khoảng 3 giờ. Kết luận: Version #1 nhiều khả năng dùng bản notebook cũ 6 ô. Cột thời gian trong nhật ký thực nghiệm ghi "(T4)", không ghi "song song" |
| E3-3 lr | Giảm còn 1e-5 ở epoch 26 (val_acc 64,59%), còn 1e-6 ở epoch 32, còn 1e-7 ở epoch 37, dừng ở epoch 43 |
| E6-1 val_loss sau epoch 17 | 1,0102 → 1,0218 → … → 1,0722 ở epoch 27: tăng dần, quá khớp |
| Run chạm trần 60 epoch | E2-1, E3-1, E5-3, E6-2: epoch tốt nhất = 60 = epoch cuối, val_loss vẫn đang giảm |
| `final.yaml` | `load_final()` đọc được `chot: True`, `override {'model.arch': 'small'}`; `load_config` ra `arch small`, lr 1e-4 như baseline |
| Agent đối chiếu số liệu mục 3 (khoảng 230 số) | 4 chỗ đã sửa: mốc train/val cắt nhau của BASE ghi rõ "lần đầu epoch 21, liên tục từ 23" và "lần đầu 26, liên tục từ 30"; câu "E1-1 thì ngược lại" đổi thành "E1-1 có train_acc vượt val_acc liên tục từ epoch 22" (đã kiểm: epoch 1–19 và 21 thì không); E2-1, E6-2 không "giảm đều" mà dao động rồi thấp nhất ở epoch 60 (sửa cả báo cáo lẫn nhật ký thực nghiệm); thêm nguồn 0.64–0.67, 24.94%; mỗi mục 3.1–3.6 có dòng Nguồn |
| Agent đối chiếu số liệu mục 4, 5 (khoảng 140 giá trị) | Đã sửa: 3 ô Bảng 4.1 lệch chữ số cuối vì làm tròn hai lần (Recall test Fear 0,4223, Recall val Happy 0,8547, F1 test Angry 0,5946; tính lại cả 42 ô từ `predictions_*.csv`), recall Fear 42,23% (2 chỗ), chênh macro F1 0,0104; ghi rõ 40 filter "< 1% ảnh" đã gồm 17 filter chết và thêm code sinh `last_conv_stats.json` (`python -m src.visualize --thong-ke`, chạy lại ra đúng 17/40); bỏ chữ "UTC" (JSON không ghi múi giờ); mô tả Grad-CAM conv cuối Fear (mắt + miệng/tay), Disgust (quanh mắt), conv giữa Happy (trán + khóe miệng); bỏ câu "không dựa vào nền hay tóc" (conv đầu vẫn sáng ở tóc/nền); bỏ câu dùng test để kết luận về việc chọn cấu hình |
| Báo cáo Word/PDF | PDF 52 trang: đổi từng trang ra ảnh (PyMuPDF) xem: bìa, mục lục 2 trang, mỗi mục lớn sang trang mới, Bảng 3.1 sang trang vẫn lặp dòng tiêu đề, Bảng A.1 đủ 100%, hình lớn không bị cắt. Word: 21 hình, 25 bảng, 6 Heading 1, 36 Heading 2, A4 + lề đúng, Times New Roman 13 pt, bảng có viền, bìa đủ 4 người, 21 chú thích hình; **chưa xem được bố cục Word bằng mắt** (máy không có trình hiển thị docx) |
| Slide | PDF 15 trang đổi ra ảnh xem từng slide; sửa: ảnh bìa sát chân trang, ô quy trình thừa chỗ, "Grad-CAM" bị ngắt ở dấu gạch (dùng gạch nối không ngắt U+2011), trục tung nhảy 55/58/61 (đặt bước 5 và 2), slide kết luận trống nửa dưới (thêm ô "Hướng tiếp theo", bảng rộng hơn). `.pptx`: `validate.py` của skill pptx **PASSED**, `slide_text.py` đủ 15 slide, không sót chữ mẫu, mỗi slide có ghi chú; 4 biểu đồ có màu riêng từng cột (`c:dPt`) và đúng bước trục. **Chưa xem .pptx bằng PowerPoint** (máy không có) |
| **Lỗi làm tròn accuracy** (phát hiện khi viết mục 3) | Log lưu val_acc 5 chữ số: BASE ghi 0.66425, nhưng thật là 2384/3589 = 0.664252, tức **66.43%**. In thẳng `0.66425 × 100` ra 66.42 (số thực máy tính ra 66.42499…). Sửa: dựng lại số ảnh đúng `round(acc × 3589)` (train: × 28709) rồi chia. Hậu quả đã sửa: BASE và E1-2 thành 66.43% (trước ghi 66.42%), E6-3 so với BASE thành −4.24 (trước −4.23). Đã sửa `NhatKyThucNghiem.md`, `final.yaml`, mục nhật ký này; `experiments/tong_hop.py` tính theo cách mới |

### Mô hình cuối

Nguồn: `results/final/final_metrics.json`, `summary.csv`, `logs/F-*.csv`.

| Run | Seed | Epoch tốt nhất / đã chạy | val_loss | val_acc | train − val | Thời gian |
|---|---|---|---|---|---|---|
| F-0 | 0 | 41 / 51 | 0,9552 | 67,01% | +4,16 | 1492 s |
| **F-1** (được chọn) | 1 | 32 / 42 | **0,9515** | 65,87% | +2,14 | 1375 s |
| F-2 | 2 | 51 / 60 | 0,9832 | 66,43% | +7,57 | 1736 s |

- **val_acc 3 seed: 66,43% ± 0,57** (độ lệch chuẩn mẫu). Trùng đúng BASE (Medium, seed 0: 66,43%). Mức +0,59 điểm của E1-1 nhiều khả năng là may mắn của seed 0: Small trung bình không hơn Medium. Đưa vào phần phân tích mục 5.
- F-1 được chọn vì val_loss nhỏ nhất, dù val_acc của nó thấp nhất trong 3 seed. Đây là hệ quả trực tiếp của quy tắc chọn theo val_loss, phải nói rõ trong báo cáo.
- **PrivateTest (1 lần): accuracy 66,09%, macro F1 0,6389.** Validation của F-1: 65,87%, macro F1 0,6284. Test và val gần nhau, không có dấu hiệu chọn mô hình làm lệch.
- F1 test từng lớp: Happy 0,86, Surprise 0,77, Neutral 0,65, Disgust 0,61, Angry 0,59, Sad 0,52, Fear 0,47. Disgust (55 ảnh test) có F1 0,61, cao hơn Fear và Sad dù rất ít ảnh.
- Confusion matrix: Fear hay bị nhầm thành Sad (21–23%), Disgust hay bị nhầm thành Angry (27% test, 39% val), Sad hay bị nhầm thành Neutral (13–15%), đúng như dự đoán từ ảnh trung bình ở mục 1.
- F-0 trùng khít E1-1 ở mọi epoch: tái lập lần thứ hai.
- **`best_model.pt` nặng 155,6 MB**, vượt giới hạn 100 MB/file của GitHub. Đã đổi `.gitignore` (bỏ dòng `!results/final/best_model.pt`) để git bỏ qua; file giữ trong máy, đưa vào zip nộp.
- Hình đã xem: Grad-CAM conv cuối tập trung vào miệng (Angry, Happy, Fear) và mắt (Disgust, Surprise, Neutral); filter conv cuối ra mẫu hình sọc, lưới, tổ ong rõ ràng.

### Ghép báo cáo: `report/ghep_bao_cao.py`

- **Word** (`build_docx`): ghép 6 file `.md` theo thứ tự, mỗi phần cách nhau một ngắt trang (khối OpenXML thô `<w:br w:type="page"/>`); thêm bìa (metadata pandoc giống BTL01: tên đề tài, "Bài tập lớn 02, học phần Học sâu · Nhóm 05", 4 thành viên + MSV, tháng năm) và **mục lục tĩnh** sinh từ các dòng `#`, `##`. Không dùng trường TOC của Word vì máy không có Word để cập nhật trường, trình xem khác có thể hiện mục lục rỗng.
- `-f markdown-implicit_figures`: tắt việc pandoc tự lấy alt của ảnh làm chú thích, vì báo cáo đã có dòng "Hình x.y. …" riêng (nếu không sẽ bị lặp chú thích).
- **File mẫu kiểu chữ** (`reference_docx`): lấy `reference.docx` mặc định của pandoc rồi sửa XML:
  - mọi `rFonts` thành Times New Roman; cỡ chữ mặc định 13 pt (`sz=26`, đơn vị nửa point); màu tiêu đề đen;
  - bảng mặc định thêm `tblBorders` (kẻ viền);
  - `sectPr`: A4 (11906 × 16838 twip), lề trên/dưới/phải 2 cm (1134), trái 3 cm (1701). `sectPr` mẫu không có `pgSz`/`pgMar` nên phải **chèn** (sau `footnotePr`, đúng thứ tự schema), thay thế thì không có gì để thay (lần đầu chạy bị như vậy, đã sửa).
- **PDF** (`build_pdf`): pandoc ra HTML độc lập (ảnh nhúng base64, `--toc` có liên kết), CSS: A4, lề 2/2/2/3 cm, số trang ở chân trang (`@page @bottom-center`), mỗi `h1` sang trang mới, không cắt ngang dòng bảng và ảnh, bảng kẻ viền; rồi Edge `--headless --print-to-pdf --no-pdf-header-footer`.

### Slide: `slides/lam_slide.js`

- **Một bản mô tả, hai đầu ra.** Mảng `SLIDES` mô tả từng slide bằng các phần tử có tọa độ inch (chữ, danh sách gạch đầu dòng, hình, ô màu, bảng, biểu đồ cột). `buildPptx` đưa vào pptxgenjs (khổ rộng 13,333 × 7,5 in, biểu đồ native sửa được trong PowerPoint, ghi chú lời nói). `buildPdf` dựng HTML đặt đúng các tọa độ đó, biểu đồ vẽ bằng SVG, rồi Edge in ra PDF. Lý do: máy không có PowerPoint/LibreOffice để xem .pptx, nên PDF vừa là file nộp vừa là bản để kiểm bố cục.
- `img()` đặt hình vừa khung theo tỉ lệ đo sẵn (`RATIO`), không méo. `stat()` là ô số lớn kèm nhãn. Font Calibri (Windows có sẵn) cho cả hai đầu ra.
- Bố cục 15 slide theo skill báo cáo: đề bài → dữ liệu (2) → mã nguồn → baseline/khởi tạo He → E1–E2 (Khang) → E3–E4 (Trung) → E5–E6 (Nhân) → tổng hợp → mô hình cuối → filter (Khang) → Grad-CAM (Nhân) → confusion matrix (Trung) → kết luận + phân công. Chân trang mỗi slide: "Trình bày: <tên>" và số trang.
- `dropFolderEntries`: bỏ các mục thư mục trong zip của .pptx (skill pptx cảnh báo file có chúng từng không mở được qua dịch vụ đám mây).

### Phân tích thêm cho mục 4, 5

- **`duplicate_effect(final_dir)`** (trong `evaluate.py`, gọi bằng `python -m src.evaluate --trung-lap results/final`):
  - Đọc lại `predictions_val.csv`, `predictions_test.csv` do `evaluate_final` đã ghi.
  - Băm MD5 ảnh để biết ảnh nào trùng train, rồi tính accuracy riêng cho phần trùng và phần còn lại.
  - **Không chạy lại mô hình trên test**, nên không phải đánh giá test lần hai.
  - Kiểm `assert` nhãn trong file dự đoán khớp nhãn trong CSV, để chắc thứ tự ảnh không lệch.
  - Đặt trong `evaluate.py` vì đây là file duy nhất được đọc PrivateTest cho mô hình cuối.
- **Kết quả** (`results/final/duplicate_effect.json`):

  | Tập | Ảnh trùng train | acc trùng | Ảnh còn lại | acc còn lại |
  |---|---|---|---|---|
  | Validation | 280 | 86,07% | 3309 | 64,16% |
  | Test | 288 | 83,33% | 3301 | **64,59%** |

  Trên ảnh test mô hình chưa từng thấy, accuracy là 64,59%, thấp hơn con số báo cáo (66,09%) 1,50 điểm.
- **Filter chết** (`results/final/last_conv_stats.json`): tính trên máy (CPU, chỉ validation). Lớp conv cuối `features.12` có 256 filter: 17 filter không kích hoạt ở ảnh val nào, 40 filter kích hoạt ở dưới 1% số ảnh. Small có 6 conv: `features.0, 2, 6, 8, 10, 12`.
- **Receptive field** của lớp conv cuối Small: 2 conv 3×3 ra 5×5, MaxPool 2×2 ra 6×6 (bước nhảy 2), 4 conv 3×3 cộng thêm 4 × 4 = 16, ra **22×22 pixel**. Giải thích vì sao ảnh activation maximization là mẫu hình lặp đều: mỗi vị trí chỉ thấy 22×22, mục tiêu là trung bình mọi vị trí.

### Giải thích `experiments/tong_hop.py`

- **`exact(acc, n)`**: dựng lại số ảnh đoán đúng `round(acc × n)` rồi chia cho n (n = 3589 với val, 28709 với train), để tránh lỗi làm tròn của log (xem bảng Đã kiểm tra).
- **`collect()`**: đọc `summary.csv` của BASE, E1–E6, thêm các cột:
  - `train_minus_val`;
  - `delta_val_acc_vs_base`;
  - `hit_max_epochs` (epoch tốt nhất = epoch cuối = 60, tức chưa hội tụ);
  - `train_time_s` cộng thẳng từ log, giữ 2 chữ số. `summary.csv` đã làm tròn 1 chữ số, làm tròn tiếp sẽ lệch: E5-2 thật 827.49 s, qua 827.5 thành 828.
- **`plot_val_acc`**: cột val_acc từng run, tô màu theo nhóm, gạch chéo nếu chạm trần, đường nét đứt là BASE. Bỏ E1-2 vì trùng BASE.
- Đường học BASE vẽ bằng `plot_curves` có sẵn trong `evaluate.py`.

### Số liệu cho báo cáo (mục 3)

- BASE 66,43% (val_loss 0,9429), ngang báo cáo mẫu 0,64–0,67.
- Chỉ E1-1 (Small) vượt BASE theo val_acc (+0,59 điểm), nhưng val_loss của nó cao hơn (0,9552). Không run nào có val_loss thấp hơn BASE.
- Augmentation quan trọng nhất trong các yếu tố đã thử:
  - Bỏ augmentation (E4-1): −6,55 điểm, khoảng cách train − val 8,62 điểm, dừng ở epoch 21.
  - Thêm zoom (E4-3): khoảng cách train − val giảm còn 1,42 điểm (BASE 4,11).
- Sigmoid (E5-3) thấp nhất mọi run: 56,59%, −9,84 điểm, vẫn chưa hội tụ sau 60 epoch.
- Bốn run chạm trần 60 epoch (E2-1, E3-1, E5-3, E6-2): phải nói rõ trong báo cáo là kết quả có thể chưa phải tốt nhất của cấu hình đó.
- Thời gian: Large chậm nhất (2124 s); E4-1 nhanh nhất (416 s) vì dừng sớm.

### Câu hỏi vấn đáp có thể gặp

- *Vì sao chọn Small dù val_loss cao hơn baseline?* Nhóm đã định trước quy tắc chọn: so val_acc tại epoch có val_loss nhỏ nhất. Theo quy tắc đó Small cao nhất (67,01% so với 66,43%). Chênh lệch nhỏ, mỗi run chỉ một seed, nên mô hình cuối chạy 3 seed để biết độ dao động.
- *Vì sao không gộp thêm LeakyReLU, zoom… vào mô hình cuối?* Các run đó đều có val_acc thấp hơn BASE (−0,14, −0,17 điểm), không có bằng chứng là chúng giúp ích. Gộp thay đổi không có bằng chứng thì không giải thích được kết quả.
- *Vì sao thêm BatchNorm (E2-1) lại kém hơn?* Run chạm trần 60 epoch, train_acc (56,12%) còn thấp hơn val_acc (62,36%), tức mạng chưa học xong. Nhóm không kết luận BN làm hại; với lr 1e-4 và 60 epoch thì cấu hình này chưa kịp hội tụ.
- *Vì sao sigmoid kém nhất?* Đạo hàm sigmoid tối đa 0,25 và gần 0 khi bão hòa. Qua 9 lớp, gradient tới các lớp đầu rất nhỏ, mạng học chậm: sau 60 epoch val_acc mới 56,59%, val_loss vẫn đang giảm.
- *Vì sao ReduceLROnPlateau (E3-3) không giúp?* Với patience 2, lr giảm 10 lần ngay ở epoch 26, rồi giảm tiếp xuống 1e-7 ở epoch 37; lr quá nhỏ nên mạng gần như ngừng học khi val_acc mới 65,20%.
- *E1-2 trùng khít BASE chứng minh gì?* Cùng cấu hình, cùng seed thì cho đúng cùng log ở mọi epoch: kết quả tái lập được, nên chênh lệch giữa các run là do biến thực nghiệm, không do ngẫu nhiên giữa các lần chạy.
- *Yếu tố nào ảnh hưởng nhiều nhất?* Trong các yếu tố đã thử: hàm kích hoạt (sigmoid kém ReLU 9,84 điểm), rồi augmentation (bỏ augmentation mất 6,55 điểm). Kiến trúc ít ảnh hưởng nhất: bốn mạng chỉ chênh 1,98 điểm (Bảng 3.8).
- *Bằng chứng augmentation chống quá khớp?* Khoảng cách train − val giảm đúng theo mức augmentation tăng: không có 8,62 → lật 7,14 → lật + xoay + dịch 4,11 → thêm zoom 1,42 điểm. Không augmentation thì val_loss nhỏ nhất ở epoch 11 rồi tăng lên 1,4957 ở epoch 21.
- *Vì sao train_acc của E2-1 (BatchNorm) thấp hơn val_acc?* Lúc train, BatchNorm chuẩn hóa bằng thống kê của từng batch 64 ảnh và dropout đang bật; lúc đánh giá, BatchNorm dùng thống kê trung bình tích lũy và dropout tắt. Hai chế độ khác nhau, cộng với run chưa hội tụ (chạm trần 60 epoch).
- *Sigmoid đứng yên mấy epoch đầu là sao?* 7 epoch đầu val_acc đúng bằng 24,94% = tỷ lệ Happy, tức mạng đoán mọi ảnh là Happy. Gradient qua 9 lớp sigmoid quá nhỏ (đạo hàm tối đa 0,25), phải sau vài epoch tín hiệu mới đủ để các lớp đầu thay đổi.

### Còn treo

- [x] Commit + push `results/BASE`, `results/E1` … `E6`, `NhatKyThucNghiem.md`, `configs/final.yaml` (`ad79a0af`)
- [x] Chạy mô hình cuối trên Kaggle (Save & Run All)
- [x] Tải kết quả mô hình cuối (`results/final/`), ghi F-0, F-1, F-2 vào `NhatKyThucNghiem.md`
- [ ] Đóng gói: nhớ chép `results/final/best_model.pt` (155,6 MB, không có trên git) vào zip nộp
- [x] Báo cáo mục 3
- [x] Báo cáo mục 4, 5
- [x] Phụ lục (bảng phân công, tỷ lệ 25% × 4, tài liệu tham khảo)
- [x] Ghép báo cáo thành `.docx` + `.pdf` (bìa, mục lục, 5 mục, phụ lục)
- [ ] Trúc mở `BaoCao_BTL02_Nhom05.docx` (Word trên máy khác, Google Docs hoặc WPS) xem bố cục; Word chưa có số trang ở chân trang (PDF có)
- [x] Slide 15 trang (.pptx + .pdf)
- [ ] Trúc mở `slides/BTL02_Nhom05.pptx` bằng PowerPoint/Google Slides xem lại; chữ có thể ngắt dòng hơi khác bản PDF
- [ ] Gửi mỗi bạn phần đứng tên để đọc trước vấn đáp
- [ ] E7: quyết định làm hay bỏ (dự phòng của kế hoạch: bỏ E7 trước)
- [ ] Ghi chú đầu `configs/baseline.yaml` vẫn nêu lý do đổi lr chưa chính xác (treo từ 23/11); baseline đã chốt nên chỉ ghi rõ trong báo cáo mục 2.4 là lr lấy theo code mẫu (đã ghi)
