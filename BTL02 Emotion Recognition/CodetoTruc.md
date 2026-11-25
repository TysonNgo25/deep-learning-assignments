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
- [ ] (tùy chọn) Đếm ảnh trùng trong FER2013 (Kaggle báo 34.034 chuỗi pixel khác nhau trên 35.887 ảnh), kiểm có ảnh nào nằm ở cả train lẫn val/test không, đưa vào mục 1 báo cáo
- [x] Xóa file `Kế hoạch.md` cũ ở gốc thư mục BTL02 (đã không còn)
- [x] `train.py`: `set_seed()` gọi **trước** `get_dataloaders()` (trong `train()`); đã kiểm tái lập, 2 lần cùng seed ra log giống hệt
- [x] `train.py`: `--workers 0` ghi đè `num_workers` khi chạy thử trên máy thiếu RAM; baseline giữ 2 cho Colab/Kaggle
- [ ] **Chạy BASE đầy đủ trên GPU** (Kaggle/Colab, notebook `experiments/chay_colab_kaggle.ipynb`), ghi `NhatKyThucNghiem.md`, chốt baseline trước hết 24/11 (từ 25/11 hook hỏi lại khi sửa `baseline.yaml`). Nếu BASE vẫn kẹt ~0,25 sau 5 epoch thì xem lại lr/khởi tạo **trước** khi chạy E1–E6
- [ ] Sửa ghi chú đầu `configs/baseline.yaml`: lý do đổi lr là "theo code mẫu", không phải "lr 1e-3 bị kẹt" (phép thử `--thu` không phân biệt được hai lr)
- [x] Trúc commit + push trước khi chạy notebook (commit `e5b06e03`, 25/11)

---

## T4 25/11/2026 – Đưa lên Kaggle (2 × T4), chạy BASE, chạy E1–E6 song song trên 2 GPU

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

### Commit trong ngày

- `e5b06e03` BTL02: models.py (4 kiến trúc, He init), train.py, run_E1–E6, notebook Colab/Kaggle; baseline lr 1e-4, 60 epoch, patience 10: toàn bộ code train và chạy thực nghiệm viết ngày 23/11 (giải thích ở mục 23/11).
- `597f4201` BTL02: notebook chạy E1–E6 song song trên 2 GPU Kaggle, nhật ký 25/11: ô 2 luồng thay 6 ô chạy tuần tự.
- `fcdf6eea` BTL02: evaluate.py (metrics, confusion valid+test, ca đúng/sai, evaluate_final đánh giá test 1 lần): thêm `src/evaluate.py`.
- _(commit visualize.py: Trúc điền hash ở mục ngày sau)_

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

- README và skill `chay-thuc-nghiem-btl02`: thêm lệnh `src.evaluate`, `src.visualize`.

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
  - Ảnh lấy từ đúng tập đang đánh giá; code mẫu vẽ nhầm `x_train[valid_idx]`.
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

### Còn treo

- [x] Xem 5 epoch đầu của BASE trên Kaggle: val_acc 0,535 ở epoch 7, đã vượt xa 0,2494
- [x] Commit + push notebook mới (`597f4201`), Import lại vào Kaggle, **Save & Run All** (Version #1)
- [ ] Kiểm tra trên Kaggle: notebook trong Version #1 là bản 1 ô chạy 2 GPU hay bản 6 ô cũ (bản cũ chỉ chậm hơn, kết quả như nhau)
- [ ] Tải `results_*.zip` của Version #1, giải nén, ghi `NhatKyThucNghiem.md` cho BASE và các run E1–E6 (`/btl02-ghi-run`)
- [x] Viết `src/evaluate.py` (accuracy, macro F1, confusion matrix valid + test, classification report, ca đúng/sai, đường học)
- [x] `src/visualize.py`: filter conv cuối (3 cách), feature map, Grad-CAM conv đầu/giữa/cuối × 7 cảm xúc
- [ ] Đưa 2 loại ảnh rác (biển cảnh báo, biểu tượng máy ảnh) vào phân tích dữ liệu/lỗi trong báo cáo; có thể đếm thêm bằng mục "ảnh trùng" còn treo ở 23/11
- [ ] `experiments/run_final.py` (27/11): train F-0/1/2 với cấu hình tốt nhất rồi gọi `evaluate_final`
