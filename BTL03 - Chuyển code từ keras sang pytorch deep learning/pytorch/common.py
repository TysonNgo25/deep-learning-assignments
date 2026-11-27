"""Phần dùng chung cho các notebook BTL03 (chuyển code Keras sang PyTorch).

Keras có sẵn `compile` / `fit` / `evaluate` / `predict` và callback; PyTorch
không có, nên file này viết lại các hàm đó để notebook PyTorch đọc gần giống
notebook Keras gốc và mọi notebook dùng chung một vòng train.

Cách dùng trong notebook (đặt ở `pytorch/<thư mục>/`):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))
    from common import *

    set_seed(0)
    device = get_device()

Nội dung:
    Đường dẫn   ROOT, PYTORCH_DIR, DATA_DIR, RESULTS_DIR, RUNS_DIR, YEU_CAU_DIR, EXAMPLES_DIR
    Môi trường  set_seed, get_device, device_name
    Dữ liệu     to_nchw, onehot_to_index, make_loader, fer2013_path, load_fer2013
    Mô hình     init_keras_like, count_params, count_all, summary
    Chỉ số      accuracy, sign_accuracy, mae
    Huấn luyện  fit, evaluate, predict
    Callback    Callback, ModelCheckpoint, EarlyStopping, TensorBoard, reduce_lr_on_plateau
    Lưu kết quả results_dir, save_history, save_metrics, plot_history, ghi_so_sanh
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Đường dẫn: tính theo vị trí file này, không ghi đường dẫn tuyệt đối, nên chạy
# được cả trên máy Trúc lẫn Colab/Kaggle sau khi clone repo.
# ---------------------------------------------------------------------------
PYTORCH_DIR = Path(__file__).resolve().parent
ROOT = PYTORCH_DIR.parent                      # thư mục BTL03
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"                       # log TensorBoard, git bỏ qua
YEU_CAU_DIR = ROOT / "Yêu cầu"                 # đề gốc, chỉ đọc
EXAMPLES_DIR = YEU_CAU_DIR / "examples"

SO_SANH_COLUMNS = [
    "notebook", "thu_muc", "chi_so", "keras", "pytorch", "chenh_lech",
    "keras_params", "pytorch_params", "epochs_keras", "epochs_pytorch",
    "optimizer", "lr", "batch", "device", "train_time_s", "seed", "ghi_chu",
]


# ---------------------------------------------------------------------------
# Môi trường
# ---------------------------------------------------------------------------
def set_seed(seed: int = 0) -> None:
    """Cố định seed cho random, numpy và torch (CPU + CUDA).

    Không bật `torch.use_deterministic_algorithms` vì một số phép trên GPU sẽ
    chậm đi nhiều; kết quả chạy lại trên cùng máy vẫn gần như trùng nhau.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(verbose: bool = True) -> torch.device:
    """Trả `cuda` nếu có GPU NVIDIA (Colab/Kaggle), ngược lại `cpu` (máy Trúc, GPU AMD)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"torch {torch.__version__} | thiết bị: {device_name(device)}")
    return device


def device_name(device: torch.device | str | None = None) -> str:
    """Tên thiết bị để ghi vào `metrics.json` và `so_sanh.csv`, ví dụ `cuda (Tesla T4)`."""
    device = torch.device(device) if device is not None else get_device(verbose=False)
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
def to_nchw(x) -> torch.Tensor:
    """Ảnh numpy kiểu Keras `(N, H, W, C)` → tensor float32 PyTorch `(N, C, H, W)`.

    Chỉ đổi thứ tự chiều và kiểu, **không** chuẩn hóa giá trị: notebook nào gốc
    chia 255 thì tự chia trước khi gọi, gốc không chia thì giữ nguyên 0–255.
    """
    # ascontiguousarray: sau np.rot90 (Gender) mảng có stride âm, torch không nhận
    x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
    if x.ndim != 4:
        raise ValueError(f"cần mảng 4 chiều (N, H, W, C), nhận {tuple(x.shape)}")
    return x.permute(0, 3, 1, 2).contiguous()


def onehot_to_index(y) -> torch.Tensor:
    """Nhãn one-hot Keras `(N, K)` (hoặc `(N, H, W, K)`) → chỉ số lớp kiểu long.

    `nn.CrossEntropyLoss` cần chỉ số lớp, còn `categorical_crossentropy` của
    Keras dùng one-hot. Hai cách cho cùng giá trị loss.
    """
    return torch.as_tensor(np.asarray(y)).argmax(dim=-1).long()


def make_loader(x, y=None, batch_size: int = 32, shuffle: bool = False, **kwargs) -> DataLoader:
    """Bọc tensor thành `DataLoader` (thay cho tham số `batch_size`, `shuffle` của `model.fit`).

    Keras `fit` mặc định xáo dữ liệu train mỗi epoch, nên train dùng `shuffle=True`,
    validation/test dùng `shuffle=False`. `num_workers=0` vì Windows hay lỗi
    khi chạy nhiều tiến trình trong notebook.
    """
    tensors = [torch.as_tensor(x)] + ([torch.as_tensor(y)] if y is not None else [])
    kwargs.setdefault("num_workers", 0)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, **kwargs)


def fer2013_path() -> Path:
    """Đường dẫn `fer2013.csv`: ưu tiên `data/emotion/fer2013.csv` (Colab/Kaggle chép vào đây),
    nếu không có thì dùng bản của BTL02 trong cùng repo (máy Trúc), khỏi chép 300 MB."""
    for p in (DATA_DIR / "emotion" / "fer2013.csv",
              ROOT.parent / "BTL02 Emotion Recognition" / "data" / "fer2013.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("Không thấy fer2013.csv: chép vào data/emotion/fer2013.csv (README mục dữ liệu)")


def load_fer2013(usage: str = "Training"):
    """Đọc FER2013 thành đúng mảng như notebook Emotion gốc.

    Gốc đọc `fer2013_training_onehot.csv` / `fer2013_publictest_onehot.csv`
    (2304 cột pixel + 7 cột one-hot). Hai file đó là `fer2013.csv` lọc theo
    cột `Usage` (Training → train, PublicTest → "test" của gốc) và đổi nhãn
    sang one-hot; đã kiểm 26/11: cùng số dòng (28.709 / 3.589), cùng thứ tự
    (3 dòng đầu, 3 dòng cuối và nhãn khớp output gốc).

    Trả `x` shape `(N, 48, 48, 1)` giá trị 0–255 (**không** chia 255 như gốc)
    và `y` one-hot `(N, 7)`. Dùng float32 thay float64 của `np.loadtxt` để đỡ
    tốn RAM; pixel là số nguyên 0–255 nên giá trị không đổi.
    """
    import pandas as pd

    df = pd.read_csv(fer2013_path())
    df = df[df["Usage"] == usage]
    if len(df) == 0:
        raise ValueError(f"usage phải là Training, PublicTest hoặc PrivateTest, nhận {usage!r}")
    x = np.stack([np.asarray(p.split(), dtype=np.float32) for p in df["pixels"]])
    x = x.reshape(-1, 48, 48, 1)
    y = np.eye(7, dtype=np.float32)[df["emotion"].to_numpy()]
    return x, y


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
def init_keras_like(model: nn.Module, forget_bias: bool = True) -> nn.Module:
    """Khởi tạo trọng số theo mặc định của Keras để so sánh công bằng.

    - `Dense`/`Conv2D` Keras: kernel `glorot_uniform`, bias 0. PyTorch mặc định
      dùng Kaiming uniform cho cả weight lẫn bias, nên phải gán lại.
    - `LSTM` Keras: kernel vào `glorot_uniform`, kernel hồi quy `orthogonal`,
      bias 0 và bias cổng quên = 1 (`unit_forget_bias=True`). PyTorch xếp các
      cổng theo thứ tự (i, f, g, o) nên cổng quên là đoạn `[h:2h]`; PyTorch có
      hai vector bias (`bias_ih`, `bias_hh`), chỉ gán 1 cho `bias_ih` để tổng
      bias cổng quên bằng 1 như Keras.
    - `BatchNormalization`: gamma 1, beta 0, giống mặc định PyTorch.
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            h = m.hidden_size
            for name, p in m.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(p)
                elif name.startswith("bias"):
                    nn.init.zeros_(p)
                    if forget_bias and name.startswith("bias_ih"):
                        with torch.no_grad():
                            p[h:2 * h].fill_(1.0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Số tham số, để `assert` với `Total params` (hoặc `Trainable params`) của `model.summary()` Keras."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def count_all(model: nn.Module) -> int:
    """Tham số + buffer (running mean/var của BatchNorm).

    Keras tính running mean/var vào `Total params` (phần "Non-trainable"),
    PyTorch để chúng là buffer, không nằm trong `parameters()`. Hàm này dùng
    để đối chiếu `Total params` của SegNet. Bỏ qua `num_batches_tracked`
    vì Keras không có biến này.
    """
    buffers = sum(b.numel() for n, b in model.named_buffers() if not n.endswith("num_batches_tracked"))
    return count_params(model, trainable_only=False) + buffers


def summary(model: nn.Module, input_size, **kwargs):
    """In bảng tóm tắt giống `model.summary()` Keras (dùng `torchinfo`).

    `input_size` gồm cả chiều batch, ví dụ `(1, 1, 28, 28)`. Nếu máy chưa cài
    `torchinfo` thì in `print(model)` và số tham số.
    """
    try:
        from torchinfo import summary as _summary
    except ImportError:
        print(model)
        print(f"Tổng tham số: {count_params(model):,} (chưa cài torchinfo nên không in shape)")
        return None
    kwargs.setdefault("col_names", ("output_size", "num_params"))
    kwargs.setdefault("depth", 3)
    # Dùng đúng thiết bị đang chứa mô hình: torchinfo gọi model.to(device), nên nếu truyền "cpu"
    # thì mô hình trên GPU bị kéo về CPU và cả notebook train bằng CPU (lỗi gặp trên Kaggle 26–27/11).
    kwargs.setdefault("device", next(model.parameters()).device)
    return _summary(model, input_size=input_size, **kwargs)


# ---------------------------------------------------------------------------
# Chỉ số (thay `metrics=[...]` trong `model.compile`)
# Mỗi hàm nhận (đầu ra mô hình, nhãn) của một batch và trả giá trị trung bình
# trên batch đó; `fit`/`evaluate` lấy trung bình có trọng số theo số mẫu.
# ---------------------------------------------------------------------------
def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Accuracy phân loại: `argmax` theo chiều lớp (chiều 1) so với chỉ số lớp.

    Dùng được cho cả `(N, K)` với nhãn `(N,)` và đầu ra theo pixel
    `(N, K, H, W)` với nhãn `(N, H, W)` (pixel accuracy của SegNet, trung bình
    trên mọi pixel như Keras). Nhãn one-hot cũng được (tự `argmax`).
    """
    if target.ndim == output.ndim:              # nhãn one-hot cùng shape với đầu ra
        target = target.argmax(dim=1)
    return (output.argmax(dim=1) == target).float().mean()


def sign_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Accuracy cho nhãn ±1 và đầu ra `tanh` (obj_detection): đúng khi cùng dấu.

    Keras báo "acc" với loss MSE theo cách khác (so khớp sau khi làm tròn),
    không phải accuracy phân loại thật; bản PyTorch dùng dấu và ghi rõ.
    """
    out = output.reshape(-1)
    tgt = target.reshape(-1).to(out.dtype)
    return (torch.sign(out) == torch.sign(tgt)).float().mean()


def mae(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sai số tuyệt đối trung bình (dùng làm chỉ số cho bài hồi quy)."""
    return (output.reshape(-1) - target.reshape(-1).to(output.dtype)).abs().mean()


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
class Callback:
    """Lớp gốc của callback, giống `Callback` của Keras (chỉ các mốc cần dùng).

    `fit` gọi `on_train_begin` một lần, `on_epoch_end(epoch, logs, model)` sau
    mỗi epoch (epoch đếm từ 1 như log Keras, `logs` có `loss`, `<metric>`,
    `val_loss`, `val_<metric>`, `lr`) và `on_train_end`. Callback đặt
    `self.stop_training = True` để dừng vòng train.
    """

    stop_training = False

    def on_train_begin(self, model: nn.Module) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict, model: nn.Module) -> None:
        pass

    def on_train_end(self, model: nn.Module) -> None:
        pass


def _better(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


class ModelCheckpoint(Callback):
    """Thay `ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)` của Keras.

    `filepath` dùng cú pháp format như Keras, ví dụ
    `str(results_dir("tu_2_Human_Gender_CNN") / "checkpoints/{epoch:06d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.pt")`.
    Epoch trong tên file đếm từ 1 như log gốc (file tốt nhất của Gender là
    `000027-...` ứng với dòng "Epoch 27/100000").
    Lưu `state_dict` (không lưu cả mô hình như `.hdf5`). Đường dẫn file tốt
    nhất nằm ở `self.best_path`, epoch tốt nhất ở `self.best_epoch`.
    """

    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min",
                 save_best_only: bool = True, verbose: int = 1):
        self.filepath = str(filepath)
        self.monitor, self.mode = monitor, mode
        self.save_best_only, self.verbose = save_best_only, verbose
        self.best = math.inf if mode == "min" else -math.inf
        self.best_path: str | None = None
        self.best_epoch: int | None = None

    def on_epoch_end(self, epoch, logs, model):
        current = logs.get(self.monitor)
        if current is None:
            raise KeyError(f"ModelCheckpoint: không có '{self.monitor}' trong logs {list(logs)}")
        improved = _better(current, self.best, self.mode)
        if self.save_best_only and not improved:
            if self.verbose:
                print(f"Epoch {epoch:05d}: {self.monitor} không cải thiện so với {self.best:.5f}")
            return
        path = Path(self.filepath.format(epoch=epoch, **logs))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f"Epoch {epoch:05d}: {self.monitor} {self.best:.5f} → {current:.5f}, lưu {path.name}")
        if improved:
            self.best, self.best_path, self.best_epoch = current, str(path), epoch


class EarlyStopping(Callback):
    """Thay `EarlyStopping(monitor='val_loss', patience=5)` của Keras.

    Dừng khi `monitor` không cải thiện (hơn `min_delta`) trong `patience`
    epoch liên tiếp: tăng bộ đếm rồi so `wait >= patience`. Cách đếm này
    khớp cả hai log gốc (đã kiểm 26/11): Gender Training Utilities tốt nhất
    epoch 27 (val_loss 0.3237) → dừng sau epoch 32; Emotion Training
    Utilities tốt nhất epoch 52 (val_loss 1.0145) → dừng sau epoch 57.

    Giống Keras (`restore_best_weights=False`): **không** tự khôi phục trọng
    số tốt nhất. Muốn dùng mô hình tốt nhất thì nạp file của `ModelCheckpoint`
    như notebook gốc.
    """

    def __init__(self, monitor: str = "val_loss", patience: int = 0, min_delta: float = 0.0,
                 mode: str = "min", verbose: int = 1):
        self.monitor, self.patience = monitor, patience
        self.min_delta, self.mode, self.verbose = abs(min_delta), mode, verbose
        self.best = math.inf if mode == "min" else -math.inf
        self.wait = 0
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self.stop_training = False

    def on_epoch_end(self, epoch, logs, model):
        current = logs.get(self.monitor)
        if current is None:
            raise KeyError(f"EarlyStopping: không có '{self.monitor}' trong logs {list(logs)}")
        if _better(current, self.best, self.mode, self.min_delta):
            self.best, self.best_epoch, self.wait = current, epoch, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.stop_training = True
            if self.verbose:
                print(f"Epoch {epoch:05d}: early stopping (tốt nhất epoch {self.best_epoch}, "
                      f"{self.monitor} = {self.best:.5f})")


class TensorBoard(Callback):
    """Thay callback `TensorBoard(log_dir=...)` của Keras: ghi loss, chỉ số, lr theo epoch.

    Log nằm ở `runs/<tên>` (git bỏ qua). Xem bằng `tensorboard --logdir runs`.
    Cần gói `tensorboard`; chỉ import khi tạo callback này.
    """

    def __init__(self, log_dir: str | Path):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def on_epoch_end(self, epoch, logs, model):
        for k, v in logs.items():
            if isinstance(v, (int, float)) and k != "epoch_time_s":
                self.writer.add_scalar(k, v, epoch)

    def on_train_end(self, model):
        self.writer.flush()
        self.writer.close()


def reduce_lr_on_plateau(optimizer, factor: float = 0.1, patience: int = 10,
                         min_lr: float = 0.0, min_delta: float = 1e-4):
    """Thay `ReduceLROnPlateau(monitor='val_loss', ...)` của Keras.

    Trả `torch.optim.lr_scheduler.ReduceLROnPlateau` đặt giống Keras: so sánh
    theo ngưỡng tuyệt đối `min_delta = 1e-4` (PyTorch mặc định ngưỡng tương
    đối), `cooldown = 0`. Truyền vào `fit(..., scheduler=...)`, `fit` gọi
    `scheduler.step(val_loss)` sau mỗi epoch.

    PyTorch giảm lr khi số epoch không cải thiện **lớn hơn** `patience` (với
    patience 5 là epoch xấu thứ 6). Trong log gốc Gender Training Utilities
    val_loss chưa lần nào xấu 6 epoch liền (EarlyStopping dừng trước) nên lr
    giữ 1e-3 suốt; cách đếm này không làm lệch so sánh.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience,
        threshold=min_delta, threshold_mode="abs", cooldown=0, min_lr=min_lr)


# ---------------------------------------------------------------------------
# Huấn luyện, đánh giá, dự đoán (thay `fit`, `evaluate`, `predict` của Keras)
# ---------------------------------------------------------------------------
def _split_batch(batch, device):
    """Tách batch của DataLoader thành (x, y); y = None nếu loader chỉ có x."""
    if isinstance(batch, (list, tuple)):
        x = batch[0].to(device)
        y = batch[1].to(device) if len(batch) > 1 else None
    else:
        x, y = batch.to(device), None
    return x, y


def _run_epoch(model, loader, loss_fn, device, metric=None, optimizer=None, max_batches=None):
    """Một lượt qua `loader`. Có `optimizer` thì train, không có thì chỉ tính loss/chỉ số.

    Loss và chỉ số lấy trung bình theo số mẫu (giống số Keras in ở cuối epoch).
    Loss train là trung bình trong lúc trọng số đang thay đổi và dropout đang
    bật, đúng như `loss` trong log Keras.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_metric, n_seen = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = _split_batch(batch, device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n = x.shape[0]
            total_loss += loss.item() * n
            if metric is not None:
                total_metric += float(metric(out.detach(), y)) * n
            n_seen += n
    if n_seen == 0:
        raise ValueError("loader rỗng")
    return total_loss / n_seen, (total_metric / n_seen if metric is not None else None)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None,
        loss_fn, optimizer, epochs: int, metric=None, metric_name: str = "acc",
        callbacks: list[Callback] | None = None, scheduler=None, device=None,
        verbose: int = 1, max_batches: int | None = None) -> dict:
    """Vòng huấn luyện thay `model.fit(x, y, batch_size, epochs, validation_data, callbacks)`.

    Tham số:
        loss_fn      ví dụ `nn.CrossEntropyLoss()` (thay `categorical_crossentropy`).
        metric       hàm chỉ số (`accuracy`, `sign_accuracy`, `mae`) hoặc None.
        metric_name  tên chỉ số trong `history` và log, mặc định `acc` như Keras 2.0.9.
        callbacks    danh sách `ModelCheckpoint`, `EarlyStopping`, `TensorBoard`...
        scheduler    ví dụ `reduce_lr_on_plateau(...)`; gọi `step(val_loss)` mỗi epoch.
        max_batches  chỉ dùng khi chạy thử (`CHAY_THU = True`): cắt số batch mỗi epoch.

    Trả `history` dạng dict giống `History.history` của Keras:
    `loss`, `<metric_name>`, `val_loss`, `val_<metric_name>`, `lr`, `epoch_time_s`,
    mỗi khóa một danh sách theo epoch. Tổng thời gian train: `sum(history["epoch_time_s"])`.
    """
    device = device or next(model.parameters()).device
    model.to(device)
    callbacks = callbacks or []
    has_val = val_loader is not None
    keys = ["loss"] + ([metric_name] if metric else [])
    if has_val:
        keys += ["val_loss"] + ([f"val_{metric_name}"] if metric else [])
    history = {k: [] for k in keys + ["lr", "epoch_time_s"]}

    if verbose:
        print(f"fit: train trên {device_name(device)}")
    for cb in callbacks:
        cb.on_train_begin(model)
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]     # lr dùng trong epoch này
        t0 = time.perf_counter()
        loss, met = _run_epoch(model, train_loader, loss_fn, device, metric, optimizer, max_batches)
        logs = {"loss": loss}
        if metric:
            logs[metric_name] = met
        if has_val:
            val_loss, val_met = _run_epoch(model, val_loader, loss_fn, device, metric, None, max_batches)
            logs["val_loss"] = val_loss
            if metric:
                logs[f"val_{metric_name}"] = val_met
        logs["lr"] = lr
        logs["epoch_time_s"] = time.perf_counter() - t0
        for k in history:
            history[k].append(logs[k])

        if verbose:
            parts = [f"{k}: {logs[k]:.4f}" for k in keys]
            print(f"Epoch {epoch}/{epochs} - {logs['epoch_time_s']:.1f}s - " + " - ".join(parts)
                  + (f" - lr: {lr:.1e}" if scheduler is not None else ""))

        for cb in callbacks:
            cb.on_epoch_end(epoch, logs, model)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(logs["val_loss"] if has_val else logs["loss"])
            else:
                scheduler.step()
        if any(getattr(cb, "stop_training", False) for cb in callbacks):
            break
    for cb in callbacks:
        cb.on_train_end(model)
    return history


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, metric=None, device=None,
             max_batches: int | None = None):
    """Thay `model.evaluate(x, y)`: `model.eval()` + `torch.no_grad()`, trả `(loss, chỉ số)`.

    Không có `metric` thì chỉ số là None. Dropout tắt và BatchNorm dùng
    running mean/var, giống Keras ở chế độ suy luận.
    """
    device = device or next(model.parameters()).device
    model.to(device)
    return _run_epoch(model, loader, loss_fn, device, metric, None, max_batches)


@torch.no_grad()
def predict(model: nn.Module, data, batch_size: int = 32, device=None,
            return_targets: bool = False):
    """Thay `model.predict(x)`: trả đầu ra mô hình (tensor trên CPU), ghép theo batch.

    `data` là `DataLoader` hoặc tensor/mảng numpy `x` (đã đúng shape PyTorch).
    Đầu ra là **logits** khi lớp cuối không có softmax; cần xác suất như Keras
    thì gọi `torch.softmax(out, dim=1)`. `return_targets=True` trả thêm nhãn
    (khi `data` là DataLoader có nhãn).
    """
    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()
    if not isinstance(data, DataLoader):
        x = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data))
        data = make_loader(x, batch_size=batch_size)
    outs, targets = [], []
    for batch in data:
        x, y = _split_batch(batch, device)
        outs.append(model(x).cpu())
        if y is not None:
            targets.append(y.cpu())
    out = torch.cat(outs)
    if return_targets:
        return out, (torch.cat(targets) if targets else None)
    return out


# ---------------------------------------------------------------------------
# Lưu kết quả (results/<notebook>/ và results/so_sanh.csv)
# ---------------------------------------------------------------------------
def results_dir(name: str) -> Path:
    """`results/<name>/`, tạo nếu chưa có. `name` là tên notebook không kèm `_pytorch`
    (bản Training Utilities thêm tiền tố `tu_`)."""
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_history(history: dict, name: str, metric_name: str = "acc") -> Path:
    """Ghi `results/<name>/history.csv` với các cột
    `epoch, train_loss, train_metric, val_loss, val_metric, lr, epoch_time_s`."""
    path = results_dir(name) / "history.csv"
    n = len(history["loss"])
    col = lambda k: history.get(k, [None] * n)
    rows = zip(range(1, n + 1), col("loss"), col(metric_name), col("val_loss"),
               col(f"val_{metric_name}"), col("lr"), col("epoch_time_s"))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_metric", "val_loss", "val_metric", "lr", "epoch_time_s"])
        for r in rows:
            w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v) for v in r])
    return path


def save_metrics(name: str, **metrics) -> Path:
    """Ghi `results/<name>/metrics.json`.

    Nên truyền: `test_loss, test_metric, n_params, epochs_ran, best_epoch,
    train_time_s, device, seed`. `torch_version` tự thêm.
    """
    metrics.setdefault("torch_version", torch.__version__)
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
    path = results_dir(name) / "metrics.json"
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def plot_history(history: dict, name: str | None = None, metric_name: str = "acc",
                 metric_label: str | None = None, title: str | None = None):
    """Vẽ loss và chỉ số theo epoch (train và validation), lưu `results/<name>/curves.png`.

    Thay cho đoạn `plt.plot(history.history['loss'])` trong notebook gốc.
    Trả `fig` để notebook hiển thị.
    """
    import matplotlib.pyplot as plt

    has_metric = metric_name in history
    fig, axes = plt.subplots(1, 2 if has_metric else 1, figsize=(11 if has_metric else 5.5, 4))
    axes = np.atleast_1d(axes)
    epochs = range(1, len(history["loss"]) + 1)
    panels = [("loss", "loss")] + ([(metric_name, metric_label or metric_name)] if has_metric else [])
    for ax, (key, label) in zip(axes, panels):
        ax.plot(epochs, history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if name:
        fig.savefig(results_dir(name) / "curves.png", dpi=150, bbox_inches="tight")
    return fig


@contextmanager
def _khoa_file(path: Path, timeout: float = 60.0):
    """Khóa đơn giản bằng file `<path>.lock` (tạo độc quyền), dùng được cả Windows và Linux.

    Nếu khóa tồn tại quá `timeout` giây (tiến trình trước chết giữa chừng) thì coi như bỏ.
    """
    lock = Path(str(path) + ".lock")
    t0 = time.time()
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if time.time() - t0 > timeout:
                lock.unlink(missing_ok=True)
            time.sleep(0.1)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def ghi_so_sanh(notebook: str, thu_muc: str, chi_so: str, keras: float | None, pytorch: float,
                keras_params=None, pytorch_params=None, epochs_keras=None, epochs_pytorch=None,
                optimizer="", lr=None, batch=None, device="", train_time_s=None, seed=0,
                ghi_chu: str = "") -> dict:
    """Thêm hoặc cập nhật **một dòng** của `results/so_sanh.csv` (bảng Keras ↔ PyTorch cho báo cáo).

    Khóa của dòng là `(notebook, thu_muc, chi_so)`: chạy lại notebook thì dòng cũ
    được thay, các dòng khác giữ nguyên. `chenh_lech = pytorch − keras` (4 chữ số).
    `keras` là số in trong output notebook gốc trong máy (Kế hoạch mục 3).
    Gốc không in số (Func_apxm_MLP chỉ có hình) thì để `keras=None`: cột
    `keras` và `chenh_lech` để trống, số đọc từ hình ghi vào `ghi_chu`.
    Func_apxm_MLP ghi một dòng mỗi vòng quét, đặt `notebook="Func_apxm_MLP_A"`...
    """
    path = RESULTS_DIR / "so_sanh.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v, nd=4):
        return "" if v is None else f"{float(v):.{nd}f}"

    def text(v):
        return "" if v is None else v

    row = {
        "notebook": notebook, "thu_muc": thu_muc, "chi_so": chi_so,
        "keras": fmt(keras), "pytorch": fmt(pytorch),
        "chenh_lech": "" if keras is None else fmt(float(pytorch) - float(keras)),
        "keras_params": text(keras_params), "pytorch_params": text(pytorch_params),
        "epochs_keras": text(epochs_keras), "epochs_pytorch": text(epochs_pytorch),
        "optimizer": optimizer, "lr": text(lr), "batch": text(batch), "device": device,
        "train_time_s": fmt(train_time_s, 1), "seed": seed, "ghi_chu": ghi_chu,
    }
    with _khoa_file(path):        # Kaggle 2 GPU chạy 2 notebook song song, tránh ghi đè lẫn nhau
        rows = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        key = (notebook, thu_muc, chi_so)
        rows = [r for r in rows if (r["notebook"], r["thu_muc"], r["chi_so"]) != key] + [row]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SO_SANH_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    print(f"so_sanh.csv: {notebook} | {chi_so} Keras {row['keras'] or '(chỉ có hình)'} "
          f"| PyTorch {row['pytorch']} | chênh {row['chenh_lech'] or '-'}")
    return row


__all__ = [
    "ROOT", "PYTORCH_DIR", "DATA_DIR", "RESULTS_DIR", "RUNS_DIR", "YEU_CAU_DIR", "EXAMPLES_DIR",
    "set_seed", "get_device", "device_name",
    "to_nchw", "onehot_to_index", "make_loader", "fer2013_path", "load_fer2013",
    "init_keras_like", "count_params", "count_all", "summary",
    "accuracy", "sign_accuracy", "mae",
    "Callback", "ModelCheckpoint", "EarlyStopping", "TensorBoard", "reduce_lr_on_plateau",
    "fit", "evaluate", "predict",
    "results_dir", "save_history", "save_metrics", "plot_history", "ghi_so_sanh",
    # để notebook không phải import lại
    "np", "torch", "nn", "DataLoader", "TensorDataset", "Path", "time",
]
