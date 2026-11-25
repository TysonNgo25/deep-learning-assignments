"""Các kiến trúc CNN cho FER2013: Small, Medium, Medium_01, Large, tùy chọn BatchNorm/activation/dropout.

Phụ trách: Phạm Trần Minh Khang

Bốn kiến trúc theo đúng cấu trúc của code mẫu (SMALL_CNN … LARGE_CNN, viết bằng Keras),
viết lại bằng PyTorch. In bảng số tham số (đứng ở thư mục BTL02):
    python -m src.models
"""
import torch
from torch import nn

# Mỗi kiến trúc: danh sách khối (số filter của từng conv trong khối) và số nơ-ron lớp FC.
# Hết mỗi khối là MaxPool 2×2 nên ảnh 48×48 nhỏ dần: 2 khối -> 12×12, 3 khối -> 6×6, 4 khối -> 3×3.
ARCHS = {
    "small": ([[64, 64], [128, 128, 256, 256]], 1024),
    "medium": ([[64] * 2, [128] * 3, [256] * 3], 1024),
    "medium_01": ([[64] * 3, [128] * 4, [256] * 4], 1024),
    "large": ([[64] * 4, [128] * 4, [256] * 4, [512] * 4], 2048),
}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
}


def make_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"activation '{name}' không hỗ trợ, chọn một trong {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class EmotionCNN(nn.Module):
    """CNN kiểu VGG: các khối [Conv3×3 (+BN) + activation]×k -> MaxPool -> Dropout, rồi FC.

    Conv dùng padding=1 (tương đương padding='same' của Keras) để chỉ MaxPool làm ảnh nhỏ lại.
    Khi có BatchNorm, conv bỏ bias vì BN đã có tham số dịch (beta), giữ bias là thừa.
    Lớp cuối trả logits (không softmax) vì nn.CrossEntropyLoss đã gồm log-softmax.
    """

    def __init__(self, arch="medium", activation="relu", batchnorm=False,
                 dropout_block=0.2, dropout_fc=0.5, fc_units=None,
                 num_classes=7, in_channels=1, image_size=48):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch '{arch}' không có, chọn một trong {list(ARCHS)}")
        blocks, default_fc = ARCHS[arch]
        fc_units = fc_units or default_fc

        layers, channels = [], in_channels
        for block in blocks:
            for out_channels in block:
                layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1,
                                        bias=not batchnorm))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(make_activation(activation))
                channels = out_channels
            layers += [nn.MaxPool2d(2), nn.Dropout(dropout_block)]
        self.features = nn.Sequential(*layers)

        side = image_size // 2 ** len(blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * side * side, fc_units),
            make_activation(activation),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_units, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo He (Kaiming) cho mọi Conv/Linear, bias = 0, dùng chung cho mọi run.

        Khởi tạo mặc định của PyTorch (Var = 1/(3·fan_in)) làm tín hiệu teo ~6 lần qua mỗi
        lớp ReLU: đo trên Medium, std sau ReLU giảm 0,37 -> 0,005 và gradient ở conv1 chỉ
        ~6e-6, mạng đứng ở mức đoán toàn lớp Happy. He (Var = 2/fan_in) giữ std ~1 qua 9 lớp,
        gradient conv1 ~7e-2.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(model_cfg, num_classes=7):
    """Tạo mô hình từ khối `model` của config (arch, activation, batchnorm, dropout, fc_units)."""
    return EmotionCNN(num_classes=num_classes, **model_cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv_layers(model):
    """Danh sách (tên, Conv2d) theo thứ tự; phần tử cuối là lớp conv cuối để vẽ filter."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def last_conv(model):
    return conv_layers(model)[-1][1]


def init_check(seed=0, batch=64):
    """So khởi tạo mặc định của PyTorch với khởi tạo He trên Medium (lý do dùng He).

    Đưa một batch nhiễu N(0, 1) (giống ảnh đã chuẩn hóa) qua mạng, đo độ lệch chuẩn đầu ra
    sau từng lớp kích hoạt và độ lớn gradient ở conv đầu tiên với nhãn ngẫu nhiên.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 1, 48, 48, generator=gen)
    y = torch.randint(0, 7, (batch,), generator=gen)
    rows = []
    for name in ("pytorch_default", "he"):
        torch.manual_seed(seed)
        model = EmotionCNN()  # dropout giữ nguyên nhưng eval() tắt nó
        if name == "pytorch_default":
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    m.reset_parameters()  # khởi tạo mặc định của PyTorch
        model.eval()
        stds = []
        hooks = [m.register_forward_hook(lambda m, i, o: stds.append(o.std().item()))
                 for m in model.modules() if isinstance(m, nn.ReLU)]
        logits = model(x)
        for h in hooks:
            h.remove()
        nn.functional.cross_entropy(logits, y).backward()
        grad = model.features[0].weight.grad.abs().mean().item()
        for k, s in enumerate(stds, start=1):
            rows.append({"init": name, "relu_layer": k, "output_std": round(s, 6)})
        rows.append({"init": name, "relu_layer": "logits", "output_std": round(logits.std().item(), 6)})
        rows.append({"init": name, "relu_layer": "grad_conv1_mean_abs", "output_std": f"{grad:.3e}"})
    return rows


def main():
    import csv

    from src.paths import RESULTS_DIR

    out_dir = RESULTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(2, 1, 48, 48)
    rows = []
    print(f"{'arch':10s} {'BN':5s} {'số conv':>7s} {'FC':>5s} {'tham số':>12s}  output")
    for arch, (blocks, fc) in ARCHS.items():
        for bn in (False, True):
            model = EmotionCNN(arch=arch, batchnorm=bn).eval()
            with torch.no_grad():
                out = model(x)
            n_conv = sum(len(b) for b in blocks)
            flatten = model.classifier[1].in_features
            rows.append({"arch": arch, "batchnorm": bn, "n_conv": n_conv,
                         "filters": "-".join(str(b[-1]) for b in blocks), "flatten": flatten,
                         "fc": fc, "n_params": count_params(model),
                         "fc1_params": flatten * fc + fc})
            print(f"{arch:10s} {str(bn):5s} {n_conv:7d} {fc:5d} {count_params(model):12,d}  "
                  f"{tuple(out.shape)}")
    with open(out_dir / "params.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    init_rows = init_check()
    with open(out_dir / "init_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(init_rows[0]))
        w.writeheader()
        w.writerows(init_rows)
    for name in ("pytorch_default", "he"):
        r = [x for x in init_rows if x["init"] == name]
        print(f"{name:16s} std sau ReLU 1 -> 9: {r[0]['output_std']} -> {r[8]['output_std']}, "
              f"grad conv1 {r[-1]['output_std']}")

    w = last_conv(EmotionCNN()).weight
    print(f"\nLớp conv cuối của Medium: weight {tuple(w.shape)} "
          f"({w.shape[0]} filter, mỗi filter {w.shape[1]}×{w.shape[2]}×{w.shape[3]})")
    print(f"Đã ghi {out_dir / 'params.csv'}, {out_dir / 'init_check.csv'}")


if __name__ == "__main__":
    main()
