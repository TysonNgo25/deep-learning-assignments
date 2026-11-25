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


def main():
    x = torch.zeros(2, 1, 48, 48)
    print(f"{'arch':10s} {'BN':5s} {'số conv':>7s} {'FC':>5s} {'tham số':>12s}  output")
    for arch, (blocks, fc) in ARCHS.items():
        for bn in (False, True):
            model = EmotionCNN(arch=arch, batchnorm=bn).eval()
            with torch.no_grad():
                out = model(x)
            n_conv = sum(len(b) for b in blocks)
            print(f"{arch:10s} {str(bn):5s} {n_conv:7d} {fc:5d} {count_params(model):12,d}  "
                  f"{tuple(out.shape)}")
    w = last_conv(EmotionCNN()).weight
    print(f"\nLớp conv cuối của Medium: weight {tuple(w.shape)} "
          f"({w.shape[0]} filter, mỗi filter {w.shape[1]}×{w.shape[2]}×{w.shape[3]})")


if __name__ == "__main__":
    main()
