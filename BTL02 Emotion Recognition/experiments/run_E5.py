"""E5 – Hàm kích hoạt (baseline: ReLU), áp dụng cho mọi Conv và lớp FC ẩn.

Phụ trách: Ngô Nguyễn Thành Nhân

    python -m experiments.run_E5
Dự phòng khi trễ: chỉ chạy Sigmoid (--runs E5-3) để so ReLU vs Sigmoid.
"""
from experiments.common import main_group

RUNS = {
    "E5-1": ({"model.activation": "leaky_relu"}, "LeakyReLU (độ dốc âm 0,01)"),
    "E5-2": ({"model.activation": "elu"}, "ELU"),
    "E5-3": ({"model.activation": "sigmoid"}, "Sigmoid"),
}

if __name__ == "__main__":
    main_group("E5", RUNS, "E5 – hàm kích hoạt")
