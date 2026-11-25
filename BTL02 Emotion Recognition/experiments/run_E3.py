"""E3 – Optimizer và learning rate (baseline: Adam lr 1e-4, không scheduler).

Phụ trách: Tạ Quang Trung

    python -m experiments.run_E3
E3-1 đổi cả optimizer lẫn lr vì SGD với lr 1e-4 gần như không học trong 60 epoch;
"biến" ở đây là cặp (optimizer, lr thường dùng với nó).
"""
from experiments.common import main_group

RUNS = {
    "E3-1": ({"train.optimizer": "sgd", "train.lr": 0.001}, "SGD momentum 0,9, lr 1e-3"),
    "E3-2": ({"train.lr": 0.001}, "Adam lr 1e-3"),
    "E3-3": ({"train.scheduler": "plateau"}, "Adam lr 1e-4 + ReduceLROnPlateau (factor 0,1, patience 2)"),
}

if __name__ == "__main__":
    main_group("E3", RUNS, "E3 – optimizer / learning rate")
