"""E6 – Regularization: dropout và CrossEntropy có trọng số lớp (baseline: dropout block 0,2, FC 0,5).

Phụ trách: Ngô Nguyễn Thành Nhân

    python -m experiments.run_E6
Dự phòng khi trễ: --runs E6-1 E6-3.
"""
from experiments.common import main_group

RUNS = {
    "E6-1": ({"model.dropout_block": 0.0, "model.dropout_fc": 0.0}, "Không dropout (block 0, FC 0)"),
    "E6-2": ({"model.dropout_block": 0.4}, "Dropout block 0,4"),
    "E6-3": ({"train.class_weighted_loss": True}, "CE có trọng số lớp w_c = N/(K·n_c)"),
}

if __name__ == "__main__":
    main_group("E6", RUNS, "E6 – regularization")
