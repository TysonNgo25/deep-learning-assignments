"""E1 – Kiến trúc: so 4 mạng Small / Medium / Medium_01 / Large, các tham số khác giữ như baseline.

Phụ trách: Phạm Trần Minh Khang

    python -m experiments.run_E1                  # cả 4 run
    python -m experiments.run_E1 --runs E1-1 E1-4
E1-2 trùng baseline: chạy lại để kiểm tra tính tái lập (cùng seed, cùng máy thì phải ra gần như BASE).
"""
from experiments.common import main_group

RUNS = {
    "E1-1": ({"model.arch": "small"}, "Kiến trúc Small (2 block, FC 1024)"),
    "E1-2": ({}, "Kiến trúc Medium (= baseline)"),
    "E1-3": ({"model.arch": "medium_01"}, "Kiến trúc Medium_01 (3/4/4 conv mỗi block)"),
    "E1-4": ({"model.arch": "large"}, "Kiến trúc Large (4 block 64-128-256-512, FC 2048)"),
}

if __name__ == "__main__":
    # E1-2 đã là baseline nên không vẽ thêm đường BASE.
    main_group("E1", RUNS, "E1 – so sánh kiến trúc", with_base=False)
