"""E4 – Mức augmentation (baseline: lật ngang + xoay ±10° + dịch 10%).

Phụ trách: Tạ Quang Trung

    python -m experiments.run_E4
"""
from experiments.common import main_group

NO_ROTATE_SHIFT = {"augmentation.rotation_deg": 0, "augmentation.shift": 0}

RUNS = {
    "E4-1": ({"augmentation.hflip": False, **NO_ROTATE_SHIFT}, "Không augmentation"),
    "E4-2": (dict(NO_ROTATE_SHIFT), "Chỉ lật ngang"),
    "E4-3": ({"augmentation.zoom": 0.1}, "Lật + xoay + dịch + zoom 10%"),
}

if __name__ == "__main__":
    main_group("E4", RUNS, "E4 – augmentation")
