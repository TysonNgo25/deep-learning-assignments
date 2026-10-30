"""Đường dẫn dùng chung, tính tương đối từ gốc project để code chạy được trên mọi máy."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"
