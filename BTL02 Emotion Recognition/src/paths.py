"""Đường dẫn dùng chung, tính tương đối theo thư mục BTL02 để chạy được trên mọi máy.

Phụ trách: Huỳnh Ngọc Trúc
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

FER_CSV = DATA_DIR / "fer2013.csv"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
