"""Kiểm tra thư mục BTL01 trước khi nộp và (tùy chọn) nén file zip.

Chạy từ thư mục "BTL01 - Function Approximation":
    python ../.claude/skills/dong-goi-btl01/check_submission.py
    python ../.claude/skills/dong-goi-btl01/check_submission.py --zip BTL01_Nhom05_HoTen_MSSV.zip
Hoặc truyền đường dẫn thư mục BTL01 bằng --root.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

REQUIRED = [
    "README.md",
    "requirements.txt",
    "configs/baseline.yaml",
    "data/train_pool.csv",
    "data/val.csv",
    "data/test.csv",
    "src/data.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/plots.py",
    "experiments/run_A.py",
    "experiments/run_B.py",
    "experiments/run_C.py",
    "results/A/summary.csv",
    "results/B/summary.csv",
    "results/C/summary.csv",
    "results/final/selected_model.pt",
    "results/final/final_metrics.json",
]
REQUIRED_GLOBS = {
    "report/*.docx": "báo cáo Word",
    "report/*.pdf": "báo cáo PDF",
    "slides/*.pptx": "slide PowerPoint",
}
EXPECTED_LOGS = {"A": 30, "B": 39, "C": 12}

# File/thư mục nội bộ không đưa vào zip. Sửa danh sách này nếu nhóm muốn nộp kèm.
EXCLUDE = {
    "Đề và yêu cầu",
    "Kế hoạch.md",
    "CodetoTruc.md",
    "CodetoNhan.md",
    "tra_loi_van_dap_K.md",
}
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".git", ".claude"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}

TF_IMPORT = re.compile(r"^\s*(import|from)\s+(tensorflow|keras)\b", re.M)
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]+(Users|Data|Documents)|/home/\w|/Users/\w")
TEST_SELECT = re.compile(r"test", re.I)
SELECT_WORDS = re.compile(r"\b(min|argmin|idxmin|sort_values|sorted|best)\b")


def code_files(root: Path):
    for ext in ("*.py", "*.yaml", "*.yml", "*.ipynb"):
        for p in root.rglob(ext):
            if not SKIP_PARTS.intersection(p.parts):
                yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Thư mục BTL01 (mặc định: thư mục hiện tại)")
    ap.add_argument("--zip", help="Tên file zip cần tạo (ghi ra thư mục cha của root)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "src").is_dir():
        print(f"Không thấy thư mục src/ trong {root}. Hãy chạy từ thư mục BTL01 hoặc dùng --root.")
        return 2

    errors: list[str] = []
    warns: list[str] = []

    for rel in REQUIRED:
        if not (root / rel).exists():
            errors.append(f"Thiếu file bắt buộc: {rel}")
    for pattern, label in REQUIRED_GLOBS.items():
        if not list(root.glob(pattern)):
            errors.append(f"Chưa có {label} ({pattern})")

    for group, n in EXPECTED_LOGS.items():
        found = len(list((root / "results" / group / "logs").glob(f"{group}_*_seed*.csv")))
        if found != n:
            errors.append(f"results/{group}/logs có {found} log, cần {n}")

    fm = root / "results/final/final_metrics.json"
    if fm.exists():
        try:
            data = json.loads(fm.read_text(encoding="utf-8"))
            if "test_metrics" not in data:
                warns.append("final_metrics.json không có khóa test_metrics")
        except json.JSONDecodeError as e:
            errors.append(f"final_metrics.json không đọc được: {e}")

    for p in code_files(root):
        rel = p.relative_to(root).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        if TF_IMPORT.search(text):
            errors.append(f"Còn import TensorFlow/Keras: {rel}")
        for i, line in enumerate(text.splitlines(), 1):
            if ABS_PATH.search(line):
                errors.append(f"Đường dẫn tuyệt đối: {rel}:{i}: {line.strip()[:100]}")
            if p.suffix == ".py" and TEST_SELECT.search(line) and SELECT_WORDS.search(line):
                warns.append(f"Đọc lại (test + chọn?): {rel}:{i}: {line.strip()[:100]}")

    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    if "best_model.pt" in readme and not (root / "models/best_model.pt").exists():
        warns.append("README nhắc models/best_model.pt nhưng file không có; checkpoint thật ở results/final/selected_model.pt")

    req = (root / "requirements.txt").read_text(encoding="utf-8") if (root / "requirements.txt").exists() else ""
    for line in req.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" not in line:
            warns.append(f"requirements.txt chưa ghim phiên bản: {line}")

    if list(root.rglob("__pycache__")):
        warns.append("Có thư mục __pycache__ (script sẽ bỏ qua khi nén)")

    print(f"Thư mục: {root}\n")
    for e in errors:
        print(f"LỖI      {e}")
    for w in warns:
        print(f"CẢNH BÁO {w}")
    print(f"\nTổng: {len(errors)} lỗi, {len(warns)} cảnh báo")

    if args.zip:
        if errors:
            print("\nCòn lỗi nên không nén. Sửa lỗi rồi chạy lại.")
            return 1
        out = root.parent / args.zip
        files = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if SKIP_PARTS.intersection(rel.parts) or p.suffix in SKIP_SUFFIXES:
                continue
            if rel.parts[0] in EXCLUDE:
                continue
            files.append((p, Path(root.name) / rel))
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for p, arc in files:
                zf.write(p, arc.as_posix())
        size = out.stat().st_size / 1e6
        print(f"\nĐã nén {len(files)} file vào {out} ({size:.1f} MB):")
        for _, arc in files:
            print(f"  {arc.as_posix()}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
