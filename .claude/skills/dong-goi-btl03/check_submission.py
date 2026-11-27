"""Kiểm tra checklist nộp BTL03 (Keras → PyTorch) và (tùy chọn) nén file zip.

Chạy từ thư mục 'BTL03 - Chuyển code từ keras sang pytorch deep learning':
    python ../.claude/skills/dong-goi-btl03/check_submission.py
    python ../.claude/skills/dong-goi-btl03/check_submission.py --zip BTL03_Nhom05_HuynhNgocTruc_3123411312.zip
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # in tiếng Việt trên console Windows

NAME = "BTL03 - Chuyển code từ keras sang pytorch deep learning"
BTL = Path.cwd()
if unicodedata.normalize("NFC", BTL.name) != NAME:
    sys.exit(f"Hãy chạy từ thư mục '{NAME}'.")

TARGETS = [
    "0. Getting Started/getting_started",
    "1. MLP/Func_apxm_MLP",
    "1. MLP/MNIST_MLP",
    "2. CNN + MLP/1_MNIST_CNN",
    "2. CNN + MLP/2_Human_Gender_CNN",
    "2. CNN + MLP/3_Emotion_Reg_CNN",
    "3. CNN + LSTM/MNIST_CNN_LSTM",
    "4. Training Utilities/2_Human_Gender_CNN",
    "4. Training Utilities/3_Emotion_Reg_CNN",
    "5. Others/1_segnet",
    "5. Others/2_obj_detection",
]
REQUIRED = ["README.md", "requirements.txt", "pytorch/common.py", "results/so_sanh.csv"]
EXCLUDE_DIRS = {"data", "Yêu cầu", "docs", "checkpoints", "runs", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_FILES = {"CodetoTruc.md", ".gitkeep"}
# Trong zip, report/ chỉ giữ bản hoàn chỉnh .docx/.pdf như BTL02 (Trúc chốt 27/11); file .md, script ghép, vẽ hình ở lại repo.
ZIP_ONLY = {"report": {".docx", ".pdf"}}
KERAS_RE = re.compile(r"^\s*(import|from)\s+(tensorflow|keras)\b", re.M)
ABS_RE = re.compile(r"[A-Za-z]:[\\/]+(Users|Data)[\\/]")

errors, warnings = [], []


def parts(p: Path) -> set:
    return {unicodedata.normalize("NFC", s) for s in p.relative_to(BTL).parts}


for rel in REQUIRED:
    if not (BTL / rel).exists():
        errors.append(f"Thiếu {rel}")
for ext in (".docx", ".pdf"):
    if not any(p.suffix == ext for p in (BTL / "report").glob("*")):
        errors.append(f"Thiếu báo cáo {ext} trong report/")

for t in TARGETS:
    nb_path = BTL / "pytorch" / f"{t}_pytorch.ipynb"
    rel = nb_path.relative_to(BTL).as_posix()
    if not nb_path.exists():
        errors.append(f"Thiếu {rel}")
        continue
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"{rel}: không đọc được JSON ({e})")
        continue
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    src = "\n".join("".join(c.get("source", "")) if isinstance(c.get("source"), list) else c.get("source", "")
                    for c in code)
    if not any(c.get("outputs") for c in code):
        errors.append(f"{rel}: chưa có output (chưa chạy hoặc chưa lưu)")
    if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
        errors.append(f"{rel}: có ô báo lỗi")
    if KERAS_RE.search(src) or "tf.keras" in src:
        errors.append(f"{rel}: ô code còn tensorflow/keras")
    if re.search(r"^\s*CHAY_THU\s*=\s*True", src, re.M):
        errors.append(f"{rel}: còn CHAY_THU = True (kết quả chạy thử, không phải bản nộp)")
    if ABS_RE.search(src):
        errors.append(f"{rel}: có đường dẫn tuyệt đối")
    if "count_params" not in src and not rel.startswith("pytorch/0."):   # getting_started không có mô hình
        warnings.append(f"{rel}: không thấy kiểm số tham số (count_params)")

common = BTL / "pytorch" / "common.py"
if common.exists():
    text = common.read_text(encoding="utf-8", errors="ignore")
    if KERAS_RE.search(text):
        errors.append("pytorch/common.py: còn tensorflow/keras")
    if ABS_RE.search(text):
        errors.append("pytorch/common.py: có đường dẫn tuyệt đối")

ss = BTL / "results" / "so_sanh.csv"
if ss.exists():
    with ss.open(encoding="utf-8") as f:
        names = {r.get("notebook", "").strip() for r in csv.DictReader(f)}
    for t in TARGETS:
        base = t.split("/")[-1]
        key = f"tu_{base}" if t.startswith("4.") else base
        if t.startswith("0."):
            continue  # getting_started không có kết quả để so
        if base == "Func_apxm_MLP":   # rule btl03-ket-qua: một dòng mỗi vòng quét A/B/C
            thieu = [f"{key}_{v}" for v in "ABC" if f"{key}_{v}" not in names]
            if thieu:
                errors.append(f"results/so_sanh.csv: thiếu dòng {', '.join(thieu)}")
        elif key not in names:
            errors.append(f"results/so_sanh.csv: thiếu dòng cho {key}")

req = BTL / "requirements.txt"
if req.exists() and not all(("==" in l or ">=" in l) for l in req.read_text().split() if l.strip()):
    warnings.append("requirements.txt chưa ghi phiên bản hết các gói")

for e in errors:
    print("LỖI     ", e)
for w in warnings:
    print("CẢNH BÁO", w)
print(f"\n{len(errors)} lỗi, {len(warnings)} cảnh báo.")

ap = argparse.ArgumentParser()
ap.add_argument("--zip")
args = ap.parse_args()
if args.zip:
    if errors:
        sys.exit("Còn LỖI, chưa nén.")
    out = BTL.parent / args.zip
    top = Path(args.zip).stem
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BTL.rglob("*")):
            if p.is_dir() or parts(p) & EXCLUDE_DIRS or p.name in EXCLUDE_FILES \
                    or p.suffix in (".pyc", ".pt", ".pth") or p.name.startswith("~$"):
                continue
            rel = p.relative_to(BTL)
            if rel.parts[0] in ZIP_ONLY and p.suffix not in ZIP_ONLY[rel.parts[0]]:
                continue
            z.write(p, Path(top) / rel)
            print("  +", rel.as_posix())
    print(f"\nĐã nén: {out}")
sys.exit(1 if errors else 0)
