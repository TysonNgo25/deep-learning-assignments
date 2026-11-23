"""Kiểm tra checklist nộp BTL02 và (tùy chọn) nén file zip.

Chạy từ thư mục 'BTL02 Emotion Recognition':
    python ../.claude/skills/dong-goi-btl02/check_submission.py
    python ../.claude/skills/dong-goi-btl02/check_submission.py --zip BTLON02_Emotion_HuynhNgocTruc_3123411312.zip
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

BTL = Path.cwd()
if BTL.name != "BTL02 Emotion Recognition":
    sys.exit("Hãy chạy từ thư mục 'BTL02 Emotion Recognition'.")

REQUIRED = [
    "README.md", "requirements.txt", "configs/baseline.yaml", "NhatKyThucNghiem.md",
    "src/paths.py", "src/data.py", "src/models.py", "src/train.py", "src/evaluate.py", "src/visualize.py",
    "experiments/run_final.py",
    "results/final/best_model.pt", "results/final/final_metrics.json",
    "results/final/figures/last_conv_filters.png", "results/final/figures/confusion_matrix.png",
]
GROUPS = ["E1", "E2", "E3", "E4", "E5", "E6"]
EXCLUDE_DIRS = {"data", "Yêu cầu", "checkpoints", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_FILES = {"CodetoTruc.md", "Kế hoạch.md"}
MIN_RUNS = 10

errors, warnings = [], []


def files(pattern):
    return [p for p in BTL.rglob(pattern) if not (set(p.relative_to(BTL).parts) & EXCLUDE_DIRS)]


for rel in REQUIRED:
    if not (BTL / rel).exists():
        errors.append(f"Thiếu {rel}")
for g in GROUPS:
    if not (BTL / "results" / g / "summary.csv").exists():
        errors.append(f"Thiếu results/{g}/summary.csv")
for d, exts, what in [("report", (".docx",), "báo cáo .docx"), ("report", (".pdf",), "báo cáo .pdf"),
                      ("slides", (".pptx",), "slide .pptx")]:
    if not any(p.suffix in exts for p in (BTL / d).glob("*")):
        errors.append(f"Thiếu {what} trong {d}/")
if not any(p.suffix == ".pdf" for p in (BTL / "slides").glob("*")):
    warnings.append("Chưa có slide .pdf")

log = BTL / "NhatKyThucNghiem.md"
if log.exists():
    rows = re.findall(r"^\|\s*((?:E\d|F)-\d+\w*|BASE)\s*\|([^|]*)\|", log.read_text(encoding="utf-8"), re.M)
    done = [r for r, day in rows if day.strip()]
    if len(done) < MIN_RUNS:
        errors.append(f"Nhật ký thực nghiệm mới có {len(done)} run (cần ít nhất {MIN_RUNS})")

for p in files("*.py") + files("*.ipynb") + files("*.yaml"):
    text = p.read_text(encoding="utf-8", errors="ignore")
    rel = p.relative_to(BTL).as_posix()
    if re.search(r"^\s*(import|from)\s+(tensorflow|keras)\b", text, re.M) or "tf.keras" in text:
        errors.append(f"{rel}: còn tensorflow/keras")
    if re.search(r"[A-Za-z]:[\\/]+(Users|Data)[\\/]", text):
        errors.append(f"{rel}: có đường dẫn tuyệt đối")
    if p.suffix == ".py" and p.name != "__init__.py" and "Phụ trách:" not in text:
        warnings.append(f"{rel}: thiếu dòng 'Phụ trách:' (phân công trong code)")
    if p.suffix == ".py" and p.name not in ("run_final.py", "evaluate.py", "data.py", "paths.py"):
        for i, line in enumerate(text.splitlines(), 1):
            if "privatetest" in line.lower() or re.search(r"\btest_(loader|set|ds)\b", line):
                warnings.append(f"{rel}:{i}: nhắc tập test, đọc lại xem có rò rỉ không: {line.strip()[:80]}")

req = BTL / "requirements.txt"
if req.exists() and not all(("==" in l or ">=" in l) for l in req.read_text().split() if l.strip()):
    warnings.append("requirements.txt chưa ghim phiên bản hết các gói")

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
            rel = p.relative_to(BTL)
            if p.is_dir() or set(rel.parts) & EXCLUDE_DIRS or p.name in EXCLUDE_FILES \
                    or p.suffix == ".pyc" or p.name.startswith("~$"):
                continue
            if p.suffix == ".pt" and rel.as_posix() != "results/final/best_model.pt":
                continue
            z.write(p, Path(top) / rel)
            print("  +", rel.as_posix())
    print(f"\nĐã nén: {out}")
sys.exit(1 if errors else 0)
