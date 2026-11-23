"""Hook bảo vệ BTL03 – Chuyển code từ Keras sang PyTorch (Python chuẩn, không cần thư viện ngoài).

Dùng:
  python btl03_guard.py pre      # PreToolUse: chặn/hỏi trước lần sửa file hoặc lệnh nguy hiểm trong BTL03
  python btl03_guard.py session  # SessionStart: nhắc hạn nộp, ngày thiếu nhật ký, số notebook đã chuyển

git push và Co-Authored-By đã do btl02_guard.py chặn cho cả repo, ở đây không lặp lại.
Hook không được làm hỏng phiên làm việc: mọi lỗi bất ngờ đều thoát 0 và im lặng.
"""
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path

BTL = "BTL03 - Chuyển code từ keras sang pytorch deep learning"
DEADLINE = date(2026, 11, 28)
ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
BTL_DIR = ROOT / BTL
THU = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

# Notebook gốc có Keras cần chuyển (thư mục con trong Yêu cầu/examples và pytorch/)
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

KERAS_RE = re.compile(r"^\s*(import|from)\s+(tensorflow|keras)\b", re.M)
ABS_RE = re.compile(r"[A-Za-z]:[\\/]+(Users|Data)[\\/]")


def norm(p: str) -> str:
    # NFC để "Yêu cầu", "Chuyển" so khớp được dù chuỗi đến ở dạng dấu tách rời (NFD)
    return unicodedata.normalize("NFC", (p or "").replace("\\", "/"))


def emit(text: str) -> None:
    """Ghi UTF-8 thẳng ra stdout: Python trên Windows có thể mặc định cp1252, in tiếng Việt sẽ lỗi."""
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.flush()


def decide(kind: str, reason: str) -> None:
    emit(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": kind,          # "deny" hoặc "ask"
        "permissionDecisionReason": reason,
    }}, ensure_ascii=False))
    sys.exit(0)


def code_cells(nb_text: str) -> str:
    """Ghép mã của các ô code trong notebook. Ô markdown được phép trích Keras gốc để đặt cạnh nhau."""
    try:
        nb = json.loads(nb_text)
    except Exception:
        return ""
    out = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            src = c.get("source", "")
            out.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(out)


def written_code(tool: str, inp: dict, rel: str) -> str:
    """Phần mã (không tính markdown) sắp được ghi."""
    if tool == "Write":
        text = inp.get("content", "")
        return code_cells(text) if rel.endswith(".ipynb") else text
    if tool == "Edit":
        return "" if rel.endswith(".ipynb") else inp.get("new_string", "")
    if tool == "MultiEdit":
        return "" if rel.endswith(".ipynb") else "\n".join(e.get("new_string", "") for e in inp.get("edits", []))
    if tool == "NotebookEdit":
        return "" if inp.get("cell_type") == "markdown" else inp.get("new_source", "")
    return ""


def check_file(tool: str, inp: dict) -> None:
    path = norm(inp.get("file_path") or inp.get("notebook_path") or "")
    if BTL not in path:
        return
    rel = path.split(BTL + "/", 1)[-1]
    if rel.startswith("Yêu cầu/"):
        decide("deny", "Thư mục 'Yêu cầu/' là đề gốc của thầy (notebook Keras + dữ liệu mẫu), không sửa.")
    if rel.endswith((".py", ".ipynb")):
        code = written_code(tool, inp, rel)
        if rel.startswith("pytorch/") and (KERAS_RE.search(code) or "tf.keras" in code):
            decide("deny", "Code trong pytorch/ phải là PyTorch: không import tensorflow/keras trong ô code. "
                           "Muốn đặt Keras gốc cạnh bản PyTorch thì để trong ô markdown.")
        if ABS_RE.search(code):
            decide("deny", "Không ghi đường dẫn tuyệt đối vào code. Lấy đường dẫn từ pytorch/common.py (pathlib tương đối).")
    if rel == "results/so_sanh.csv" and tool == "Write" and (BTL_DIR / rel).exists():
        decide("ask", "Đang ghi đè cả results/so_sanh.csv (bảng Keras ↔ PyTorch dùng cho báo cáo). "
                      "Thêm/sửa từng dòng thì an toàn hơn. Xác nhận ghi đè?")


def check_command(cmd: str) -> None:
    low = norm(cmd).lower()
    if BTL.lower() not in low and "btl03" not in low:
        return
    if re.search(r"\b(rm|rmdir|del|remove-item|rd)\b", low) and re.search(r"/(results|data)\b", low):
        decide("ask", "Lệnh này xóa file trong results/ hoặc data/ của BTL03 (kết quả so sánh, dữ liệu Gender ~586 MB). Xác nhận?")
    if re.search(r"\b(rm|rmdir|del|remove-item|rd|mv|move|move-item)\b", low) and "yêu cầu" in low:
        decide("deny", "Không xóa/di chuyển file trong 'Yêu cầu/' (đề gốc).")


def pre() -> None:
    data = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    tool = data.get("tool_name", "")
    inp = data.get("tool_input") or {}
    if tool in ("Bash", "PowerShell"):
        check_command(inp.get("command", ""))
    elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        check_file(tool, inp)
    sys.exit(0)


def session() -> None:
    today = date.today()
    if today > DEADLINE:
        return
    lines = [f"[BTL03 Keras → PyTorch] Hôm nay {THU[today.weekday()]} {today:%d/%m/%Y}, "
             f"còn {(DEADLINE - today).days} ngày đến hạn nộp {DEADLINE:%d/%m/%Y} (trùng BTL02). "
             f"Lịch: '{BTL}/docs/Kế hoạch.md' mục 7."]
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "--date=short", "--format=%ad", "--", BTL],
                             capture_output=True, text=True, timeout=10).stdout.split()
        commit_days = sorted({datetime.strptime(d, "%Y-%m-%d").date() for d in out})
        diary = BTL_DIR / "CodetoTruc.md"
        have = set()
        if diary.exists():
            for m in re.finditer(r"^## \S+ (\d{2}/\d{2}/\d{4})", diary.read_text(encoding="utf-8"), re.M):
                have.add(datetime.strptime(m.group(1), "%d/%m/%Y").date())
        missing = [d for d in commit_days if d not in have]
        if missing:
            lines.append("Ngày có commit BTL03 nhưng chưa có mục trong CodetoTruc.md: "
                         + ", ".join(f"{THU[d.weekday()]} {d:%d/%m}" for d in missing)
                         + ". Báo Trúc và bổ sung (skill nhat-ky-btl03).")
    except Exception:
        pass
    try:
        done = [t for t in TARGETS if (BTL_DIR / "pytorch" / f"{t}_pytorch.ipynb").exists()]
        todo = [f"{t.split('.')[0]}/{t.split('/')[-1]}" for t in TARGETS if t not in done]
        lines.append(f"Notebook đã chuyển: {len(done)}/{len(TARGETS)}."
                     + (f" Còn: {', '.join(todo)}." if todo else ""))
    except Exception:
        pass
    try:
        rows = (BTL_DIR / "results" / "so_sanh.csv").read_text(encoding="utf-8").strip().splitlines()[1:]
        lines.append(f"results/so_sanh.csv: {len(rows)} dòng kết quả.")
    except Exception:
        pass
    emit("\n".join(lines))


if __name__ == "__main__":
    try:
        {"pre": pre, "session": session}[sys.argv[1]]()
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
