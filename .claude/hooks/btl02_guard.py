"""Hook bảo vệ BTL02 (chạy bằng Python chuẩn, không cần thư viện ngoài).

Dùng:
  python btl02_guard.py pre      # PreToolUse: chặn/hỏi trước lệnh hoặc lần sửa file nguy hiểm
  python btl02_guard.py session  # SessionStart: nhắc hạn nộp, ngày thiếu nhật ký, số run đã ghi

Hook không được làm hỏng phiên làm việc: mọi lỗi bất ngờ đều thoát 0 và im lặng.
"""
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

BTL = "BTL02 Emotion Recognition"
DEADLINE = date(2026, 11, 28)
BASELINE_LOCK = date(2026, 11, 25)  # baseline chốt hết ngày 24/11; từ 25/11 sửa thì hỏi lại
ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
BTL_DIR = ROOT / BTL
THU = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


def norm(p: str) -> str:
    return (p or "").replace("\\", "/")


def decide(kind: str, reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": kind,          # "deny" hoặc "ask"
        "permissionDecisionReason": reason,
    }}, ensure_ascii=False))
    sys.exit(0)


def check_command(cmd: str) -> None:
    c = norm(cmd)
    low = c.lower()
    if re.search(r"\bgit\s+push\b", low):
        decide("deny", "CLAUDE.md mục 2: Trúc tự push. Hãy đưa lệnh git push cho Trúc chạy.")
    if re.search(r"\bgit\s+commit\b", low) and "co-authored-by" in low:
        decide("deny", "CLAUDE.md mục 2: không thêm Co-Authored-By vào commit (repo nhóm, có chấm điểm).")
    if "run_final" in low and (BTL_DIR / "results/final/final_metrics.json").exists():
        decide("deny", "results/final/final_metrics.json đã có: mô hình cuối đã đánh giá PrivateTest một lần. "
                       "Chạy lại run_final là đánh giá test lần hai. Chỉ chạy khi Trúc tự gõ lệnh.")
    if re.search(r"\b(rm|rmdir|del|remove-item|rd)\b", low) and BTL.lower() in low and \
            re.search(r"/(results|data)\b", low):
        decide("ask", "Lệnh này xóa file trong results/ hoặc data/ của BTL02 (log, dữ liệu dùng cho báo cáo). Xác nhận?")


def written_text(tool: str, inp: dict) -> str:
    if tool == "Write":
        return inp.get("content", "")
    if tool == "Edit":
        return inp.get("new_string", "")
    if tool == "MultiEdit":
        return "\n".join(e.get("new_string", "") for e in inp.get("edits", []))
    if tool == "NotebookEdit":
        return inp.get("new_source", "")
    return ""


def check_file(tool: str, inp: dict) -> None:
    path = norm(inp.get("file_path") or inp.get("notebook_path") or "")
    if BTL not in path:
        return
    rel = path.split(BTL + "/", 1)[-1]
    text = written_text(tool, inp)
    if rel.endswith((".py", ".ipynb")):
        if re.search(r"^\s*(import|from)\s+(tensorflow|keras)\b", text, re.M) or "tf.keras" in text:
            decide("deny", "Đề yêu cầu PyTorch: không được dùng tensorflow/keras trong code BTL02.")
    if rel.endswith((".py", ".ipynb", ".yaml", ".yml")):
        if re.search(r"[A-Za-z]:[\\/]+(Users|Data)[\\/]", text):
            decide("deny", "Không ghi đường dẫn tuyệt đối vào code/config. Dùng src/paths.py (pathlib tương đối).")
    if rel.startswith("Yêu cầu/"):
        decide("deny", "Thư mục 'Yêu cầu/' là đề gốc của thầy, không sửa.")
    if rel.startswith("results/final/"):
        decide("ask", "Đang sửa results/final/ (kết quả mô hình cuối, đã đánh giá test). Xác nhận?")
    if rel == "configs/baseline.yaml" and date.today() >= BASELINE_LOCK:
        decide("ask", "baseline.yaml đã chốt ngày 24/11. Đổi baseline làm mọi run E1–E6 không còn so sánh được. Xác nhận?")


def pre() -> None:
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    inp = data.get("tool_input") or {}
    if tool in ("Bash", "PowerShell"):
        check_command(inp.get("command", ""))
    elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        check_file(tool, inp)
    sys.exit(0)


def session() -> None:
    today = date.today()
    lines = [f"[BTL02 Emotion Recognition] Hôm nay {THU[today.weekday()]} {today:%d/%m/%Y}, "
             f"còn {(DEADLINE - today).days} ngày đến hạn nộp {DEADLINE:%d/%m/%Y}. "
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
            lines.append("Ngày có commit BTL02 nhưng chưa có mục trong CodetoTruc.md: "
                         + ", ".join(f"{THU[d.weekday()]} {d:%d/%m}" for d in missing)
                         + ". Báo Trúc và bổ sung (skill nhat-ky-btl02).")
    except Exception:
        pass
    try:
        log = (BTL_DIR / "NhatKyThucNghiem.md").read_text(encoding="utf-8")
        # Mã run: BASE, E1-1 … E7-2, F-0/1/2, kể cả bản chạy lại có hậu tố _r2.
        runs = re.findall(r"^\|\s*((?:(?:E\d|F)-\d+|BASE)(?:_r\d+)?)\s*\|([^|]*)\|", log, re.M)
        done = [r for r, day in runs if day.strip()]
        lines.append(f"Nhật ký thực nghiệm: {len(done)} run đã ghi ({', '.join(done) or 'chưa có'}).")
    except Exception:
        pass
    if (BTL_DIR / "results/final/final_metrics.json").exists():
        lines.append("Mô hình cuối ĐÃ đánh giá PrivateTest. Không chạy lại run_final.")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        {"pre": pre, "session": session}[sys.argv[1]]()
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
