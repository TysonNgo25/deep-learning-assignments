"""Ghép các file report/section_*.md thành một báo cáo report/BTL01_Nhom05.md.

Các file mục của từng người giữ nguyên. Mọi chỉnh sửa chỉ áp lên bản ghép:
- bỏ dòng ghi chú "Nháp ngày ..., người viết: ...",
- đổi tiêu đề cho khớp bố cục 6 mục của đề,
- đánh số lại bảng/hình theo mục (Bảng 4.1, Hình 4.1, ...), thêm chú thích còn thiếu,
- đổi link giữa các file mục thành chữ ("mục 4.2"), link tới file kết quả thành đường dẫn.

Cuối cùng script kiểm tra: số bảng/hình liên tục và không trùng, mỗi bảng/hình được nhắc
trong văn trước khi xuất hiện, mọi ảnh tồn tại. Có lỗi thì in ra và trả mã thoát 1.

Chạy từ thư mục BTL01:  python -m scripts.build_report
Xuất Word (khi có pandoc):
  pandoc report/BTL01_Nhom05.md -f markdown+tex_math_single_backslash
         --resource-path=report -o report/BTL01_Nhom05.docx
"""

import re
import sys
from pathlib import Path

from src.paths import REPORT_DIR, ROOT

OUT_PATH = REPORT_DIR / "BTL01_Nhom05.md"

FRONT_MATTER = """---
title: "Xấp xỉ hàm mật độ Gaussian hai chiều bằng mạng MLP"
subtitle: "Bài tập lớn 01, học phần Học sâu · Nhóm 05"
author:
  - Huỳnh Ngọc Trúc (3123411312)
  - Phạm Trần Minh Khang (3123411143)
  - Tạ Quang Trung (3123411318)
  - Ngô Nguyễn Thành Nhân (3123411205)
date: "Bản ghép 1, 06/11/2026"
---
"""

# Tên mục dùng khi đổi link [..](section_x.md) thành chữ.
SECTION_NAMES = {
    "section_1_intro.md": "mục 1",
    "section_2_data.md": "mục 2",
    "section_3_model.md": "mục 3",
    "section_4_1_setup.md": "mục 4.1",
    "section_4_2_experiments_AB.md": "mục 4.2",
    "section_4_3_experiment_C.md": "mục 4.3",
    "section_5_discussion_AB.md": "mục 5.1",
    "section_5_discussion_C.md": "mục 5.2",
    "section_5_final_model_evaluation.md": "mục 5.3",
    "section_6_conclusion.md": "mục 6",
}

DRAFT_NOTE = r"^Nháp ngày [^\n]*?người viết: [^.\n]+(?:\([^)\n]*\))?\.[ ]?"

# Nhân viết số kiểu "1.000 mẫu", "27,4 lần"; các mục khác dùng dấu chấm thập phân.
VN_NUMBERS = [(r"\b(\d)\.000\b", r"\g<1>000"), (r"(\d),(\d)", r"\1.\2")]

# Mỗi phần: file, các cặp (regex, thay bằng) áp theo thứ tự, bảng đổi số bảng/hình.
PARTS = [
    {
        "file": "section_1_intro.md",
        "subs": [(r"^Nháp ngày [^\n]*\n\n", "")],
    },
    {
        "file": "section_2_data.md",
        "subs": [(DRAFT_NOTE, "")],
    },
    {
        "file": "section_3_model.md",
        "subs": [
            (DRAFT_NOTE, ""),
            (r"Kiến trúc baseline:\n", "Hình 3.1 là sơ đồ kiến trúc baseline.\n"),
            (
                r"(!\[[^\]]*\]\([^)]+\)\n)",
                r"\1\nHình 3.1. Sơ đồ MLP baseline: 2 đầu vào, hai tầng ẩn 16 nơ-ron "
                "dùng tanh, 1 đầu ra tuyến tính (337 tham số).\n",
            ),
            (
                r"Các giá trị baseline dùng chung cho A, B, C được chốt trong `configs/baseline.yaml`:\n\n",
                "Bảng 3.1 liệt kê các giá trị baseline dùng chung cho A, B, C, chốt trong "
                "`configs/baseline.yaml`.\n\nBảng 3.1. Cấu hình baseline.\n\n",
            ),
            (
                r"(## 3\.5\. Tổ chức mã nguồn\n\n)",
                r"\1Bảng 3.2 tóm tắt vai trò của các module chính.\n\nBảng 3.2. Các module mã nguồn.\n\n",
            ),
        ],
    },
    {
        "file": "section_4_1_setup.md",
        "subs": [(DRAFT_NOTE, "")],
    },
    {
        "file": "section_4_2_experiments_AB.md",
        "demote": 1,
        "subs": [
            (r"^Nháp ngày 03/11/2026, phần phụ trách của Tạ Quang Trung\. ", ""),
            (
                r"Các tham số khác giữ nguyên để so sánh tác động của biến đang khảo sát\.\n\n",
                "Các tham số khác giữ nguyên để so sánh tác động của biến đang khảo sát. "
                "Bảng 4.1 tóm tắt thiết lập.\n\nBảng 4.1. Thiết lập thực nghiệm A và B.\n\n",
            ),
            (
                r"Bảng sau ghi metric trên cả train và validation\.",
                "Bảng 4.2 ghi metric trên cả train và validation, Hình 4.1 vẽ metric validation.",
            ),
            (
                r"Bảng dưới trình bày metric train và validation\.",
                "Bảng 4.3 trình bày metric train và validation, Hình 4.2 vẽ cùng số liệu.",
            ),
        ],
        "renumber": {
            ("Bảng", "4.2.1"): "4.2",
            ("Bảng", "4.2.2"): "4.3",
            ("Hình", "4.2.1"): "4.1",
            ("Hình", "4.2.2"): "4.2",
        },
    },
    {
        "file": "section_4_3_experiment_C.md",
        "demote": 1,
        "subs": [
            (
                r"^Nháp ngày [^\n]*\n",
                "Bảng kết quả đầy đủ và liên kết tới 12 log nằm ở `results/C/table_C_mean_std.md`.\n",
            ),
            *VN_NUMBERS,
            (
                r"Các tham số còn lại giữ theo baseline đã chốt\.\n",
                "Các tham số còn lại giữ theo baseline đã chốt. Bảng 4.4 ghi thiết lập dùng chung.\n",
            ),
            (
                r"(### 4\.3\.2\. [^\n]*\n\n|## 4\.3\.2\. [^\n]*\n\n)",
                r"\1Bảng 4.5 ghi MSE trên ba tập, Hình 4.3 vẽ MSE test theo số tầng.\n\n",
            ),
            (
                r"\n(\*\*Bảng 4\.3\.3\.\*\*)",
                r"\nBảng 4.6 bổ sung MAE và RMSE trên test.\n\n\1",
            ),
            (
                r"(## 4\.3\.3\. [^\n]*\n\n)",
                r"\1Hình 4.4 vẽ train loss và validation loss theo epoch.\n\n",
            ),
            (
                r"(## 4\.3\.4\. [^\n]*\n\n)",
                r"\1Bảng 4.7 và Hình 4.5 ghi thời gian train và best epoch.\n\n",
            ),
            (r"## Nguồn số liệu để ghép báo cáo", "## Nguồn số liệu"),
        ],
        "renumber": {
            ("Bảng", "4.3.1"): "4.4",
            ("Bảng", "4.3.2"): "4.5",
            ("Bảng", "4.3.3"): "4.6",
            ("Bảng", "4.3.4"): "4.7",
            ("Hình", "4.3.1"): "4.3",
            ("Hình", "4.3.2"): "4.4",
            ("Hình", "4.3.3"): "4.5",
        },
    },
    {
        "file": "section_5_discussion_AB.md",
        "subs": [
            # Hạ cấp tiêu đề con trước, để không bắt nhầm tiêu đề "## 5.1." chèn ở dòng sau.
            (r"^## 5\.1\. ", "### 5.1.1. "),
            (r"^## 5\.2\. ", "### 5.1.2. "),
            (r"^## 5\.3\. ", "### 5.1.3. "),
            (r"^# Mục 5\. [^\n]*\n", "# 5. Thảo luận\n\n## 5.1. Thực nghiệm A và B\n"),
        ],
    },
    {
        "file": "section_5_discussion_C.md",
        "subs": [
            (r"^# Mục 5\. [^\n]*\n", "## 5.2. Thực nghiệm C\n"),
            (r"^## Ảnh hưởng", "### 5.2.1. Ảnh hưởng"),
            (r"^## Hội tụ", "### 5.2.2. Hội tụ"),
            (r"^## Giới hạn", "### 5.2.3. Giới hạn"),
            *VN_NUMBERS,
        ],
    },
    {
        "file": "section_5_final_model_evaluation.md",
        "subs": [
            (r"^# Đánh giá chung mô hình cuối\n", "## 5.3. Đánh giá chung mô hình cuối\n"),
            (
                r"^## Đồ thị giá trị thật và dự đoán\n\n",
                "### 5.3.1. Đồ thị giá trị thật và dự đoán\n\n"
                "Hình 5.1 đặt giá trị thật cạnh giá trị dự đoán trên 500 điểm validation.\n\n",
            ),
            (
                r"\n(\| Tập validation \(500 điểm\))",
                r"\nBảng 5.1 ghi sai số của checkpoint trên validation.\n\n"
                r"Bảng 5.1. Sai số của checkpoint cuối trên validation.\n\n\1",
            ),
            (
                r"^## So sánh bề mặt\n\n",
                "### 5.3.2. So sánh bề mặt\n\nHình 5.2 đặt hai bề mặt cạnh nhau trên cùng thang trục.\n\n",
            ),
            (r"tại 6400 tọa độ là:\n\n", "tại 6400 tọa độ ghi ở Bảng 5.2.\n\nBảng 5.2. Sai số trên lưới 80 × 80.\n\n"),
            (r"^## Kết luận và giới hạn", "### 5.3.3. Kết luận và giới hạn"),
            (r"^## Tái tạo hình", "### 5.3.4. Tái tạo hình"),
        ],
    },
    {
        "file": "section_6_conclusion.md",
        "subs": [(DRAFT_NOTE, "")],
    },
]

LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
LABEL = re.compile(r"(Bảng|Hình) (\d+(?:\.\d+)+)")
CAPTION = re.compile(r"^\**(Bảng|Hình) (\d+)\.(\d+)\.\**\s")
IMAGE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)")


def demote(text: str, levels: int) -> str:
    """Thêm `levels` dấu # vào mọi tiêu đề, bỏ qua dòng trong khối code."""
    out, in_code = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,5} ", line):
            line = "#" * levels + line
        out.append(line)
    return "\n".join(out)


def rewrite_link(match: re.Match) -> str:
    text, target = match.group(1), match.group(2)
    if target.startswith(("http://", "https://")):
        return match.group(0)
    path = target.split("#")[0]
    name = Path(path).name
    if name in SECTION_NAMES:
        return text if text.lower().startswith("mục") else f"{text} ({SECTION_NAMES[name]})"
    rel = (REPORT_DIR / path).resolve().relative_to(ROOT.resolve()).as_posix()
    return f"`{rel}`" if text in (name, f"`{name}`") else f"{text} (`{rel}`)"


def build_part(part: dict) -> str:
    text = (REPORT_DIR / part["file"]).read_text(encoding="utf-8").replace("\r\n", "\n")
    if part.get("demote"):
        text = demote(text, part["demote"])
    for pattern, repl in part.get("subs", []):
        text, n = re.subn(pattern, repl, text, flags=re.M)
        if n == 0 and (pattern, repl) not in VN_NUMBERS:
            raise ValueError(f"{part['file']}: không tìm thấy mẫu {pattern!r}, có thể file đã đổi")
    mapping = part.get("renumber", {})
    text = LABEL.sub(lambda m: f"{m.group(1)} {mapping.get((m.group(1), m.group(2)), m.group(2))}", text)
    text = LINK.sub(rewrite_link, text)
    return text.strip() + "\n"


def check(lines: list[str]) -> list[str]:
    """Kiểm tra đánh số và tham chiếu trên bản ghép, trả về danh sách lỗi."""
    problems = []
    captions = {}  # (loại, số) -> dòng chú thích
    expected = {}  # (loại, mục) -> số thứ tự kế tiếp
    for i, line in enumerate(lines):
        m = CAPTION.match(line)
        if not m:
            continue
        kind, sec, num = m.group(1), int(m.group(2)), int(m.group(3))
        label = f"{sec}.{num}"
        if (kind, label) in captions:
            problems.append(f"dòng {i + 1}: {kind} {label} bị trùng")
        captions[(kind, label)] = i
        want = expected.get((kind, sec), 1)
        if num != want:
            problems.append(f"dòng {i + 1}: {kind} {label} nhảy số, đáng lẽ là {kind} {sec}.{want}")
        expected[(kind, sec)] = num + 1

    for (kind, label), i in captions.items():
        anchor = i
        if kind == "Hình":  # chú thích hình nằm dưới hình: phải nhắc trước chính hình đó
            for j in range(i - 1, max(i - 25, -1), -1):
                if IMAGE.match(lines[j]) or lines[j].startswith("```"):
                    anchor = j
                    break
        mentions = [
            j for j, line in enumerate(lines)
            if j not in captions.values() and re.search(rf"{kind} {re.escape(label)}(?![\d.]\d)", line)
        ]
        if not mentions:
            problems.append(f"{kind} {label}: không được nhắc trong văn")
        elif mentions[0] > anchor:
            problems.append(f"{kind} {label}: lần nhắc đầu (dòng {mentions[0] + 1}) nằm sau hình/bảng (dòng {anchor + 1})")

    for i, line in enumerate(lines):
        for kind, label in LABEL.findall(line):
            if re.fullmatch(r"\d+\.\d+", label) and (kind, label) not in captions and i not in captions.values():
                problems.append(f"dòng {i + 1}: nhắc {kind} {label} nhưng không có hình/bảng này")
        m = IMAGE.match(line)
        if m and not (REPORT_DIR / m.group(1)).exists():
            problems.append(f"dòng {i + 1}: không thấy ảnh {m.group(1)}")
        if "section_" in line and ".md" in line:
            problems.append(f"dòng {i + 1}: còn link tới file mục")
        if line.startswith("Nháp ngày"):
            problems.append(f"dòng {i + 1}: còn dòng ghi chú nháp")
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # in tiếng Việt cả khi chuyển hướng output
    body = "\n".join(build_part(part) for part in PARTS)
    text = FRONT_MATTER + "\n" + body
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")

    lines = text.split("\n")
    problems = check(lines)
    tables = sum(1 for line in lines if CAPTION.match(line) and line.lstrip("*").startswith("Bảng"))
    figures = sum(1 for line in lines if CAPTION.match(line) and line.lstrip("*").startswith("Hình"))
    words = len(re.findall(r"\w+", body))
    print(f"Đã ghi {OUT_PATH.relative_to(ROOT).as_posix()}: {len(PARTS)} phần, {tables} bảng, {figures} hình, khoảng {words} từ")
    for p in problems:
        print("  LỖI:", p)
    print(f"PROBLEMS: {len(problems)}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
