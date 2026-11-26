"""Ghép report/section_*.md và appendix.md thành báo cáo Word (.docx) và PDF.

Phụ trách: Huỳnh Ngọc Trúc

Cần pandoc (`pip install pypandoc_binary` có kèm sẵn pandoc, hoặc pandoc trong PATH) và Microsoft Edge
hoặc Google Chrome để in PDF. Đứng ở thư mục BTL02:
    python report/ghep_bao_cao.py
Ghi: report/BaoCao_BTL02_Nhom05.docx, report/BaoCao_BTL02_Nhom05.pdf.
Nội dung chỉ sửa trong các file .md rồi chạy lại script; không sửa tay file .docx/.pdf.
"""
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

REPORT = Path(__file__).resolve().parent
PARTS = ["section_1_intro_data.md", "section_2_code.md", "section_3_experiments.md",
         "section_4_visualize.md", "section_5_final.md", "appendix.md"]
OUT = REPORT / "BaoCao_BTL02_Nhom05"

META = """---
title: "Nhận dạng cảm xúc khuôn mặt trên FER2013 bằng mạng nơ-ron tích chập"
subtitle: "Bài tập lớn 02, học phần Học sâu · Nhóm 05"
author:
  - Huỳnh Ngọc Trúc (3123411312), nhóm trưởng
  - Phạm Trần Minh Khang (3123411143)
  - Tạ Quang Trung (3123411318)
  - Ngô Nguyễn Thành Nhân (3123411205)
date: "Tháng 11 năm 2026"
lang: vi
---
"""
# Ngắt trang trong Word (khối OpenXML thô, pandoc chèn nguyên văn).
PAGE_BREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'
TNR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman" />'

CSS = """
@page { size: A4; margin: 2cm 2cm 2cm 3cm;
        @bottom-center { content: counter(page); font: 10pt "Times New Roman", serif; } }
body { font-family: "Times New Roman", serif; font-size: 12.5pt; line-height: 1.4; color: #000;
       max-width: none; margin: 0; padding: 0; hyphens: none; }
header#title-block-header { text-align: center; padding-top: 6cm; break-after: page; }
header .title { font-size: 22pt; margin-bottom: 0.4em; }
header .subtitle { font-size: 15pt; font-weight: normal; margin-bottom: 2.5em; }
header .author { font-size: 13pt; margin: 0.2em 0; }
header .date { font-size: 13pt; margin-top: 2.5em; }
nav#TOC { break-after: page; }
nav#TOC > ul { list-style: none; padding-left: 0; }
nav#TOC ul ul { list-style: none; padding-left: 1.5em; }
nav#TOC a { color: #000; text-decoration: none; }
h1 { font-size: 17pt; break-before: page; margin-top: 0; }
h2 { font-size: 14pt; margin-top: 1.2em; }
h3 { font-size: 12.5pt; }
h1, h2, h3 { break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0 1em; font-size: 10.5pt; }
th, td { border: 1px solid #555; padding: 3px 5px; vertical-align: top; }
th { background: #eee; }
tr, img { break-inside: avoid; }
img { max-width: 100%; display: block; margin: 0.4em auto; }
p:has(> img) { break-after: avoid; margin-bottom: 0.2em; }
code { font-family: Consolas, monospace; font-size: 0.88em; }
pre { font-size: 9.5pt; background: #f6f6f6; padding: 6px; white-space: pre-wrap; break-inside: avoid; }
"""


def pandoc_exe():
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except Exception:
        found = shutil.which("pandoc")
        if not found:
            raise SystemExit("Không tìm thấy pandoc: chạy `pip install pypandoc_binary` hoặc cài pandoc.")
        return found


def browser_exe():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(p).exists():
            return p
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("Không tìm thấy Edge hoặc Chrome để in PDF.")


def read_parts():
    texts = [(REPORT / name).read_text(encoding="utf-8").strip() + "\n" for name in PARTS]
    return texts


def static_toc(texts):
    """Mục lục tĩnh cho Word: không phụ thuộc vào việc trình xem cập nhật trường TOC."""
    lines = ["**MỤC LỤC**", ""]
    for t in texts:
        for m in re.finditer(r"^(#{1,2}) (.+)$", t, re.M):
            level, title = len(m.group(1)), m.group(2).strip()
            title = re.sub(r"^(\d+)\.", r"\1\\.", title)  # để "1." không thành danh sách đánh số
            lines.append(("" if level == 1 else "    ") + f"- {title}")
    return "\n".join(lines) + "\n"


def reference_docx(pandoc, workdir):
    """Mẫu kiểu chữ: Times New Roman 13 pt, chữ đen, A4, lề 3/2/2/2 cm, bảng kẻ viền."""
    ref = workdir / "reference.docx"
    subprocess.run([pandoc, "-o", str(ref), "--print-default-data-file", "reference.docx"], check=True)
    out = workdir / "reference_btl02.docx"
    with zipfile.ZipFile(ref) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                s = data.decode("utf-8")
                s = re.sub(r"<w:rFonts [^>]*/>", TNR, s)
                # Cỡ chữ mặc định 13 pt (đơn vị nửa point), chỉ sửa trong khối rPrDefault.
                s = re.sub(r"(<w:rPrDefault>.*?)<w:sz w:val=\"\d+\"\s*/>\s*<w:szCs w:val=\"\d+\"\s*/>",
                           r'\1<w:sz w:val="26" /><w:szCs w:val="26" />', s, count=1, flags=re.S)
                s = re.sub(r"<w:color [^>]*/>", '<w:color w:val="000000" />', s)
                borders = ("<w:tblBorders>" + "".join(
                    f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="555555" />'
                    for b in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>")
                s = re.sub(r'(<w:style w:type="table" w:default="1" w:styleId="Table">.*?<w:tblInd [^>]*/>)',
                           r"\1" + borders, s, count=1, flags=re.S)
                data = s.encode("utf-8")
            elif item.filename == "word/document.xml":
                # sectPr mẫu của pandoc không có khổ giấy, lề; pandoc lấy sectPr này cho file ra.
                # Theo thứ tự của schema, pgSz và pgMar đứng sau footnotePr.
                s = data.decode("utf-8")
                page = ('<w:pgSz w:w="11906" w:h="16838" />'
                        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1701" '
                        'w:header="567" w:footer="567" w:gutter="0" />')
                s = re.sub(r"<w:pgSz [^>]*/>|<w:pgMar [^>]*/>", "", s)
                s, n = re.subn(r"(<w:sectPr>\s*<w:footnotePr>.*?</w:footnotePr>)", r"\1" + page, s,
                               count=1, flags=re.S)
                if n == 0:
                    s = s.replace("<w:sectPr>", "<w:sectPr>" + page, 1)
                data = s.encode("utf-8")
            zout.writestr(item, data)
    return out


def build_docx(pandoc, texts, workdir):
    body = PAGE_BREAK.join(texts)
    md = META + "\n" + PAGE_BREAK + static_toc(texts) + PAGE_BREAK + body
    src = workdir / "bao_cao_docx.md"
    src.write_text(md, encoding="utf-8")
    subprocess.run([pandoc, str(src), "-f", "markdown-implicit_figures", "-o", str(OUT.with_suffix(".docx")),
                    "--reference-doc", str(reference_docx(pandoc, workdir)),
                    "--resource-path", str(REPORT)], check=True)


def build_pdf(pandoc, texts, workdir):
    src = workdir / "bao_cao_html.md"
    src.write_text(META + "\n" + "\n".join(texts), encoding="utf-8")
    css = workdir / "bao_cao.css"
    css.write_text(CSS, encoding="utf-8")
    html = workdir / "bao_cao.html"
    subprocess.run([pandoc, str(src), "-f", "markdown-implicit_figures", "-s", "--embed-resources",
                    "--toc", "--toc-depth=2", "--css", str(css), "--resource-path", str(REPORT),
                    "--metadata", "toc-title=Mục lục", "-o", str(html)], check=True)
    pdf = OUT.with_suffix(".pdf")
    subprocess.run([browser_exe(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    if not pdf.exists():
        raise SystemExit("Trình duyệt không tạo được file PDF.")


def main():
    pandoc = pandoc_exe()
    texts = read_parts()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_docx(pandoc, texts, tmp)
        build_pdf(pandoc, texts, tmp)
    for ext in (".docx", ".pdf"):
        f = OUT.with_suffix(ext)
        print(f"Đã ghi {f} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
