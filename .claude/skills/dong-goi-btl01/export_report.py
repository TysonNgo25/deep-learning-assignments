"""Xuất report/BTL01_Nhom05.md thành report/BTL01_Nhom05.docx và docs/BTL01_Nhom05_BaoCao.pdf.

Chạy từ thư mục "BTL01 - Function Approximation" (sau `python -m scripts.build_report`):
    python ../.claude/skills/dong-goi-btl01/export_report.py --pandoc <đường dẫn pandoc.exe>

Cần: pandoc, python-docx (thêm vào PYTHONPATH nếu không cài trong môi trường btl01),
và Word hoặc WPS Office trên Windows (dùng qua COM để cập nhật mục lục, xuất PDF).
Các công cụ này chỉ dùng để xuất báo cáo nên không có trong requirements.txt.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor

FONT = "Times New Roman"
BODY_PT = 12
LINE_SPACING = 1.0
TEXT_WIDTH_CM = 16.5
MAX_IMG_W_CM = 14.0
MAX_IMG_H_CM = 6.6
CM_PER_CHAR = {8: 0.142, 8.5: 0.151, 9.5: 0.169, 10: 0.178}  # độ rộng trung bình một ký tự, kể cả chữ số
CAPTION = re.compile(r"^(Bảng|Hình) \d+\.\d+\.")

PS_SCRIPT = r"""
param([string]$Src, [string]$Docx, [string]$Pdf)
$ErrorActionPreference = 'Stop'
# Khong goi Quit va khong dat Visible: WPS co the dung chung tien trinh voi cua so nguoi dung dang mo.
$w = New-Object -ComObject Word.Application
$d = $w.Documents.Open($Src)
foreach ($t in $d.TablesOfContents) { $t.Update() }
$d.Repaginate()
foreach ($t in $d.TablesOfContents) { $t.UpdatePageNumbers() }
$d.SaveAs2($Docx, 16)
$d.ExportAsFixedFormat($Pdf, 17)
try { $d.Close(0) } catch { Write-Host "Bo qua loi dong tai lieu: $_" }
"""


def set_fonts(style, name=FONT, size=None, bold=None, italic=None, color=None):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = RGBColor(*color)


def style_if(doc, name):
    try:
        return doc.styles[name]
    except KeyError:
        return None


def setup_styles(doc):
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text"):
        st = style_if(doc, name)
        if st is None:
            continue
        set_fonts(st, size=BODY_PT)
        pf = st.paragraph_format
        pf.line_spacing = LINE_SPACING
        pf.space_before = Pt(0)
        pf.space_after = Pt(0 if name == "Compact" else 5)
        if name != "Compact":
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size, before, after in (("Heading 1", 15, 12, 5), ("Heading 2", 13, 8, 3), ("Heading 3", 12, 6, 2)):
        st = style_if(doc, name)
        if st is not None:
            set_fonts(st, size=size, bold=True, italic=False, color=(0x1F, 0x2A, 0x44))
            st.paragraph_format.space_before = Pt(before)
            st.paragraph_format.space_after = Pt(after)
            st.paragraph_format.keep_with_next = True
    for name, size, bold in (("Title", 22, True), ("Subtitle", 14, False), ("Author", 13, False), ("Date", 13, False)):
        st = style_if(doc, name)
        if st is not None:
            set_fonts(st, size=size, bold=bold, italic=False, color=(0x1F, 0x2A, 0x44) if bold else (0, 0, 0))
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for name in ("Verbatim Char", "Source Code"):
        st = style_if(doc, name)
        if st is not None:
            set_fonts(st, name="Consolas", size=10)
    toc = style_if(doc, "TOC Heading")
    if toc is not None:
        set_fonts(toc, size=15, bold=True, color=(0x1F, 0x2A, 0x44))


def setup_page(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin, sec.right_margin = Cm(2.5), Cm(2.0)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
    sec.different_first_page_header_footer = True  # bìa không có số trang
    para = sec.footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.text = text
        run._r.append(el)


def page_break_before(par):
    par.paragraph_format.page_break_before = True


def format_cover(doc, cover_lines):
    """Pandoc xếp Title, Subtitle, Author, Date ở đầu. Đặt thêm khoảng trống và ngắt trang sau bìa."""
    for p in doc.paragraphs:
        if p.style.name == "Title":
            p.paragraph_format.space_before = Pt(170)
            p.paragraph_format.space_after = Pt(14)
        elif p.style.name == "Subtitle":
            p.paragraph_format.space_after = Pt(40)
        elif p.style.name == "Author":
            p.paragraph_format.space_after = Pt(4)
        elif p.style.name == "Date":
            p.paragraph_format.space_before = Pt(50)
            br = OxmlElement("w:br")
            br.set(qn("w:type"), "page")
            p.add_run()._r.append(br)
            break


def format_toc(doc):
    for name, indent in (("TOC 1", 0), ("TOC 2", 0.6)):
        st = style_if(doc, name)
        if st is None:
            st = doc.styles.add_style(name, 1)
        set_fonts(st, size=11.5)
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.line_spacing = 1.0
        st.paragraph_format.left_indent = Cm(indent)
    for par in doc.paragraphs:
        if par.style.name == "Heading 1":
            par.paragraph_format.page_break_before = True
            break


def set_cell_borders(table):
    tblpr = table._tbl.tblPr
    old = tblpr.find(qn("w:tblBorders"))
    if old is not None:
        tblpr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    tblpr.append(borders)


def cell_text(cell):
    """Lấy cả chữ trong công thức (m:t), vì cell.text của python-docx bỏ qua công thức."""
    parts = [t.text or "" for t in cell._tc.iter(qn("w:t"), qn("m:t"))]
    return "".join(parts).strip()


def format_table(table):
    n = len(table.columns)
    longest, total = [1] * n, [1] * n
    for row in table.rows:
        for i, cell in enumerate(row.cells[:n]):
            text = cell_text(cell)
            total[i] = max(total[i], len(text))
            is_code = any(r.style is not None and r.style.name == "Verbatim Char" for par in cell.paragraphs for r in par.runs)
            factor = 1.3 if is_code else 1.15 if row is table.rows[0] else 1.0
            for word in text.split():
                longest[i] = max(longest[i], len(word) * factor)
    # Chọn cỡ chữ lớn nhất mà mọi từ dài nhất vẫn vừa ô, không gãy giữa từ.
    total_w = TEXT_WIDTH_CM
    pad = 0.45
    size = 8
    for candidate in (10, 9.5, 8.5, 8):
        if n >= 7 and candidate > 8:
            continue
        if sum(c * CM_PER_CHAR[candidate] + pad for c in longest) <= total_w:
            size = candidate
            break
    mins = [max(c * CM_PER_CHAR[size] + pad, 1.5) for c in longest]
    spare = max(total_w - sum(mins), 0)
    extra = [min(max(total[i] - longest[i], 0), 45) for i in range(n)]  # chặn trên để cột dài không lấy hết chỗ
    widths = [m + spare * (e / sum(extra) if sum(extra) else 1 / n) for m, e in zip(mins, extra)]
    scale = total_w / sum(widths)
    widths = [w * scale for w in widths]

    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_cell_borders(table)
    tblpr = table._tbl.tblPr
    tblw = tblpr.find(qn("w:tblW"))
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tblpr.append(tblw)
    tblw.set(qn("w:type"), "dxa")
    tblw.set(qn("w:w"), str(int(total_w * 567)))
    grid = table._tbl.tblGrid
    for col, w in zip(grid.findall(qn("w:gridCol")), widths):
        col.set(qn("w:w"), str(int(w * 567)))

    keep_together = len(table.rows) <= 16
    for r, row in enumerate(table.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if r == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for i, cell in enumerate(row.cells[:n]):
            cell.width = Cm(widths[i])
            if r == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "E7EAF0")
                cell._tc.get_or_add_tcPr().append(shd)
            for par in cell.paragraphs:
                pf = par.paragraph_format
                pf.space_before, pf.space_after = Pt(1), Pt(1)
                pf.line_spacing = 1.0
                pf.keep_with_next = keep_together and r < len(table.rows) - 1
                if pf.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY or pf.alignment is None:
                    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in par.runs:
                    run.font.size = Pt(size if not run.style or run.style.name != "Verbatim Char" else size - 0.5)
                    if r == 0:
                        run.font.bold = True


def format_images(doc):
    for shape_par in doc.paragraphs:
        drawings = shape_par._p.findall(f".//{qn('wp:inline')}")
        if not drawings:
            continue
        shape_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        shape_par.paragraph_format.keep_with_next = True
        shape_par.paragraph_format.space_after = Pt(3)
        for inline in drawings:
            extent = inline.find(qn("wp:extent"))
            cx, cy = int(extent.get("cx")), int(extent.get("cy"))
            scale = min(Cm(MAX_IMG_W_CM) / cx, Cm(MAX_IMG_H_CM) / cy)
            new_cx, new_cy = int(cx * scale), int(cy * scale)
            extent.set("cx", str(new_cx))
            extent.set("cy", str(new_cy))
            for ext in inline.iter(qn("a:ext")):
                if ext.get("cx"):
                    ext.set("cx", str(new_cx))
                    ext.set("cy", str(new_cy))


def format_captions(doc):
    paragraphs = doc.paragraphs
    for idx, par in enumerate(paragraphs):
        text = par.text.strip()
        match = CAPTION.match(text)
        if not match or par._p.getparent().tag != qn("w:body"):
            continue
        is_table = text.startswith("Bảng")
        # Chú thích thật ngay sát bảng (trên) hoặc ảnh (dưới), câu văn nhắc tới bảng thì không.
        nxt = par._p.getnext()
        prv = par._p.getprevious()
        if is_table and (nxt is None or nxt.tag != qn("w:tbl")):
            continue
        if not is_table and not (prv is not None and prv.findall(f".//{qn('wp:inline')}")):
            continue
        par.paragraph_format.keep_with_next = is_table
        par.paragraph_format.space_before = Pt(6 if is_table else 2)
        par.paragraph_format.space_after = Pt(3 if is_table else 10)
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(text) < 90 else WD_ALIGN_PARAGRAPH.JUSTIFY
        label = match.group(0)
        first = par.runs[0] if par.runs else None
        if first is not None and first.text.startswith(label) and not first.bold:
            rest = first.text[len(label):]
            first.text = label
            first.bold = True
            if rest:
                new = par.add_run(rest)
                first._r.addnext(new._r)
        for run in par.runs:
            run.font.size = Pt(11)


def relax_code_paragraphs(doc):
    """Dòng có nhiều đường dẫn không ngắt được: canh trái để Word khỏi giãn khoảng trắng."""
    for par in doc.paragraphs:
        code = [r for r in par.runs if r.style is not None and r.style.name == "Verbatim Char"]
        if len(code) >= 2 or any(len(r.text) > 30 for r in code):
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT


def add_table_spacing(doc):
    """Pandoc đặt đoạn văn sát dưới bảng; thêm khoảng cách trước đoạn kế tiếp."""
    body = doc.element.body
    for tbl in body.findall(qn("w:tbl")):
        nxt = tbl.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            ppr = nxt.get_or_add_pPr()
            spacing = ppr.find(qn("w:spacing"))
            if spacing is None:
                spacing = OxmlElement("w:spacing")
                ppr.append(spacing)
            spacing.set(qn("w:before"), "160")


def run_pandoc(pandoc, md, out):
    subprocess.run(
        [
            pandoc, str(md.name),
            "-f", "markdown-implicit_figures+tex_math_single_backslash",
            "--resource-path=.",
            "--toc", "--toc-depth=2",
            "-M", "toc-title=Mục lục",
            "-M", "lang=vi-VN",
            "-o", str(out),
        ],
        check=True,
        cwd=md.parent,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pandoc", default="pandoc")
    ap.add_argument("--no-office", action="store_true", help="Chỉ tạo docx trung gian, không gọi Word/WPS")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    md = root / "report" / "BTL01_Nhom05.md"
    docx_out = root / "report" / "BTL01_Nhom05.docx"
    pdf_out = root / "docs" / "BTL01_Nhom05_BaoCao.pdf"
    pdf_out.parent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw.docx"
        run_pandoc(args.pandoc, md, raw)

        doc = Document(str(raw))
        setup_styles(doc)
        setup_page(doc)
        format_cover(doc, None)
        format_toc(doc)
        for table in doc.tables:
            format_table(table)
        format_images(doc)
        format_captions(doc)
        add_table_spacing(doc)
        relax_code_paragraphs(doc)
        doc.core_properties.title = "Xấp xỉ hàm mật độ Gaussian hai chiều bằng mạng MLP"
        doc.core_properties.author = "Nhóm 05"
        styled = tmp / "styled.docx"
        doc.save(str(styled))

        if args.no_office:
            styled.replace(docx_out)
            print(f"Đã ghi {docx_out}")
            return
        ps1 = tmp / "export.ps1"
        ps1.write_text(PS_SCRIPT, encoding="utf-8")
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
             "-Src", str(styled), "-Docx", str(docx_out), "-Pdf", str(pdf_out)],
            check=True,
            timeout=300,
        )
    print(f"Đã ghi {docx_out.name} ({docx_out.stat().st_size // 1024} KB), {pdf_out.name} ({pdf_out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
