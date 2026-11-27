"""Bìa báo cáo theo mẫu "TRƯỜNG ĐẠI HỌC SÀI GÒN.docx" ở gốc repo (Trúc thêm 27/11).

Phụ trách: Huỳnh Ngọc Trúc

Dùng chung cho BTL02 và BTL03 (mỗi bài có một bản giống hệt trong report/). ghep_bao_cao.py gọi:
- tao_bia_docx(...): chép file bìa, sửa nội dung (tên bài, Nhóm 05 + 4 thành viên, ngày) rồi ghép vào
  đầu báo cáo Word; giữ nguyên định dạng, logo, lề trang của file bìa.
- bia_html(...): bìa cùng nội dung và định dạng cho bản PDF (in từ HTML), logo lấy từ file bìa.
Nội dung bìa sửa ở COVER_* trong ghep_bao_cao.py, không sửa tay file .docx/.pdf.
"""
import copy
import html
import re
import zipfile
from pathlib import Path

BIA_GOC = Path(__file__).resolve().parents[2] / "TRƯỜNG ĐẠI HỌC SÀI GÒN.docx"
THANH_VIEN = [
    ("Huỳnh Ngọc Trúc (nhóm trưởng)", "3123411312"),
    ("Phạm Trần Minh Khang", "3123411143"),
    ("Tạ Quang Trung", "3123411318"),
    ("Ngô Nguyễn Thành Nhân", "3123411205"),
]
LOP = "DCT123C3"
NGAY = "TP. HCM, THÁNG 11/2026"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def _dat_chu(p, *dong):
    """Giữ pPr và định dạng run đầu tiên của đoạn p, thay nội dung bằng các dòng (ngắt dòng mềm)."""
    runs = [r for r in p if r.tag == W + "r"]
    rpr = copy.deepcopy(runs[0].find(W + "rPr")) if runs and runs[0].find(W + "rPr") is not None else None
    for r in runs:
        p.remove(r)
    for i, d in enumerate(dong):
        r = p.makeelement(W + "r", {})
        if rpr is not None:
            r.append(copy.deepcopy(rpr))
        if i:
            r.append(r.makeelement(W + "br", {}))
        t = r.makeelement(W + "t", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
        t.text = d
        r.append(t)
        p.append(r)


def tao_bia_docx(bao_cao_docx: Path, ten_bai: str, ten_de_tai: str, workdir: Path) -> None:
    """Ghép bìa (đã điền nội dung) vào đầu bao_cao_docx, ghi đè file đó."""
    from docx import Document
    from docxcompose.composer import Composer

    bia = Document(str(BIA_GOC))
    ps = bia.paragraphs
    tim = lambda s: next(p for p in ps if _text(p._p).strip().startswith(s))

    p_bc = tim("BÁO CÁO BÀI TẬP")
    _dat_chu(p_bc._p, ten_bai, ten_de_tai)
    _dat_chu(tim("Sinh viên thực hiện")._p, "Sinh viên thực hiện: Nhóm 05",
             *[f"{ten} – {msv}" for ten, msv in THANH_VIEN], f"Lớp: {LOP}")
    _dat_chu(tim("TP. HCM")._p, NGAY)

    # Bìa thêm 5 dòng (tên đề tài, 4 thành viên thay 2 dòng cũ): bỏ bớt đoạn trống phía trên dòng ngày
    # để dòng ngày vẫn nằm cuối trang 1.
    body = bia.element.body
    i_ngay = list(body).index(tim("TP. HCM")._p)
    trong = [e for e in list(body)[:i_ngay][::-1] if e.tag == W + "p" and not _text(e).strip()
             and e.find(".//" + W + "drawing") is None and e.find(".//" + W + "sectPr") is None]
    for e in trong[:6]:
        body.remove(e)

    tmp = workdir / "bia_da_dien.docx"
    bia.save(str(tmp))
    bc = Document(str(bao_cao_docx))
    comp = Composer(bc)                       # báo cáo làm gốc: giữ style, khổ giấy của báo cáo
    comp.insert(0, Document(str(tmp)))        # bìa lên đầu; đoạn cuối bìa có sectPr riêng (lề của bìa)
    comp.save(str(bao_cao_docx))


def luu_logo(workdir: Path) -> Path:
    """Chép logo trong file bìa ra workdir để pandoc nhúng vào HTML (pandoc không nhận ảnh dạng data URI)."""
    z = zipfile.ZipFile(BIA_GOC)
    ten = next(n for n in z.namelist() if n.startswith("word/media/") and not n.endswith("/"))
    out = workdir / ("logo_bia." + ten.rsplit(".", 1)[-1])
    out.write_bytes(z.read(ten))
    return out


def bia_html(ten_bai: str, ten_de_tai: str, logo: Path) -> str:
    """Bìa cho bản PDF: cùng nội dung, Times New Roman 13 pt đậm, căn giữa như file bìa."""
    e = html.escape
    tv = "".join(f"<div>{e(ten)} – {e(msv)}</div>" for ten, msv in THANH_VIEN)
    return f"""
<section class="bia">
  <div>TRƯỜNG ĐẠI HỌC SÀI GÒN</div>
  <div>KHOA CÔNG NGHỆ THÔNG TIN</div>
  <img src="{logo.as_posix()}" alt="Logo Trường Đại học Sài Gòn">
  <div class="hp">HỌC PHẦN: HỌC SÂU</div>
  <div>{e(ten_bai)}</div>
  <div class="de-tai">{e(ten_de_tai)}</div>
  <div class="gv">Giảng viên hướng dẫn: Đỗ Như Tài</div>
  <div class="sv"><div>Sinh viên thực hiện: Nhóm 05</div>{tv}<div>Lớp: {e(LOP)}</div></div>
  <div class="ngay">{e(NGAY)}</div>
</section>
"""


CSS_BIA = """
@page :first { margin: 1.15cm 1.5cm; @bottom-center { content: none; } }
section.bia { height: 27.2cm; display: flex; flex-direction: column; align-items: center;
              text-align: center; font: bold 13pt/1.45 "Times New Roman", serif; break-after: page; }
section.bia img { width: 3.41cm; height: 3.41cm; margin: 0.8cm 0 1.6cm; }
section.bia .hp { margin-bottom: 0.4cm; }
section.bia .de-tai { max-width: 15cm; }
section.bia .gv { margin-top: 2.2cm; }
section.bia .sv { margin-top: 0.8cm; }
section.bia .ngay { margin-top: auto; }
"""
