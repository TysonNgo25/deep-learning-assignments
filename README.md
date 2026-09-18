# Deep Learning Assignments

## Bài 03 — Kỹ thuật lập trình Python với Numpy

Repository chứa đề bài gốc, notebook lời giải và hướng dẫn chạy trên Anaconda cho bài thực hành số 03.

## Phân loại file

### Đề bài gốc

- [`bai03/my_lesson3.html`](bai03/my_lesson3.html): nội dung đề bài được xuất dưới dạng HTML.
- [`bai03/my_lesson3.ipynb`](bai03/my_lesson3.ipynb): notebook đề bài gốc với các vị trí cần điền lời giải.
- `bai03/4a.png`, `bai03/4b.png`, `bai03/pi.png`: hình kết quả/minh họa đi kèm notebook gốc.

### Đáp án

- [`bai03/bai03_solution.ipynb`](bai03/bai03_solution.ipynb): lời giải hoàn chỉnh theo thứ tự Bài 1 → Bài 2 → Bài 3. Notebook đã được chạy và lưu output.
- [`bai03/outputs/`](bai03/outputs/): các biểu đồ được sinh từ notebook lời giải.

### Cài đặt và hướng dẫn chạy

- [`bai03/environment.yml`](bai03/environment.yml): cấu hình môi trường Conda.
- [`bai03/README_DEMO.md`](bai03/README_DEMO.md): hướng dẫn tạo môi trường, chạy notebook và kiểm tra kết quả.

## Chạy nhanh

Mở Anaconda Prompt và thực hiện:

```powershell
cd /d "C:\Users\TGC\OneDrive\Documents\ChatGPT\python-deep-learning\bai03"
conda env create -f environment.yml
conda activate bai03-python
jupyter notebook bai03_solution.ipynb
```

Trong Jupyter, chọn **Kernel → Restart Kernel and Run All Cells**. Bài chạy đúng khi cell cuối in:

```text
TẤT CẢ KIỂM TRA ĐỀU ĐẠT
```

## Nội dung chính

- Phép toán vector và ma trận với Numpy.
- Broadcasting và khoảng cách giữa hai tập giá trị.
- Phân bố đều, phân bố chuẩn và trực quan hóa bằng Matplotlib.
- Giải hệ phương trình tuyến tính bằng `numpy.linalg.solve`.
- Xấp xỉ số π bằng phương pháp Monte Carlo.

## Cấu trúc nộp bài

```text
python-deep-learning/
├── README.md
├── .gitignore
└── bai03/
    ├── my_lesson3.html          # Đề bài dạng HTML
    ├── my_lesson3.ipynb         # Notebook đề bài gốc
    ├── 4a.png                   # Hình của đề gốc
    ├── 4b.png
    ├── pi.png
    ├── bai03_solution.ipynb     # Đáp án chính
    ├── environment.yml          # Môi trường Conda
    ├── README_DEMO.md           # Hướng dẫn demo
    └── outputs/                 # Biểu đồ do đáp án sinh ra
```

Các tài liệu `python_advance.*`, `img1.png`, `img2.png`, `plot2d.png` và `scater2d.png` là tài liệu Python mở rộng. Chúng không phải đáp án chính nhưng vẫn được lưu trong `bai03` để repository chứa đầy đủ thư mục bài học gốc.
