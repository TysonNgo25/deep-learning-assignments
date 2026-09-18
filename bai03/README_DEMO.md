# Hướng dẫn chạy bài 03 trên Anaconda

## 1. Nội dung bài làm

- `my_lesson3.html`, `my_lesson3.ipynb`: đề bài gốc, được giữ nguyên.
- `bai03_solution.ipynb`: notebook lời giải, chia theo đúng thứ tự Bài 1 → Bài 2 → Bài 3.
- `environment.yml`: cấu hình môi trường Conda.
- `outputs/`: các biểu đồ do notebook tạo ra.

Bài này chỉ sử dụng Numpy và Matplotlib; không cần GPU, CUDA, TensorFlow, Keras hoặc PyTorch.

## 2. Tạo môi trường bằng Anaconda Prompt

Mở **Anaconda Prompt**, chuyển đến thư mục `bai03`, sau đó chạy:

```powershell
cd "DUONG_DAN_DEN_PROJECT\python-deep-learning\bai03"
conda env create -f environment.yml
conda activate bai03-python
python -m ipykernel install --user --name bai03-python --display-name "Python (bai03-python)"
```

Nếu môi trường đã tồn tại và cần đồng bộ lại thư viện:

```powershell
conda env update -f environment.yml --prune
```

Phương án tạo thủ công tương đương:

```powershell
conda create -n bai03-python python=3.11 -y
conda activate bai03-python
conda install -c conda-forge numpy=1.26 matplotlib=3.8 notebook=7 jupyterlab=4 nbconvert=7 ipykernel=6 -y
```

## 3. Mở và chạy notebook

### Từ Anaconda Prompt

```powershell
conda activate bai03-python
cd "DUONG_DAN_DEN_PROJECT\python-deep-learning\bai03"
jupyter notebook bai03_solution.ipynb
```

Trong Jupyter, chọn kernel **Python (bai03-python)** rồi chọn **Kernel → Restart Kernel and Run All Cells**. Chạy cell theo thứ tự từ trên xuống; các cell sau sử dụng biến được tạo ở cell trước.

### Từ Anaconda Navigator

1. Mở Anaconda Navigator.
2. Chọn môi trường `bai03-python` trong mục **Environments**.
3. Quay lại **Home**, chọn đúng môi trường rồi mở **Jupyter Notebook** hoặc **JupyterLab**.
4. Mở thư mục dự án, vào `bai03`, mở `bai03_solution.ipynb`.
5. Chọn **Run All Cells**.

### Chạy kiểm thử không cần giao diện

```powershell
conda activate bai03-python
cd "DUONG_DAN_DEN_PROJECT\python-deep-learning\bai03"
jupyter nbconvert --to notebook --execute bai03_solution.ipynb --output bai03_solution_executed.ipynb --ExecutePreprocessor.timeout=300
```

## 4. Dữ liệu

Không cần tải dataset. Tất cả dãy số, ma trận và điểm Monte Carlo được sinh trong notebook. Biến `SEED = 42` giúp kết quả có thể tái lập. Có thể đổi seed hoặc tạo lại `rng` để quan sát kết quả ngẫu nhiên khác.

## 5. Checklist đối chiếu đề bài

- [ ] Bài 1.1 tạo được mảng `a` gồm 10 số nguyên.
- [ ] Bài 1.2 in bình phương, độ dài, min, max, trung bình, phương sai mẫu và độ lệch chuẩn.
- [ ] Bài 1.3 in số chẵn/lẻ, khoảng cách tại chỉ số lẻ và khoảng cách nhỏ nhất giữa hai tập.
- [ ] Bài 1.4 tạo đủ ba biểu đồ phân bố trong `outputs/`.
- [ ] Bài 2.1 tạo ma trận `6×4`.
- [ ] Bài 2.2 hoàn thành chuyển vị, truy xuất, đảo dòng, tổng dòng và trung bình cột.
- [ ] Bài 2.3 hoàn thành phép toán từng phần tử và phép nhân ma trận `4×3 @ 3×4`.
- [ ] Bài 2.4 nghiệm hệ có `||Ax-y||₂` gần bằng 0.
- [ ] Bài 3 báo cáo xấp xỉ π cho `N=100`, `10.000`, `1.000.000` và cả sai số có dấu/lỗi tuyệt đối.
- [ ] Có biểu đồ điểm Monte Carlo và biểu đồ sai số trong `outputs/`.
- [ ] Cell kiểm tra cuối cùng in `TẤT CẢ KIỂM TRA ĐỀU ĐẠT`.

## 6. Các file kết quả mong đợi

Sau khi chạy notebook, thư mục `outputs/` gồm:

```text
outputs/
├── bai1_4a_tan_so_phan_bo_deu.png
├── bai1_4b_ham_mat_do_chuan.png
├── bai1_4c_histogram_phan_bo_chuan.png
├── bai3_monte_carlo_points.png
└── bai3_sai_so_pi.png
```

Do dữ liệu ngẫu nhiên được cố định seed, số liệu sẽ giống nhau giữa các lần chạy trong cùng phiên bản Numpy. Chênh lệch rất nhỏ giữa phiên bản thư viện không ảnh hưởng đến kết luận.

