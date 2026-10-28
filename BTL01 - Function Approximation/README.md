# BTL01 – Xấp xỉ hàm Gaussian 2 chiều bằng MLP (PyTorch) · Nhóm 5

Kế hoạch và phân vai: [Kế hoạch.md](Kế%20hoạch.md). Đề bài: [Đề và yêu cầu/](Đề%20và%20yêu%20cầu/).

## Cài môi trường

Dùng Anaconda Prompt (hoặc PowerShell đã `conda init`), đứng ở thư mục này:

```powershell
conda create -n btl01 python=3.11 -y
conda activate btl01
# Máy không có GPU NVIDIA: cài PyTorch bản CPU trước (nhẹ hơn nhiều)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python -m ipykernel install --user --name btl01 --display-name "Python (btl01)"
```

Kiểm tra:

```powershell
python -c "import torch, numpy, pandas, matplotlib, yaml; print(torch.__version__)"
```

Mở notebook thì chọn kernel **Python (btl01)**.

## Cấu trúc thư mục

```text
configs/baseline.yaml   baseline chung (chốt 29/10)
data/                   train_pool.csv, val.csv, test.csv
src/                    paths.py, data.py, model.py, train.py, evaluate.py, plots.py
scripts/                plot_data.py (hình dữ liệu cho mục 2)
experiments/            run_A.py, run_B.py, run_C.py
results/A|B|C/          logs/*.csv, figures/*.png, summary.csv
models/                 best_model.pt
report/                 BTL01_Nhom05.docx, .pdf
slides/                 BTL01_Nhom05.pptx, .pdf
```

Mọi đường dẫn lấy từ `src/paths.py` (tương đối so với thư mục này), không ghi đường dẫn tuyệt đối.

## Chạy

Chạy từ thư mục này dưới dạng module để import được `src`, ví dụ:

```powershell
python -m src.data
python -m experiments.run_A
```

Phần này cập nhật khi pipeline xong (mốc 31/10).
