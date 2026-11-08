# BTL01 – Xấp xỉ hàm Gaussian 2 chiều bằng MLP (PyTorch) · Nhóm 5

Bài tập lớn 01, học phần Học sâu. Báo cáo: [report/BTL01_Nhom05.pdf](report/BTL01_Nhom05.pdf). Slide: [slides/BTL01_Nhom05.pptx](slides/BTL01_Nhom05.pptx).

## Cài môi trường

Dùng Anaconda Prompt (hoặc PowerShell đã `conda init`), đứng ở thư mục này:

```powershell
conda create -n btl01 python=3.11 -y
conda activate btl01
# Máy không có GPU NVIDIA: cài PyTorch bản CPU trước (nhẹ hơn nhiều)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
pip install -r requirements.txt
```

Kiểm tra:

```powershell
python -c "import torch, numpy, pandas, matplotlib, yaml; print(torch.__version__)"
```

## Cấu trúc thư mục

```text
configs/baseline.yaml   baseline chung (chốt 29/10)
data/                   train_pool.csv, val.csv, test.csv, stats.csv
src/                    paths.py, data.py, model.py, train.py, evaluate.py, plots.py
scripts/                plot_data.py (hình mục 2), run_baseline.py (chạy thử end-to-end),
                        run_final.py (train mô hình cuối + đánh giá test một lần),
                        plot_final_evaluation.py (hình y–ŷ và bề mặt của mô hình cuối),
                        build_report.py (ghép các mục báo cáo)
experiments/            run_A.py, run_B.py, run_C.py, common.py, export_C.py
results/A|B|C/          logs/*.csv, figures/*.png, summary.csv
results/final/          selected_model.pt (trọng số mô hình được chọn), final_metrics.json,
                        logs/, figures/
report/                 BTL01_Nhom05.docx, .pdf (báo cáo nộp), section_*.md (từng mục),
                        BTL01_Nhom05.md (bản ghép), figures/
slides/                 BTL01_Nhom05.pptx, .pdf
```

Mô hình được chọn (theo val MSE): 2 tầng ẩn × 16 nơ-ron, tanh, 2000 mẫu train, seed 1.
Trọng số nằm ở `results/final/selected_model.pt`. Nạp lại bằng `src.train.load_checkpoint`.

Mọi đường dẫn lấy từ `src/paths.py` (tương đối so với thư mục này), không ghi đường dẫn tuyệt đối.

## Chạy

Chạy từ thư mục này dưới dạng module để import được `src`, ví dụ:

```powershell
python -m src.data                      # sinh data/*.csv (seed 42)
python -m scripts.plot_data             # hình dữ liệu cho mục 2 -> report/figures/
python -m scripts.run_baseline          # chạy thử baseline end-to-end -> results/baseline/
python -m experiments.run_A             # 10 cỡ mẫu x 3 seed -> results/A/
python -m experiments.run_B             # 13 độ rộng x 3 seed -> results/B/
python -m experiments.run_C             # 4 độ sâu x 3 seed -> results/C/
python -m experiments.export_C          # kiểm 12 log C, xuất bảng mean ± std
python -m scripts.run_final             # mô hình cuối -> results/final/ (xem lưu ý dưới)
python -m scripts.plot_final_evaluation # hình y–ŷ và bề mặt -> results/final/figures/
python -m scripts.plot_mlp_architecture # sơ đồ kiến trúc -> report/figures/
python -m scripts.build_report          # ghép báo cáo -> report/BTL01_Nhom05.md
```

Các lệnh sinh dữ liệu, runner A/B/C và `plot_final_evaluation` ghi đè file trong `data/` và `results/`. Bài nộp đã có sẵn kết quả, nên muốn thử chạy lại thì chép thư mục này sang chỗ khác rồi chạy trên bản sao. Bản ghim trong `requirements.txt` là môi trường đã kiểm cho log trùng với log A (Python 3.11, torch 2.15.0 bản CPU).

`scripts/run_final.py` từ chối chạy nếu `results/final/final_metrics.json` đã có, để tập
test không bị đánh giá lần thứ hai. `build_report.py` in `PROBLEMS: 0` khi số bảng/hình
liên tục và mỗi bảng/hình đều được nhắc trong văn trước khi xuất hiện.

`scripts/run_baseline.py` train một lần cấu hình baseline (2 × 16, tanh, 1000 mẫu,
seed 0), lưu checkpoint, nạp lại và đánh giá trên train/val, vẽ hình y–ŷ và bề mặt
xấp xỉ. Script không dùng tập test.

Các runner A, B đọc cấu hình ở `configs/baseline.yaml`, dùng các seed `0, 1, 2`,
ghi log loss theo epoch cùng metric train/validation vào `results/A/logs/` và
`results/B/logs/`, bảng mean ± std vào `summary.csv`, và đồ thị vào `figures/`.
Runner C ghi thêm sai số test và thời gian train của từng lượt vào `results/C/summary.csv`,
vì đề yêu cầu báo test error theo số tầng. Không runner nào dùng test để chọn cấu hình.
Hai runner dùng chung `experiments/common.py` để thực hiện từng tổ hợp giá trị/seed
và tổng hợp metric bằng trung bình cùng độ lệch chuẩn mẫu.
Không có runner nào dùng tập test để chọn mô hình. Có thể chạy một nhóm nhỏ
để kiểm tra hoặc chạy lại một phần thí nghiệm bằng các tùy chọn bên dưới. Khi chỉ
chạy một phần, runner ghi bảng ra `summary_partial.csv` và giữ nguyên `summary.csv`
cùng các hình của bản chạy đầy đủ:

```powershell
python -m experiments.run_A --values 200 400 --seeds 0
python -m experiments.run_B --values 1 4 10 --seeds 0
python -m experiments.run_C --values 1 2 --seeds 0
```
