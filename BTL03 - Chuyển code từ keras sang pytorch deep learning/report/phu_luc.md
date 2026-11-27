# Phụ lục

Người viết: Huỳnh Ngọc Trúc

## A. Bảng phân công và tỷ lệ đóng góp

Nhóm 5. Nhóm trưởng: Huỳnh Ngọc Trúc. Mỗi notebook có dòng `Phụ trách:` ở ô đầu, `common.py` có dòng này ở đầu file; cả hai khớp với Bảng A.1. Mỗi mục báo cáo ghi người viết ở dòng đầu; mục 3 ghi thêm ở đầu từng tiểu mục 3.1–3.10.

Bảng A.1. Phân công thành viên và tỷ lệ đóng góp.

| Thành viên | STT | MSV | Notebook và mã nguồn | Phần báo cáo | Tỷ lệ |
| --- | ---: | --- | --- | --- | ---: |
| Huỳnh Ngọc Trúc (nhóm trưởng) | 43 | 3123411312 | `common.py`; `getting_started`, `Func_apxm_MLP`, `MNIST_MLP`; `chay_kaggle.ipynb` (chạy GPU trên Kaggle); `results/so_sanh.csv`; đóng gói | Mục 1, 2, 3.1, 3.2, 4, 6, phụ lục | 25% |
| Phạm Trần Minh Khang | 19 | 3123411143 | `1_MNIST_CNN` (kèm PCA đặc trưng), `MNIST_CNN_LSTM` | Mục 3.3, 3.6 | 25% |
| Tạ Quang Trung | 44 | 3123411318 | `2_Human_Gender_CNN`, `4. Training Utilities/2_Human_Gender_CNN` (callback), `2_obj_detection` | Mục 3.4, 3.7, 3.10 | 25% |
| Ngô Nguyễn Thành Nhân | 29 | 3123411205 | `3_Emotion_Reg_CNN`, `4. Training Utilities/3_Emotion_Reg_CNN` (augmentation), `1_segnet` | Mục 3.5, 3.8, 3.9, 5 | 25% |
| **Tổng** | | | | | **100%** |

## B. Cách chạy lại

**Môi trường.** Python 3.11, cài gói trong `requirements.txt` (PyTorch, torchvision, torchinfo, tensorboard, numpy, scipy, scikit-learn, scikit-image, matplotlib, seaborn, pandas, notebook):

```bat
conda activate btl02
cd /d "<thư mục BTL03>"
pip install -r requirements.txt
```

**Dữ liệu** (không nộp kèm vì nặng):

| Bộ | Đặt ở | Ghi chú |
| --- | --- | --- |
| MNIST | `data/mnist/` | `torchvision` tự tải |
| Gender (WIKI) | `data/gender/` | 4 file `64_64_*_onehot.npy` của đề |
| FER2013 | `data/emotion/fer2013.csv` | `load_fer2013` lọc theo cột Usage |
| Ảnh bàn tay, ảnh xe, `test.jpg` | `Yêu cầu/examples/5. Others/` | có sẵn trong đề |

**Chạy một notebook** (lưu output vào chính file):

```bat
cd "pytorch\1. MLP"
python -m nbconvert --to notebook --execute --inplace MNIST_MLP_pytorch.ipynb
```

Mỗi notebook có huấn luyện có biến `CHAY_THU` ở ô đầu: `True` chạy thử với ít dữ liệu để kiểm code, `False` (mặc định) chạy đầy đủ. Bảy notebook CNN, LSTM, SegNet nên chạy trên GPU; `chay_kaggle.ipynb` chạy cả 7 notebook trên Kaggle (GPU T4 x2) và nén kết quả thành một file zip.

## C. Danh sách file nộp

| File | Nội dung |
| --- | --- |
| `pytorch/common.py` | hàm dùng chung (mục 2.2) |
| `pytorch/<thư mục>/<tên>_pytorch.ipynb` | 11 notebook PyTorch đã có output |
| `chay_kaggle.ipynb` | notebook chạy 7 notebook GPU trên Kaggle |
| `results/so_sanh.csv` | bảng so sánh Keras và PyTorch (Bảng 4.1) |
| `results/<notebook>/` | `history.csv`, `metrics.json`, hình của từng notebook |
| `report/` | báo cáo `BaoCao_BTL03_Nhom05.docx` và `.pdf` (file nguồn `.md`, script vẽ hình và ghép báo cáo để ở kho mã, không kèm trong file nộp) |
| `README.md`, `requirements.txt` | hướng dẫn cài đặt và chạy |

## D. Tài liệu tham khảo

1. Slide `01_Introduction-to-Python-TensorFlow-and-Keras.pdf` và `02_KerasExamples.pdf` của học phần Học sâu (thư mục `Yêu cầu/`).
2. PyTorch Documentation, <https://pytorch.org/docs/stable/>: `torch.nn`, `torch.optim`, `torch.utils.data`, `torch.nn.init`.
3. Keras 2.0.9 Documentation và mã nguồn (`keras/callbacks.py`, `keras/preprocessing/image.py`, `keras/layers/recurrent.py`), <https://github.com/keras-team/keras/tree/2.0.9>.
4. torchinfo, <https://github.com/TylerYep/torchinfo>.
5. V. Badrinarayanan, A. Kendall, R. Cipolla, "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation", IEEE TPAMI, 2017.
6. S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997.
