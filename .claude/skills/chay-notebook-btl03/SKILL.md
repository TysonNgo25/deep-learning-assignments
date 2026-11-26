---
name: chay-notebook-btl03
description: Cách cài môi trường và chạy notebook PyTorch của BTL03 an toàn - trên máy Trúc (Windows, env btl02, chỉ CPU, hay thiếu RAM) hoặc GPU Colab/Kaggle (đưa dữ liệu Gender/FER2013 lên, tải kết quả về) - và notebook nào chạy ở đâu. Dùng trước khi chạy bất kỳ notebook hay lệnh python nào trong BTL03.
---

# Chạy notebook BTL03

## 1. Môi trường trên máy Trúc

Dùng chung env `btl02` (Python 3.11, torch CPU). **Máy không có GPU NVIDIA** (GPU AMD) và **hay thiếu RAM**. Cài thêm gói của BTL03 (Anaconda Prompt):

```bat
conda activate btl02
cd /d "D:\Data\GitHub\deep-learning-assignments\BTL03 - Chuyển code từ keras sang pytorch deep learning"
pip install -r requirements.txt
```

Python của env: `C:\Users\hntruc\anaconda3\envs\btl02\python.exe`. Đặt `PYTHONIOENCODING=utf-8`.

Chạy notebook không cần mở trình duyệt (lưu output vào chính file):

```powershell
$py = "C:\Users\hntruc\anaconda3\envs\btl02\python.exe"
Set-Location "D:\Data\GitHub\deep-learning-assignments\BTL03 - Chuyển code từ keras sang pytorch deep learning\pytorch\1. MLP"
& $py -X utf8 -m nbconvert --to notebook --execute --inplace "MNIST_MLP_pytorch.ipynb" --ExecutePreprocessor.timeout=3600
```

Dùng `-m nbconvert`, **không** dùng `-m jupyter nbconvert`: lệnh `jupyter` gọi subcommand theo PATH và trên máy có Python 3.12 cài ngoài conda (`AppData\Local\Programs\Python\Python312`) nên chạy nhầm bản đó, báo `NoSuchKernel: python3` (gặp 26/11). Env `btl02` đã cài đủ `torchinfo tensorboard scikit-image nbconvert ipykernel notebook seaborn` (26/11).

Báo `MemoryError`/out of memory hoặc shell Cowork không khởi động: nhờ Trúc tắt bớt ứng dụng (AMDChat, VS Code thừa), `wsl --shutdown`, rồi chạy lại. `DataLoader(num_workers=0)` trên Windows.

## 2. Notebook nào chạy ở đâu

| Notebook | Máy Trúc (CPU) | GPU | Ghi chú thời gian (gốc) |
|---|---|---|---|
| `getting_started`, `Func_apxm_MLP`, `MNIST_MLP` | chạy đầy đủ | – | MNIST_MLP ~3 phút CPU |
| `1_MNIST_CNN` | chỉ chạy thử (1 epoch, ít batch) | đầy đủ | 4 phút GTX 1080, 105 phút i7 |
| `2_Human_Gender_CNN` (cả 2 bản) | chạy thử | đầy đủ | 3 phút GPU, 109 phút i7 |
| `3_Emotion_Reg_CNN` (cả 2 bản) | chạy thử | đầy đủ | 6 phút / 23 phút GPU, 5 giờ i7 |
| `MNIST_CNN_LSTM` | chạy thử | đầy đủ | 27 phút GPU, 13,5 giờ i7 |
| `1_segnet` | chạy thử | đầy đủ | 12,5 phút GPU |
| `2_obj_detection` | có thể đầy đủ (5 epoch, mô hình nhỏ) | tùy | vài phút |

"Chạy thử" là để kiểm code (shape, số tham số, một vòng train), **không** lưu làm kết quả nộp. Mỗi notebook nên có biến `CHAY_THU = False` ở ô đầu: `True` thì cắt dữ liệu và epoch; bản nộp phải chạy với `False`.

## 3. Chạy trên GPU (Colab hoặc Kaggle)

1. Trúc **commit + push** code mới nhất (repo công khai `TysonNgo25/deep-learning-assignments`).
2. Ô đầu trên Colab/Kaggle: `git clone` repo, `cd` vào thư mục BTL03, `pip install torchinfo`.
3. Dữ liệu không có trong git:
   - MNIST: torchvision tự tải.
   - Gender (~586 MB, 4 file `.npy`): đưa lên Google Drive `MyDrive/BTL03/gender/`, mount Drive rồi chép (hoặc tạo symlink) vào `data/gender/`. Kaggle: tạo Dataset riêng từ 4 file.
   - FER2013: Kaggle Add Input `nicolejyt/facialexpressionrecognition` (như BTL02) hoặc `MyDrive/fer2013.csv`; chép vào `data/emotion/fer2013.csv`.
   - SegNet, xe: có sẵn trong repo (`Yêu cầu/examples/5. Others/`) nếu thư mục dữ liệu đã được commit; nếu không, nén và đưa lên Drive như Gender.
4. Chạy notebook bằng `jupyter nbconvert --execute --inplace` (hoặc mở trực tiếp trên Colab), rồi tải về: notebook đã có output + `results/<notebook>/` (không kèm `checkpoints/`, `runs/`).
5. Giải nén đè vào đúng chỗ trong repo, `git status --short .` để xem file đổi, rồi `/btl03-ghi-ket-qua <notebook>`.
6. Gom nhiều notebook vào một phiên GPU; có thể chạy chung phiên với BTL02 nếu không vượt RAM/thời gian.

## 4. Không được làm nếu Trúc chưa đồng ý rõ ràng

- Không sửa, xóa, di chuyển gì trong `Yêu cầu/` (hook chặn).
- Không ghi đè cả `results/so_sanh.csv` (hook hỏi lại); cập nhật từng dòng.
- Không lưu notebook nộp ở chế độ `CHAY_THU = True`.
- Không chạy notebook nặng đầy đủ trên máy Trúc.
- Không commit `data/`, `*.pt`, `runs/` (đã có `.gitignore`).

## 5. Sau khi chạy

`git status --short .` để báo Trúc file nào đổi. Ghi `so_sanh.csv`, ghi nhật ký (skill `nhat-ky-btl03`). Gọi agent `btl03-doi-chieu-keras` cho notebook vừa chạy.
