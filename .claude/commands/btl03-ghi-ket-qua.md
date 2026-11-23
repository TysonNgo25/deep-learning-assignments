---
description: Ghi kết quả một hoặc nhiều notebook PyTorch vào results/so_sanh.csv, đặt cạnh số Keras gốc (ví dụ /btl03-ghi-ket-qua MNIST_MLP 1_MNIST_CNN)
argument-hint: <tên notebook> [tên notebook ...]
---

Ghi kết quả các notebook `$ARGUMENTS` của BTL03 vào `results/so_sanh.csv` theo `.claude/rules/btl03-ket-qua.md`:

- Số PyTorch: đọc `results/<notebook>/metrics.json` và `history.csv`; đối chiếu với output đã lưu trong notebook `_pytorch.ipynb`. Hai nguồn lệch nhau thì dừng, báo Trúc.
- Số Keras: lấy từ output của notebook gốc trong `Yêu cầu/examples/` (đọc file `.ipynb`, tìm dòng `Test accuracy`, `Total params`, `Mean absolute error`), đối chiếu `docs/Kế hoạch.md` mục 3.
- Điền đủ cột; `ghi_chu` nêu khác biệt có chủ đích. Chênh lệch vượt ngưỡng trong rule thì ghi rõ và gợi ý nguyên nhân cần kiểm.
- Cập nhật dòng đã có của notebook đó, không thêm dòng trùng. Không bịa số: thiếu file kết quả thì báo.

Sau khi ghi, in lại các dòng vừa thêm/sửa.
