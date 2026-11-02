# CodetoNhan — Phần việc Nhân đến 02/11/2026

Người phụ trách: Ngô Nguyễn Thành Nhân, STT 29, MSSV 3123411205.
Đối chiếu theo [Kế hoạch.md](Kế%20hoạch.md). Tài liệu này ghi bằng chứng đang có trong repo; không xem lịch giao việc là bằng chứng ngày hoàn thành thực tế.

## Đối chiếu công việc đến hôm nay

| Ngày giao | Việc của Nhân | Bằng chứng/trạng thái khi rà ngày 02/11 |
| --- | --- | --- |
| 28/10 | Đọc đề, cài môi trường | Đã đọc yêu cầu trong kế hoạch; môi trường cục bộ đã cài và dùng chạy C trong phiên trước. Không có nhật ký chứng minh đã hoàn thành ngay 28/10. |
| 29/10 | Đọc tài liệu, phác plots.py; cùng nhóm chốt baseline | `src/plots.py` hiện đã hoàn thiện cho C; baseline được ghi là đã chốt trong YAML. Không thay nhóm xác nhận việc họp/chốt. |
| 30/10 | Hoàn thiện plots.py, phác run_C.py | Đã có runner, đồ thị loss, MSE/RMSE, thời gian và hình hai panel. |
| 31/10 | Chạy thử C một giá trị, một seed; review code chéo | Phiên trước đã chạy thử C=1, seed=0 rồi chạy đủ 12 lượt. Ngày 02/11 đã đọc lại model/train và rà tương thích runner; chưa có bằng chứng thành viên khác đã review phần Nhân. |
| 01/11 | Chạy đủ C: 4 độ sâu × 3 seed | Có 12 log và 12 dòng `results/C/summary.csv`; lượt đầy đủ đã chạy sớm từ phiên trước, không cần chạy lại theo ngày lịch. |
| 02/11 | Xuất bảng và đồ thị C | Đã đối chiếu 12 log, xuất bảng đủ metric và dựng lại 5 hình bằng `experiments/export_C.py`. |

## Bàn giao ngày 02/11

- [Bảng đọc trực tiếp, dẫn từng log và liên kết hình](results/C/table_C_mean_std.md).
- [CSV đầy đủ mean/std](results/C/table_C_mean_std.csv): MSE, MAE, RMSE của train/val/test; thời gian train; best epoch.
- [Số liệu từng lượt](results/C/summary.csv), [bảng mean/std cũ](results/C/summary_mean_std.csv), 12 CSV trong `results/C/logs/`.
- 5 hình trong `results/C/figures/`: loss, test MSE, test RMSE, thời gian, RMSE và thời gian trên hai panel.

Bảng mới bổ sung mean/std của train MAE/RMSE và validation MAE/RMSE; các metric này vốn đã nằm trong summary từng lượt. Không sinh số liệu huấn luyện mới.

## Kết quả đối chiếu

- Đủ 4 độ sâu (1–4) và seed 0, 1, 2, không thiếu/trùng cặp trong summary.
- Epoch trong mỗi log liên tiếp từ 1; loss hữu hạn; số epoch không quá 1000.
- `config_json` và seed trong log khớp độ sâu, 16 nơ-ron/tầng, 1000 mẫu, tanh, Adam, lr=0.001, batch=32, patience=100.
- Best epoch là epoch có val loss thấp nhất; `val_mse` của mô hình đã khôi phục khớp min val loss trong sai số số thực.
- Mean và độ lệch chuẩn mẫu tính lại từ summary khớp bảng mean/std đã lưu.
- So với commit kết quả C `48fbf36`, model/train/data, CSV dữ liệu và baseline chưa thay đổi trong checkout ngày rà.

Thời gian và metric lấy từ lượt chạy đã lưu. Log C không lưu dự đoán hoặc checkpoint từng lượt, nên đối chiếu này không thay thế việc tính lại metric độc lập từ trọng số. Test đã được đọc cho cả 12 lượt để báo cáo C; không dùng test để chốt mô hình. Việc đánh giá test của mô hình cuối cần phối hợp nhóm ngày 03/11 và mô tả trung thực việc test đã được xem trước đó.

## Hiểu model.py và train.py để vấn đáp

1. **Đầu vào và đầu ra:** X có shape `(n, 2)`, y `(n, 1)`. `MLP(hidden=[16] * C)` tạo C tầng ẩn, mỗi tầng gồm `Linear` rồi `Tanh`; tầng cuối `Linear(..., 1)` cho đầu ra liên tục, không thêm activation.
2. **Một mini-batch:** đặt gradient về 0, forward để dự đoán, tính MSE, `backward()` tính gradient, `optimizer.step()` cập nhật trọng số bằng Adam.
3. **Train loss theo epoch:** cộng loss trung bình từng batch nhân số mẫu batch, rồi chia tổng số mẫu. Batch cuối ít mẫu vẫn có trọng số đúng. Đây là loss trong lúc trọng số đang thay đổi, khác train MSE tính lại trên checkpoint tốt nhất.
4. **Validation:** `eval()` và `torch.no_grad()` đánh giá tập val, không cập nhật trọng số. `no_grad()` mới là phần tắt theo dõi gradient; `eval()` điều chỉnh chế độ hoạt động của các lớp.
5. **Early stopping và checkpoint:** lưu bản sao trọng số khi val loss giảm; sau 100 epoch không cải thiện thì dừng. Cuối vòng lặp khôi phục trọng số có val loss thấp nhất, không mặc định lấy epoch cuối.
6. **Seed:** seed quyết định khởi tạo trọng số và thứ tự mini-batch. Ba seed giúp đo dao động; std là độ lệch chuẩn mẫu với mẫu số n−1.
7. **Thời gian:** `perf_counter()` đo vòng huấn luyện, không gồm ghi CSV và vẽ hình. Early stopping làm thời gian phụ thuộc cả độ sâu lẫn số epoch thực chạy.
8. **Test:** không truyền test vào `train_model()`. Runner C chỉ tính test metric sau khi train trả về checkpoint được chọn bằng validation; runner không chọn độ sâu dựa trên test.

Hàm metric cục bộ của C dùng float32 như lượt chạy gốc. `src/evaluate.py` hiện đã có hàm chung dùng float64; giữ C như hiện tại để không âm thầm đổi chữ số của kết quả đã lưu.

## Cách xuất lại bảng và hình, không huấn luyện lại

Đứng trong thư mục `BTL01 - Function Approximation`, với môi trường theo README đã kích hoạt:

```powershell
python -m experiments.export_C
```

Trên máy này, môi trường được cài trong `.venv` bằng runtime Codex, dùng:

```powershell
$env:PYTHONPATH = (Resolve-Path '.venv\Lib\site-packages').Path
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = (Join-Path (Get-Location).Path '.cache\matplotlib')
& '.venv\Scripts\python.exe' -m experiments.export_C
```

Script kiểm tra các dữ kiện trước khi xuất bảng, dùng helper tổng hợp có sẵn trong `experiments/common.py` và hàm vẽ có sẵn trong `src/plots.py`. Các log, summary gốc và thời gian đã đo được giữ nguyên.

## Phần cần con người xác nhận và việc sau hôm nay

- Nhóm xác nhận review code chéo và việc Nhân đã hiểu/trình bày được phần code; tài liệu này là nội dung ôn tập, không chứng nhận thay thành viên.
- Đã chuẩn bị sớm ngày 02/11: [nháp mục 4.3](report/section_4_3_experiment_C.md), có bảng, hình, phân tích hội tụ/thời gian và giới hạn kết luận. Ngày 03/11 rà lại khi ghép; phối hợp Trúc/Khang về mô hình cuối, checkpoint và protocol test.
- 04/11: viết thảo luận C, hạn chế/cải tiến; dựng khung slide.
- 05–07/11: đưa hình/bảng vào slide, tập trình bày, xuất PPTX/PDF.
- Nếu cần bổ sung checkpoint hoặc tính lại metric từ trọng số của 12 lượt C, trình phương án chạy lại để Nhân duyệt trước; việc xuất bảng ngày 02/11 không cần bước này.
