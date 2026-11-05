# 4. Thực nghiệm

## 4.1. Thiết lập chung và quy trình đánh giá

Nháp ngày 06/11/2026, người viết: Huỳnh Ngọc Trúc (khi ghép báo cáo bản 1). Phần thiết lập riêng của từng nhóm nằm ở đầu mục 4.2 (Bảng 4.1) và 4.3 (Bảng 4.4).

Cả ba nhóm thực nghiệm dùng chung baseline ở Bảng 3.1: Adam [4] với learning rate 10⁻³, batch 32, tối đa 1000 epoch, dừng sớm khi val loss 100 epoch liên tiếp không giảm [3], kích hoạt tanh, đầu ra tuyến tính. Các runner `run_A.py`, `run_B.py`, `run_C.py` đều đọc tham số từ `configs/baseline.yaml` và chỉ thay đúng yếu tố đang khảo sát (Bảng 1.1). Validation 500 điểm và test 100 điểm giữ nguyên cho mọi lượt chạy.

Quy trình đánh giá giống nhau ở cả ba nhóm:

1. Mỗi cấu hình chạy với ba seed huấn luyện 0, 1, 2. Seed quyết định trọng số khởi tạo và thứ tự xáo mini-batch.
2. Sau khi train, mạng được nạp lại trọng số ở epoch có val loss thấp nhất, rồi mới tính MSE, MAE, RMSE. Số trong bảng không lấy từ epoch cuối.
3. Mỗi ô trong bảng là trung bình ± độ lệch chuẩn mẫu (mẫu số n − 1) qua ba seed. RMSE tính riêng cho từng lượt rồi mới lấy trung bình, nên RMSE trung bình không nhất thiết bằng căn của MSE trung bình.
4. So sánh và xếp hạng cấu hình chỉ dựa trên val MSE. Sai số test chỉ được tính ở từng lượt C, vì đề yêu cầu báo test error theo số tầng, và ở mô hình cuối (mục 1.5). Không bước chọn nào đọc cột test.

Ba nhóm có hai cấu hình trùng nhau. Mạng 1 tầng 16 nơ-ron vừa là B = 16 vừa là C = 1. Mạng 2 tầng × 16 nơ-ron với 1000 mẫu vừa là A = 1000 vừa là C = 2. Nhóm so log của từng cặp ở cả ba seed. Train loss và val loss trùng nhau ở cả 1000 epoch, chênh lệch bằng 0, dù A/B và C do hai runner khác nhau chạy. Vì vậy các bảng của A, B và C so được trực tiếp với nhau. Giữa Bảng 4.2, 4.3 và Bảng 4.5, vài số lệch một đơn vị ở chữ số có nghĩa thứ bảy, ví dụ val MSE của A = 1000 là 4.350544e-07, còn của C = 2 là 4.350545e-07. Mỗi runner tính lại val MSE từ trọng số tốt nhất sau khi train. Ở seed 1, hai lần tính này lệch nhau cỡ 10⁻¹⁴ do sai số số thực, đủ để trung bình làm tròn đến bảy chữ số có nghĩa lệch một đơn vị ở chữ số cuối. Log loss của hai runner vẫn trùng nhau.

Nguồn số liệu: `configs/baseline.yaml`; log `results/B/logs/B_16_seed{0,1,2}.csv` so với `results/C/logs/C_1_seed{0,1,2}.csv`, và `results/A/logs/A_1000_seed{0,1,2}.csv` so với `results/C/logs/C_2_seed{0,1,2}.csv`.
