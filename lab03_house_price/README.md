# Lab 03: Dự đoán giá nhà Ames

## Dữ liệu và bài toán

`data/train.csv` (1.460 căn nhà có `SalePrice`) và `data/test.csv` (1.459 căn nhà chưa có nhãn) là dữ liệu cuộc thi [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Bản CSV dùng trong bài được lấy từ [kho lưu trữ công khai](https://github.com/Ayeshaa-Aslam/House-Prices-Advanced-Regression-Techniques) vì thư mục nguồn ban đầu chỉ có hai PDF. Paper De Cock mô tả bộ Ames gốc 2.930 giao dịch; tập Kaggle là phiên bản/chia dữ liệu dùng trong PDF thứ hai, không phải toàn bộ 2.930 dòng.

## Chạy lại

Tạo môi trường Python 3.12, cài `requirements.txt`, rồi ở thư mục `lab3` chạy:

```powershell
python -m pip install -r requirements.txt
python run_experiments.py
python test_pipeline.py
```

Mã cố định seed 42. `run_experiments.py` chia 80% tập có nhãn để phát triển và 20% giữ lại để kiểm tra cuối. Trên phần phát triển, ba nếp kiểm định chéo so sánh bốn nhóm đặc trưng với Ridge và HistGradientBoosting của scikit-learn, hai nhóm với MLP PyTorch. Mọi bộ điền thiếu, mã hóa và chuẩn hóa chỉ học trên phần huấn luyện của từng nếp. MLP còn tách một phần nhỏ từ nếp huấn luyện để dừng sớm. Tiêu chí chính là RMSE giữa log giá thật và log giá dự đoán; MAE tính theo USD là tiêu chí phụ.

`results/` chứa số liệu từng nếp, bảng tóm tắt, điểm trên tập giữ lại, dự đoán cho tập giữ lại, và `submission.csv` gồm hai cột `Id,SalePrice` cho Kaggle. Mô hình tạo `submission.csv` được chọn bằng điểm kiểm định chéo trên phần phát triển, sau đó học lại trên toàn bộ 1.460 dòng có nhãn. Không có điểm Kaggle cho tập 1.459 dòng vì nhãn không được công bố.

## Tài liệu tham khảo

- Dean De Cock (2011), *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*, Journal of Statistics Education 19(3), file `decock.pdf` do đề bài cung cấp.
- Rishabh Nimje, *House Prices: Advanced Regression Techniques*, PDF do đề bài cung cấp. Ví dụ gốc dùng XGBoost và Keras; bài này viết lại bằng scikit-learn và PyTorch.
- Kaggle, [quy tắc đánh giá](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation).
