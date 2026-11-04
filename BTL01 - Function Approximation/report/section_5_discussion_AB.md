# Mục 5. Thảo luận kết quả thực nghiệm A/B

Phần này thảo luận kết quả A/B trong [mục 4.2](./section_4_2_experiments_AB.md). Các kết luận dựa trên trung bình và độ lệch chuẩn của ba seed, chủ yếu dùng sai số validation để so sánh. Không dùng test để lựa chọn hay giải thích cấu hình A/B.

## 5.1. Ảnh hưởng của số lượng mẫu huấn luyện

Trong thực nghiệm A, số mẫu train tăng từ 200 lên 2000, còn tập validation được giữ cố định. Validation MSE trung bình giảm từ `1.219636e-05` xuống `1.681281e-07`, tức giảm khoảng 72.5 lần khi số mẫu tăng 10 lần. Xu hướng giảm xuất hiện xuyên suốt các cỡ mẫu đã thử; chẳng hạn, MSE giảm từ `2.224003e-06` ở 400 mẫu xuống `4.350544e-07` ở 1000 mẫu, rồi còn `1.681281e-07` ở 2000 mẫu.

Kết quả phù hợp với kỳ vọng rằng nhiều mẫu hơn giúp mạng quan sát hàm mục tiêu trên nhiều vị trí hơn trong miền dữ liệu, nhờ đó khớp tốt hơn cả dữ liệu train lẫn các điểm validation chưa dùng cập nhật trọng số. Sai số validation không tăng khi tăng dữ liệu, nên trong phạm vi thí nghiệm này không thấy dấu hiệu rằng thêm mẫu làm giảm khả năng tổng quát hóa. Mức cải thiện tương đối ở các cỡ mẫu lớn không còn mạnh như giai đoạn đầu; điều đó gợi ý lợi ích biên có thể giảm dần, nhưng chưa thể kết luận đã đạt ngưỡng bão hòa vì sai số vẫn tiếp tục giảm ở 2000 mẫu.

Độ lệch chuẩn qua seed cho thấy kết quả không hoàn toàn giống nhau giữa các lần khởi tạo. Chẳng hạn, ở 200 mẫu, validation MSE là `1.219636e-05 ± 7.14e-06`, còn ở 2000 mẫu là `1.681281e-07 ± 5.89e-08`. Vì mỗi cỡ mẫu chỉ có ba seed, độ lệch chuẩn này hữu ích để mô tả biến động nhưng chưa đủ để kiểm định ý nghĩa thống kê. Thêm nữa, phần lớn cấu hình chạy gần giới hạn 1000 epoch; vì vậy kết quả còn phản ánh cả giới hạn ngân sách huấn luyện và tiêu chí dừng sớm, không chỉ riêng số lượng dữ liệu.

## 5.2. Năng lực mô hình, thiếu khớp và dấu hiệu quá khớp

Trong thực nghiệm B, số mẫu train được giữ ở 1000 và thay đổi độ rộng của một tầng ẩn. Với width 1, validation MSE trung bình là `2.915716e-03`; khi tăng lên width 64, giá trị này giảm còn `2.810769e-06`. Sai số train cũng giảm mạnh, từ `3.194696e-03` xuống `2.244774e-06`. Mạng quá hẹp có ít tham số và khả năng biểu diễn hạn chế so với hàm mục tiêu, nên kết quả ở width 1 là dấu hiệu thiếu năng lực mô hình (underfitting) trong thiết lập này. Tăng width giúp mạng biểu diễn hàm tốt hơn, và các cấu hình rộng hơn 10 tiếp tục cải thiện validation MSE.

Không phải mọi độ rộng đều tạo ra đường cong giảm đều. Ở width 8, 9 và 10, validation MSE lần lượt là `9.535581e-05`, `1.073820e-04` và `1.099850e-04`. Độ lệch chuẩn tại width 9 đặc biệt lớn (`9.70e-05`), cho thấy kết quả của cấu hình này rất nhạy với seed. Điểm đáng chú ý là train MSE ở width 9 (`1.033432e-04`) cũng cao hơn width 8 (`8.285447e-05`), thay vì thấp hơn trong khi validation error tăng. Do đó, các điểm này không thể hiện mẫu hình điển hình của quá khớp là train error tiếp tục giảm nhưng validation error tăng. Chúng phù hợp hơn với biến động tối ưu hóa giữa các seed hoặc việc các lượt train chưa hội tụ đồng đều; dữ liệu hiện có không đủ để xác định nguyên nhân cụ thể.

Nhìn toàn dải width, train và validation MSE tương đối gần nhau, và cả hai tiếp tục giảm rõ ở width 16, 32, 64. Vì vậy, trong các cấu hình đã thử, chưa có bằng chứng mạnh để kết luận mô hình rộng hơn bị quá khớp. Đây không phải khẳng định rằng quá khớp không thể xảy ra: dữ liệu được sinh từ hàm xác định, chỉ có ba seed, và phép thử chỉ khảo sát một khoảng kiến trúc cùng một quy trình train. Cần thêm seed, cấu hình hoặc dữ liệu nhiễu để kết luận tổng quát hơn.

## 5.3. Giới hạn khi diễn giải

Các kết quả chỉ áp dụng cho thiết lập đã cố định: dữ liệu Gaussian không nhiễu, kiến trúc MLP tanh, Adam với learning rate `1e-3`, batch size 32, tối đa 1000 epoch và patience 100. Vì vậy, không nên suy rộng rằng cứ tăng số mẫu hoặc width sẽ luôn cải thiện kết quả trong mọi bài toán. Một khảo sát tiếp theo có thể tăng số seed và kiểm tra riêng ảnh hưởng của ngân sách epoch hoặc siêu tham số tối ưu hóa, nhưng cần giữ các thay đổi đó tách biệt để biết yếu tố nào tạo ra khác biệt.

Việc so sánh và lựa chọn trong A/B chỉ dựa trên validation. Các số test không được dùng để chọn cấu hình; giới hạn về việc test đã được truy vấn trong thực nghiệm C cần được trình bày như đã ghi ở mục 1.2 và mục 4.3.
