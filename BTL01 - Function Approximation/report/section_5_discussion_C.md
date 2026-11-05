# Mục 5. Thảo luận kết quả thực nghiệm C

## Ảnh hưởng của số tầng ẩn

Khi tăng từ một lên hai tầng ẩn, MSE trung bình trên train giảm từ 9,281 × 10⁻⁶ xuống 4,136 × 10⁻⁷ và trên validation giảm từ 1,193 × 10⁻⁵ xuống 4,351 × 10⁻⁷. Mức giảm khoảng 27,4 lần trên validation cho thấy thêm tầng thứ hai có ích rõ rệt trong thiết lập 1.000 mẫu train và 16 nơ-ron mỗi tầng. Sai số cao đồng thời trên train và validation của mạng một tầng phù hợp với khả năng xấp xỉ còn hạn chế ở cấu hình này. Ba lượt đều chạm 1.000 epoch, vì vậy giới hạn tối ưu cũng có thể góp phần vào kết quả. [Bảng kết quả thực nghiệm C](../results/C/table_C_mean_std.md).

Khi tăng tiếp độ sâu, validation MSE trung bình tăng ở ba tầng rồi giảm ở bốn tầng. Cấu hình bốn tầng đạt 4,098 × 10⁻⁷, thấp hơn khoảng 5,8% so với hai tầng. Độ lệch chuẩn của bốn tầng là 1,852 × 10⁻⁷, khoảng 7,6 lần so với hai tầng. Với ba seed, chênh lệch trung bình này chỉ mô tả các lượt đã chạy. Hai tầng ít dao động hơn giữa các seed và mất trung bình 21,55 giây, so với 25,72 giây ở bốn tầng trên máy chạy thí nghiệm. [Kết quả từng seed](../results/C/summary.csv).

## Hội tụ và dấu hiệu quá khớp

Cấu hình một và hai tầng chạy đủ 1.000 epoch ở cả ba seed. Hai lượt của cấu hình ba tầng và hai lượt của cấu hình bốn tầng dừng sớm. Chẳng hạn, cấu hình ba tầng ở seed 0 đạt validation loss tốt nhất tại epoch 437 rồi dừng tại epoch 537. Do số seed còn chạy thay đổi ở cuối đường loss, cần xem từng log khi so sánh các đoạn cuối của biểu đồ. Số epoch thực chạy cũng ảnh hưởng đến tổng thời gian, nên thời gian trong bảng không đo trực tiếp chi phí mỗi epoch. [Đường loss](../results/C/figures/C_loss_curves.png) · [Đối chiếu 12 log](../results/C/table_C_mean_std.md#đối-chiếu-log).

Từ hai lên ba tầng, cả train MSE và validation MSE trung bình cùng tăng. Từ ba lên bốn tầng, cả hai cùng giảm. Kết quả hiện có chưa cho thấy mẫu hình train MSE giảm trong khi validation MSE tăng dần theo độ sâu. Số tham số tăng từ 65 lên 881 khi giữ cố định độ rộng và tăng số tầng, nên chưa thể tách ảnh hưởng của độ sâu khỏi ảnh hưởng của năng lực biểu diễn và quá trình tối ưu. [Mục 4.3](section_4_3_experiment_C.md).

## Giới hạn và hướng cải tiến

Các kết quả mô tả bài toán Gaussian hai chiều trên miền và tập mẫu đã chọn. Một lượt khảo sát tiếp theo có thể tăng số seed để ước lượng biến động tốt hơn, so sánh các kiến trúc có số tham số gần nhau và ghi thời gian mỗi epoch bên cạnh tổng thời gian. Lưu checkpoint cùng dự đoán theo seed sẽ giúp đối chiếu trực tiếp những lượt có sai số khác biệt.

Nhóm đã chốt checkpoint của thực nghiệm A với 2.000 mẫu train dựa trên validation. Các chỉ số test trong mục C dùng để mô tả những lượt đã chạy. Tập test đó đã được truy vấn trong C, nên khi trình bày kết quả checkpoint cuối cần ghi rõ lịch sử sử dụng tập dữ liệu này. [Kết quả checkpoint cuối](../results/final/final_metrics.json) · [Đánh giá mô hình cuối](section_5_final_model_evaluation.md).
