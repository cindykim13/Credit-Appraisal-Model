# Hệ Thống Chấm Điểm Tín Dụng Doanh Nghiệp 

## Tóm Tắt Dự Án
Dự án là một hệ thống chấm điểm tín dụng hiện đại được thiết kế theo kiến trúc 2 tầng (2-Tier Architecture) nhằm đánh giá khả năng vỡ nợ (Probability of Default - PD) của các doanh nghiệp (SME). Dự án mô phỏng việc tích hợp đa chiều các yếu tố rủi ro của doanh nghiệp thông qua 4 mô hình đánh giá độc lập (Tầng 1) kết hợp cùng mô hình tổng hợp Meta-Model (Tầng 2). Báo cáo này đặc tả cấu trúc của hệ thống, bao gồm nguồn dữ liệu, hệ thống mã nguồn và các kết quả nghiên cứu.

---

## 1. Dữ Liệu Đầu Vào (`1_Data`)
Thư mục `1_Data` chứa toàn bộ tài nguyên dữ liệu và các báo cáo chất lượng dữ liệu ban đầu cho việc huấn luyện mô hình. Bộ dữ liệu được thiết kế công phu và phản ánh sát với thực tế thông qua 67 đặc trưng (features) bao gồm thông tin tài chính, hành vi tín dụng, chất lượng kinh doanh, độ ổn định và rủi ro.

- **Quy mô tập dữ liệu**: Tổng số 3,095 mẫu doanh nghiệp với 67 đặc trưng và nhãn vỡ nợ (Default rate: 8.50%).
- **Phân chia tập dữ liệu**:
  - `train.csv` (69.9%): 2,164 mẫu dùng để huấn luyện mô hình.
  - `val.csv` (14.9%): 462 mẫu dùng cho quá trình tinh chỉnh (tuning).
  - `test.csv` (15.2%): 469 mẫu dùng làm tập đánh giá khách quan cuối cùng.
- **Báo cáo Data Quality** (`FINAL_QA_REPORT.txt`): Chứng nhận toàn bộ dữ liệu (Module 1 đến Module 4) bao phủ 100% không gian đặc trưng và đã sẵn sàng cho huấn luyện.

---

## 2. Cấu Trúc Mã Nguồn (`2_Code`)
Thư mục `2_Code` là lõi mã nguồn của dự án, được thiết kế dưới dạng module hoá và tái sử dụng, phân tách rõ ràng giữa thành phần logic cốt lõi và các kịch bản thực thi.

- **`src/` (Mã nguồn cốt lõi)**:
  - `modules/`: Chứa định nghĩa Abstract Base Class (`base_module.py`) cho mọi module, cùng logic chuyên biệt cho 4 module tầng 1 (Financial, Credit, Business, Stability) và Meta-Model tầng 2.
  - `utils/`: Bao gồm Data Loader (hỗ trợ đọc tách dữ liệu) và Preprocessor (cho việc chuẩn hoá - normalize, mã hoá - encode và loại bỏ nhiễu).
  - `pipeline.py`: Pipeline Orchestrator cho việc thực thi luồng học máy End-to-End từ dữ liệu thô đến điểm tín dụng PD cuối cùng.
  - `config.py`: Khai báo các đường dẫn cố định và hyperparameters của hệ thống.

- **`scripts/` (Kịch bản thực thi)**: Bao gồm các script chạy theo trình tự, từ tiền xử lý dữ liệu (`01_organize_and_clean_data.py`), trích xuất feature, huấn luyện độc lập cho từng module tầng 1 (`train_module1.py`, `train_module3.py`, `train_module4.py`), trích xuất điểm kết quả tầng 1 (`score_collector.py`) đến huấn luyện mô hình cuối cùng `train_meta_model.py`.

---

## 3. Kết Quả Nghiên Cứu và Phân Tích (`3_KetQua`)
Thư mục `3_KetQua` cung cấp bằng chứng minh bạch và kết quả định lượng chi tiết cho từng chặng phát triển. Việc thiết kế mô hình 2 tầng cho thấy tính tối ưu mạnh mẽ với việc cải thiện từng thành phần.

### 3.1. Phase 3: Kết Quả Tầng 1 (4 Modules Cơ Sở)
Báo cáo chi tiết `phase3_results.md` mô tả thiết kế và kết quả đào tạo cho 4 thành phần đánh giá rủi ro chuyên sâu thông qua thuật toán như LightGBM, XGBoost và Random Forest, kết hợp cùng Semi-supervised blend để gia tăng tín hiệu nhận diện vỡ nợ trên data giả lập:
- **Module 1 (Financial Health - S1)**: Ensemble XGBoost + LightGBM. Mô hình đạt chỉ số R² (val) là 0.7014. Các chỉ số như DSCR, ROA là yếu tố quyết định.
- **Module 2 (Credit Behavior - S2)**: Ensemble XGBoost + Random Forest. R² (val) đạt mức cực cao 0.9705 với MAE 0.0152. Hành vi tín dụng được chứng minh là yếu tố cực kỳ hiệu quả trong việc dự đoán vỡ nợ (chênh lệch sanity test +0.043).
- **Module 3 (Business Quality - S3)**: LightGBM (Single). R² (val) = 0.6717 với lợi thế vượt trội khi bắt được các mối quan hệ phi tuyến tính trong độ chín muồi kinh doanh, vượt RF đến +0.1317 (ΔR²).
- **Module 4 (Stability & Risk - S4)**: XGBoost + LightGBM Ensemble. Điểm số tập trung vào ổn định và rủi ro ngoại sinh (Ngành nghề/Địa lý), đã nhận diện và cô lập các rủi ro nghiêm trọng như tiền sử phá sản (hard rules).

### 3.2. Phase 4: Kết Quả Tầng 2 (Meta-Model)
Sau khi thu thập kết quả điểm thành phần (thông qua `scores_train.csv`, `scores_val.csv`, `scores_test.csv`), Meta-Model dự đoán xác suất vỡ nợ thực tế. Kết quả ghi lại trong `phase4_results.md`:
- Meta-model **vượt tất cả các chỉ tiêu KPI đề ra** nhờ khả năng của XGBoost bắt được tín hiệu tương tác phi tuyến (ví dụ: điểm tài chính tốt nhưng có rủi ro ngoại cảnh cao).
- **Các Chỉ Số Thành Tựu Nổi Bật**:
  - **AUC-ROC (Val)**: 0.8773 (Mục tiêu > 0.87).
  - **KS Statistic**: 0.7338, cho thấy khả năng tách biệt lớp xuất sắc (Industry standard chỉ yêu cầu > 0.4).
  - **Brier Score**: 0.0637 (Mục tiêu < 0.09).
  - **F1-Score**: Tối ưu hóa ngưỡng Threshold (0.8451) giúp tăng F1 lên tới 0.7436.
- Sử dụng độ tin cậy của SHAP (Feature Importance), mô hình đã chứng minh các đặc tính tương tác (như S1_x_S2) và chất lượng doanh nghiệp (S3) có tác động mạnh nhất.

---

## Tổng Kết
Hệ thống thành công trong việc xây dựng một luồng ống dẫn dữ liệu có tính khoa học, độ tin cậy và tuân thủ các chuẩn mực quản trị rủi ro. Phương pháp tiếp cận 2 tầng giúp bóc tách độ phức tạp thành 4 chuyên môn đánh giá nhỏ, trước khi kết nối chúng lại để cung cấp quyết định cho vay định hướng dữ liệu mang lại giá trị cao trong hệ thống ngân hàng. 
