# Assignment 6: Email Spam Filtering

### Objectives
Mục tiêu của bài thực hành này là xây dựng spam email filter sử dụng pandas và scikit-learn.

### Dataset
URL: https://drive.google.com/uc?id=1bTJKchSInd3IgLs41b1_-Gd-T36a_pal

### Instructions
1. Download dataset từ URL trên và giải nén, đặt tên thư mục là `/spam_data`.
2. Sử dụng pandas để load dữ liệu từ file `spam_data/spam_ham_dataset.csv`.
3. Tiền xử lý dữ liệu:
    - Thay thế các ký tự `\n`, `\r`, và `\t` trong cột `text` bằng dấu cách.
    - Xóa subject từ đầu mỗi email.
    - Nhặt ra dòng đầu tiên của mỗi email và lưu vào cột mới `first_line`.
    - Chuyển đổi nhãn `spam` và `ham` trong cột `label` thành 1 và 0.
4. Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% - 20%).
5. Xây dựng model:
    - Sử dụng `CountVectorizer` để chuyển đổi cột `text` thành ma trận đặc trưng.
    - Huấn luyện một mô hình phân loại (ví dụ: `MultinomialNB`, `LogisticRegression`, v.v.) trên tập huấn luyện. Trong bài này sử dụng Naive Bayes.
6. Đánh giá mô hình trên tập kiểm tra và in ra các chỉ số: accuracy, precision, recall, F1-score.
7. (Tùy chọn) Visualize kết quả bằng biểu đồ. Đường cong ROC-AUC có thể là một lựa chọn tốt.
8. (Tùy chọn) Thử nghiệm với các mô hình khác nhau và so sánh hiệu suất. Ở đây sử dụng `HashingVectorizer` thay vì `CountVectorizer`.