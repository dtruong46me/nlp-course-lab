# Assignment 2: Preprocessing Text Data

### Description

Mục tiêu của bài thực hành là xây dựng một hàm xử lý văn bản bao gồm các bước như loại bỏ ký tự không mong muốn, chuyển đổi chữ hoa thành chữ thường, tách từ, loại bỏ từ dừng và đem các từ về dạng gốc.

### Instructions

1. Cài đặt các thư viện cần thiết:
   - Sử dụng thư viện NLTK để hỗ trợ các bước xử lý ngôn ngữ tự nhiên.
    - Sử dụng thư viện re để xử lý các biểu thức chính quy.

2. Xây dựng hàm xử lý văn bản với các bước sau:
   - Loại bỏ các ký tự không phải chữ cái (A-Z, a-z)
    - Chuyển đổi tất cả văn bản thành chữ thường.
    - Loại bỏ dấu chấm câu.
    - Tách từ (tokenization).
    - Loại bỏ các từ dừng (stop words).
    - Đem các từ về dạng gốc (stemming hoặc lemmatization).

3. Hàm này nhận vào một câu và trả về một danh sách các từ đã được xử lý.

4. Kiểm tra hàm với một số câu ví dụ và in ra kết quả.