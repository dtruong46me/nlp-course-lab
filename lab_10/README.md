# Assignment 10: Machine Translation with Seq2Seq Models

### Objectives

Mục tiêu của bài thực hành này là xây dựng một mô hình dịch máy (Machine Translation) sử dụng kiến trúc Seq2Seq với cơ chế Attention. Dữ liệu sử dụng là bộ Standford NMT, bao gồm các cặp câu Tiếng Anh và Tiếng Việt.

### Dataset

Dữ liệu được sử dụng trong bài thực hành này là bộ Standford NMT, có thể tải về từ [đây](https://nlp.stanford.edu/projects/nmt/data/). Bộ dữ liệu này bao gồm các cặp câu Tiếng Anh và Tiếng Việt, được chia thành tập train, validation và test.

Do trang web gốc không thể truy cập được, do đó chúng ta sẽ sử dụng bản sao lưu của bộ dataset này tại URL: https://github.com/stefan-it/nmt-en-vi

Sau khi download và giải nén, cấu trúc thư mục sẽ như sau:

```
/data_iwslt15
    ├── train.en
    ├── train.vi
    ├── tst2012.en
    ├── tst2012.vi
    ├── tst2013.en
    └── tst2013.vi
```

### Instructions

1. Tải dữ liệu từ URL trên và giải nén, đặt tên thư mục là `/data_iwslt15`.
> Lưu ý: 
> - Nếu bạn sử dụng Google Colab, bạn có thể tải dữ liệu trực tiếp vào thư mục `/content/data_iwslt15`.

2. Thêm các token đặc biệt: `<start>`, `<end>` vào đầu và cuối mỗi câu trong tập train, validation và test.

3. Tiền xử lý dữ liệu:
    - Tokenize các câu, sử dụng Tokenizer của tensorflow.

4. Xây dựng mô hình Seq2Seq với Attention:
    - Sử dụng LSTM hoặc GRU cho cả Encoder và Decoder.
    - Áp dụng cơ chế Attention để cải thiện hiệu suất dịch. Trong bài này sử dụng Bahdanau Attention.
    - Optimize mô hình sử dụng Adam optimizer và hàm mất mát là Sparse Categorical Crossentropy.
    - Training mô hình trong 10 epochs

5. Đánh giá mô hình:
    - Sử dụng BLEU score để đánh giá chất lượng dịch trên tập test.
    - In ra một số câu dịch mẫu từ tập test để so sánh với câu gốc.

### Submission
Nộp file notebook đã hoàn thành bài thực hành này lên hệ thống LMS của khóa học.