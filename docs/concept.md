# NLP Concepts — Kiến thức nền tảng

Tài liệu này giải thích các khái niệm NLP được dùng trong PyVi.

---

## 1. Tại sao tiếng Việt khó xử lý?

### Đặc điểm đặc trưng

| Đặc điểm | Tiếng Anh | Tiếng Việt |
|---------|-----------|-----------|
| Ranh giới từ | Dấu cách rõ ràng | Dấu cách giữa **âm tiết**, không phải từ |
| Morphology | Inflection (runs/ran/running) | **Isolating** — từ không biến đổi hình thái |
| Số lượng âm tiết/từ | Đa âm tiết (beautiful, wonderful) | Hầu hết đơn âm tiết (đẹp, tốt, xấu) |
| Dấu thanh | Không có | 6 thanh (ngang, huyền, sắc, hỏi, ngã, nặng) |

### Vấn đề word segmentation đặc trưng

Tiếng Việt là **ngôn ngữ phân tích/cô lập** (analytic/isolating language): mỗi âm tiết mang một nghĩa, và nghĩa của "từ" phụ thuộc vào sự kết hợp.

```
"bò" = cow    "ăn" = eat    "bò ăn" = cow eats
"bò ăn cỏ" = cow eats grass

nhưng:
"học" = study    "sinh" = born/live
"học sinh" = student  (không phải "studying + living")
```

Không có quy tắc cứng nào để biết khi nào hai âm tiết là một từ — cần học từ corpus.

---

## 2. Conditional Random Field (CRF)

### CRF là gì?

CRF (Conditional Random Field) là một **discriminative probabilistic model** cho sequence labeling. Nó mô hình hóa phân phối có điều kiện:

```
P(y | x) = P(y₁, y₂, ..., yₙ | x₁, x₂, ..., xₙ)
```

Trong đó:
- `x = [x₁, ..., xₙ]` = sequence of observations (âm tiết và features)
- `y = [y₁, ..., yₙ]` = sequence of labels (B_W / I_W)

### So sánh với các model khác

| Model | Loại | Phụ thuộc output? | Dùng trong PyVi? |
|-------|------|-------------------|-----------------|
| Naive Bayes | Generative | Không | Không |
| MaxEnt/Logistic Regression | Discriminative | Không | Không |
| HMM | Generative | Có | Không |
| **CRF** | **Discriminative** | **Có** | **Có** |
| BiLSTM-CRF | Discriminative + Neural | Có | Không |
| Transformer (BERT) | Neural | Có | Không |

**Tại sao CRF tốt hơn Naive Bayes/MaxEnt?**
- CRF xét **toàn bộ sequence** khi predict, không chỉ từng vị trí độc lập
- Tránh được label bias problem của HMM
- Feature engineering linh hoạt — dùng bất kỳ feature nào

### Linear-chain CRF (dùng trong PyVi)

```
x₁  → x₂  → x₃  → ... → xₙ
|      |      |            |
y₁ → y₂ → y₃ → ... → yₙ
```

Mỗi nhãn `yᵢ` phụ thuộc vào tất cả `x` và nhãn lân cận `yᵢ₋₁`.

### Hàm score

```
score(y, x) = Σᵢ Σₖ λₖ fₖ(yᵢ₋₁, yᵢ, x, i)
```

`fₖ` là các feature functions (như "âm tiết này có trong bi_grams với âm tiết trước không?")
`λₖ` là trọng số được học trong training.

### L-BFGS Optimization

PyVi dùng L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno):

- Quasi-Newton method: xấp xỉ Hessian thay vì tính trực tiếp
- "Limited memory": chỉ lưu m cặp vector gần nhất (m ~ 10-20)
- Phù hợp cho large-scale ML với hàng ngàn features

### Regularization

```python
c1 = 0.1   # L1: ||w||₁ → ép một số weights = 0 (sparse)
c2 = 0.01  # L2: ||w||₂² → giảm magnitude tất cả weights
```

L1 tốt cho feature selection (loại bỏ feature không quan trọng).
L2 tốt cho smooth weights.

---

## 3. Sequence Labeling — IOB/BIO scheme

### BIO tagging

PyVi dùng biến thể đơn giản gọi là **B_W / I_W**:

```
B_W = Beginning of Word  (bắt đầu từ mới)
I_W = Inside Word        (tiếp tục từ trước đó)
```

Ví dụ:

```
Sentence: "Trường đại học Bách Khoa Hà Nội"
Labels:    B_W    B_W  I_W  B_W   I_W  B_W  I_W
Result:   "Trường đại_học Bách_Khoa Hà_Nội"
```

Trong NER, người ta thường dùng B/I/O (Outside):
```
B-PER = bắt đầu tên người
I-PER = tiếp tục tên người
O     = không phải entity
```

PyVi chỉ có 2 nhãn B_W/I_W vì mục tiêu chỉ là tách từ, không phân loại thực thể.

---

## 4. Unicode và Dấu Thanh Tiếng Việt

### 6 thanh điệu

| Thanh | Ký hiệu | Ví dụ | Telex |
|-------|---------|-------|-------|
| Ngang (flat) | — | ma | (không có) |
| Huyền | ` | mà | f |
| Sắc | ´ | má | s |
| Hỏi | ˀ | mả | r |
| Ngã | ~ | mã | x |
| Nặng | . | mạ | j |

### Nguyên âm đặc biệt

| Nguyên âm | Ký hiệu | Telex |
|-----------|---------|-------|
| ă | breve | aw |
| â | circumflex | aa |
| ê | circumflex | ee |
| ô | circumflex | oo |
| ơ | horn | ow |
| ư | horn | uw |
| đ | stroke | dd |

### Unicode normalization

Python có 4 forms:
- **NFC** (Composed): "ọ" = 1 code point. Dùng trong tokenization.
- **NFD** (Decomposed): "ọ" = "o" + combining_dot_below. 
- **NFKC/NFKD**: thêm compatibility decomposition.

```python
# remove_accents dùng NFKD + encode('ascii', 'ignore')
unicodedata.normalize('NFKD', 'ọ')  # → 'o' + '̣'
.encode('ascii', 'ignore')           # → b'o'
```

---

## 5. Feature Engineering trong CRF

### Tại sao feature engineering quan trọng?

CRF "truyền thống" không học representation như neural networks. Nó học **linear combination of features**. Vì vậy, feature design quyết định performance.

### Features dùng trong ViTokenizer

```python
features = {
    # Surface features
    'word.lower()':     'học',     # lowercase
    'word.isupper()':   False,     # không phải viết hoa
    'word.istitle()':   False,     # không phải Title case
    'word.isdigit()':   False,     # không phải số

    # Context (nhìn trái)
    '-1:word.lower()':  'đại',     # âm tiết trước
    '-1:word.istitle()': False,
    '-1:word.bi_gram()': True,     # "đại học" có trong từ điển!

    # Context (nhìn phải)
    '+1:word.lower()':  'bách',    # âm tiết sau
    '+1:word.bi_gram()': True,     # "học bách" không có trong từ điển
}
```

Vị trí `học` trong "Trường đại **học** Bách Khoa":
- `-1:word.bi_gram()` = True → "đại học" là bi-gram → CRF học pattern này → predict `I_W`
- `+1:word.bi_gram()` = False → "học Bách" không phải bi-gram → không gợi ý join tiếp

### Features dùng trong ViPosTagger

Thêm một số features:
```python
'word[:1].isdigit()':   # ký tự đầu là số
'word[:3].isupper()':   # 3 ký tự đầu viết hoa (VNM, GDP, ...)
'word.isfiltered':      # là dấu câu
```

Không có bi-gram lookup vì POS tagging không cần biết ranh giới từ (đã tokenize rồi).

---

## 6. Evaluation Metrics

### F1 Score

```
Precision = TP / (TP + FP)   # trong những gì predict là positive, bao nhiêu đúng
Recall    = TP / (TP + FN)   # trong những positive thực, bao nhiêu được tìm ra
F1        = 2 × P × R / (P + R)   # harmonic mean
```

Với sequence labeling, dùng **flat F1** — so sánh từng token:
- TP: predict B_W đúng là B_W
- FP: predict B_W nhưng thực ra là I_W
- FN: predict I_W nhưng thực ra là B_W

**Weighted F1:** Tính F1 riêng cho B_W và I_W rồi weight theo tần suất.

### Kết quả PyVi

```
Tokenizer:  F1 = 0.985  (trên VLSP dataset)
POS Tagger: F1 = 0.925
```

SOTA cho tiếng Việt (năm 2016-2017) vào khoảng 0.97-0.98 cho tokenizer.
