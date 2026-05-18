# Training — Huấn luyện Word Segmentation

Đây là phần chi tiết nhất về cách PyVi train model tokenizer (word segmentation).

---

## Tổng quan pipeline training

```
Vietnamese Treebank (vtb.txt)
           │
    [Parse corpus]
           │
    [Syllabelize]  ← tách từng câu thành âm tiết
           │
    [Label generation]  ← B_W / I_W cho mỗi âm tiết
           │
    [Feature extraction]  ← word2features()
           │
  [pickle → X_train, y_train]
           │
    [CRF training]  ← sklearn-crfsuite, Lbfgs
           │
  [RandomizedSearchCV]  ← tối ưu c1, c2
           │
    [Pickle model]  ← pyvi3.pkl
```

---

## Bước 1 — Parse Vietnamese Treebank

File VTB có format: `word1/POS word2/POS word3/POS ...`

Từ ghép (compound) dùng `_` nối âm tiết: `học_sinh/N`, `đại_học/N`

**Cách đọc:**

```python
# Mỗi dòng là một câu
# Tách thành (syllable, label) pairs
for line in vtb.txt:
    for token in line.split():
        word, pos = token.rsplit('/', 1)
        syllables = word.split('_')
        # syllables[0] → label B_W (beginning of word)
        # syllables[1:] → label I_W (inside word)
```

**Ví dụ:**

```
Input:  "học_sinh/N trường/N đại_học/N"
Output: [("học",   "B_W"),
         ("sinh",  "I_W"),
         ("trường","B_W"),
         ("đại",   "B_W"),
         ("học",   "I_W")]
```

---

## Bước 2 — Feature Extraction (`word2features`)

Source: [`pyvi/ViTokenizer.py:27-71`](../pyvi/ViTokenizer.py#L27-L71)

Với mỗi âm tiết tại vị trí `i`, ta tạo ra một dict feature:

```python
features = {
    'bias': 1.0,                        # constant term
    'word.lower()': word.lower(),       # lowercase của âm tiết hiện tại
    'word.isupper()': word.isupper(),   # có phải all-caps không? (VNM, IMB, ...)
    'word.istitle()': word.istitle(),   # có phải title case? (Hà, Nội, ...)
    'word.isdigit()': word.isdigit(),   # có phải số không?
}
```

**Context features (nhìn sang trái/phải):**

```python
# i-1 (âm tiết liền trước)
'-1:word.lower()': word1.lower()
'-1:word.istitle()': word1.istitle()
'-1:word.isupper()': word1.isupper()
'-1:word.bi_gram()': ' '.join([word1, word]).lower() in bi_grams  # ← KEY!

# i-2, i-1, i (tri-gram lookup)
'-2:word.tri_gram()': ' '.join([word2, word1, word]).lower() in tri_grams

# i+1 (âm tiết liền sau)
'+1:word.lower()': word1.lower()
'+1:word.istitle()': word1.istitle()
'+1:word.isupper()': word1.isupper()
'+1:word.bi_gram()': ' '.join([word, word1]).lower() in bi_grams

# i, i+1, i+2 (tri-gram phải)
'+2:word.tri_gram()': ' '.join([word, word1, word2]).lower() in tri_grams
```

**Tại sao bi-gram/tri-gram quan trọng?**

CRF không "biết" tiếng Việt. Nó học từ pattern. Bi-gram lookup cho nó biết:
- "học sinh" → có trong từ điển → khả năng cao là một từ → `I_W`
- "học máy" → có trong từ điển → CRF sẽ join → "học_máy"
- "học ăn" → không có trong từ điển → thường là hai từ riêng

---

## Bước 3 — CRF Model (`train_tokenizer.py`)

Source: [`pyvi/train_tokenizer.py`](../pyvi/train_tokenizer.py)

### Thuật toán: L-BFGS

```python
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',       # Limited-memory Broyden–Fletcher–Goldfarb–Shanno
    max_iterations=100,
    all_possible_transitions=True
)
```

**L-BFGS là gì?** Thuật toán quasi-Newton optimization, hiệu quả với large-scale sparse problems như NLP. Nó xấp xỉ Hessian matrix thay vì tính trực tiếp → memory-efficient.

### Nhãn (Labels)

```python
labels = ['B_W', 'I_W']
```

- `B_W` — Beginning of Word: âm tiết bắt đầu một từ (mới)
- `I_W` — Inside Word: âm tiết tiếp tục của từ ghép

### Hyperparameter Search

```python
params_space = {
    'c1': scipy.stats.expon(scale=0.5),   # L1 regularization
    'c2': scipy.stats.expon(scale=0.05),  # L2 regularization
}

rs = RandomizedSearchCV(
    crf,
    params_space,
    cv=5,           # 5-fold cross-validation
    n_iter=50,      # 50 random combinations
    n_jobs=12,      # parallel jobs
    scoring=f1_scorer
)
rs.fit(X_train, y_train)
```

**c1 (L1 regularization):** Giảm overfitting bằng cách ép nhiều weight → 0 (sparse model). Thang đo: exponential với mean=0.5.

**c2 (L2 regularization):** Giảm magnitude của tất cả weights. Thang đo: exponential với mean=0.05 (nhỏ hơn c1 → ít aggressive hơn).

**Tại sao RandomizedSearch thay vì GridSearch?**
- GridSearch: thử mọi combination → O(N²)
- RandomizedSearch: sample ngẫu nhiên 50 điểm → nhanh hơn nhiều, đủ tốt

### Evaluation Metric

```python
f1_scorer = make_scorer(
    metrics.flat_f1_score,
    average='weighted',
    labels=labels
)
```

F1 weighted: tính F1 cho mỗi nhãn rồi weight theo tần suất → phù hợp khi B_W nhiều hơn I_W.

---

## Bước 4 — Inference (Prediction)

Source: [`pyvi/ViTokenizer.py:114-128`](../pyvi/ViTokenizer.py#L114-L128)

```python
@staticmethod
def tokenize(str):
    text, tmp = ViTokenizer.sylabelize(str)   # tách âm tiết
    labels = ViTokenizer.model.predict(
        [ViTokenizer.sent2features(tmp, False)]
    )  # → [['B_W', 'I_W', 'B_W', 'B_W', 'I_W', ...]]
    
    output = tmp[0]
    for i in range(1, len(labels[0])):
        if (labels[0][i] == 'I_W'
                and tmp[i] not in string.punctuation       # không join dấu câu
                and tmp[i-1] not in string.punctuation
                and not tmp[i][0].isdigit()                # không join số
                and not tmp[i-1][0].isdigit()
                and not (tmp[i][0].istitle()               # không join nếu chữ hoa mới
                         and not tmp[i-1][0].istitle())):
            output = output + '_' + tmp[i]   # join bằng _
        else:
            output = output + ' ' + tmp[i]   # tách bằng space
    return output
```

**Post-processing rules (hardcoded):** Dù model predict `I_W`, vẫn tách ra nếu:
1. Âm tiết là dấu câu
2. Âm tiết là chữ số
3. Âm tiết mới bắt đầu bằng chữ hoa (Title case) trong khi âm tiết trước không viết hoa

Rule 3 là heuristic quan trọng: "Hà Nội" thường là 2 từ riêng (tên riêng có viết hoa), không phải "Hà_Nội" (trừ khi CRF đủ tự tin).

---

## Sylabelize — Pre-processing

Source: [`pyvi/ViTokenizer.py:77-111`](../pyvi/ViTokenizer.py#L77-L111)

Trước khi đưa vào CRF, text phải được tách thành units (âm tiết). Đây không đơn giản là `.split()` vì có nhiều trường hợp đặc biệt:

```python
patterns = [
    # Xử lý trước (priority cao hơn)
    "[A-ZĐ]+\.",        # Abbreviations: TP., IMB.
    "Tp\.",
    "Mr\.", "Mrs\.", "Ms\.", "Dr\.", "ThS\.",

    # Specials
    "==>", "->", "\.\.\.", ">>"

    # Entities
    "\w+://[^\s]+",     # URLs: https://example.com
    "([a-zA-Z0-9_.+-]+@[...]+)",  # Emails

    # Digits (với separator): 5.2%, 1,000,000
    "\d+([\.,_]\d+)+",

    # Regular
    "[^\w\s]",          # Non-word chars (punctuation)
    "\w+",              # Words/syllables
]
```

**Ví dụ sylabelize:**

```python
"Mr. Trung đang học ThS. tại ĐHBK"
→ ["Mr.", "Trung", "đang", "học", "ThS.", "tại", "ĐHBK"]
#   ^^^^ giữ nguyên abbreviation, không tách thành "Mr" + "."
```

```python
"test@gmail.com và https://example.com"
→ ["test@gmail.com", "và", "https://example.com"]
#   ^^^^^^^^^^^^^^^ email giữ nguyên, URL giữ nguyên
```

---

## Kết quả training

```
Tokenizer (word segmentation):  F1 = 0.985
POS Tagger:                     F1 = 0.925
```

Đây là kết quả trên VLSP benchmark. Khá cao so với baseline.

---

## Train lại model từ đầu

Xem hướng dẫn chi tiết tại [06-usage.md](06-usage.md#train-model-riêng).

Tóm tắt:

```bash
# 1. Chuẩn bị dữ liệu từ VTB hoặc VLSP
python prepare_data.py  # parse VTB → X_train.pkl, y_train.pkl

# 2. Train
python pyvi/train_tokenizer.py
# → tạo ra tokenizer_model.pkl và tokenizer_model_py3.pkl

# 3. Copy vào models/
cp tokenizer_model_py3.pkl pyvi/models/pyvi3.pkl
```
