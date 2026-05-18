# Data — Dữ liệu huấn luyện

## Tổng quan các nguồn dữ liệu

PyVi sử dụng 3 nguồn dữ liệu chính:

| File | Kích thước | Mục đích |
|------|-----------|---------|
| `data/vtb.txt` | 1.9 MB, 10,382 dòng | Corpus gán nhãn POS — dùng để extract training data cho tokenizer |
| `data/dataset.zip` | 26.4 MB | VLSP 2016/2018 corpus (nén) |
| `pyvi/models/words.txt` | 377 KB, 31,157 entries | Từ điển bi-gram/tri-gram — dùng làm feature |

---

## Vietnamese Treebank (vtb.txt)

### Format

Mỗi dòng là một câu. Mỗi token có dạng `word/POS_TAG`. Từ ghép (compound word) dùng dấu `_` nối các âm tiết:

```
Hải_tặc/N eo_biển/N Malacca/Np (/( kỳ/N 1/M )/) .../...
Đó/P là/V con/Nc đường/N biển/N ngắn/A nhất/A ...
```

### Ý nghĩa với việc training

VTB là nguồn "sự thật" (ground truth) cho tokenizer. Từ corpus này:

1. **Extract sequences** — tách từng câu thành danh sách âm tiết
2. **Generate labels** — âm tiết đầu của từ ghép → `B_W`, âm tiết tiếp theo → `I_W`

Ví dụ dòng đầu tiên:

```
Hải_tặc/N  →  Hải → B_W,  tặc → I_W
eo_biển/N  →  eo  → B_W,  biển → I_W
Malacca/Np →  Malacca → B_W
```

→ Sequence labels: `[B_W, I_W, B_W, I_W, B_W, ...]`

### Thống kê VTB

- **Số câu:** 10,382
- **Ngữ liệu:** Báo tiếng Việt (Tuổi Trẻ, ...)
- **POS tags được dùng:** 17 loại (xem [05-nlp-concepts.md](05-nlp-concepts.md))
- **Từ loại phổ biến nhất:** N (danh từ), V (động từ)

---

## Từ điển words.txt

### Format

Mỗi dòng là một cụm từ gồm 2 hoặc 3 âm tiết (bi-gram hoặc tri-gram):

```
a dua
a ha
học sinh
đại học
bách khoa hà nội
xử lý ngôn ngữ tự nhiên
...
```

### Cách dùng trong feature extraction

Trong `ViTokenizer.word2features()`:

```python
# Tại vị trí i, kiểm tra xem âm tiết [i-1, i] có trong từ điển không
'-1:word.bi_gram()': ' '.join([word1, word]).lower() in ViTokenizer.bi_grams,

# Kiểm tra tri-gram [i-2, i-1, i]
'-2:word.tri_gram()': ' '.join([word2, word1, word]).lower() in ViTokenizer.tri_grams,
```

**Tại sao quan trọng?** CRF biết rằng nếu hai âm tiết liền nhau có trong từ điển, khả năng cao chúng là một từ ghép → gán `I_W`.

### Thống kê words.txt

- Tổng: 31,157 entries
- Phần lớn là bi-gram (2 âm tiết)
- Một phần nhỏ tri-gram (3 âm tiết): "bách khoa hà nội", "xử lý ngôn ngữ", ...

---

## VLSP Dataset (dataset.zip)

**VLSP** = Vietnam Language and Speech Processing — tổ chức tổ chức các cuộc thi NLP tiếng Việt hàng năm.

- **VLSP 2016:** Shared task về word segmentation, POS tagging
- **VLSP 2018:** Mở rộng hơn, thêm NER (Named Entity Recognition)

Dataset này được dùng trong `train_tokenizer.ipynb` để tạo ra các file:
- `tokenized_X_train.pkl` — feature vectors cho training
- `tokenized_y_train.pkl` — nhãn B_W/I_W tương ứng

> **Lưu ý:** Dataset zip không có sẵn trong repo (chỉ có file zip), cần đăng ký tại [vlsp.org.vn](http://vlsp.org.vn) để tải.

---

## Cách dữ liệu được chuẩn bị (pipeline)

```
VTB corpus (vtb.txt)
        │
        ▼
[Parse câu] → tách thành list (âm_tiết, label_POS)
        │
        ▼
[Extract syllables] → ["Hải", "tặc", "eo", "biển", ...]
        │
        ▼
[Generate B_W/I_W labels] → ["B_W", "I_W", "B_W", "I_W", ...]
        │
        ▼
[word2features()] → [{bias:1.0, word.lower:'hải', bi_gram:False, ...}, ...]
        │
        ▼
tokenized_X_train.pkl (features) + tokenized_y_train.pkl (labels)
        │
        ▼
     CRF training (train_tokenizer.py)
```

---

## Vì sao tiếng Việt khó tokenize?

1. **Không có dấu cách phân tách từ rõ ràng** — âm tiết nào ghép với nhau là quy ước, không phải quy tắc cứng
2. **Đồng âm khác nghĩa** — "đường" một mình = road/sugar, "đường biển" = sea route
3. **Từ mượn & tên riêng** — "Malacca", "Mr.", "TP.HCM" cần xử lý đặc biệt
4. **Viết tắt** — "IMB", "km", "km/h" không phải từ ghép

Đây là lý do PyVi dùng CRF thay vì rule-based hay simple dictionary lookup.
