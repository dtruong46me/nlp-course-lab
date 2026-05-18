# Pipeline — Luồng xử lý NLP đầy đủ

## Kiến trúc pipeline

PyVi không enforce một pipeline cứng — các module độc lập với nhau. Nhưng thứ tự tự nhiên khi dùng là:

```
Raw Text (string)
     │
     ▼ ViTokenizer.tokenize()
Tokenized Text ("word1 com_pound word2")
     │
     ▼ ViPosTagger.postagging()
(tokens, pos_tags)
     │
     ▼ (optional) downstream NLP tasks
NER / Parsing / Sentiment / ...
```

Diacritics (ViDiac/ViUtils) là bước độc lập, có thể chạy trước hoặc sau.

---

## Module 1: ViTokenizer — Word Segmentation

**Vấn đề:** Tiếng Việt viết âm tiết cách nhau bằng dấu cách. Nhưng "từ" (word) có thể gồm nhiều âm tiết.

```
"học sinh" ≠ "học" + "sinh"   →   "học_sinh" (1 từ = student)
"đại học"  ≠ "đại" + "học"    →   "đại_học"  (1 từ = university)
```

**Output convention:** Từ đơn âm tiết viết bình thường. Từ đa âm tiết nối bằng `_`.

```python
from pyvi import ViTokenizer

# Trường hợp cơ bản
ViTokenizer.tokenize("Trường đại học bách khoa hà nội")
# → "Trường đại học bách khoa hà_nội"
# Lưu ý: "bách khoa" không join được vì model không chắc chắn

# Tên riêng
ViTokenizer.tokenize("Ông Nguyễn Văn An là Thủ tướng")
# → "Ông Nguyễn Văn_An là Thủ tướng"

# Công ty, tổ chức
ViTokenizer.tokenize("Chính phủ Việt Nam đã ban hành")
# → "Chính_phủ Việt_Nam đã ban hành"

# URL, email không bị tách vỡ
ViTokenizer.tokenize("Email test@gmail.com và https://example.com")
# → "Email test@gmail.com và https://example.com"

# Viết tắt
ViTokenizer.tokenize("Mr. Trung đang học ThS.")
# → "Mr._Trung đang học ThS."   ← Mr. bị join với tên (quirk)
```

**Output cho spaCy:**

```python
tokens, spaces = ViTokenizer.spacy_tokenize("Trường đại học bách khoa hà nội")
# tokens = ['Trường', 'đại', 'học', 'bách', 'khoa', 'hà_nội']
# spaces = [True, True, True, True, True, False]
```

---

## Module 2: ViPosTagger — POS Tagging

**Input:** Chuỗi token cách nhau bằng dấu cách (output của ViTokenizer)

**Output:** `(list_tokens, list_tags)`

```python
from pyvi import ViPosTagger

# Method 1: từ string tokenized
tokens, tags = ViPosTagger.postagging("Chính_phủ Việt_Nam đã ban hành")
# tokens = ['Chính_phủ', 'Việt_Nam', 'đã', 'ban', 'hành']
# tags   = ['N', 'Np', 'R', 'V', 'V']

# Method 2: từ list tokens
tokens, tags = ViPosTagger.postagging_tokens(['Trường_đại_học', 'Bách_Khoa', 'tuyển_sinh'])
```

**Bảng POS Tags (Vietnamese Tagset):**

| Tag | Tiếng Việt | English | Ví dụ |
|-----|-----------|---------|-------|
| `N` | Danh từ | Common noun | nhà, trường, nước |
| `Np` | Danh từ riêng | Proper noun | Hà_Nội, Việt_Nam |
| `Nc` | Danh từ chỉ loại | Classifier | con, cái, chiếc, ông |
| `Nu` | Danh từ đơn vị | Unit noun | km, kg, m² |
| `Ny` | Danh từ viết tắt | Abbreviated noun | IMB, VNM |
| `V` | Động từ | Verb | là, đi, học, làm |
| `A` | Tính từ | Adjective | đẹp, lớn, nhanh |
| `P` | Đại từ | Pronoun | tôi, anh, họ, đó |
| `R` | Trạng từ | Adverb | rất, đã, đang, sẽ |
| `M` | Số từ | Numeral | một, hai, 5, 100 |
| `L` | Định từ | Determiner | các, những, mỗi |
| `E` | Giới từ | Preposition | của, từ, tại, trong |
| `C` | Liên từ bình đẳng | Coord. conj. | và, hoặc, nhưng |
| `S` | Liên từ phụ thuộc | Subord. conj. | vì, nếu, khi |
| `T` | Tình thái từ | Modal/Auxiliary | hãy, đừng, nhé |
| `I` | Thán từ | Interjection | ôi, ồ, chao |
| `X` | Không xác định | Unknown | |
| `F` | Dấu câu | Punctuation | ., ,, !, ? |

---

## Module 3: ViUtils / ViDiac — Diacritics

### Remove accents

```python
from pyvi import ViUtils

ViUtils.remove_accents("Việt Nam")   # → b'Viet Nam'
# Lưu ý: trả về bytes trong Python 3 (quirk của implementation)
# Nếu cần string: ViUtils.remove_accents("Việt Nam").decode()
```

**Cơ chế:**

```python
def remove_accents(s):
    s = unicodedata.normalize('NFKD', s)    # decompose: "ọ" → "o" + combining_diacritical
    s = s.encode('ascii', 'ignore')          # drop non-ASCII (diacritical marks)
    return s
```

### Add accents (restoration)

```python
ViUtils.add_accents("truong dai hoc bach khoa ha noi")
# → "Trường Đại học Bách Khoa hà nội"   ← không hoàn hảo, có lỗi context
```

**Cơ chế:** CRFsuite character-level model

1. Lowercase toàn bộ input
2. Với mỗi ký tự, tạo feature từ ký tự xung quanh (trong và ngoài từ)
3. CRF predict label: `L` (lowercase), `U` (uppercase), hoặc dấu thanh
4. Reconstruct output bằng `reversed_mapping` (Telex → Unicode Vietnamese)

**Label format cho diacritics:**

```
"ọ" → base char 'o', label 'Lj'  (L=lowercase, j=dot below)
"Ầ" → base char 'a', label 'UAAm'  (U=uppercase, AA=double-a=â, m=circumflex doubled, s/f/r/x/j=tone)
```

Telex encoding dùng ký hiệu: `s`=sắc, `f`=huyền, `r`=hỏi, `x`=ngã, `j`=nặng, `w`=ư/ơ, `aa`=â, `ee`=ê, `oo`=ô

---

## Pipeline đầy đủ — Ví dụ thực tế

```python
from pyvi import ViTokenizer, ViPosTagger, ViUtils

# 1. Input
text = "Chính phủ Việt Nam đã ban hành Nghị quyết 42 về phát triển kinh tế"

# 2. Word segmentation
segmented = ViTokenizer.tokenize(text)
# → "Chính_phủ Việt_Nam đã ban_hành Nghị_quyết 42 về phát_triển kinh_tế"

# 3. POS tagging
words, tags = ViPosTagger.postagging(segmented)
# → words: ['Chính_phủ', 'Việt_Nam', 'đã', 'ban_hành', 'Nghị_quyết', '42', ...]
# → tags:  ['N',        'Np',       'R',  'V',         'N',          'M', ...]

# 4. Hiển thị kết quả
for w, t in zip(words, tags):
    print(f"{w:25s} → {t}")
```

---

## Tích hợp với spaCy

PyVi cung cấp `spacy_tokenize()` để tích hợp:

```python
import spacy
from pyvi import ViTokenizer

# Custom tokenizer cho tiếng Việt
nlp = spacy.blank("vi")

def vi_tokenizer(text):
    tokens, spaces = ViTokenizer.spacy_tokenize(text)
    return spacy.tokens.Doc(nlp.vocab, words=tokens, spaces=spaces)

nlp.tokenizer = vi_tokenizer

doc = nlp("Trường đại học bách khoa hà nội")
for token in doc:
    print(token.text, token.whitespace_)
```

---

## Giới hạn pipeline

1. **Không có dependency parsing** — không phân tích cấu trúc ngữ pháp (subject/object)
2. **Không có NER** — không phân biệt loại tên riêng (người/địa danh/tổ chức)
3. **Không pipeline end-to-end** — các module không auto-chain
4. **Stateless** — không có ngữ cảnh giữa các câu

Xem thêm tại [07-strengths-weaknesses.md](07-strengths-weaknesses.md).
