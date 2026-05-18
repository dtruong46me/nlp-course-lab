# Strengths & Weaknesses — Ưu/Nhược điểm

## Tóm tắt nhanh

| | PyVi |
|-|------|
| **Ưu** | Nhẹ, dễ cài, không cần GPU, F1 tốt cho thời điểm ra đời |
| **Nhược** | Cũ (2017), không có neural, giới hạn tác vụ, một số quirks |

---

## Ưu điểm

### 1. Không cần GPU

PyVi dùng CRF thuần — chạy tốt trên bất kỳ máy CPU nào, không cần setup CUDA, PyTorch, hay bất kỳ heavy dependency nào.

```bash
pip install pyvi  # chỉ cần thế này
```

So với PhoBERT (BERT-based Vietnamese NLP) cần:
- GPU (hoặc đợi rất lâu trên CPU)
- PyTorch 1.x+ 
- transformers library
- 400+ MB model weights

### 2. Latency thấp

CRF inference rất nhanh — predict toàn bộ câu chỉ mất vài milliseconds.

```
PyVi tokenizer:    ~1-5ms/câu
PhoBERT tokenizer: ~50-200ms/câu (CPU), ~5-20ms (GPU)
```

Phù hợp cho production systems cần throughput cao.

### 3. Accuracy tốt cho thời điểm ra đời (2016-2017)

```
Tokenizer F1 = 0.985  (trên VLSP 2016)
POS Tagger F1 = 0.925
```

Đây là near-SOTA vào thời điểm đó. Hiện nay neural models đạt ~0.99+ nhưng khoảng cách không quá lớn với tokenizer.

### 4. Pipeline đầy đủ trong 1 package

Tokenizer + POS tagger + Diacritics trong 1 gói nhỏ (~30 MB total).

### 5. Dễ tích hợp

API đơn giản:

```python
from pyvi import ViTokenizer
ViTokenizer.tokenize("...")  # xong
```

### 6. Xử lý tốt các trường hợp đặc biệt

Regex-based sylabelize xử lý được:
- Email: `test@gmail.com` → không bị tách
- URL: `https://example.com` → không bị tách
- Abbreviations: `Mr.`, `Dr.`, `ThS.`, `TP.` → không mất dấu chấm
- Số có separator: `5.2%`, `1,000,000`

---

## Nhược điểm & Lỗi thực tế

### 1. Model cũ, không còn được maintain

PyVi được train trên VLSP 2016/2017. Từ đó tiếng Việt (đặc biệt ngôn ngữ mạng xã hội, tiếng lóng, từ mới) đã thay đổi nhiều.

Repo GitHub last commit: **2019**. Không có plan update.

### 2. Word segmentation không nhất quán

Quan sát từ test thực tế:

```python
# "Trường đại học" — nên là "Trường_đại_học" (1 từ)
ViTokenizer.tokenize("Trường đại học bách khoa hà nội")
# → "Trường đại học bách khoa hà_nội"
#     ^^^^^^^^^^^^^^^^ chỉ join "hà_nội", bỏ sót phần còn lại!

# "Nguyễn Văn An" — nên là "Nguyễn_Văn_An"
ViTokenizer.tokenize("Ông Nguyễn Văn An là Thủ tướng")
# → "Ông Nguyễn Văn_An là Thủ tướng"
#             ^^^^^^^ chỉ join "Văn_An", bỏ "Nguyễn"

# "Thủ tướng" — nên là "Thủ_tướng"
# → "Thủ tướng"  ✗ không join được
```

**Nguyên nhân:**
- Bi-gram/tri-gram lookup trong words.txt không đủ coverage
- Model học từ VTB có thể không đủ đa dạng

### 3. `remove_accents` trả về bytes thay vì str

```python
ViUtils.remove_accents("Việt Nam")
# → b'Viet Nam'   ← bytes!

# Phải decode thủ công:
ViUtils.remove_accents("Việt Nam").decode('ascii')
# → 'Viet Nam'
```

Đây là bug tiềm ẩn: code cũ viết cho Python 2 (bytes = str), không được update cho Python 3.

### 4. `add_accents` không chính xác với ngữ cảnh ngắn

```python
ViUtils.add_accents("toi di hoc o truong")
# → "Tôi đi học ở truông"   ← "truông" sai, phải là "trường"

ViUtils.add_accents("thu do ha noi")
# → "Thư đó hạ nội"   ← sai hoàn toàn, phải là "Thủ đô Hà Nội"
```

CRFsuite character-level model cần đủ ngữ cảnh. Câu ngắn, từ đơn lẻ thường sai.

### 5. `Mr.` bị join với tên tiếp theo

```python
ViTokenizer.tokenize("Mr. Trung đang học")
# → "Mr._Trung đang học"   ← không mong muốn
```

Regex pattern `Mr\.` match "Mr." và đặt nó là một token, rồi CRF join với token tiếp theo.

### 6. Không có neural/contextual representations

CRF với hand-crafted features bị giới hạn bởi:
- Không học được long-range dependencies
- Feature sparsity: từ mới không có trong features → predict kém
- Không transfer learning: train lại từ đầu cho domain mới

### 7. Giới hạn tác vụ

PyVi **không có:**
- Named Entity Recognition (NER)
- Dependency Parsing
- Sentence segmentation (chia câu)
- Sentiment Analysis
- Coreference Resolution
- Text Classification

---

## So sánh với thư viện khác

### Vietnamese NLP landscape

| Thư viện | Approach | Tokenizer F1 | GPU? | Maintain? |
|---------|---------|-------------|------|----------|
| **PyVi** | CRF | ~0.985 | Không | ❌ Dừng 2019 |
| **underthesea** | Rule + CRF + Neural | ~0.972 | Tùy module | ✅ Active |
| **VnCoreNLP** (Java) | Max-Margin | ~0.975 | Không | ✅ Active |
| **PhoBERT** | BERT | ~0.990+ | Cần GPU | ✅ Active |
| **PhoNLP** | PhoBERT-based | ~0.990+ | Cần GPU | ✅ Active |

### Khi nào dùng PyVi?

**Nên dùng PyVi khi:**
- Cần deploy nhanh, không có GPU
- Production system cần latency thấp
- Domain chính thức (báo chí, văn bản hành chính) — gần với training data
- Prototype/research không cần độ chính xác tuyệt đối

**Không nên dùng PyVi khi:**
- Cần NER, parsing, hay các tác vụ nâng cao
- Data là ngôn ngữ mạng xã hội, tiếng lóng, text không chuẩn
- Cần accuracy cao nhất cho production
- Project long-term (dependency không được maintain)

### underthesea — alternative được maintain tốt

```bash
pip install underthesea
```

```python
from underthesea import word_tokenize, pos_tag, ner

word_tokenize("Tôi yêu Việt Nam")
# → ['Tôi', 'yêu', 'Việt Nam']

pos_tag("Tôi yêu Việt Nam")
# → [('Tôi', 'P'), ('yêu', 'V'), ('Việt Nam', 'Np')]

ner("Nguyễn Huệ là vị anh hùng dân tộc")
# → [('Nguyễn Huệ', 'PER'), ...]
```

---

## Kết luận

PyVi là một **thư viện tốt cho mục đích học tập và prototyping** với Vietnamese NLP. Code sạch, dễ đọc, dễ hiểu — lý tưởng để hiểu cách CRF hoạt động với tiếng Việt.

Cho production systems quan trọng, nên dùng **underthesea** (maintained, nhiều tác vụ hơn) hoặc **PhoBERT/PhoNLP** (accuracy cao hơn nếu có GPU).
