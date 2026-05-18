# Go Inference — Chạy Word Segmentation hoàn toàn trong Go

**Không cần Python khi inference.** Bạn chỉ cần chạy Python **một lần duy nhất** để export
model weights ra JSON, rồi Go tự lo toàn bộ phần còn lại.

---

## Tại sao được?

File `pyvi3.pkl` là Python pickle của object `sklearn_crfsuite.CRF`. Go không đọc được
trực tiếp, nhưng CRF chỉ là tập hợp số thực (weights). Sau khi extract ra, model trở thành
một file JSON 221 KB — hoàn toàn language-agnostic.

### Hai phát hiện quan trọng từ khi inspect model

**1. Toàn bộ model chỉ có 5803 con số:**

| Thành phần | Số lượng |
|-----------|---------|
| State feature weights | 5,799 |
| Transition weights | 4 (B_W→B_W, B_W→I_W, I_W→B_W, I_W→I_W) |
| Total | 5,803 |

**2. Tính phản đối xứng (antisymmetry) — quan trọng nhất:**

```
B_W weight cho feature f  =  -(I_W weight cho feature f)
```

Verified thực tế:
```
bias:          B_W = +2.444936,  I_W = -2.444936  ✓
word.isupper(): B_W = +0.680798, I_W = -0.680798  ✓
word.istitle(): B_W = -0.559908, I_W = +0.559908  ✓
-1:word.bi_gram(): B_W = -4.181370, I_W = +4.181370 ✓
```

→ Chỉ cần lưu B_W weights. Score I_W = -Score B_W. Viterbi chỉ cần track 1 giá trị mỗi vị trí.

---

## Bước 1 — Export model weights (chạy một lần duy nhất)

```python
# export_model.py  —  chạy 1 lần với Python, sau đó Go dùng JSON mãi mãi
import pickle, json, re, tempfile, os
import sys; sys.path.insert(0, '.')

with open('pyvi/models/pyvi3.pkl', 'rb') as f:
    crf = pickle.load(f)

# Dump CRFsuite binary model sang text format (pycrfsuite built-in)
tmp = tempfile.mktemp(suffix='.txt')
crf.tagger_.dump(tmp)
with open(tmp, encoding='utf-8') as f:
    content = f.read()
os.unlink(tmp)

# Parse TRANSITIONS
transitions = {}
m = re.search(r'TRANSITIONS = \{(.*?)\}', content, re.DOTALL)
for entry in re.finditer(r'\(\d+\) (\S+) --> (\S+): ([-\d.]+)', m.group(1)):
    transitions[entry.group(1) + '\t' + entry.group(2)] = float(entry.group(3))

# Parse STATE_FEATURES — chỉ giữ B_W weights (I_W = -B_W)
state_features = {}
m = re.search(r'STATE_FEATURES = \{(.*)\}', content, re.DOTALL)
for entry in re.finditer(r'\(\d+\) (.*?) --> (\S+): ([-\d.]+)', m.group(1)):
    feat, label, weight = entry.group(1), entry.group(2), float(entry.group(3))
    if label == 'B_W':
        state_features[feat] = weight

export = {
    "labels": ["B_W", "I_W"],
    "transitions": transitions,      # 4 entries
    "state_features": state_features # ~5799 entries
}

with open('pyvi/models/model_weights.json', 'w', encoding='utf-8') as f:
    json.dump(export, f, ensure_ascii=False)

print(f"Exported: {len(state_features)} state features, {len(transitions)} transitions")
# Output: Exported: 5799 state features, 4 transitions
```

**Kết quả:** file `model_weights.json` (221 KB) — đây là toàn bộ thứ Go cần.

### Format JSON

```json
{
  "labels": ["B_W", "I_W"],
  "transitions": {
    "B_W\tB_W": -0.99668,
    "B_W\tI_W":  1.918715,
    "I_W\tB_W": -2.969562,
    "I_W\tI_W": -0.43685
  },
  "state_features": {
    "bias":                   2.444936,
    "word.isupper()":         0.680798,
    "word.istitle()":        -0.559908,
    "word.isdigit()":         1.815288,
    "word.lower():học":       0.35...,
    "-1:word.bi_gram()":     -4.181370,
    "+1:word.bi_gram()":      0.884030,
    "-2:word.tri_gram()":     1.408808,
    "+2:word.tri_gram()":    ...
  }
}
```

**Quy ước feature key:**
- Boolean feature = `True` → key là tên feature, ví dụ `"word.isupper()"`
- String feature → key là `"tên_feature:giá_trị"`, ví dụ `"word.lower():học"`
- Float feature (`bias`) → key là `"bias"`, nhân với value (luôn là 1.0)

---

## Bước 2 — Go package đầy đủ

Tạo package `vitoken` với 3 files:

### `vitoken/model.go` — Load model và Viterbi

```go
package vitoken

import (
    "encoding/json"
    "math"
    "os"
    "strings"
)

// CRFModel holds the exported weights from pyvi3.pkl.
// Only B_W weights are stored; I_W score = -B_W score (antisymmetry).
type CRFModel struct {
    Labels        []string           `json:"labels"`
    Transitions   map[string]float64 `json:"transitions"`
    StateFeatures map[string]float64 `json:"state_features"`
}

// transKey builds the map key for a label transition.
func transKey(from, to string) string { return from + "\t" + to }

// LoadModel reads model_weights.json produced by export_model.py.
func LoadModel(path string) (*CRFModel, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }
    var m CRFModel
    if err := json.Unmarshal(data, &m); err != nil {
        return nil, err
    }
    return &m, nil
}

// StateScoreBW computes the B_W score for a feature dict.
// Score for I_W = -StateScoreBW (antisymmetry property of the trained model).
//
// Feature encoding rules (mirrors sklearn-crfsuite convention):
//   - bool True  → key = feature name alone,   e.g. "word.isupper()"
//   - bool False → feature inactive, contributes 0
//   - string     → key = name + ":" + value,   e.g. "word.lower():học"
//   - float 1.0  → key = feature name alone,   e.g. "bias"
func (m *CRFModel) StateScoreBW(features map[string]any) float64 {
    var total float64
    for key, val := range features {
        switch v := val.(type) {
        case bool:
            if v {
                total += m.StateFeatures[key]
            }
        case string:
            total += m.StateFeatures[key+":"+v]
        case float64:
            total += m.StateFeatures[key] * v
        }
    }
    return total
}

// Viterbi decodes the most likely label sequence for a syllable sequence.
// Returns a slice of "B_W" / "I_W" labels, one per syllable.
//
// Since score(I_W) = -score(B_W), each position only needs one score value.
// The DP tracks two paths (ending in B_W or I_W) across the sequence.
func (m *CRFModel) Viterbi(syllables []string, features []map[string]any) []string {
    n := len(syllables)
    if n == 0 {
        return nil
    }

    // Transition weights
    trBB := m.Transitions[transKey("B_W", "B_W")]
    trBI := m.Transitions[transKey("B_W", "I_W")]
    trIB := m.Transitions[transKey("I_W", "B_W")]
    trII := m.Transitions[transKey("I_W", "I_W")]

    // Precompute B_W state scores; I_W score = -bwScore
    bwScores := make([]float64, n)
    for i, f := range features {
        bwScores[i] = m.StateScoreBW(f)
    }

    // dp[0]=score ending in B_W, dp[1]=score ending in I_W
    dp := [2]float64{bwScores[0], -bwScores[0]}
    // backtrack[i][label] = which label at i-1 led to best score
    back := make([][2]int, n)

    for i := 1; i < n; i++ {
        bw := bwScores[i]
        iw := -bw

        // From B_W (dp[0])
        fromB_toB := dp[0] + trBB + bw
        fromB_toI := dp[0] + trBI + iw
        // From I_W (dp[1])
        fromI_toB := dp[1] + trIB + bw
        fromI_toI := dp[1] + trII + iw

        newB, predB := maxOf(fromB_toB, 0, fromI_toB, 1)
        newI, predI := maxOf(fromB_toI, 0, fromI_toI, 1)

        dp = [2]float64{newB, newI}
        back[i] = [2]int{predB, predI}
    }

    // Backtrack
    path := make([]int, n)
    if dp[0] >= dp[1] {
        path[n-1] = 0 // B_W
    } else {
        path[n-1] = 1 // I_W
    }
    for i := n - 1; i > 0; i-- {
        path[i-1] = back[i][path[i]]
    }

    // Convert int indices to label strings
    labels := make([]string, n)
    lmap := [2]string{"B_W", "I_W"}
    for i, idx := range path {
        labels[i] = lmap[idx]
    }
    return labels
}

func maxOf(a float64, ai int, b float64, bi int) (float64, int) {
    if a >= b {
        return a, ai
    }
    return b, bi
}

// init guard for unused import
var _ = math.MaxFloat64
```

### `vitoken/features.go` — Feature extraction

```go
package vitoken

import (
    "bufio"
    "os"
    "strings"
    "unicode"
)

// Tokenizer wraps the CRF model and the bi/trigram dictionaries.
type Tokenizer struct {
    model    *CRFModel
    bigrams  map[string]bool
    trigrams map[string]bool
}

// New creates a Tokenizer from model JSON + words.txt paths.
func New(modelPath, wordsPath string) (*Tokenizer, error) {
    m, err := LoadModel(modelPath)
    if err != nil {
        return nil, err
    }
    bi, tri, err := loadWords(wordsPath)
    if err != nil {
        return nil, err
    }
    return &Tokenizer{model: m, bigrams: bi, trigrams: tri}, nil
}

// loadWords parses words.txt into bigram and trigram sets.
// Keys are lowercase, space-separated syllables: "học sinh", "bách khoa hà nội".
func loadWords(path string) (bigrams, trigrams map[string]bool, err error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, nil, err
    }
    defer f.Close()

    bigrams = make(map[string]bool)
    trigrams = make(map[string]bool)
    scanner := bufio.NewScanner(f)
    for scanner.Scan() {
        line := strings.TrimSpace(scanner.Text())
        parts := strings.Fields(line)
        switch len(parts) {
        case 2:
            bigrams[strings.ToLower(line)] = true
        case 3:
            trigrams[strings.ToLower(line)] = true
        }
    }
    return bigrams, trigrams, scanner.Err()
}

// word2features builds the CRF feature dict for syllable at position i.
// Mirrors pyvi/ViTokenizer.py:ViTokenizer.word2features() exactly.
//
// Example (inference, sent = ["Chính", "phủ", "Việt", "Nam"], i=1):
//
//	{
//	  "bias":             1.0,
//	  "word.lower()":     "phủ",
//	  "word.isupper()":   false,
//	  "word.istitle()":   false,
//	  "word.isdigit()":   false,
//	  "-1:word.lower()":  "chính",
//	  "-1:word.istitle()": true,
//	  "-1:word.isupper()": false,
//	  "-1:word.bi_gram()": false,    // "chính phủ" IS in bigrams → True for this example
//	  "+1:word.lower()":  "việt",
//	  "+1:word.istitle()": true,
//	  "+1:word.isupper()": false,
//	  "+1:word.bi_gram()": false,
//	  "-2:word.tri_gram()": false,
//	  "+2:word.tri_gram()": false,
//	}
func (t *Tokenizer) word2features(sent []string, i int) map[string]any {
    word := sent[i]
    lower := strings.ToLower(word)

    f := map[string]any{
        "bias":           1.0,
        "word.lower()":   lower,
        "word.isupper()": isUpper(word),
        "word.istitle()": isTitle(word),
        "word.isdigit()": isDigit(word),
    }

    // ── Left context ────────────────────────────────────────────────
    if i > 0 {
        w1 := sent[i-1]
        l1 := strings.ToLower(w1)
        f["-1:word.lower()"] = l1
        f["-1:word.istitle()"] = isTitle(w1)
        f["-1:word.isupper()"] = isUpper(w1)
        f["-1:word.bi_gram()"] = t.bigrams[l1+" "+lower]

        if i > 1 {
            w2 := sent[i-2]
            l2 := strings.ToLower(w2)
            f["-2:word.tri_gram()"] = t.trigrams[l2+" "+l1+" "+lower]
        }
    }

    // ── Right context ────────────────────────────────────────────────
    if i < len(sent)-1 {
        w1 := sent[i+1]
        l1 := strings.ToLower(w1)
        f["+1:word.lower()"] = l1
        f["+1:word.istitle()"] = isTitle(w1)
        f["+1:word.isupper()"] = isUpper(w1)
        f["+1:word.bi_gram()"] = t.bigrams[lower+" "+l1]

        if i < len(sent)-2 {
            w2 := sent[i+2]
            l2 := strings.ToLower(w2)
            f["+2:word.tri_gram()"] = t.trigrams[lower+" "+l1+" "+l2]
        }
    }

    return f
}

// sentFeatures applies word2features to every syllable in sent.
func (t *Tokenizer) sentFeatures(sent []string) []map[string]any {
    out := make([]map[string]any, len(sent))
    for i := range sent {
        out[i] = t.word2features(sent, i)
    }
    return out
}

// ── Unicode helpers ────────────────────────────────────────────────────────
// These mirror Python's str.isupper() / str.istitle() / str.isdigit() semantics.

// isUpper returns true if every cased rune is uppercase and there is at least one cased rune.
// Mirrors Python's str.isupper().
func isUpper(s string) bool {
    hasCased := false
    for _, r := range s {
        if unicode.IsLetter(r) {
            hasCased = true
            if !unicode.IsUpper(r) {
                return false
            }
        }
    }
    return hasCased
}

// isTitle returns true if the first cased rune is uppercase and the rest are lowercase.
// Mirrors Python's str.istitle() for single-word tokens (good enough for syllables).
func isTitle(s string) bool {
    runes := []rune(s)
    if len(runes) == 0 {
        return false
    }
    firstCased := false
    for i, r := range runes {
        if unicode.IsLetter(r) {
            if !firstCased {
                if !unicode.IsUpper(r) {
                    return false
                }
                firstCased = true
            } else {
                // After first uppercase, must be lowercase
                if i > 0 && unicode.IsUpper(r) {
                    return false
                }
            }
        }
    }
    return firstCased
}

// isDigit returns true if all runes are digits.
// Mirrors Python's str.isdigit().
func isDigit(s string) bool {
    if len(s) == 0 {
        return false
    }
    for _, r := range s {
        if !unicode.IsDigit(r) {
            return false
        }
    }
    return true
}
```

### `vitoken/tokenize.go` — Syllabelize + post-processing

```go
package vitoken

import (
    "regexp"
    "strings"
    "unicode"

    "golang.org/x/text/unicode/norm"
)

// Compiled regex patterns — built at init time, same priority order as Python.
// Key difference from Python re: Go RE2 uses \p{L} for Unicode letters (not \w).
var (
    // Combined alternation pattern (priority: abbreviations first, word last)
    syllableRE = regexp.MustCompile(
        `(?:` +
            // 1. Abbreviations (highest priority — must come before word rule)
            `[A-ZĐ]+\.` + `|` +
            `Tp\.` + `|` +
            `Mr\.|Mrs\.|Ms\.|Dr\.|ThS\.` + `|` +

            // 2. Special arrows / symbols
            `==>|->|\.\.\.|\n|>>` + `|` +

            // 3. URL  (e.g. https://vnexpress.net)
            `[\p{L}\p{N}_]+://\S+` + `|` +

            // 4. Email  (e.g. user@domain.com)
            `[\w.+-]+@[\w-]+(?:\.[\w-]+)+` + `|` +

            // 5. Numbers with separators  (e.g. 1,000,000 or 5.2)
            `\d+(?:[.,_]\d+)+` + `|` +

            // 6. Non-word characters (punctuation, symbols)
            `[^\p{L}\p{N}_\s]` + `|` +

            // 7. Words / syllables (lowest priority)
            `[\p{L}\p{N}_]+` +
            `)`,
    )

    // Punctuation set for post-processing join suppression
    asciiPunct = func() map[rune]bool {
        m := make(map[rune]bool)
        for _, r := range `!"#$%&'()*+,-./:;<=>?@[\]^_{|}~` + "`" {
            m[r] = true
        }
        return m
    }()
)

// Sylabelize splits text into atomic syllable units.
// Returns (normalizedText, syllables).
//
// Examples:
//
//	Sylabelize("Hà Nội là thủ đô")
//	→ ("Hà Nội là thủ đô", ["Hà","Nội","là","thủ","đô"])
//
//	Sylabelize("Mr. Trung gửi abc@gmail.com")
//	→ (..., ["Mr.","Trung","gửi","abc@gmail.com"])
//
//	Sylabelize("1,000,000 VND tăng 5.2%")
//	→ (..., ["1,000,000","VND","tăng","5.2","%"])
func Sylabelize(text string) (string, []string) {
    // NFC: precomposed form — ensures "ọ" is 1 code point, not "o" + combining mark
    normalized := norm.NFC.String(text)
    matches := syllableRE.FindAllString(normalized, -1)
    return normalized, matches
}

// Tokenize performs full Vietnamese word segmentation.
//
// Pipeline: Sylabelize → word2features → Viterbi → post-process
//
// Output: space-separated tokens; compound words joined with "_".
//
// Examples:
//
//	Tokenize("Chính phủ Việt Nam") → "Chính_phủ Việt_Nam"
//	Tokenize("con bò đang ăn cỏ") → "con bò đang ăn cỏ"
//	Tokenize("test@gmail.com và https://example.com") → (preserved)
func (t *Tokenizer) Tokenize(text string) string {
    _, syllables := Sylabelize(text)
    if len(syllables) == 0 {
        return text
    }

    features := t.sentFeatures(syllables)
    labels := t.model.Viterbi(syllables, features)

    // Post-process: join I_W syllables with "_", separate B_W with " "
    var b strings.Builder
    b.WriteString(syllables[0])

    for i := 1; i < len(labels); i++ {
        curr := syllables[i]
        prev := syllables[i-1]

        join := labels[i] == "I_W" &&
            !isPunct(prev) &&
            !isPunct(curr) &&
            !startsWithDigit(curr) &&
            !startsWithDigit(prev) &&
            // Suppress join when curr starts a new proper noun
            !(isTitle(curr) && !isTitle(prev))

        if join {
            b.WriteByte('_')
        } else {
            b.WriteByte(' ')
        }
        b.WriteString(curr)
    }
    return b.String()
}

// isPunct reports whether s is a single ASCII punctuation character.
func isPunct(s string) bool {
    runes := []rune(s)
    return len(runes) == 1 && asciiPunct[runes[0]]
}

func startsWithDigit(s string) bool {
    if len(s) == 0 {
        return false
    }
    return unicode.IsDigit([]rune(s)[0])
}
```

### `vitoken/main_example.go` — Cách dùng

```go
package main

import (
    "fmt"
    "log"

    "your_module/vitoken"
)

func main() {
    t, err := vitoken.New(
        "pyvi/models/model_weights.json",
        "pyvi/models/words.txt",
    )
    if err != nil {
        log.Fatal(err)
    }

    sentences := []string{
        "Chính phủ Việt Nam ban hành nghị quyết",
        "Tôi đang học máy học tại đại học bách khoa",
        "Email test@gmail.com và https://vnexpress.net",
        "Mr. Trung gửi báo cáo cho ThS. Lan",
    }

    for _, s := range sentences {
        fmt.Printf("IN : %s\n", s)
        fmt.Printf("OUT: %s\n\n", t.Tokenize(s))
    }
}
```

---

## Bước 3 — Go module setup

```bash
# Tạo module
go mod init your_module
go get golang.org/x/text/unicode/norm

# Build
go build ./...

# Test
go test ./vitoken/...
```

---

## Bước 4 — Validation: so sánh Go vs Python

```python
# validate.py — chạy song song để so sánh
import subprocess, sys, json
sys.path.insert(0, '.')
from pyvi import ViTokenizer

test_sentences = [
    "Chính phủ Việt Nam ban hành nghị quyết",
    "học sinh đại học bách khoa",
    "Mr. Trung gửi email tới abc@gmail.com",
    "Hà Nội là thủ đô của Việt Nam",
]

for s in test_sentences:
    py_out = ViTokenizer.tokenize(s)
    # go_out = subprocess.check_output(['./vitoken_cli', s]).decode().strip()
    print(f"Input: {s}")
    print(f"  Python: {py_out}")
    # print(f"  Go:     {go_out}")
    # print(f"  Match:  {py_out == go_out}")
    print()
```

---

## Phân tích khả năng sai lệch Python vs Go

Các điểm có thể cho output khác nhau:

| Vấn đề | Python | Go | Mức độ ảnh hưởng |
|--------|--------|-----|-----------------|
| Unicode normalization | `unicodedata.normalize('NFC')` | `norm.NFC.String()` | Thấp (cùng chuẩn NFC) |
| `str.isupper()` | Built-in Python | Custom Go function | Trung bình — test kỹ |
| `str.istitle()` | Built-in Python | Custom Go function | Trung bình — test kỹ |
| Regex `\w+` (UNICODE) | Matches Vietnamese letters | `[\p{L}\p{N}_]+` | Thấp nếu dùng đúng |
| Float precision | 64-bit Python float | `float64` Go | Cực thấp (cùng IEEE 754) |
| JSON number precision | Python json | Go encoding/json | Thấp |

**Khuyến nghị:** Verify 100+ câu trước khi deploy. Đặc biệt test `isTitle()` và `isUpper()` vì Python's `str.istitle()` có edge cases với mixed Unicode scripts.

---

## Kết quả kỳ vọng

```
Exported model:   221 KB (model_weights.json) + 377 KB (words.txt) = ~600 KB total
Go binary:        ~5-8 MB (static linked)
Startup time:     ~50-100ms (load JSON + words.txt vào memory)
Inference speed:  >10,000 sentences/second (ước tính — Go nhanh hơn Python CRF ~3-5x)
Memory:           ~30-50 MB RSS (bigrams/trigrams + model in memory)
```

---

## Tóm tắt

```
Python (một lần)        Go (mãi mãi)
─────────────────       ──────────────────────────────────────────
pyvi3.pkl  ──export──►  model_weights.json  ──load──►  CRFModel
vtb.txt                 words.txt           ──load──►  bi/trigrams
                                                         │
                                            Raw text ──► Sylabelize
                                                         │
                                                       word2features
                                                         │
                                                       Viterbi (DP)
                                                         │
                                                       Post-process
                                                         │
                                                    "Chính_phủ Việt_Nam"
```

**Không có Python runtime nào được gọi sau bước export.**
