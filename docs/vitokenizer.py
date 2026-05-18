"""
ViTokenizer — Vietnamese word segmentation using CRF (Conditional Random Field).

Architecture
------------
1. sylabelize()   : split raw text into syllable units via regex
2. word2features(): convert each syllable into a feature dict for CRF
3. sent2features(): apply word2features over the whole sequence
4. tokenize()     : run the full pipeline, return segmented string

Output convention
-----------------
Compound words (multiple syllables) are joined with underscores:
  "học sinh"  →  "học_sinh"
  "Việt Nam"  →  "Việt_Nam"
Single-syllable words and punctuation are separated by spaces.
"""

import sys
import os
import codecs
import pickle
import re
import string
import unicodedata as ud
from typing import Dict, List, Set, Tuple, Union


class ViTokenizer:
    """CRF-based Vietnamese word segmenter.

    Class-level attributes are loaded once at import time:
      - ``bi_grams``  : set of known 2-syllable compound words from words.txt
      - ``tri_grams`` : set of known 3-syllable compound words from words.txt
      - ``model``     : pre-trained CRF classifier (sklearn-crfsuite)

    All public methods are ``@staticmethod`` — no instance needed.

    Example
    -------
    >>> from pyvi import ViTokenizer
    >>> ViTokenizer.tokenize("Chính phủ Việt Nam ban hành nghị quyết")
    'Chính_phủ Việt_Nam ban hành nghị_quyết'
    """

    # ── Class-level data loaded once at import ────────────────────────────
    bi_grams: Set[str] = set()   # e.g. {"học sinh", "đại học", "Việt Nam", ...}
    tri_grams: Set[str] = set()  # e.g. {"bách khoa hà nội", "xử lý ngôn ngữ", ...}

    model_file: str = 'models/pyvi.pkl'
    if sys.version_info[0] == 3:
        model_file = 'models/pyvi3.pkl'

    # Load bi-gram / tri-gram dictionary from words.txt
    with codecs.open(os.path.join(os.path.dirname(__file__), 'models/words.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            tmp = token.split(' ')
            if len(tmp) == 2:
                bi_grams.add(token)
            elif len(tmp) == 3:
                tri_grams.add(token)

    # Load pre-trained CRF model (pyvi3.pkl for Python 3, pyvi.pkl for Python 2)
    with open(os.path.join(os.path.dirname(__file__), model_file), 'rb') as fin:
        model = pickle.load(fin)

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def word2features(
        sent: Union[List[str], List[Tuple[str, str]]],
        i: int,
        is_training: bool,
    ) -> Dict[str, Union[str, bool, float]]:
        """Build the CRF feature dictionary for the syllable at position ``i``.

        Features capture the surface form of the current syllable plus left/right
        context (up to 2 positions) and dictionary lookup against known bi-grams
        and tri-grams.

        Parameters
        ----------
        sent : list
            During **inference** (``is_training=False``): ``List[str]``, e.g.
            ``['Chính', 'phủ', 'Việt', 'Nam']``.
            During **training** (``is_training=True``): ``List[Tuple[str, str]]``,
            e.g. ``[('Chính', 'B_W'), ('phủ', 'I_W'), ...]``, where each element
            is a (syllable, label) pair.
        i : int
            Index of the target syllable within ``sent`` (0-based).
        is_training : bool
            ``True``  → ``sent`` contains (syllable, label) tuples.
            ``False`` → ``sent`` contains plain syllable strings.

        Returns
        -------
        dict
            Feature dictionary with string keys and bool/float/str values,
            ready to pass directly to ``sklearn_crfsuite``.

        Examples
        --------
        Inference mode — 4-syllable sentence, extracting features at each position:

        >>> sent = ['Chính', 'phủ', 'Việt', 'Nam']

        Position 0 — first syllable, no left context:
        >>> ViTokenizer.word2features(sent, 0, False)
        {
          'bias': 1.0,
          'word.lower()': 'chính',
          'word.isupper()': False,
          'word.istitle()': True,    # title-case → likely start of proper noun
          'word.isdigit()': False,
          '+1:word.lower()': 'phủ',
          '+1:word.istitle()': False,
          '+1:word.isupper()': False,
          '+1:word.bi_gram()': False, # "chính phủ" IS in bi_grams → True when lowercased
          '+2:word.tri_gram()': False
        }
        # Note: 'chính phủ' must be lowercase to match words.txt entries.

        Position 1 — middle syllable, full left+right context:
        >>> ViTokenizer.word2features(sent, 1, False)
        {
          'bias': 1.0,
          'word.lower()': 'phủ',
          'word.isupper()': False,
          'word.istitle()': False,
          'word.isdigit()': False,
          '-1:word.lower()': 'chính',
          '-1:word.istitle()': True,
          '-1:word.isupper()': False,
          '-1:word.bi_gram()': False,  # "chính phủ" checked lowercase
          '+1:word.lower()': 'việt',
          '+1:word.istitle()': True,
          '+1:word.isupper()': False,
          '+1:word.bi_gram()': False,
          '-2:word.tri_gram()': False,
          '+2:word.tri_gram()': False
        }

        Training mode — same sentence with labels:
        >>> sent_train = [('Chính', 'B_W'), ('phủ', 'I_W'), ('Việt', 'B_W'), ('Nam', 'I_W')]
        >>> ViTokenizer.word2features(sent_train, 0, True)
        # identical output — only the word extraction line differs:
        #   inference: word = sent[i]
        #   training:  word = sent[i][0]
        """
        # In training mode each element is (syllable, label); in inference mode it's just a string.
        word: str = sent[i][0] if is_training else sent[i]

        # ── Current-syllable surface features ────────────────────────────
        features: Dict[str, Union[str, bool, float]] = {
            'bias': 1.0,                        # constant offset term for CRF
            'word.lower()': word.lower(),        # lowercase form — main lexical signal
            'word.isupper()': word.isupper(),    # ALL-CAPS (e.g. VNM, IMB) → likely abbreviation
            'word.istitle()': word.istitle(),    # Title-case (e.g. Hà, Nội) → proper noun hint
            'word.isdigit()': word.isdigit(),    # digit-only → suppresses compound joining
        }

        # ── Left context: i-1 ────────────────────────────────────────────
        if i > 0:
            word1: str = sent[i - 1][0] if is_training else sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                # Dictionary lookup: is (prev, curr) a known compound?
                # True  → strong signal for I_W (current syllable continues prev word)
                # False → no evidence from dictionary, rely on other features
                '-1:word.bi_gram()': ' '.join([word1, word]).lower() in ViTokenizer.bi_grams,
            })

            # ── Left context: i-2 (tri-gram check) ───────────────────────
            if i > 1:
                word2: str = sent[i - 2][0] if is_training else sent[i - 2]
                features.update({
                    # True if (i-2, i-1, i) forms a known 3-syllable compound
                    '-2:word.tri_gram()': ' '.join([word2, word1, word]).lower() in ViTokenizer.tri_grams,
                })

        # ── Right context: i+1 ───────────────────────────────────────────
        if i < len(sent) - 1:
            word1 = sent[i + 1][0] if is_training else sent[i + 1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                # Dictionary lookup: is (curr, next) a known compound?
                '+1:word.bi_gram()': ' '.join([word, word1]).lower() in ViTokenizer.bi_grams,
            })

            # ── Right context: i+2 (tri-gram check) ──────────────────────
            if i < len(sent) - 2:
                word2 = sent[i + 2][0] if is_training else sent[i + 2]
                features.update({
                    # True if (i, i+1, i+2) forms a known 3-syllable compound
                    '+2:word.tri_gram()': ' '.join([word, word1, word2]).lower() in ViTokenizer.tri_grams,
                })

        return features

    @staticmethod
    def sent2features(
        sent: Union[List[str], List[Tuple[str, str]]],
        is_training: bool,
    ) -> List[Dict[str, Union[str, bool, float]]]:
        """Convert a full syllable sequence into a list of CRF feature dicts.

        This is a simple map of :func:`word2features` over every position,
        producing the feature matrix for one sentence that ``sklearn_crfsuite``
        can consume directly.

        Parameters
        ----------
        sent : list
            Same dual-mode list as in :func:`word2features`.
            - Inference: ``List[str]``       e.g. ``['Chính', 'phủ', 'Việt', 'Nam']``
            - Training : ``List[Tuple[str,str]]``  e.g. ``[('Chính','B_W'), ...]``
        is_training : bool
            Passed through to every :func:`word2features` call.

        Returns
        -------
        list of dict
            One feature dict per syllable, in order.
            ``len(result) == len(sent)`` always.

        Examples
        --------
        >>> sent = ['Chính', 'phủ', 'Việt', 'Nam']
        >>> feats = ViTokenizer.sent2features(sent, False)
        >>> len(feats)
        4
        >>> feats[0]['word.lower()']
        'chính'
        >>> feats[1]['-1:word.lower()']
        'chính'
        >>> feats[3].get('+1:word.lower()')   # last position has no right context
        # key absent → None
        """
        return [ViTokenizer.word2features(sent, i, is_training) for i in range(len(sent))]

    @staticmethod
    def sylabelize(text: str) -> Tuple[str, List[str]]:
        """Tokenize raw text into atomic syllable units before CRF inference.

        Applies Unicode NFC normalization then uses a priority-ordered regex to
        keep "atomic" tokens intact — abbreviations, URLs, emails, and numbers
        with separators are never split further.

        Regex priority order (high → low)
        ----------------------------------
        1. Abbreviations: ``[A-ZĐ]+.``  ``Tp.``  ``Mr.``  ``Mrs.``  ``Ms.``
                          ``Dr.``  ``ThS.``
        2. Arrow/special symbols: ``==>``  ``->``  ``...``  ``>>``  newlines
        3. URLs:  ``\\w+://[^\\s]+``   (e.g. ``https://vnexpress.net``)
        4. Emails: ``user@domain.tld``
        5. Numbers with separators: ``\\d+([.,_]\\d+)+``  (e.g. ``1,000,000``)
        6. Punctuation/non-word: ``[^\\w\\s]``
        7. Words/syllables: ``\\w+``

        Parameters
        ----------
        text : str
            Raw Vietnamese text (may contain diacritics, URLs, numbers, etc.)

        Returns
        -------
        tuple[str, list[str]]
            ``(normalized_text, syllables)`` where:
            - ``normalized_text`` is the NFC-normalized version of ``text``
              (used later by :func:`spacy_tokenize` for space alignment).
            - ``syllables`` is the list of atomic tokens; each will become one
              row in the CRF feature matrix.

        Examples
        --------
        >>> ViTokenizer.sylabelize('Hà Nội là thủ đô')
        ('Hà Nội là thủ đô', ['Hà', 'Nội', 'là', 'thủ', 'đô'])

        >>> ViTokenizer.sylabelize('Mr. Trung gửi email tới abc@gmail.com')
        ('Mr. Trung gửi email tới abc@gmail.com',
         ['Mr.', 'Trung', 'gửi', 'email', 'tới', 'abc@gmail.com'])
        # 'Mr.' kept intact (abbreviation rule); email kept intact (email rule)

        >>> ViTokenizer.sylabelize('Giá: 1,000,000 VND tăng 5.2%')
        ('Giá: 1,000,000 VND tăng 5.2%',
         ['Giá', ':', '1,000,000', 'VND', 'tăng', '5.2', '%'])
        # '1,000,000' kept intact (number-with-separator rule)
        # '5.2' and '%' are separate — digit rule matches '5.2' but '%' is non-word

        >>> ViTokenizer.sylabelize('Link: https://vnexpress.net/tin-tuc')
        ('Link: https://vnexpress.net/tin-tuc',
         ['Link', ':', 'https://vnexpress.net/tin-tuc'])
        # URL kept intact (URL rule fires before word rule)

        >>> ViTokenizer.sylabelize('TP. Hồ Chí Minh')
        ('TP. Hồ Chí Minh', ['TP.', 'Hồ', 'Chí', 'Minh'])
        # 'TP.' matched by abbreviation rule [A-ZĐ]+.
        """
        # NFC normalization: precomposed form ensures consistent byte sequences
        # for Vietnamese diacritics (ợ as single code point, not o + combining mark)
        text = ud.normalize('NFC', text)

        # ── Pattern definitions (order matters — earlier patterns win) ────
        specials   = ["==>", "->", r"\.\.\.", ">>", '\n']
        digit      = r"\d+([\.,_]\d+)+"           # numbers with decimal/thousand sep
        email      = r"([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
        web        = r"\w+://[^\s]+"               # URLs (http://, https://, ftp://, ...)
        word       = r"\w+"                        # any word character sequence (incl. Vietnamese)
        non_word   = r"[^\w\s]"                    # punctuation / symbols
        abbreviations = [
            r"[A-ZĐ]+\.",       # all-caps abbrev: TP., IMB., GDP., ĐHBK.
            r"Tp\.",            # title-case city: Tp.
            r"Mr\.", r"Mrs\.", r"Ms\.",
            r"Dr\.", r"ThS\.",
        ]

        # Build combined alternation pattern; abbreviations have highest priority
        patterns: List[str] = []
        patterns.extend(abbreviations)
        patterns.extend(specials)
        patterns.extend([web, email])
        patterns.extend([digit, non_word, word])

        combined: str = "(" + "|".join(patterns) + ")"

        # Python 2 compat: decode bytes → str (no-op in Python 3)
        if sys.version_info < (3, 0):
            combined = combined.decode('utf-8')

        # re.findall returns list of tuples because the outer group + inner groups
        # each match produces a tuple; token[0] is the full outer match.
        tokens = re.findall(combined, text, re.UNICODE)
        return text, [token[0] for token in tokens]

    @staticmethod
    def tokenize(str: str) -> str:
        """Segment Vietnamese text into words, joining compound syllables with ``_``.

        Full pipeline:
          1. ``sylabelize`` → split into syllable units
          2. ``sent2features`` → convert to CRF feature matrix
          3. ``model.predict`` → get B_W / I_W label per syllable
          4. Post-process → join consecutive I_W syllables with ``_``, skip if:
             - either syllable is punctuation
             - either syllable starts with a digit
             - the next syllable is Title-case while the current is not
               (heuristic: new proper noun starting → break the compound)

        Parameters
        ----------
        str : str
            Raw Vietnamese input sentence.

        Returns
        -------
        str
            Space-separated tokens where multi-syllable compounds use ``_``.
            Returns the original ``str`` unchanged if no syllables are found.

        Examples
        --------
        >>> ViTokenizer.tokenize('học sinh đại học')
        'học sinh đại học'
        # 'học sinh' and 'đại học' NOT joined — both missing from bi_grams in this test
        # (accuracy depends on the trained model; results may vary)

        >>> ViTokenizer.tokenize('Chính phủ Việt Nam')
        'Chính_phủ Việt_Nam'
        # 'chính phủ' and 'việt nam' are in bi_grams → joined

        >>> ViTokenizer.tokenize('xử lý ngôn ngữ tự nhiên')
        'xử_lý_ngôn ngữ tự nhiên'
        # Known limitation: 'xử_lý_ngôn' is a mis-segmentation; correct is 'xử_lý ngôn_ngữ'

        >>> ViTokenizer.tokenize('con bò đang ăn cỏ')
        'con bò đang ăn cỏ'

        >>> ViTokenizer.tokenize('Email test@gmail.com và https://example.com')
        'Email test@gmail.com và https://example.com'
        # URLs and emails are never split by sylabelize → pass through as-is

        >>> ViTokenizer.tokenize('')
        ''
        # Empty or whitespace-only input returns original string

        Notes
        -----
        Compound joining is **suppressed** even when the model predicts I_W if:
        - ``tmp[i]`` or ``tmp[i-1]`` is ASCII punctuation (``string.punctuation``)
        - Either syllable starts with a digit (avoids "5_km", "2_người")
        - ``tmp[i]`` is Title-case but ``tmp[i-1]`` is not (new proper noun boundary)
        """
        text, tmp = ViTokenizer.sylabelize(str)
        if len(tmp) == 0:
            return str

        # CRF predicts a label sequence for the whole sentence at once
        # labels[0] is the label list for the first (only) sentence in the batch
        labels: List[List[str]] = ViTokenizer.model.predict(
            [ViTokenizer.sent2features(tmp, False)]
        )

        output: str = tmp[0]
        for i in range(1, len(labels[0])):
            # Decide whether to join syllable i onto the previous token
            join: bool = (
                labels[0][i] == 'I_W'
                and tmp[i]   not in string.punctuation   # no punct on right
                and tmp[i-1] not in string.punctuation   # no punct on left
                and not tmp[i][0].isdigit()              # right not a digit
                and not tmp[i-1][0].isdigit()            # left not a digit
                # Suppress joining when the next syllable starts a new proper noun
                # (Title-case after non-Title-case strongly suggests a word boundary)
                and not (tmp[i][0].istitle() and not tmp[i-1][0].istitle())
            )
            output += ('_' if join else ' ') + tmp[i]

        return output

    @staticmethod
    def spacy_tokenize(str: str) -> Tuple[List[str], List[bool]]:
        """Segment text and return spaCy-compatible (tokens, spaces) pair.

        Identical segmentation logic to :func:`tokenize` but returns the result
        in the format expected by ``spacy.tokens.Doc(vocab, words=..., spaces=...)``.

        Parameters
        ----------
        str : str
            Raw Vietnamese input sentence.

        Returns
        -------
        tuple[list[str], list[bool]]
            ``(tokens, spaces)`` where:
            - ``tokens`` : list of word strings (compound words joined with ``_``)
            - ``spaces`` : parallel bool list; ``True`` if the original text had a
              space **after** that token, ``False`` otherwise.
            Returns ``([], [])`` if no syllables are found.

        Examples
        --------
        >>> tokens, spaces = ViTokenizer.spacy_tokenize('Chính phủ Việt Nam ban hành')
        >>> tokens
        ['Chính_phủ', 'Việt_Nam_ban', 'hành']
        >>> spaces
        [True, True, False]
        # Note: 'Việt_Nam_ban' is a mis-segmentation — illustrates model limitations

        >>> tokens, spaces = ViTokenizer.spacy_tokenize('Trường đại học bách khoa hà nội')
        >>> tokens
        ['Trường', 'đại', 'học', 'bách', 'khoa', 'hà_nội']
        >>> spaces
        [True, True, True, True, True, False]

        Notes
        -----
        The space detection walks through ``normalized_text`` character by character,
        advancing by ``len(token)`` after each token and checking whether the next
        character is a space. This is slightly fragile when a token contains ``_``
        (the underscore is not in the original text), so space alignment may be
        off for compound tokens — use with care when exact whitespace matters.
        """
        text, tmp = ViTokenizer.sylabelize(str)
        if len(tmp) == 0:
            return [], []

        labels: List[List[str]] = ViTokenizer.model.predict(
            [ViTokenizer.sent2features(tmp, False)]
        )

        # Build token list (same join logic as tokenize())
        token: str = tmp[0]
        tokens: List[str] = []
        spaces: List[bool] = []

        for i in range(1, len(labels[0])):
            join: bool = (
                labels[0][i] == 'I_W'
                and tmp[i]   not in string.punctuation
                and tmp[i-1] not in string.punctuation
                and not tmp[i][0].isdigit()
                and not tmp[i-1][0].isdigit()
                and not (tmp[i][0].istitle() and not tmp[i-1][0].istitle())
            )
            if join:
                token = token + '_' + tmp[i]
            else:
                tokens.append(token)
                token = tmp[i]
        tokens.append(token)

        # ── Space alignment: walk normalized_text to find post-token spaces ──
        # We advance cursor i by len(token) (without the _ glue chars) and check
        # whether the original text has a space at that position.
        i = 0
        for token in tokens:
            i = i + len(token)
            if i < len(text) and text[i] == ' ':
                spaces.append(True)
                i += 1
            else:
                spaces.append(False)

        return tokens, spaces


# ── Module-level convenience wrappers ────────────────────────────────────────
# These allow ``from pyvi.ViTokenizer import tokenize`` without instantiating
# the class or using the fully qualified ``ViTokenizer.tokenize(...)`` form.

def spacy_tokenize(str: str) -> Tuple[List[str], List[bool]]:
    """Module-level alias for :meth:`ViTokenizer.spacy_tokenize`."""
    return ViTokenizer.spacy_tokenize(str)


def tokenize(str: str) -> str:
    """Module-level alias for :meth:`ViTokenizer.tokenize`."""
    return ViTokenizer.tokenize(str)
