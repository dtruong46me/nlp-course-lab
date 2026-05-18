"""
train_tokenizer.py — Train the CRF word-segmentation model for PyVi.

This is a **run-once script**, not an importable module.  It expects four
pre-built pickle files in the working directory and writes two model files:

Input files (must exist before running)
----------------------------------------
tokenized_X_train.pkl  : List[List[Dict]]  — feature matrix for training sentences
tokenized_X_test.pkl   : List[List[Dict]]  — feature matrix for test sentences
tokenized_y_train.pkl  : List[List[str]]   — B_W/I_W label sequences (train)
tokenized_y_test.pkl   : List[List[str]]   — B_W/I_W label sequences (test)

Each element in X is one sentence represented as a list of feature dicts
(one dict per syllable), as produced by ``ViTokenizer.sent2features()``.
Each element in y is the corresponding list of 'B_W' / 'I_W' labels.

Output files
------------
tokenizer_model.pkl     : CRF model serialized with pickle protocol 2 (Python 2 compat)
tokenizer_model_py3.pkl : CRF model serialized with default protocol (Python 3)

Usage
-----
    python pyvi/train_tokenizer.py

Typical training time: 5-20 min (50 CV iterations × 5 folds, 12 parallel jobs).
Expected best CV F1: ~0.985 on VLSP 2016 data.

Label vocabulary
----------------
B_W  Beginning of Word  — first syllable of every word/compound
I_W  Inside Word        — 2nd, 3rd, ... syllable of a multi-syllable compound

Example sequence
----------------
Input:   ['học', 'sinh', 'trường', 'đại', 'học']   (5 syllables)
Labels:  ['B_W', 'I_W', 'B_W',   'B_W', 'I_W']
Output:  "học_sinh trường đại_học"
"""

from pyvi import ViTokenizer

import pickle
from typing import List, Dict, Any

import sklearn_crfsuite
from sklearn_crfsuite import metrics          # flat_f1_score for sequence labels
import scipy.stats                            # expon distribution for hyperparam search
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV


# ── 1. Load pre-built feature matrices and label sequences ────────────────────
# These files are generated externally (e.g. from train_tokenizer.ipynb) by
# parsing the Vietnamese Treebank (vtb.txt) or VLSP corpus through
# ViTokenizer.sent2features() in training mode.

tokenized_X_train = open('tokenized_X_train.pkl', 'rb')
tokenized_X_test  = open('tokenized_X_test.pkl',  'rb')
tokenized_y_train = open('tokenized_y_train.pkl',  'rb')
tokenized_y_test  = open('tokenized_y_test.pkl',   'rb')

# List[List[Dict[str, Any]]]  — one inner list per sentence, one dict per syllable
X_train: List[List[Dict[str, Any]]] = pickle.load(tokenized_X_train)
X_test:  List[List[Dict[str, Any]]] = pickle.load(tokenized_X_test)

# List[List[str]]  — one inner list per sentence, each str is 'B_W' or 'I_W'
y_train: List[List[str]] = pickle.load(tokenized_y_train)
y_test:  List[List[str]] = pickle.load(tokenized_y_test)

tokenized_X_train.close()
tokenized_X_test.close()
tokenized_y_train.close()
tokenized_y_test.close()


# ── 2. Define the CRF estimator ───────────────────────────────────────────────
# sklearn-crfsuite wraps the CRFsuite C library with a scikit-learn compatible API.
#
# algorithm='lbfgs'
#   L-BFGS (Limited-memory BFGS) — quasi-Newton optimizer.  Memory-efficient for
#   sparse, high-dimensional feature spaces typical of NLP.  Supports L1+L2
#   regularization jointly via the c1/c2 hyperparameters.
#
# max_iterations=100
#   Cap on L-BFGS gradient-descent steps.  RandomizedSearchCV will override this
#   indirectly via the best_estimator; the cap is a safety guard during CV folds.
#
# all_possible_transitions=True
#   Generate transition features for every (label_i, label_j) pair even if that
#   transition never appears in training data.  Without this, unseen transitions
#   at inference can cause assertion errors in CRFsuite.

labels: List[str] = ['B_W', 'I_W']

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True,
)


# ── 3. Hyperparameter search space ────────────────────────────────────────────
# c1 — L1 regularization coefficient (λ₁ · ||w||₁)
#   Promotes sparsity: pushes irrelevant feature weights to exactly 0.
#   Sampled from Exponential(scale=0.5) → mostly small values (mean=0.5),
#   tail extends to ~2–3, rarely higher.
#
# c2 — L2 regularization coefficient (λ₂ · ||w||₂²)
#   Shrinks all weights toward 0, preventing any single feature from dominating.
#   Sampled from Exponential(scale=0.05) → smaller values than c1 (mean=0.05),
#   meaning L2 is applied more gently than L1.
#
# Together c1+c2 form an elastic-net penalty, balancing sparsity and smoothness.

params_space: Dict[str, Any] = {
    'c1': scipy.stats.expon(scale=0.5),   # L1 strength — e.g. sampled ~[0.01, 2.0]
    'c2': scipy.stats.expon(scale=0.05),  # L2 strength — e.g. sampled ~[0.001, 0.2]
}


# ── 4. Evaluation metric ──────────────────────────────────────────────────────
# flat_f1_score flattens all label sequences into a single list before computing
# precision/recall/F1 — equivalent to token-level accuracy weighted by label freq.
# 'weighted' average: F1 per class weighted by support (B_W is more frequent than
# I_W, so its F1 contributes more to the final score).

f1_scorer = make_scorer(
    metrics.flat_f1_score,
    average='weighted',
    labels=labels,
)


# ── 5. Randomized hyperparameter search with cross-validation ─────────────────
# RandomizedSearchCV samples n_iter=50 random (c1, c2) pairs from params_space,
# evaluates each with 5-fold CV, and keeps the best.
#
# n_iter=50   — 50 random combinations to try (trade-off: quality vs. time)
# cv=5        — 5-fold stratified cross-validation on X_train
# n_jobs=12   — run folds in parallel; set to -1 to use all available CPUs
# verbose=1   — print one line per CV fold so you can monitor progress
#
# After .fit(), rs.best_params_ holds the optimal (c1, c2) and
# rs.best_score_ holds the mean CV weighted-F1 for those parameters.

rs = RandomizedSearchCV(
    crf,
    params_space,
    cv=5,
    verbose=1,
    n_jobs=12,
    n_iter=50,
    scoring=f1_scorer,
)
rs.fit(X_train, y_train)

print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)


# ── 6. Persist the trained model ──────────────────────────────────────────────
# Two pickle files are written:
#
#   tokenizer_model.pkl     — protocol=2 for Python 2 compatibility (pyvi.pkl)
#   tokenizer_model_py3.pkl — default protocol (protocol=4+ in Python 3.8+) (pyvi3.pkl)
#
# Copy whichever file matches your Python version into pyvi/models/.

# Python 2 compatible pickle (protocol 2 is the highest supported by Python 2)
with open('tokenizer_model.pkl', 'wb') as tokenizer_model:
    pickle.dump(rs.best_estimator_, tokenizer_model, protocol=2)

# Python 3 native pickle (smaller file due to more efficient encoding)
with open('tokenizer_model_py3.pkl', 'wb') as tokenizer_model_py3:
    pickle.dump(rs.best_estimator_, tokenizer_model_py3)
