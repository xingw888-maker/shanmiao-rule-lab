# -*- coding: utf-8 -*-
"""Clause classifier using TF-IDF + Naive Bayes for clause type prediction.

Provides two implementations:
  - TfidfClauseClassifier: sklearn TfidfVectorizer + MultinomialNB (primary)
  - PurePythonClauseClassifier: Pure Python count-TFIDF + NaiveBayes (fallback)

Both implement the same interface: fit(X_texts, y_labels) + predict(text) + predict_proba(text).
Integration point: clause_splitter.py split(classifier=...) accepts any object with a predict() method.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional sklearn import ─────────────────────────────────────────────

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── CJK n-gram helper ──────────────────────────────────────────────────

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]+")


def _cjk_ngrams(text: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    """Extract CJK character n-grams from text."""
    cjk_chars = "".join(_CJK_RE.findall(text))
    ngrams: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(len(cjk_chars) - n + 1):
            ngrams.append(cjk_chars[i : i + n])
    return ngrams


# ═══════════════════════════════════════════════════════════════════════════
# sklearn-based classifier
# ═══════════════════════════════════════════════════════════════════════════


class TfidfClauseClassifier:
    """TF-IDF + MultinomialNB clause type classifier (sklearn required).

    Features: CJK character 1-3 grams via TfidfVectorizer with sublinear tf.
    Model: MultinomialNB with additive (Laplace) smoothing.

    Usage:
        clf = TfidfClauseClassifier()
        clf.fit(texts, labels)        # texts: list[str], labels: list[str]
        label = clf.predict(text)     # → str
        probs = clf.predict_proba(text)  # → dict[str, float]
    """

    def __init__(self):
        if not _HAS_SKLEARN:
            raise ImportError(
                "sklearn is required for TfidfClauseClassifier. "
                "Install with: pip install scikit-learn"
            )
        self._pipeline: Optional[Pipeline] = None
        self._classes: list[str] = []
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def classes_(self) -> list[str]:
        return list(self._classes)

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfClauseClassifier":
        """Train the classifier on labeled clause texts.

        Args:
            texts: List of clause text strings.
            labels: List of corresponding clause type labels.

        Returns:
            self for chaining.
        """
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must have same length "
                f"({len(texts)} vs {len(labels)})"
            )
        if len(texts) < 2:
            raise ValueError("Need at least 2 training samples")

        self._classes = sorted(set(labels))

        # Build pipeline: char n-gram TF-IDF → MultinomialNB
        self._pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    max_df=0.95,
                    min_df=1,
                ),
            ),
            ("clf", MultinomialNB(alpha=1.0)),
        ])

        self._pipeline.fit(texts, labels)
        self._fitted = True
        logger.info(
            "TfidfClauseClassifier fitted: %d samples, %d classes",
            len(texts),
            len(self._classes),
        )
        return self

    def predict(self, text: str) -> str:
        """Predict the clause type for a single text."""
        if not self._fitted or self._pipeline is None:
            return "unknown"
        return str(self._pipeline.predict([text])[0])

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return class probability dict for a single text."""
        if not self._fitted or self._pipeline is None:
            return {"unknown": 1.0}
        probs = self._pipeline.predict_proba([text])[0]
        return {
            cls: float(prob)
            for cls, prob in zip(self._pipeline.classes_, probs)
        }

    def predict_top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (label, probability) predictions."""
        proba = self.predict_proba(text)
        sorted_items = sorted(proba.items(), key=lambda x: -x[1])
        return sorted_items[:k]


# ═══════════════════════════════════════════════════════════════════════════
# Pure Python fallback classifier
# ═══════════════════════════════════════════════════════════════════════════


class PurePythonClauseClassifier:
    """Pure Python clause classifier (no sklearn dependency).

    Uses count-based TF-IDF with CJK n-grams + Multinomial Naive Bayes
    implemented from scratch.  Works identically to TfidfClauseClassifier
    but with no external dependencies.

    Usage:
        clf = PurePythonClauseClassifier()
        clf.fit(texts, labels)
        label = clf.predict(text)
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}  # ngram → index
        self._idf: dict[str, float] = {}  # ngram → idf
        self._class_log_prior: dict[str, float] = {}
        self._class_feature_log_prob: dict[str, list[float]] = {}
        self._classes: list[str] = []
        self._fitted = False
        self._alpha = 1.0  # Laplace smoothing

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def classes_(self) -> list[str]:
        return list(self._classes)

    def fit(
        self, texts: list[str], labels: list[str],
    ) -> "PurePythonClauseClassifier":
        """Train on labeled clause texts."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        if len(texts) < 2:
            raise ValueError("Need at least 2 training samples")

        n_docs = len(texts)
        self._classes = sorted(set(labels))

        # Step 1: Build vocabulary (CJK 1-3 grams, min_df=1)
        doc_ngrams: list[set[str]] = []
        df: dict[str, int] = defaultdict(int)
        for text in texts:
            ngrams = set(_cjk_ngrams(text, 1, 3))
            doc_ngrams.append(ngrams)
            for ng in ngrams:
                df[ng] += 1

        # Build vocab index
        vocab_ngrams = sorted(df.keys())
        self._vocab = {ng: i for i, ng in enumerate(vocab_ngrams)}
        self._idx_to_ngram = {i: ng for ng, i in self._vocab.items()}
        vocab_size = len(self._vocab)

        # Step 2: Compute IDF (both by ngram key and by index)
        self._ngram_idf: dict[str, float] = {
            ng: math.log((n_docs + 1) / (df[ng] + 1)) + 1.0
            for ng in vocab_ngrams
        }
        self._idf: dict[int, float] = {
            self._vocab[ng]: v for ng, v in self._ngram_idf.items()
        }

        # Step 3: Compute per-class feature counts for Naive Bayes
        class_docs: dict[str, list[set[str]]] = defaultdict(list)
        for text, label, ngrams in zip(texts, labels, doc_ngrams):
            class_docs[label].append(ngrams)

        # Class priors (log)
        for cls in self._classes:
            self._class_log_prior[cls] = math.log(
                len(class_docs.get(cls, [])) / n_docs
            )

        # Per-class feature log probabilities (with Laplace smoothing)
        for cls in self._classes:
            docs = class_docs.get(cls, [])
            # Count features per class
            feat_counts = defaultdict(float)
            total_count = 0.0
            for ngrams in docs:
                for ng in ngrams:
                    idx = self._vocab.get(ng)
                    if idx is not None:
                        feat_counts[idx] += 1.0
                        total_count += 1.0

            # Log probabilities
            denom = total_count + self._alpha * vocab_size
            log_probs = [0.0] * vocab_size
            for idx in range(vocab_size):
                count = feat_counts.get(idx, 0.0)
                log_probs[idx] = math.log(
                    (count + self._alpha) / denom
                )
            self._class_feature_log_prob[cls] = log_probs

        self._fitted = True
        logger.info(
            "PurePythonClauseClassifier fitted: %d samples, %d classes, %d features",
            n_docs,
            len(self._classes),
            vocab_size,
        )
        return self

    def export_model(self) -> dict:
        """Export trained model parameters as a serializable dict.

        Returns a dict suitable for json.dump() that can be loaded
        by ClauseModelInference for zero-training prediction.
        """
        if not self._fitted:
            raise RuntimeError("Cannot export unfitted model")
        # Use uniform priors on export to counteract class imbalance
        # (prevents dominant classes like 其他 from drowning out minorities)
        n_classes = len(self._classes)
        uniform_prior = math.log(1.0 / n_classes) if n_classes > 0 else 0.0
        balanced_priors = {cls: uniform_prior for cls in self._classes}

        return {
            "model_type": "pure_python_naive_bayes",
            "version": "1.0.0",
            "vocab": self._vocab,
            "idf": self._idf,
            "classes": self._classes,
            "class_log_prior": balanced_priors,
            "class_feature_log_prob": self._class_feature_log_prob,
            "alpha": self._alpha,
            "ngram_min": 1,
            "ngram_max": 3,
        }

    def _tfidf_vector(self, text: str) -> dict[int, float]:
        """Compute TF-IDF vector for a text as sparse index→weight dict."""
        ngrams = _cjk_ngrams(text, 1, 3)
        tf: dict[int, float] = defaultdict(float)
        for ng in ngrams:
            idx = self._vocab.get(ng)
            if idx is not None:
                tf[idx] += 1.0
                # Also collect IDF on first encounter
                if idx not in self._idf:
                    self._idf[idx] = self._ngram_idf.get(ng, 1.0)

        # Normalize TF and multiply by IDF
        total = sum(tf.values()) or 1.0
        vec: dict[int, float] = {}
        for idx, count in tf.items():
            idf = self._idf.get(idx, 1.0)
            vec[idx] = (count / total) * idf
        return vec

    def predict(self, text: str) -> str:
        """Predict clause type."""
        probs = self.predict_proba(text)
        if not probs:
            return "unknown"
        return max(probs, key=probs.get)  # type: ignore[arg-type]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return class probability dict."""
        if not self._fitted:
            return {"unknown": 1.0}

        vec = self._tfidf_vector(text)
        log_probs: dict[str, float] = {}
        for cls in self._classes:
            lp = self._class_log_prior.get(cls, 0.0)
            feat_probs = self._class_feature_log_prob.get(cls, [])
            for idx, weight in vec.items():
                if idx < len(feat_probs):
                    lp += feat_probs[idx] * weight
            log_probs[cls] = lp

        # Softmax
        max_lp = max(log_probs.values()) if log_probs else 0.0
        exp_sum = 0.0
        exp_vals: dict[str, float] = {}
        for cls, lp in log_probs.items():
            v = math.exp(lp - max_lp)
            exp_vals[cls] = v
            exp_sum += v

        if exp_sum > 0:
            return {cls: v / exp_sum for cls, v in exp_vals.items()}
        return {cls: 1.0 / len(self._classes) for cls in self._classes}

    def predict_top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (label, probability) predictions."""
        proba = self.predict_proba(text)
        sorted_items = sorted(proba.items(), key=lambda x: -x[1])
        return sorted_items[:k]


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid V2: TF-IDF + Dynamic prototype fallback
# ═══════════════════════════════════════════════════════════════════════════


class HybridClauseClassifierV2:
    """Hybrid clause classifier combining TF-IDF/NaiveBayes with prototype fallback.

    If sklearn is available, uses TfidfClauseClassifier (TF-IDF + MultinomialNB).
    Otherwise, uses PurePythonClauseClassifier.
    Falls back to DynamicClauseClassifier when confidence is below threshold.

    Implements the same predict(text) → str interface as HybridClauseClassifier.
    """

    MIN_CONFIDENCE = 0.25  # Below this, fall back to dynamic classifier

    def __init__(
        self,
        dynamic_classifier=None,
        training_jsonl: Optional[str] = None,
    ):
        """Initialize the hybrid classifier.

        Args:
            dynamic_classifier: Optional DynamicClauseClassifier for fallback.
            training_jsonl: Path to JSONL training data.  If None, looks for
                            data/training/clauses.jsonl relative to project root.
        """
        self._dynamic = dynamic_classifier

        # Choose backend
        if _HAS_SKLEARN:
            self._backend = TfidfClauseClassifier()
            logger.info("HybridClauseClassifierV2: using sklearn backend")
        else:
            self._backend = PurePythonClauseClassifier()
            logger.info("HybridClauseClassifierV2: using pure Python backend")

        self._fitted = False
        self._training_path = training_jsonl

    # ── Properties ───────────────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def classes_(self) -> list[str]:
        if self._fitted:
            return self._backend.classes_
        return []

    # ── Training ─────────────────────────────────────────────────────

    def fit_from_jsonl(self, jsonl_path: Optional[str] = None) -> "HybridClauseClassifierV2":
        """Train from a JSONL file produced by build_clause_dataset.py.

        Args:
            jsonl_path: Path to clauses.jsonl.  Uses self._training_path if None.

        Returns:
            self for chaining.
        """
        path = jsonl_path or self._training_path
        if path is None:
            raise ValueError("No training JSONL path provided")

        texts: list[str] = []
        labels: list[str] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text", "")
                label = rec.get("label", "")
                if text and label and label not in ("unknown", "未分类"):
                    texts.append(text)
                    labels.append(label)

        if len(texts) < 2:
            raise ValueError(
                f"Not enough labeled samples ({len(texts)}). "
                "Need at least 2 with non-generic labels."
            )

        self._backend.fit(texts, labels)
        self._fitted = True
        logger.info(
            "HybridClauseClassifierV2 trained on %d samples from %s",
            len(texts),
            path,
        )
        return self

    def fit(self, texts: list[str], labels: list[str]) -> "HybridClauseClassifierV2":
        """Train directly from lists of texts and labels."""
        self._backend.fit(texts, labels)
        self._fitted = True
        return self

    @classmethod
    def from_model(
        cls,
        model_path: str,
        dynamic_classifier=None,
    ) -> "HybridClauseClassifierV2":
        """Create a pre-trained classifier from an exported model JSON.

        Args:
            model_path: Path to clause_model.json.
            dynamic_classifier: Optional DynamicClauseClassifier for fallback.

        Returns:
            A fitted HybridClauseClassifierV2 ready for prediction.
        """
        from app.engine.model_inference import ClauseModelInference

        instance = cls(dynamic_classifier=dynamic_classifier)
        instance._backend = ClauseModelInference(model_path)
        instance._fitted = True
        logger.info(
            "HybridClauseClassifierV2 loaded from model: %s", model_path
        )
        return instance

    # ── Prediction ──────────────────────────────────────────────────

    def predict(self, text: str) -> str:
        """Predict clause type, falling back to dynamic classifier if uncertain."""
        if not self._fitted:
            if self._dynamic is not None:
                return self._dynamic.predict(text)
            return "unknown"

        proba = self._backend.predict_proba(text)
        top_label = max(proba, key=proba.get)  # type: ignore[arg-type]
        top_prob = proba[top_label]

        if top_prob < self.MIN_CONFIDENCE and self._dynamic is not None:
            dyn_label = self._dynamic.predict(text)
            if dyn_label and dyn_label != "未分类":
                return dyn_label

        return top_label

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return class probability dict."""
        if not self._fitted:
            return {"unknown": 1.0}
        return self._backend.predict_proba(text)

    def predict_top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (label, probability) predictions."""
        if not self._fitted:
            return [("unknown", 1.0)]
        return self._backend.predict_top(text, k)


# ═══════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════


def create_classifier(
    training_jsonl: Optional[str] = None,
    dynamic_classifier=None,
    prefer_sklearn: bool = True,
) -> HybridClauseClassifierV2:
    """Create and optionally train a hybrid clause classifier.

    Args:
        training_jsonl: Path to clauses.jsonl training data.
        dynamic_classifier: Fallback DynamicClauseClassifier instance.
        prefer_sklearn: If True and sklearn available, use sklearn backend.

    Returns:
        A fitted HybridClauseClassifierV2, or unfitted if no training data.
    """
    clf = HybridClauseClassifierV2(
        dynamic_classifier=dynamic_classifier,
        training_jsonl=training_jsonl,
    )

    if training_jsonl and os.path.isfile(training_jsonl):
        try:
            clf.fit_from_jsonl(training_jsonl)
        except Exception as exc:
            logger.warning("Failed to train classifier: %s", exc)

    return clf
