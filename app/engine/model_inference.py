# -*- coding: utf-8 -*-
"""Lightweight clause model inference — loads exported JSON, predicts with zero training.

Usage:
    from app.engine.model_inference import ClauseModelInference
    model = ClauseModelInference("data/models/clause_model.json")
    label = model.predict("屋面防水保修期限为五年。")
    probs = model.predict_proba("屋面防水保修期限为五年。")
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Optional

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]+")


def _cjk_ngrams(text: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    """Extract CJK character n-grams from text."""
    cjk_chars = "".join(_CJK_RE.findall(text))
    ngrams: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(len(cjk_chars) - n + 1):
            ngrams.append(cjk_chars[i : i + n])
    return ngrams


class ClauseModelInference:
    """Load a pre-trained clause model and predict with zero training.

    Implements the same predict/predict_proba interface as
    PurePythonClauseClassifier, but loads pre-computed parameters
    from a JSON file.
    """

    def __init__(self, model_path: str):
        """Load model parameters from a JSON file.

        Args:
            model_path: Path to clause_model.json.
        """
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._vocab: dict[str, int] = data["vocab"]
        self._idf: dict[str, float] = {
            int(k): float(v) for k, v in data["idf"].items()
        }
        self._classes: list[str] = data["classes"]
        self._class_log_prior: dict[str, float] = data["class_log_prior"]
        self._class_feature_log_prob: dict[str, list[float]] = (
            data["class_feature_log_prob"]
        )
        self._alpha: float = data.get("alpha", 1.0)
        self._ngram_min: int = data.get("ngram_min", 1)
        self._ngram_max: int = data.get("ngram_max", 3)

    # ── Properties ─────────────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        return True

    @property
    def classes_(self) -> list[str]:
        return list(self._classes)

    # ── Prediction ─────────────────────────────────────────────────

    def predict(self, text: str) -> str:
        """Predict clause type."""
        probs = self.predict_proba(text)
        if not probs:
            return "unknown"
        return max(probs, key=probs.get)  # type: ignore[arg-type]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return class probability dict."""
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

    # ── Internal ───────────────────────────────────────────────────

    def _tfidf_vector(self, text: str) -> dict[int, float]:
        """Compute TF-IDF vector as sparse index→weight dict."""
        ngrams = _cjk_ngrams(text, self._ngram_min, self._ngram_max)
        tf: dict[int, float] = defaultdict(float)
        for ng in ngrams:
            idx = self._vocab.get(ng)
            if idx is not None:
                tf[idx] += 1.0

        total = sum(tf.values()) or 1.0
        vec: dict[int, float] = {}
        for idx, count in tf.items():
            idf = self._idf.get(str(idx), 1.0)
            vec[idx] = (count / total) * float(idf)
        return vec
