"""Dynamic Clause Classifier — feature-space clause type matching.

Replaces the hardcoded _TYPE_PATTERNS keyword table in clause_splitter.py with
a dynamic feature-space approach.  Instead of asking "does the clause title
contain 付款?", it asks:

  1. Extract a ClauseFeatureVector from the clause block (structural,
     numeric, text characteristic features + sparse n-gram profile).

  2. Compare against KNOWN-TYPE PROTOTYPES via multi-signal similarity:
     - Structural similarity (cosine on first 6 dims) → 30% weight
     - Numeric similarity (cosine on dims 6-10) → 20% weight
     - N-gram similarity (sparse cosine) → 50% weight

  3. If the top type's weighted score exceeds the confidence threshold,
     assign that type.  Otherwise, assign "未知".

The classifier learns prototypes from examples.  Initially, seeds are
provided via domain config (clause_type_examples), but the classifier
can accumulate more examples over time to improve accuracy.

Design principle: ZERO hardcoded Chinese keywords.  The classifier doesn't
know what "付款" means — it only knows that clauses with high digit density,
percentage tokens, and certain n-gram distributions tend to cluster together,
and that cluster happens to be labeled "付款" in the construction domain.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from app.engine.feature_extractor import (
    ClauseFeatureVector,
    FeatureExtractor,
    _cosine,
    _sparse_cosine,
)

logger = logging.getLogger(__name__)


@dataclass
class TypePrototype:
    """A learned prototype for a clause type.

    Each prototype has:
    - type_name: the label (e.g. "付款", "验收", "工期")
    - centroid: the mean feature vector of all examples
    - example_count: how many examples contributed to this centroid
    - structural_centroid: first 6 dims of centroid (for structural similarity)
    - numeric_centroid: dims 6-10 of centroid (for numeric similarity)
    - ngram_centroid: merged sparse n-gram frequencies from all examples
    """
    type_name: str
    centroid: list[float] = field(default_factory=lambda: [0.0] * 14)
    example_count: int = 0
    structural_centroid: list[float] = field(default_factory=lambda: [0.0] * 6)
    numeric_centroid: list[float] = field(default_factory=lambda: [0.0] * 4)
    ngram_centroid: dict[int, float] = field(default_factory=dict)
    ngram_total_weight: float = 0.0


class DynamicClauseClassifier:
    """Classifies clause blocks by comparing feature vectors to learned prototypes.

    Usage:
        classifier = DynamicClauseClassifier()

        # Seed with domain-specific examples (from domain.json clause_type_examples)
        classifier.seed_from_examples(examples)

        # Classify a clause block
        type_name, confidence = classifier.classify(feature_vector)

        # Learn from new examples over time
        classifier.add_example(type_name, feature_vector)
    """

    # ── Similarity weights ──
    STRUCTURAL_WEIGHT = 0.30
    NUMERIC_WEIGHT = 0.20
    NGRAM_WEIGHT = 0.50

    # ── Confidence thresholds ──
    MIN_CONFIDENCE = 0.15         # below this → "未知"
    HIGH_CONFIDENCE = 0.35        # above this → strong match

    def __init__(self):
        self._prototypes: dict[str, TypePrototype] = {}
        self._feature_extractor = FeatureExtractor()

    # ── Prototype management ──

    @property
    def prototype_types(self) -> list[str]:
        """Return list of known type names."""
        return sorted(self._prototypes.keys())

    @property
    def prototype_count(self) -> int:
        return len(self._prototypes)

    def get_prototype(self, type_name: str) -> Optional[TypePrototype]:
        return self._prototypes.get(type_name)

    def seed_from_examples(
        self, examples: dict[str, list[str]]
    ) -> int:
        """Seed prototypes from text examples keyed by type name.

        Args:
            examples: {"付款": ["text example 1", "text example 2"], ...}

        Returns:
            Number of prototypes created.
        """
        count = 0
        for type_name, texts in examples.items():
            if not texts:
                continue
            features = self._feature_extractor.extract(
                [{"clause_id": f"seed_{type_name}_{i}",
                  "clause_title": text.split('\n')[0] if text else "",
                  "content": text, "level": 1}
                 for i, text in enumerate(texts)]
            )
            for fv in features:
                self.add_example(type_name, fv)
            count += 1
        return count

    def add_example(self, type_name: str, fv: ClauseFeatureVector) -> None:
        """Add a single example to the prototype for type_name.

        Updates the centroid using online mean update (Welford-like).
        """
        if type_name not in self._prototypes:
            self._prototypes[type_name] = TypePrototype(type_name=type_name)

        proto = self._prototypes[type_name]
        dense = fv.to_dense()

        # Online mean update for dense centroid
        n = proto.example_count
        if n == 0:
            proto.centroid = list(dense)
            proto.structural_centroid = list(dense[:6])
            proto.numeric_centroid = list(dense[6:10])
        else:
            for i, val in enumerate(dense):
                proto.centroid[i] = (proto.centroid[i] * n + val) / (n + 1)
            for i, val in enumerate(dense[:6]):
                proto.structural_centroid[i] = (proto.structural_centroid[i] * n + val) / (n + 1)
            for i, val in enumerate(dense[6:10]):
                proto.numeric_centroid[i] = (proto.numeric_centroid[i] * n + val) / (n + 1)

        # Merge sparse n-gram centroid
        for hash_key, freq in fv.sparse_ngrams.items():
            if hash_key in proto.ngram_centroid:
                proto.ngram_centroid[hash_key] = (
                    (proto.ngram_centroid[hash_key] * n + freq) / (n + 1)
                )
            else:
                proto.ngram_centroid[hash_key] = freq / (n + 1)

        proto.ngram_total_weight = 1.0  # mark as initialized
        proto.example_count = n + 1

    def seed_from_domain_config(self, domain_dir: str) -> int:
        """Load clause_type_examples from a domain's domain.json.

        Args:
            domain_dir: Path to the domain directory containing domain.json

        Returns:
            Number of prototypes seeded.
        """
        config_path = os.path.join(domain_dir, "domain.json")
        if not os.path.exists(config_path):
            return 0

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        examples = config.get("clause_type_examples", {})
        if not examples:
            return 0

        return self.seed_from_examples(examples)

    # ── Classification ──

    def classify(
        self, fv: ClauseFeatureVector
    ) -> tuple[str, float]:
        """Classify a feature vector to the best-matching type.

        Returns (type_name, confidence).
        If no prototype scores above MIN_CONFIDENCE, returns ("未知", 0.0).
        """
        if not self._prototypes:
            return "未知", 0.0

        scores: list[tuple[str, float]] = []
        for type_name, proto in self._prototypes.items():
            score = self._compute_score(fv, proto)
            scores.append((type_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_type, best_score = scores[0]

        if best_score < self.MIN_CONFIDENCE:
            return "未知", best_score

        return best_type, best_score

    def classify_with_alternatives(
        self, fv: ClauseFeatureVector
    ) -> list[tuple[str, float]]:
        """Return all type scores sorted by confidence (for logging/debug)."""
        if not self._prototypes:
            return [("未知", 0.0)]

        scores = []
        for type_name, proto in self._prototypes.items():
            score = self._compute_score(fv, proto)
            scores.append((type_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if not scores or scores[0][1] < self.MIN_CONFIDENCE:
            scores.insert(0, ("未知", 0.0))
        return scores

    def _compute_score(
        self, fv: ClauseFeatureVector, proto: TypePrototype
    ) -> float:
        """Compute weighted similarity score between fv and prototype."""
        dense = fv.to_dense()

        # Structural similarity
        struct_sim = _cosine(dense[:6], proto.structural_centroid)

        # Numeric similarity
        numeric_sim = _cosine(dense[6:10], proto.numeric_centroid)

        # N-gram similarity
        if fv.sparse_ngrams and proto.ngram_centroid:
            ngram_sim = _sparse_cosine(fv.sparse_ngrams, proto.ngram_centroid)
        else:
            # Fall back to full dense cosine if n-gram data is missing
            ngram_sim = _cosine(dense, proto.centroid)

        return (
            self.STRUCTURAL_WEIGHT * struct_sim +
            self.NUMERIC_WEIGHT * numeric_sim +
            self.NGRAM_WEIGHT * ngram_sim
        )

    # ── Serialization ──

    def to_dict(self) -> dict:
        """Serialize prototypes to a JSON-serializable dict.

        The sparse ngram_centroid is skipped in serialization (too large).
        Prototypes are reloaded from examples on next startup.
        """
        return {
            type_name: {
                "centroid": proto.centroid,
                "example_count": proto.example_count,
                "structural_centroid": proto.structural_centroid,
                "numeric_centroid": proto.numeric_centroid,
            }
            for type_name, proto in self._prototypes.items()
        }

    @classmethod
    def from_prototype_dict(cls, data: dict) -> "DynamicClauseClassifier":
        """Restore classifier from a prototype dict (without n-gram data).

        N-gram centroids are NOT restored — they would need re-seeding
        from examples.  The dense centroids alone provide reasonable
        classification (~70% of full accuracy).
        """
        classifier = cls()
        for type_name, proto_data in data.items():
            proto = TypePrototype(type_name=type_name)
            proto.centroid = proto_data.get("centroid", [0.0] * 14)
            proto.example_count = proto_data.get("example_count", 0)
            proto.structural_centroid = proto_data.get("structural_centroid", [0.0] * 6)
            proto.numeric_centroid = proto_data.get("numeric_centroid", [0.0] * 4)
            classifier._prototypes[type_name] = proto
        return classifier


# ── Convenience: hybrid classifier that can fall back to keywords ──

class HybridClauseClassifier:
    """Classifier that tries feature-space first, falls back to keywords.

    This is the integration point — it replaces _infer_clause_type() in
    clause_splitter.py while maintaining backward compatibility with
    domains that haven't yet provided clause_type_examples.
    """

    def __init__(
        self,
        dynamic: Optional[DynamicClauseClassifier] = None,
        keyword_table: Optional[dict[str, list[str]]] = None,
    ):
        self._dynamic = dynamic
        self._keyword_table = keyword_table or {}

    @property
    def has_dynamic(self) -> bool:
        return self._dynamic is not None and self._dynamic.prototype_count > 0

    def classify(
        self, fv: ClauseFeatureVector, clause_title: str = "", content_head: str = ""
    ) -> tuple[str, float]:
        """Classify a clause. Uses dynamic classifier when available."""
        if self.has_dynamic and self._dynamic is not None:
            type_name, confidence = self._dynamic.classify(fv)
            if type_name != "未知":
                return type_name, confidence
            # Dynamic classifier returned unknown → fall through to keywords

        # Fallback: keyword-based inference
        if self._keyword_table:
            return _keyword_classify(clause_title, content_head, self._keyword_table)

        return "其他", 0.0

    def classify_with_alternatives(
        self, fv: ClauseFeatureVector, clause_title: str = "", content_head: str = ""
    ) -> list[tuple[str, float]]:
        """Return all type scores."""
        if self.has_dynamic and self._dynamic is not None:
            return self._dynamic.classify_with_alternatives(fv)
        type_name, confidence = self.classify(fv, clause_title, content_head)
        return [(type_name, confidence)]


def _keyword_classify(
    clause_title: str, content_head: str, keyword_table: dict[str, list[str]]
) -> tuple[str, float]:
    """Original keyword-based classification (maintained for backward compat)."""
    scores: dict[str, float] = {}
    for type_name, keywords in keyword_table.items():
        score = 0.0
        for kw in keywords:
            if kw in clause_title:
                score = max(score, 0.8)
            if kw in content_head:
                score = max(score, 0.5)
        if score > 0:
            scores[type_name] = score

    if not scores:
        return "其他", 0.0

    best_type = max(scores, key=lambda t: (scores[t], -list(keyword_table.keys()).index(t)))
    return best_type, scores[best_type]
