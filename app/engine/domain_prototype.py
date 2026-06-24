"""DomainPrototypeStore --- Self-bootstrap Ring 1: domain-level feature prototypes.

This module builds domain-level feature prototypes from contract corpora and
classifies new texts by Mahalanobis distance to known domains.

Usage:
    store = DomainPrototypeStore()
    store.build("construction", [text_01, text_05, text_08])
    store.build("purchase", [purchase_text])
    results = store.classify(new_text)
    store.save("prototypes.json")
    store = DomainPrototypeStore.load("prototypes.json")
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

# Guarded imports
try:
    from app.engine.feature_extractor import FeatureExtractor
except ImportError:
    FeatureExtractor = None

try:
    from app.engine.clause_splitter import ClauseSplitter
except ImportError:
    ClauseSplitter = None

# Guarded scipy import
_HAS_SCIPY = False
try:
    import scipy.spatial.distance as sp_distance
    _HAS_SCIPY = True
except ImportError:
    pass


@dataclass
class DomainPrototype:
    """Prototype (centroid + covariance) for a domain's text-level feature space."""
    domain_id: str = ""
    sample_count: int = 0
    dense_centroid: list[float] = field(default_factory=lambda: [0.0] * 14)
    dense_covariance: list[list[float]] = field(default_factory=lambda: [[0.0] * 14 for _ in range(14)])
    block_count_typical: float = 0.0
    text_level_centroid: list[float] = field(default_factory=lambda: [0.0] * 14)


# --- Math helpers ---


def _mean_of_vectors(vectors):
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    result = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            result[i] += vec[i]
    return [v / n for v in result]


def _covariance_matrix(vectors, mean):
    n = len(vectors)
    dim = len(mean)
    if n < 2:
        identity = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            identity[i][i] = 1.0
        return identity
    cov = [[0.0] * dim for _ in range(dim)]
    for vec in vectors:
        for i in range(dim):
            for j in range(dim):
                cov[i][j] += (vec[i] - mean[i]) * (vec[j] - mean[j])
    for i in range(dim):
        for j in range(dim):
            cov[i][j] /= (n - 1)
    return cov


def _invert_matrix(m):
    n = len(m)
    if n == 0:
        return []
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = m[i][j]
        aug[i][n + i] = 1.0
    for col in range(n):
        pivot_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                pivot_row = row
        if max_val < 1e-12:
            raise ValueError("Singular matrix")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                if abs(factor) > 1e-15:
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inv[i][j] = aug[i][n + j]
    return inv


def _mahalanobis_pure_python(x, mean, cov):
    dim = len(x)
    diff = [x[i] - mean[i] for i in range(dim)]
    if dim == 0:
        return 0.0
    try:
        inv_cov = _invert_matrix(cov)
    except ValueError:
        return math.sqrt(sum(d * d for d in diff))
    temp = [0.0] * dim
    for i in range(dim):
        s = 0.0
        for j in range(dim):
            s += inv_cov[i][j] * diff[j]
        temp[i] = s
    result = sum(diff[i] * temp[i] for i in range(dim))
    if result < 0:
        result = 0.0
    return math.sqrt(result)


def _mahalanobis_distance(x, mean, cov):
    if _HAS_SCIPY:
        try:
            import numpy as np
            x_arr = np.array(x, dtype=float)
            mean_arr = np.array(mean, dtype=float)
            cov_arr = np.array(cov, dtype=float)
            try:
                inv_cov = np.linalg.inv(cov_arr)
                return float(sp_distance.mahalanobis(x_arr, mean_arr, inv_cov))
            except np.linalg.LinAlgError:
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, mean)))
        except ImportError:
            return _mahalanobis_pure_python(x, mean, cov)
    return _mahalanobis_pure_python(x, mean, cov)


def _text_level_vector(text):
    if ClauseSplitter is None or FeatureExtractor is None:
        raise ImportError("ClauseSplitter or FeatureExtractor not available")
    blocks = ClauseSplitter.split(text)
    if not blocks:
        return [0.0] * 14
    blocks = [b for b in blocks if b is not None]
    features = FeatureExtractor.extract(blocks, full_text=text)
    if not features:
        return [0.0] * 14
    dense_vectors = [fv.to_dense() for fv in features]
    return _mean_of_vectors(dense_vectors)


# --- DomainPrototypeStore ---


class DomainPrototypeStore:
    """Store of domain prototypes, with build/classify/save/load."""

    def __init__(self):
        self.prototypes = {}

    def build(self, domain_id, corpus_texts):
        if not corpus_texts:
            raise ValueError("corpus_texts must be non-empty")
        if ClauseSplitter is None or FeatureExtractor is None:
            raise ImportError("ClauseSplitter or FeatureExtractor not available")

        text_vectors = []
        block_counts = []
        all_block_vectors = []

        for text in corpus_texts:
            blocks = ClauseSplitter.split(text)
            blocks = [b for b in blocks if b is not None]
            block_counts.append(len(blocks))
            if not blocks:
                text_vectors.append([0.0] * 14)
                continue
            features = FeatureExtractor.extract(blocks, full_text=text)
            if not features:
                text_vectors.append([0.0] * 14)
                continue
            dense_vectors = [fv.to_dense() for fv in features]
            all_block_vectors.extend(dense_vectors)
            text_vectors.append(_mean_of_vectors(dense_vectors))

        dense_centroid = _mean_of_vectors(all_block_vectors) if all_block_vectors else [0.0] * 14
        text_level_centroid = _mean_of_vectors(text_vectors) if text_vectors else [0.0] * 14
        dense_covariance = _covariance_matrix(text_vectors, text_level_centroid)
        block_count_typical = sum(block_counts) / max(len(block_counts), 1)

        proto = DomainPrototype(
            domain_id=domain_id,
            sample_count=len(corpus_texts),
            dense_centroid=dense_centroid,
            dense_covariance=dense_covariance,
            block_count_typical=block_count_typical,
            text_level_centroid=text_level_centroid,
        )
        self.prototypes[domain_id] = proto
        return proto

    def classify(self, text):
        if not self.prototypes:
            return []
        text_vec = _text_level_vector(text)
        results = []
        for domain_id, proto in self.prototypes.items():
            dist = _mahalanobis_distance(text_vec, proto.text_level_centroid, proto.dense_covariance)
            results.append((domain_id, dist))
        results.sort(key=lambda x: x[1])
        return results

    def store_prototype(self, proto):
        self.prototypes[proto.domain_id] = proto

    def get_prototype(self, domain_id):
        return self.prototypes.get(domain_id)

    def save(self, path):
        data = {}
        for domain_id, proto in self.prototypes.items():
            data[domain_id] = {
                "domain_id": proto.domain_id,
                "sample_count": proto.sample_count,
                "dense_centroid": proto.dense_centroid,
                "dense_covariance": proto.dense_covariance,
                "block_count_typical": proto.block_count_typical,
                "text_level_centroid": proto.text_level_centroid,
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for domain_id, d in data.items():
            proto = DomainPrototype(
                domain_id=d["domain_id"],
                sample_count=d["sample_count"],
                dense_centroid=d["dense_centroid"],
                dense_covariance=d["dense_covariance"],
                block_count_typical=d["block_count_typical"],
                text_level_centroid=d["text_level_centroid"],
            )
            store.prototypes[domain_id] = proto
        return store
