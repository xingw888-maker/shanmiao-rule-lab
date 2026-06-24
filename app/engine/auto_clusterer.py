"""Auto-Clusterer — unsupervised DBSCAN clustering of clause blocks.

This module clusters ClauseBlock objects by their 14-dim feature vectors
(from FeatureExtractor), then automatically generates human-readable labels
for each cluster based on CJK bigram profiles, digit density, and position
distribution.

This replaces the hardcoded _TYPE_PATTERNS keyword table in clause_splitter.py
with a data-driven approach: clusters emerge from the feature space rather
than from hand-written keyword lists.

Design principle: zero hardcoded Chinese keywords in the clustering logic.
The auto-labeler uses purely statistical properties of the member blocks.

Distance metric: hybrid cosine distance combining structural (30%),
numeric (20%), and n-gram (50%) similarity, matching the approach in
DynamicClauseClassifier.  Pure dense-feature cosine distance is insufficient
for separating Chinese contract clause types since they share very similar
structural and numeric profiles.
"""

import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from app.engine.clause_splitter import ClauseBlock
from app.engine.feature_extractor import (
    FeatureExtractor,
    ClauseFeatureVector,
    _cosine,
    _sparse_cosine,
)

# ── Try to import sklearn; fall back to pure-Python DBSCAN ──
try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class AutoCluster:
    """A discovered cluster of clause blocks with auto-generated label.

    Attributes:
        cluster_id: Unique identifier like "C_000", "C_001", ...
        centroid: Mean of the 14-dim feature vectors of member blocks.
        member_count: Number of blocks in this cluster.
        top_bigrams: Top-5 CJK bigrams across member blocks (most common first).
        digit_density_avg: Average digit_density across member blocks.
        position_zone: "前段" (front), "中段" (mid), or "后段" (back).
        auto_label: Human-readable label, e.g. "数字密集-保修防水年限-后段".
        block_indices: Indices of member blocks in the original block list.
    """

    cluster_id: str
    centroid: list[float] = field(default_factory=lambda: [0.0] * 14)
    member_count: int = 0
    top_bigrams: list[str] = field(default_factory=list)
    digit_density_avg: float = 0.0
    position_zone: str = "中段"
    auto_label: str = ""
    block_indices: list[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Hybrid cosine distance (matches DynamicClauseClassifier weights)
# ═══════════════════════════════════════════════════════════════

_STRUCTURAL_WEIGHT = 0.30
_NUMERIC_WEIGHT = 0.20
_NGRAM_WEIGHT = 0.50


def _hybrid_distance(
    a_dense: list[float],
    b_dense: list[float],
    a_sparse: dict[int, float],
    b_sparse: dict[int, float],
) -> float:
    """Hybrid cosine distance combining structural, numeric and n-gram signals.

    This mirrors the scoring function in DynamicClauseClassifier:
      - Structural dims [0:6]:   weight 0.30
      - Numeric   dims [6:10]:  weight 0.20
      - N-gram sparse cosine:    weight 0.50

    Returns cosine-distance (1 - weighted_similarity) in [0, 2].
    """
    struct_sim = _cosine(a_dense[:6], b_dense[:6])
    numeric_sim = _cosine(a_dense[6:10], b_dense[6:10])

    if a_sparse and b_sparse:
        ngram_sim = _sparse_cosine(a_sparse, b_sparse)
    else:
        ngram_sim = _cosine(a_dense, b_dense)

    weighted_sim = (
        _STRUCTURAL_WEIGHT * struct_sim
        + _NUMERIC_WEIGHT * numeric_sim
        + _NGRAM_WEIGHT * ngram_sim
    )
    return max(0.0, 1.0 - weighted_sim)


# ═══════════════════════════════════════════════════════════════
# Pure-Python DBSCAN implementation
# ═══════════════════════════════════════════════════════════════

class PurePythonDBSCAN:
    """Pure-Python DBSCAN using a precomputed distance matrix.

    This is a straightforward implementation of the DBSCAN algorithm.
    It computes the full distance matrix upfront (O(n^2) memory).
    For the expected scale (dozens to low hundreds of blocks) this is fine.

    Reference: Ester et al. (1996).
    """

    def __init__(self, eps: float = 0.35, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: list[int] = []

    def fit_predict(self, dist_matrix: list[list[float]]) -> list[int]:
        """Run DBSCAN clustering on a precomputed distance matrix.

        Args:
            dist_matrix: n x n matrix of pairwise distances (dist_matrix[i][j]).

        Returns:
            List of cluster labels (-1 = noise, 0..n-1 = cluster index).
        """
        n = len(dist_matrix)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Find neighbours for each point
        neighbours: list[list[int]] = []
        for i in range(n):
            neigh = [
                j for j in range(n) if j != i and dist_matrix[i][j] <= self.eps
            ]
            neighbours.append(neigh)

        # Classify each point
        labels = [-1] * n  # -1 = unvisited
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue  # already visited / assigned

            # Check if core point
            if len(neighbours[i]) < self.min_samples:
                labels[i] = -2  # temporarily mark as noise
                continue

            # Start a new cluster (BFS expansion)
            labels[i] = cluster_id
            seed_set = list(neighbours[i])

            while seed_set:
                q = seed_set.pop()
                if labels[q] == -2:
                    # Previously marked as noise — now part of a cluster
                    labels[q] = cluster_id
                if labels[q] != -1:
                    continue  # already assigned
                labels[q] = cluster_id

                # If q is a core point, add its neighbours to seed set
                if len(neighbours[q]) >= self.min_samples:
                    for nb in neighbours[q]:
                        if labels[nb] in (-1, -2):
                            seed_set.append(nb)

            cluster_id += 1

        # Convert noise label from -2 to -1
        labels = [-1 if l == -2 else l for l in labels]
        self.labels_ = labels
        return labels


class HybridDBSCAN:
    """DBSCAN that uses the hybrid (structural + numeric + n-gram) distance.

    Accepts a list of (dense_vector, sparse_ngrams) tuples and computes
    the hybrid distance between them.
    """

    def __init__(self, eps: float = 0.35, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples
        self._dbscan = PurePythonDBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(
        self,
        dense_vectors: list[list[float]],
        sparse_ngrams: list[dict[int, float]],
    ) -> list[int]:
        """Run hybrid-distance DBSCAN clustering.

        Args:
            dense_vectors: List of 14-dim dense feature vectors.
            sparse_ngrams: List of sparse n-gram dicts.

        Returns:
            List of cluster labels (-1 = noise).
        """
        n = len(dense_vectors)
        if n == 0:
            return []
        if n == 1:
            return [0]

        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _hybrid_distance(
                    dense_vectors[i], dense_vectors[j],
                    sparse_ngrams[i], sparse_ngrams[j],
                )
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d

        return self._dbscan.fit_predict(dist_matrix)


# ═══════════════════════════════════════════════════════════════
# Auto-labeling helpers
# ═══════════════════════════════════════════════════════════════

_CJK_CHAR = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def _extract_cjk_bigrams(text: str) -> Counter:
    """Extract character bigrams from CJK-only characters in text."""
    cjk_chars = _CJK_CHAR.findall(text)
    bigrams = Counter()
    for i in range(len(cjk_chars) - 1):
        bigrams[cjk_chars[i] + cjk_chars[i + 1]] += 1
    return bigrams


def _top_bigrams_across_blocks(
    blocks: list[ClauseBlock], top_n: int = 5
) -> list[str]:
    """Collect top-N CJK bigrams across a list of clause blocks.

    Returns bigram strings sorted by frequency (most common first).
    """
    all_bigrams: Counter = Counter()
    for block in blocks:
        text = (block.clause_title or "") + "\n" + (block.content or "")
        all_bigrams += _extract_cjk_bigrams(text)
    return [bg for bg, _ in all_bigrams.most_common(top_n)]


def _compute_position_zone(centroid: list[float]) -> str:
    """Determine position zone from position_ratio (dim 0).

    position_ratio < 0.3 -> "前段" (front)
    0.3 <= position_ratio <= 0.7 -> "中段" (mid)
    position_ratio > 0.7 -> "后段" (back)
    """
    pos = centroid[0] if centroid else 0.5
    if pos < 0.3:
        return "前段"
    elif pos <= 0.7:
        return "中段"
    else:
        return "后段"


def _compute_digit_characteristic(centroid: list[float]) -> str:
    """Determine digit characteristic from digit_density (dim 6).

    avg digit_density > 0.01 -> "数字密集" (digit-dense)
    else -> "文本为主" (text-dominant)
    """
    dd = centroid[6] if len(centroid) > 6 else 0.0
    return "数字密集" if dd > 0.01 else "文本为主"


def _auto_label(
    digit_char: str,
    top_bigrams: list[str],
    position_zone: str,
    max_bigrams_in_label: int = 2,
) -> str:
    """Generate a human-readable auto-label for a cluster.

    Format: "{digit_char}-{bigram1}{bigram2}-{zone}"

    Args:
        digit_char: "数字密集" or "文本为主"
        top_bigrams: Ordered list of top bigrams
        position_zone: "前段", "中段", or "后段"
        max_bigrams_in_label: How many bigrams to include in the label

    Returns:
        Human-readable label string.
    """
    bg_part = "".join(top_bigrams[:max_bigrams_in_label])
    if not bg_part:
        bg_part = "未分类"
    return f"{digit_char}-{bg_part}-{position_zone}"


# ═══════════════════════════════════════════════════════════════
# AutoClusterer
# ═══════════════════════════════════════════════════════════════

class AutoClusterer:
    """Unsupervised clause block clusterer with auto-labeling.

    Uses a hybrid distance metric (structural + numeric + n-gram) for
    DBSCAN clustering, then auto-generates human-readable labels for
    each cluster based on bigram profiles, digit density, and position.

    Usage:
        clusterer = AutoClusterer()
        clusters = clusterer.cluster(clause_blocks)
        for c in clusters:
            print(c.cluster_id, c.auto_label, c.member_count)

        fv = FeatureExtractor.extract([new_block])[0]
        match = clusterer.match_cluster(fv, clusters)
        if match:
            print(f"Matches: {match.auto_label}")
    """

    # Default DBSCAN parameters
    EPS = 0.35
    MIN_SAMPLES = 1

    # Noise assignment: hybrid-distance threshold for reassigning noise
    # to the nearest cluster centroid.  Only applies to the few points
    # that DBSCAN labels as -1.
    NOISE_MAX_DISTANCE = 0.35

    def __init__(
        self,
        eps: float = EPS,
        min_samples: int = MIN_SAMPLES,
        noise_max_distance: float = NOISE_MAX_DISTANCE,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.noise_max_distance = noise_max_distance
        self._feature_extractor = FeatureExtractor()
        self._last_labels: list[int] = []
        self._last_fvs: list[ClauseFeatureVector] = []

    # ── Public API ──

    def cluster(self, blocks: list[ClauseBlock]) -> list[AutoCluster]:
        """Cluster clause blocks by their hybrid-distance feature vectors.

        Steps:
          1. Extract feature vectors via FeatureExtractor
          2. Run HybridDBSCAN (structural + numeric + n-gram distance)
          3. Assign noise points to nearest cluster if within threshold
          4. Auto-name each cluster

        Args:
            blocks: List of ClauseBlock objects to cluster.

        Returns:
            List of AutoCluster objects, sorted by member_count descending.
        """
        if not blocks:
            return []

        # Step 1: Extract features
        feature_vectors = self._feature_extractor.extract(blocks)
        dense_vectors = [fv.to_dense() for fv in feature_vectors]
        sparse_ngrams = [fv.sparse_ngrams for fv in feature_vectors]

        self._last_fvs = list(feature_vectors)

        # Step 2: Run HybridDBSCAN
        hybrid = HybridDBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = hybrid.fit_predict(dense_vectors, sparse_ngrams)
        self._last_labels = list(labels)

        # Step 3: Assign noise to nearest cluster
        labels = self._assign_noise(dense_vectors, sparse_ngrams, labels)

        # Step 4: Build cluster objects
        clusters = self._build_clusters(
            blocks, feature_vectors, dense_vectors, labels
        )

        return clusters

    def match_cluster(
        self,
        fv: ClauseFeatureVector,
        clusters: list[AutoCluster],
    ) -> Optional[AutoCluster]:
        """Find the nearest cluster for a feature vector.

        Uses hybrid distance (structural + numeric + n-gram).

        Args:
            fv: A ClauseFeatureVector to match.
            clusters: List of AutoCluster objects.

        Returns:
            The nearest AutoCluster, or None if clusters is empty.
        """
        if not clusters:
            return None

        dense = fv.to_dense()
        sparse = fv.sparse_ngrams

        best_cluster = None
        best_dist = float("inf")

        for cluster in clusters:
            dist = _hybrid_distance(dense, cluster.centroid, sparse, {})
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster

        return best_cluster

    # ── Serialization ──

    def to_dict(self, clusters: list[AutoCluster]) -> dict:
        """Serialize clusters to a JSON-compatible dict."""
        return {
            "params": {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "noise_max_distance": self.noise_max_distance,
            },
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "centroid": c.centroid,
                    "member_count": c.member_count,
                    "top_bigrams": c.top_bigrams,
                    "digit_density_avg": c.digit_density_avg,
                    "position_zone": c.position_zone,
                    "auto_label": c.auto_label,
                    "block_indices": c.block_indices,
                }
                for c in clusters
            ],
        }

    @classmethod
    def from_dict(
        cls, data: dict
    ) -> tuple["AutoClusterer", list[AutoCluster]]:
        """Restore an AutoClusterer and its clusters from a dict.

        Returns:
            (AutoClusterer, list[AutoCluster])
        """
        params = data.get("params", {})
        clusterer = cls(
            eps=params.get("eps", cls.EPS),
            min_samples=params.get("min_samples", cls.MIN_SAMPLES),
            noise_max_distance=params.get(
                "noise_max_distance", cls.NOISE_MAX_DISTANCE
            ),
        )

        clusters = [
            AutoCluster(
                cluster_id=c["cluster_id"],
                centroid=c["centroid"],
                member_count=c["member_count"],
                top_bigrams=c["top_bigrams"],
                digit_density_avg=c["digit_density_avg"],
                position_zone=c["position_zone"],
                auto_label=c["auto_label"],
                block_indices=c["block_indices"],
            )
            for c in data.get("clusters", [])
        ]

        return clusterer, clusters

    # ── Internal methods ──

    def _assign_noise(
        self,
        dense_vectors: list[list[float]],
        sparse_ngrams: list[dict[int, float]],
        labels: list[int],
    ) -> list[int]:
        """Assign noise points to the nearest cluster centroid.

        A noise point is reassigned if its hybrid distance to the nearest
        cluster centroid is <= self.noise_max_distance.
        """
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return labels

        # Compute centroids of non-noise clusters
        centroids: dict[int, list[float]] = {}
        for label in unique_labels:
            members = [
                dense_vectors[i]
                for i in range(len(dense_vectors))
                if labels[i] == label
            ]
            if members:
                dim = len(members[0])
                centroid = [
                    sum(v[j] for v in members) / len(members)
                    for j in range(dim)
                ]
                centroids[label] = centroid

        if not centroids:
            return labels

        new_labels = list(labels)
        for i, label in enumerate(labels):
            if label != -1:
                continue

            best_label = -1
            best_dist = self.noise_max_distance
            for clabel, centroid in centroids.items():
                dist = _hybrid_distance(
                    dense_vectors[i], centroid,
                    sparse_ngrams[i], {},
                )
                if dist < best_dist:
                    best_dist = dist
                    best_label = clabel

            if best_label != -1:
                new_labels[i] = best_label

        return new_labels

    def _build_clusters(
        self,
        blocks: list[ClauseBlock],
        feature_vectors: list[ClauseFeatureVector],
        dense_vectors: list[list[float]],
        labels: list[int],
    ) -> list[AutoCluster]:
        """Build AutoCluster objects from DBSCAN labels."""
        # Group block indices by label
        label_to_indices: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            label_to_indices.setdefault(label, []).append(i)

        if not label_to_indices:
            all_indices = list(range(len(blocks)))
            label_to_indices[0] = all_indices

        clusters: list[AutoCluster] = []
        for sorted_idx, (db_label, indices) in enumerate(
            sorted(label_to_indices.items())
        ):
            member_blocks = [blocks[i] for i in indices]
            member_features = [dense_vectors[i] for i in indices]

            dim = len(member_features[0]) if member_features else 14
            centroid = [
                sum(fv[j] for fv in member_features) / len(member_features)
                for j in range(dim)
            ]

            top_bigrams = _top_bigrams_across_blocks(member_blocks, top_n=5)
            digit_density_avg = centroid[6] if dim > 6 else 0.0
            position_zone = _compute_position_zone(centroid)
            digit_char = _compute_digit_characteristic(centroid)
            label_text = _auto_label(digit_char, top_bigrams, position_zone)

            cluster = AutoCluster(
                cluster_id=f"C_{sorted_idx:03d}",
                centroid=centroid,
                member_count=len(member_blocks),
                top_bigrams=top_bigrams,
                digit_density_avg=digit_density_avg,
                position_zone=position_zone,
                auto_label=label_text,
                block_indices=sorted(indices),
            )
            clusters.append(cluster)

        # Sort by member_count descending, then re-assign IDs
        clusters.sort(key=lambda c: c.member_count, reverse=True)
        for i, cluster in enumerate(clusters):
            cluster.cluster_id = f"C_{i:03d}"

        return clusters


# ═══════════════════════════════════════════════════════════════
# Verification helpers
# ═══════════════════════════════════════════════════════════════

def _adjusted_rand_index(
    labels_true: list[int],
    labels_pred: list[int],
) -> float:
    """Compute Adjusted Rand Index (ARI) between two label assignments.

    Pure-Python implementation following Hubert & Arabie (1985).

    ARI ranges from -1 to 1, where 1 = perfect agreement.
    """
    n = len(labels_true)
    if n <= 1:
        return 0.0

    true_unique = sorted(set(labels_true))
    pred_unique = sorted(set(labels_true) | set(labels_pred))

    contingency: dict[int, dict[int, int]] = {}
    for t in true_unique:
        contingency[t] = {}
        for p in pred_unique:
            contingency[t][p] = 0

    for t, p in zip(labels_true, labels_pred):
        if t not in contingency:
            contingency[t] = {}
        contingency[t][p] = contingency[t].get(p, 0) + 1

    row_sums = [
        sum(row.values()) for row in contingency.values()
    ]
    col_sums_list = []
    for p in pred_unique:
        col_sum = sum(
            contingency[t].get(p, 0) for t in true_unique
        )
        col_sums_list.append(col_sum)

    def _comb2(x: int) -> int:
        return x * (x - 1) // 2 if x >= 2 else 0

    sum_row_comb = sum(_comb2(rs) for rs in row_sums)
    sum_col_comb = sum(_comb2(cs) for cs in col_sums_list)

    sum_ij_comb = 0
    for t in true_unique:
        for p in pred_unique:
            sum_ij_comb += _comb2(contingency[t].get(p, 0))

    total_comb = _comb2(n)
    expected = sum_row_comb * sum_col_comb / max(total_comb, 1)
    numerator = sum_ij_comb - expected
    denominator = (sum_row_comb + sum_col_comb) / 2.0 - expected

    if denominator == 0:
        return 0.0
    return numerator / denominator


def _build_expected_labels(
    examples: dict[str, list[str]],
) -> tuple[list[ClauseBlock], list[int]]:
    """Build ClauseBlock list and ground-truth integer labels from examples."""
    type_names = list(examples.keys())
    type_to_int = {name: i for i, name in enumerate(type_names)}

    blocks: list[ClauseBlock] = []
    labels: list[int] = []

    for type_name, texts in examples.items():
        for text in texts:
            title_line = text.split("\n")[0] if text else type_name
            block = ClauseBlock(
                clause_id=title_line,
                clause_title=title_line,
                clause_type=type_name,
                type_confidence=1.0,
                content=text,
                level=1,
            )
            blocks.append(block)
            labels.append(type_to_int[type_name])

    return blocks, labels


def _report_results(
    clusters: list[AutoCluster],
    ari: float,
    labels_true: list[int],
    type_names: list[str],
    examples: dict[str, list[str]],
) -> None:
    """Print a detailed verification report."""
    print("=" * 64)
    print("  DBSCAN Auto-Clusterer -- Verification Report")
    print("=" * 64)

    print(f"\n  Number of auto-clusters found:  {len(clusters)}")
    print(f"  Total blocks clustered:         {sum(c.member_count for c in clusters)}")
    print(f"  ARI vs manual labels:           {ari:.4f}")
    print()

    # Per-cluster details
    print("  -- Per-cluster auto-labels --")
    for cluster in clusters:
        print(f"    {cluster.cluster_id} | {cluster.auto_label}")
        print(f"          members={cluster.member_count}, "
              f"bigrams={cluster.top_bigrams[:3]}, "
              f"zone={cluster.position_zone}")
    print()

    # Ground truth -> auto-cluster mapping
    print("  -- Manual type -> auto-cluster membership --")
    label_to_cid: dict[int, str] = {}
    for cluster in clusters:
        for idx in cluster.block_indices:
            label_to_cid[idx] = cluster.cluster_id

    for type_name in type_names:
        type_examples = examples.get(type_name, [])
        start_idx = sum(
            len(examples[t])
            for t in type_names[: type_names.index(type_name)]
        )
        end_idx = start_idx + len(type_examples)
        cluster_ids = set()
        for i in range(start_idx, end_idx):
            cid = label_to_cid.get(i, "NOISE")
            cluster_ids.add(cid)
        print(
            f"    {type_name:<8} -> {', '.join(sorted(cluster_ids))}"
        )
    print()

    # ARI analysis
    print("  -- ARI Analysis --")
    if ari >= 0.9:
        print("    EXCELLENT: near-perfect agreement with manual labels.")
    elif ari >= 0.7:
        print("    GOOD: strong agreement with manual labels.")
    elif ari >= 0.5:
        print("    ACCEPTABLE: moderate agreement with manual labels.")
    else:
        print("    POOR: low agreement with manual labels.")
        print("    Analysis of merging/splitting issues:")

        type_to_clusters: dict[str, set[str]] = {}
        for type_name in type_names:
            type_to_clusters[type_name] = set()
            start_idx = sum(
                len(examples[t])
                for t in type_names[: type_names.index(type_name)]
            )
            for i in range(
                start_idx, start_idx + len(examples[type_name])
            ):
                cid = label_to_cid.get(i, "NOISE")
                type_to_clusters[type_name].add(cid)

        cluster_to_types: dict[str, set[str]] = {}
        for tn, cids in type_to_clusters.items():
            for cid in cids:
                cluster_to_types.setdefault(cid, set()).add(tn)

        for cid, types in sorted(cluster_to_types.items()):
            if len(types) > 1:
                print(
                    f"      MERGED: Cluster {cid} contains types: "
                    + ", ".join(sorted(types))
                )

        for tn, cids in sorted(type_to_clusters.items()):
            if len(cids) > 1:
                print(
                    f"      SPLIT: Type '{tn}' spans clusters: "
                    + ", ".join(sorted(cids))
                )

    print()
    print("  -- Conclusion --")
    if ari > 0.6:
        print("    Auto-clustering CAN replace the keyword table (ARI > 0.6).")
    else:
        print(
            "    Auto-clustering needs improvement before replacing keywords."
        )
        print(f"    (ARI = {ari:.4f}, threshold = 0.6)")

    print("=" * 64)


def main() -> None:
    """Run verification: load domain examples, cluster, compare with ARI."""
    # Resolve domain.json path:
    # script is at <project_root>/app/engine/auto_clusterer.py
    # domains are at <project_root>/domains/construction/domain.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    domain_path = os.path.join(
        project_root, "domains", "construction", "domain.json"
    )

    if not os.path.exists(domain_path):
        print(f"ERROR: domain.json not found at {domain_path}", file=sys.stderr)
        sys.exit(1)

    with open(domain_path, "r", encoding="utf-8") as f:
        domain_config = json.load(f)

    examples = domain_config.get("clause_type_examples", {})
    if not examples:
        print(
            "ERROR: No clause_type_examples found in domain.json",
            file=sys.stderr,
        )
        sys.exit(1)

    type_names = list(examples.keys())
    total_blocks = sum(len(texts) for texts in examples.values())

    print(
        f"Loaded {len(type_names)} types, {total_blocks} example texts"
    )
    print(f"Types: {', '.join(type_names)}")
    print(f"Using sklearn: {_HAS_SKLEARN}")
    print()

    # Build ClauseBlock list with ground-truth labels
    blocks, labels_true = _build_expected_labels(examples)

    # Cluster
    clusterer = AutoClusterer()
    clusters = clusterer.cluster(blocks)

    # Map auto-cluster labels to integer labels for ARI
    cluster_id_to_int: dict[str, int] = {}
    for i, cluster in enumerate(clusters):
        for idx in cluster.block_indices:
            cluster_id_to_int[idx] = i

    labels_pred = [
        cluster_id_to_int.get(i, -1) for i in range(len(blocks))
    ]

    ari = _adjusted_rand_index(labels_true, labels_pred)
    _report_results(
        clusters, ari, labels_true, type_names, examples
    )


if __name__ == "__main__":
    main()
