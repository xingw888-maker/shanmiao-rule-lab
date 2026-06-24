"""Feature Extractor — domain-agnostic clause block feature vectors.

This module extracts structured feature vectors from ClauseBlock objects.
The features are purely statistical/structural — zero domain knowledge, zero
keyword tables, zero Chinese-specific assumptions beyond Unicode CJK ranges.

A feature vector is a flat list of floats that can be compared via cosine
similarity.  Clauses with similar structure, similar character n-gram profiles,
and similar numeric density cluster together — whether they come from a
construction contract, a philosophy paper, or a medical guideline.

Design principle: extract what the text IS, not what it MEANS.  The downstream
classifier decides which clusters correspond to which types.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# ── CJK character ranges ──
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
]


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


# ── Structural marker detection ──
_RE_ARTICLE = re.compile(r'第\s*[一二三四五六七八九十\d]+\s*[条章]')
_RE_SECTION = re.compile(r'^[一二三四五六七八九十]+[、]')
_RE_PAREN = re.compile(r'^[（(][一二三四五六七八九十]+[）)]')
_RE_ARABIC = re.compile(r'^\d+\s*[.、]')


def _detect_structural_marker(title: str) -> tuple[str, int]:
    """Return (marker_type, level) for a clause title line."""
    if _RE_ARTICLE.search(title):
        return "article", 1
    if _RE_SECTION.match(title.strip()):
        return "section", 2
    if _RE_PAREN.match(title.strip()):
        return "paren", 3
    if _RE_ARABIC.match(title.strip()):
        return "arabic", 2
    return "none", 1


@dataclass
class ClauseFeatureVector:
    """Feature vector for a single clause block.

    All features are numeric and normalized to [0,1] range where possible,
    making them directly usable for cosine similarity and clustering.
    """

    clause_id: str = ""
    clause_title: str = ""

    # ── Structural features (6 dims) ──
    position_ratio: float = 0.0          # 0-1 position in document
    block_length_chars: float = 0.0      # normalized by max_block_len
    block_length_lines: float = 0.0      # normalized by max_lines
    structural_level: float = 0.0        # normalized: 0=head, 0.33=article, 0.66=section, 1=paren
    structural_marker_type: float = 0.0  # one-hot-ish: article=1.0, section=0.66, arabic=0.66, paren=0.33, none=0.0
    avg_line_length: float = 0.0         # normalized by 200 chars

    # ── Numeric features (4 dims) ──
    digit_density: float = 0.0           # ratio of digit chars to total chars
    number_token_count: float = 0.0      # normalized count of numeric tokens (Arabic + Chinese)
    chinese_numeral_count: float = 0.0   # normalized count of specifically Chinese numerals
    percentage_count: float = 0.0        # normalized count of % or ％ tokens

    # ── Text characteristic features (4 dims) ──
    cjk_ratio: float = 0.0              # ratio of CJK chars to total chars
    punctuation_density: float = 0.0    # ratio of punctuation to total chars
    unique_char_ratio: float = 0.0      # type/token ratio for characters
    line_variation: float = 0.0         # std dev of line lengths / mean (measures structured vs prose)

    # ── Character n-gram profile (top-K bigrams as sparse indices) ──
    # Instead of full n-gram vectors (which are huge), we keep the top-20
    # bigrams and their normalized frequencies.  For similarity, we use
    # Jaccard on the bigram sets weighted by frequency.
    top_bigrams: list[tuple[str, float]] = field(default_factory=list)
    top_trigrams: list[tuple[str, float]] = field(default_factory=list)

    # ── Full sparse vector for precise similarity ──
    # This is a sparse dict: {ngram_hash: normalized_frequency}
    # Only populated when needed for similarity computation.
    sparse_ngrams: dict[int, float] = field(default_factory=dict)

    def to_dense(self) -> list[float]:
        """Return dense feature vector (first 14 numeric dims) for clustering.

        The n-gram profiles are NOT included in the dense vector — they're
        used separately via sparse cosine similarity.
        """
        return [
            self.position_ratio,
            self.block_length_chars,
            self.block_length_lines,
            self.structural_level,
            self.structural_marker_type,
            self.avg_line_length,
            self.digit_density,
            self.number_token_count,
            self.chinese_numeral_count,
            self.percentage_count,
            self.cjk_ratio,
            self.punctuation_density,
            self.unique_char_ratio,
            self.line_variation,
        ]

    @property
    def dense_dim(self) -> int:
        return 14

    def structural_similarity(self, other: "ClauseFeatureVector") -> float:
        """Cosine similarity of structural features only (first 6 dims)."""
        return _cosine(self.to_dense()[:6], other.to_dense()[:6])

    def numeric_similarity(self, other: "ClauseFeatureVector") -> float:
        """Cosine similarity of numeric features only (dims 6-10)."""
        return _cosine(self.to_dense()[6:10], other.to_dense()[6:10])

    def ngram_similarity(self, other: "ClauseFeatureVector") -> float:
        """Jaccard-weighted similarity of bigram profiles."""
        if not self.sparse_ngrams or not other.sparse_ngrams:
            return _cosine(self.to_dense(), other.to_dense())
        return _sparse_cosine(self.sparse_ngrams, other.sparse_ngrams)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Character n-gram extraction ──
_CJK_CHAR = re.compile(r'[一-鿿㐀-䶿豈-﫿]')
_DIGIT = re.compile(r'\d')
_CN_NUM = re.compile(r'[零一二三四五六七八九十百千万亿]')
_PUNCT = re.compile(r'[，。、；：！？·「」『』（）【】《》—…\-,.;:!?()\[\]{}""'']')
_PERCENT = re.compile(r'[%％]')
_NUMBER_TOKEN = re.compile(r'\d+(?:\.\d+)?|[零一二三四五六七八九十百千万亿]+')


def _char_bigrams(text: str) -> Counter:
    """Extract character bigrams from text, CJK-only."""
    cjk_chars = _CJK_CHAR.findall(text)
    bigrams = Counter()
    for i in range(len(cjk_chars) - 1):
        bigrams[cjk_chars[i] + cjk_chars[i + 1]] += 1
    return bigrams


def _char_trigrams(text: str) -> Counter:
    """Extract character trigrams from text, CJK-only."""
    cjk_chars = _CJK_CHAR.findall(text)
    trigrams = Counter()
    for i in range(len(cjk_chars) - 2):
        trigrams[cjk_chars[i] + cjk_chars[i + 1] + cjk_chars[i + 2]] += 1
    return trigrams


def _ngram_hash(ngram: str) -> int:
    """Simple hash for n-gram strings."""
    h = 0
    for ch in ngram:
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return h


class FeatureExtractor:
    """Extract ClauseFeatureVector from clause blocks.

    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract(clause_blocks)
        for fv in features:
            print(fv.clause_id, fv.to_dense())
    """

    @classmethod
    def extract(cls, blocks: list, full_text: str = "") -> list[ClauseFeatureVector]:
        """Extract feature vectors from a list of clause blocks.

        Args:
            blocks: List of ClauseBlock objects or dicts with clause_id,
                    clause_title, content, level fields.
            full_text: Full document text (for position calculation). If empty,
                       position is estimated from block order.

        Returns:
            List of ClauseFeatureVector, one per block.
        """
        if not blocks:
            return []

        total_chars = len(full_text) if full_text else sum(
            len(b.content if hasattr(b, 'content') else b.get('content', ''))
            for b in blocks
        )

        # Find global max for normalization
        max_block_len = max(
            len(b.content if hasattr(b, 'content') else b.get('content', ''))
            for b in blocks
        ) or 1
        max_lines = max(
            (b.content if hasattr(b, 'content') else b.get('content', '')).count('\n') + 1
            for b in blocks
        ) or 1

        accumulated_chars = 0
        features: list[ClauseFeatureVector] = []

        for i, block in enumerate(blocks):
            if hasattr(block, 'content'):
                content = block.content
                clause_id = block.clause_id
                clause_title = block.clause_title
                level = block.level
            else:
                content = block.get('content', '')
                clause_id = block.get('clause_id', f'block_{i}')
                clause_title = block.get('clause_title', '')
                level = block.get('level', 1)

            fv = cls._extract_single(
                clause_id=clause_id,
                clause_title=clause_title,
                content=content,
                level=level,
                block_index=i,
                total_blocks=len(blocks),
                total_chars=total_chars,
                accumulated_chars=accumulated_chars,
                max_block_len=max_block_len,
                max_lines=max_lines,
                top_n_ngrams=20,
            )
            features.append(fv)
            accumulated_chars += len(content)

        return features

    @classmethod
    def _extract_single(
        cls,
        clause_id: str,
        clause_title: str,
        content: str,
        level: int,
        block_index: int,
        total_blocks: int,
        total_chars: int,
        accumulated_chars: int,
        max_block_len: int,
        max_lines: int,
        top_n_ngrams: int = 20,
    ) -> ClauseFeatureVector:
        """Extract features from a single clause block."""

        text = content
        text_len = len(text)
        lines = text.splitlines()
        n_lines = len(lines) if lines and any(l.strip() for l in lines) else 1

        # ── Structural features ──
        position_ratio = accumulated_chars / max(total_chars, 1)
        block_length_chars = min(text_len / max_block_len, 1.0)
        block_length_lines = min(n_lines / max_lines, 1.0)

        # Level normalization: 0→0, 1→0.33, 2→0.66, 3+→1.0
        structural_level = min(level / 3.0, 1.0) if level > 0 else 0.0

        marker_type, _ = _detect_structural_marker(clause_title)
        marker_map = {"article": 1.0, "section": 0.66, "arabic": 0.66, "paren": 0.33, "none": 0.0}
        structural_marker_type = marker_map.get(marker_type, 0.0)

        # Average line length (normalized to 200 chars)
        line_lengths = [len(l.strip()) for l in lines if l.strip()]
        avg_line = sum(line_lengths) / max(len(line_lengths), 1)
        avg_line_length = min(avg_line / 200.0, 1.0)

        # ── Numeric features ──
        digit_chars = len(_DIGIT.findall(text))
        cjk_chars = _CJK_CHAR.findall(text)
        cjk_char_count = len(cjk_chars)
        total_chars_in_block = max(text_len, 1)

        digit_density = digit_chars / total_chars_in_block

        num_tokens = _NUMBER_TOKEN.findall(text)
        number_token_count = min(len(num_tokens) / 30.0, 1.0)  # cap at ~30 numbers

        cn_nums = _CN_NUM.findall(text)
        chinese_numeral_count = min(len(cn_nums) / 30.0, 1.0)

        pct_count = len(_PERCENT.findall(text))
        percentage_count = min(pct_count / 10.0, 1.0)

        # ── Text characteristic features ──
        cjk_ratio = cjk_char_count / total_chars_in_block

        punct_chars = len(_PUNCT.findall(text))
        punctuation_density = punct_chars / total_chars_in_block

        unique_chars = len(set(text))
        unique_char_ratio = unique_chars / max(total_chars_in_block, 1)

        # Line variation (std / mean) — higher = more varied structure
        if len(line_lengths) >= 2:
            mean_ll = sum(line_lengths) / len(line_lengths)
            if mean_ll > 0:
                var_ll = sum((l - mean_ll) ** 2 for l in line_lengths) / len(line_lengths)
                line_variation = min(math.sqrt(var_ll) / mean_ll, 2.0) / 2.0
            else:
                line_variation = 0.0
        else:
            line_variation = 0.0

        # ── Character n-gram profiles ──
        bigrams = _char_bigrams(text)
        trigrams = _char_trigrams(text)

        total_bigrams = sum(bigrams.values()) or 1
        total_trigrams = sum(trigrams.values()) or 1

        top_bigrams = [
            (bg, count / total_bigrams)
            for bg, count in bigrams.most_common(top_n_ngrams)
        ]
        top_trigrams = [
            (tg, count / total_trigrams)
            for tg, count in trigrams.most_common(top_n_ngrams)
        ]

        # ── Sparse n-gram vector for similarity ──
        sparse: dict[int, float] = {}
        for bg, count in bigrams.items():
            sparse[_ngram_hash(bg)] = count / total_bigrams
        for tg, count in trigrams.items():
            # Use a different hash offset to avoid collisions
            sparse[_ngram_hash(tg) ^ 0x80000000] = count / total_trigrams

        return ClauseFeatureVector(
            clause_id=clause_id,
            clause_title=clause_title,
            position_ratio=position_ratio,
            block_length_chars=block_length_chars,
            block_length_lines=block_length_lines,
            structural_level=structural_level,
            structural_marker_type=structural_marker_type,
            avg_line_length=avg_line_length,
            digit_density=digit_density,
            number_token_count=number_token_count,
            chinese_numeral_count=chinese_numeral_count,
            percentage_count=percentage_count,
            cjk_ratio=cjk_ratio,
            punctuation_density=punctuation_density,
            unique_char_ratio=unique_char_ratio,
            line_variation=line_variation,
            top_bigrams=top_bigrams,
            top_trigrams=top_trigrams,
            sparse_ngrams=sparse,
        )
