"""VocabularyExtractor -- domain-agnostic, synchronous, LLM-free CJK term extraction.

Extracts meaningful multi-character terms from raw text using pure statistics:
- Sliding-window bigram/trigram frequencies
- Pointwise Mutual Information (PMI) cohesion scoring
- Distribution uniformity across text segments
- Composite ranking combining frequency, cohesion, and distribution

Output is compatible with CandidatePrototypeStore's bigram interface.

HAS_TRADITIONAL_CHINESE_SUPPORT = True  # Fallback when pmma unavailable
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

_CJK_RE = re.compile(r"[一-鿿]+")
_NON_CJK_WORD_RE = re.compile(r"[A-Za-z]{3,}(?:[-'][A-Za-z]+)*")

# ── PMI / cohesion computing ──
try:
    from pmma import bigram_frequencies as _pmma_bg
    _HAS_PMMA = True
except ImportError:
    _HAS_PMMA = False

# ── Common CJK stop bigrams ──
_STOP_BIGRAMS: set[str] = {
    "一个", "这个", "可以", "进行", "没有", "他们", "我们",
    "因为", "所以", "但是", "如果", "什么", "怎么", "已经",
    "还是", "不过", "或者", "而且", "虽然", "可是", "只是",
    "的话", "就是", "不是", "也是", "都是", "还有", "不会",
    "不能", "不要", "不可", "一种", "一些", "不同", "之间",
    "其中", "所有", "任何", "这样", "那样", "如何", "什么",
    "知道", "需要", "能够", "应该", "必须", "可能", "通过",
    "成为", "作为", "因此", "由于", "为了", "对于", "关于",
    "包括", "以及", "及其", "等等", "就是", "即",
    # Possessive / auxiliary / measure
    "的X", "X的", "是X", "X是", "在X", "X在", "和X",
    "了X", "着X", "过X", "把X", "被X", "从X", "向X",
    # Numeric
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
}

def _extract_bigrams(text: str) -> Counter[str]:
    """Extract CJK character bigrams; skip non-CJK spans.

    Returns a Counter of bigram -> count.
    """
    spans = _CJK_RE.findall(text)
    bg: Counter[str] = Counter()
    for span in spans:
        chars = list(span)
        for i in range(len(chars) - 1):
            bg[chars[i] + chars[i + 1]] += 1
    return bg

def _extract_trigrams(text: str) -> Counter[str]:
    """Extract CJK character trigrams; skip non-CJK spans."""
    spans = _CJK_RE.findall(text)
    tg: Counter[str] = Counter()
    for span in spans:
        chars = list(span)
        for i in range(len(chars) - 2):
            tg[chars[i] + chars[i + 1] + chars[i + 2]] += 1
    return tg


@dataclass
class ExtractionResult:
    """Result of a vocabulary extraction run."""

    terms: list[dict] = field(default_factory=list)
    bigrams: dict[str, int] = field(default_factory=dict)
    trigrams: dict[str, int] = field(default_factory=dict)
    domain_label: str = ""
    entity_groups: dict[str, list[str]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "terms": self.terms,
            "bigrams": self.bigrams,
            "trigrams": self.trigrams,
            "domain_label": self.domain_label,
            "entity_groups": self.entity_groups,
            "stats": self.stats,
        }


class VocabularyExtractor:
    """Statistical CJK vocabulary extractor.

    Extracts candidate terms by computing PMI-based cohesion scores,
    distribution uniformity, and composite ranking — no LLM, no API calls.

    Args:
        min_term_len: Minimum term length in characters (default 2).
        max_term_len: Maximum term length in characters (default 4).
        min_freq: Minimum frequency for a term to be considered (default 2).
        max_terms: Maximum number of terms in the ranked output (default 200).
        num_segments: Number of segments for distribution scoring (default 10).
    """

    def __init__(
        self,
        min_term_len: int = 2,
        max_term_len: int = 4,
        min_freq: int = 2,
        max_terms: int = 200,
        num_segments: int = 10,
    ):
        if min_term_len < 2:
            raise ValueError("min_term_len must be >= 2")
        if max_term_len < min_term_len:
            raise ValueError("max_term_len must be >= min_term_len")
        self.min_term_len = min_term_len
        self.max_term_len = max_term_len
        self.min_freq = min_freq
        self.max_terms = max_terms
        self.num_segments = num_segments

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, text: str) -> ExtractionResult:
        """Extract vocabulary from the given text.

        Args:
            text: Raw text to analyze.

        Returns:
            ExtractionResult with ranked terms, bigrams, and trigrams.
        """
        if not text.strip():
            return ExtractionResult(stats={"total_chars": 0, "total_bigrams": 0, "total_trigrams": 0})

        # 1. Extract CJK sequences and non-CJK word-like sequences
        cjk_sequences = self._extract_cjk_sequences(text)

        # 2. Compute character and bigram/trigram frequencies
        char_freq, bigram_freq, trigram_freq = self._compute_ngram_frequencies(cjk_sequences)
        total_chars = sum(char_freq.values())
        total_bigrams = sum(bigram_freq.values())
        total_trigrams = sum(trigram_freq.values())

        # 3. Generate candidate terms (sliding window over CJK sequences)
        candidate_terms: dict[str, int] = Counter()
        for seq in cjk_sequences:
            seq_len = len(seq)
            for term_len in range(self.min_term_len, min(self.max_term_len, seq_len) + 1):
                for i in range(seq_len - term_len + 1):
                    term = seq[i : i + term_len]
                    candidate_terms[term] += 1

        # 4. Add non-CJK terms (word-like sequences 3+ chars)
        non_cjk_terms: dict[str, int] = Counter()
        for m in _NON_CJK_WORD_RE.finditer(text):
            t = m.group()
            non_cjk_terms[t] += 1

        # 5. Filter by minimum frequency
        candidate_terms = Counter({t: f for t, f in candidate_terms.items() if f >= self.min_freq})

        # 6. Prepare text segments for distribution scoring
        text_len = len(text)
        segment_size = max(1, text_len // self.num_segments)
        segments = [text[i : i + segment_size] for i in range(0, text_len, segment_size)]

        # 7. Score each candidate
        scored_terms: list[dict] = []
        for term, freq in candidate_terms.items():
            cohesion = self._compute_cohesion(term, char_freq, bigram_freq, total_chars, total_bigrams)
            distribution = self._compute_distribution(term, segments)
            log_freq = math.log(freq + 1)
            log_cohesion = math.log(cohesion + 1e-10) if cohesion > 0 else -10.0
            score = log_freq * log_cohesion * distribution

            scored_terms.append({
                "term": term,
                "frequency": freq,
                "cohesion": round(cohesion, 4),
                "distribution": round(distribution, 4),
                "score": round(score, 4),
                "is_candidate": True,
            })

        # 8. Add non-CJK terms with a default score (no PMI available)
        for term, freq in non_cjk_terms.items():
            if freq < self.min_freq:
                continue
            distribution = self._compute_distribution(term, segments)
            score = math.log(freq + 1) * distribution
            scored_terms.append({
                "term": term,
                "frequency": freq,
                "cohesion": 0.0,
                "distribution": round(distribution, 4),
                "score": round(score, 4),
                "is_candidate": True,
            })

        # 9. Sort by score descending, take top max_terms
        scored_terms.sort(key=lambda x: -x["score"])
        scored_terms = scored_terms[: self.max_terms]

        # 10. Build stats
        stats = {
            "total_chars": total_chars,
            "total_bigrams": total_bigrams,
            "total_trigrams": total_trigrams,
            "candidates_scored": len(scored_terms),
            "unique_chars": len(char_freq),
            "unique_bigrams": len(bigram_freq),
            "unique_trigrams": len(trigram_freq),
        }

        return ExtractionResult(
            terms=scored_terms,
            bigrams=dict(bigram_freq.most_common()),
            trigrams=dict(trigram_freq.most_common()),
            stats=stats,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract_cjk_sequences(self, text: str) -> list[str]:
        """Split text into CJK character sequences (non-CJK chars act as delimiters)."""
        return _CJK_RE.findall(text)

    def _compute_ngram_frequencies(
        self, sequences: list[str]
    ) -> tuple[Counter, Counter, Counter]:
        """Compute character, bigram, and trigram frequencies from CJK sequences.

        Returns:
            Tuple of (char_freq, bigram_freq, trigram_freq) as Counters.
            Bigrams and trigrams use sliding windows with step 1.
        """
        char_freq: Counter = Counter()
        bigram_freq: Counter = Counter()
        trigram_freq: Counter = Counter()

        for seq in sequences:
            seq_len = len(seq)

            # Single characters
            for ch in seq:
                char_freq[ch] += 1

            # Bigra
            # Bigrams (sliding window, step 1)
            for i in range(seq_len - 1):
                bigram_freq[seq[i : i + 2]] += 1

            # Trigrams (sliding window, step 1)
            for i in range(seq_len - 2):
                trigram_freq[seq[i : i + 3]] += 1

        return char_freq, bigram_freq, trigram_freq

    def _compute_cohesion(
        self,
        term: str,
        char_freq: dict[str, int],
        bigram_freq: dict[str, int],
        total_chars: int = 0,
        total_bigrams: int = 0,
    ) -> float:
        """Compute PMI-based cohesion score for a multi-character term.

        For 2-character terms: log2(p(xy) / (p(x) * p(y)))
        For 3+ character terms: average of all adjacent bigram PMIs.

        Args:
            term: The candidate term (2-4 CJK characters).
            char_freq: Character frequency map.
            bigram_freq: Bigram frequency map.
            total_chars: Total character count (computed from char_freq if 0).
            total_bigrams: Total bigram count (computed from bigram_freq if 0).

        Returns:
            Cohesion score (float). Higher means more likely to be a real term.
        """
        n = len(term)
        if n < 2:
            return 0.0

        # Character probabilities
        total_chars = total_chars or sum(char_freq.values()) or 1
        total_bigrams = total_bigrams or sum(bigram_freq.values()) or 1

        if n == 2:
            # PMI for bigram
            p_xy = bigram_freq.get(term, 0) / total_bigrams
            p_x = char_freq.get(term[0], 0) / total_chars
            p_y = char_freq.get(term[1], 0) / total_chars
            if p_xy <= 0 or p_x <= 0 or p_y <= 0:
                return 0.0
            return math.log2(p_xy / (p_x * p_y))

        # For longer terms, average adjacent bigram PMIs
        cohesion = 0.0
        count = 0
        for i in range(n - 1):
            bg = term[i : i + 2]
            p_xy = bigram_freq.get(bg, 0) / total_bigrams
            p_x = char_freq.get(term[i], 0) / total_chars
            p_y = char_freq.get(term[i + 1], 0) / total_chars
            if p_xy > 0 and p_x > 0 and p_y > 0:
                cohesion += math.log2(p_xy / (p_x * p_y))
                count += 1
        return cohesion / count if count > 0 else 0.0

    def _compute_distribution(
        self,
        term: str,
        segments: list[str],
    ) -> float:
        """Compute how uniformly a term is distributed across text segments.
        
        Terms that appear evenly across all segments are more likely to be
        domain terms rather than incidental mentions.
        
        Args:
            term: The candidate term.
            segments: List of text segments (e.g., 10 equal chunks).
        
        Returns:
            Distribution uniformity score (0-1). Higher = more uniform.
        """
        if not segments:
            return 0.0
        
        n_segments = len(segments)
        seg_counts = [seg.count(term) for seg in segments]
        total = sum(seg_counts)
        if total == 0:
            return 0.0
        
        # Normalized entropy: 1 = perfectly uniform, 0 = concentrated in one segment
        entropy = 0.0
        for c in seg_counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(n_segments) if n_segments > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _compute_composite_score(
        self,
        freq: int,
        cohesion: float,
        uniformity: float,
    ) -> float:
        """Composite score combining frequency, cohesion, and uniformity.
        
        Args:
            freq: Raw term frequency.
            cohesion: PMI-based cohesion score.
            uniformity: Distribution uniformity (0-1).
        
        Returns:
            Composite score.
        """
        # Frequency component (log-scaled to dampen long-tail effects)
        freq_score = math.log2(1 + freq) / 10.0  # max ~0.5 for freq=15
        
        # Cohesion component (clamped to reasonable range)
        cohesion_score = min(max(cohesion / 10.0, 0.0), 1.0)
        
        # Weighted combination
        return 0.4 * freq_score + 0.4 * cohesion_score + 0.2 * uniformity
