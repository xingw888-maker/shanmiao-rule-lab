"""Clause Splitter — contract structural preprocessor.

This module sits BEFORE the validation engine's handler layer.  It splits
a full contract text into independent clause blocks based on Chinese legal
document structure (articles, sections, subsections).

Each clause block carries its own id, title, inferred type, and isolated
content text — enabling downstream rules to operate on clause-scoped
context rather than a ±N-character global window, which reduces
cross-clause noise (e.g. catching a "50-day construction period" as a
"warranty period" because both appear within a fixed character radius).

Design principle: structural parsing, NOT semantic understanding.  The
splitter relies on explicit formatting markers (numbered clauses, headers)
rather than NLP or AI.  This keeps it deterministic, fast, and auditable.

Classification: type inference uses dynamic feature-space classification
via AutoClusterer prototypes or a HybridClauseClassifier injected through
the split() method's classifier parameter.  No hardcoded Chinese keywords.

Usage:
    from app.engine.clause_splitter import ClauseSplitter
    blocks = ClauseSplitter.split(contract_text)
    for block in blocks:
        print(block.clause_id, block.clause_type, block.clause_title[:40])

    # With cluster-built classifier:
    from app.engine.clause_splitter import ClauseSplitter
    pool = ClauseSplitter.split(pool_text)
    classifier = ClauseSplitter.build_cluster_prototypes(pool)
    blocks = ClauseSplitter.split(text, classifier=classifier)

    # With domain-seeded dynamic classifier:
    from app.engine.dynamic_classifier import DynamicClauseClassifier
    dc = DynamicClauseClassifier()
    dc.seed_from_domain_config("/path/to/domain")
    from app.engine.dynamic_classifier import HybridClauseClassifier
    hybrid = HybridClauseClassifier(dynamic=dc)
    blocks = ClauseSplitter.split(text, classifier=hybrid)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Fallback type for anything that doesn't match
_DEFAULT_TYPE = "其他"

# ── Clause boundary patterns ──
# These regex patterns define the start of a new clause.
# Priority-ordered: more specific patterns first, general patterns later.

# Pattern 1: "第X条" / "第X章" where X is Chinese digits or Arabic digits
_RE_CN_ARTICLE = re.compile(
    r'第\s*'               # "第" + optional whitespace
    r'(?:'
    r'[一二三四五六七八九十\d]+'   # Chinese numerals or Arabic digits
    r')'
    r'\s*[条章]\s*'        # "条" or "章" + optional whitespace
)

# Pattern 2: "一、" / "二、" etc. (Chinese numeral with Chinese comma)
_RE_CN_SECTION = re.compile(
    r'^[一二三四五六七八九十]+[、]',
    re.MULTILINE,
)

# Pattern 3: "（一）" / "（二）" etc. (parenthesized Chinese numerals, at line start)
_RE_CN_PAREN = re.compile(
    r'^[（(][一二三四五六七八九十]+[）)]',
    re.MULTILINE,
)

# Pattern 4: "1." / "2." etc. (Arabic digit numbering at line start)
# Only match at line start to avoid matching numbers in running text
_RE_ARABIC_SECTION = re.compile(
    r'^\d+\s*[.、]',
    re.MULTILINE,
)

# Combine all article-level patterns into one for unified matching
# We keep them separate for priority matching but also combine for first-pass
_RE_CLAUSE_BOUNDARY = re.compile(
    r'(?:'
    r'第\s*(?:[一二三四五六七八九十\d]+)\s*[条章]'
    r')',
)


@dataclass
class ClauseBlock:
    """A single clause block extracted from a contract.

    Attributes:
        clause_id: Structural id (e.g. "第一条", "第七条", "三、", "（一）")
        clause_title: Full title line text (e.g. "第一条  工程概况")
        clause_type: Inferred type category (e.g. "工程概况", "付款", "验收")
        type_confidence: Confidence score of the type inference (0.0 to 1.0)
        content: Full text of the clause block (header + body)
        raw_lines: List of individual lines in this block
        level: Nesting level (0=header, 1=article/条, 2=subsection/款, etc.)
    """
    clause_id: str
    clause_title: str
    clause_type: str = _DEFAULT_TYPE
    type_confidence: float = 0.0
    content: str = ""
    raw_lines: list[str] = field(default_factory=list)
    level: int = 1
    # ── Structured extraction field (for Road 2 A/B validation) ──
    # Populated by external extractors (MockRuleExtractor, LLMRuleExtractor) for
    # comparative analysis against the regex-based handler path.
    # Format: {"numeric_values": [{"value": 50, "unit": "年"}], "subject": "...", "is_applicable": true}
    # Defaults to None — the field is a no-op in the normal engine flow.
    structured: Optional[dict] = None


def _extract_clause_id(title_line: str) -> str:
    """Extract the structural id from a clause title line.

    Strips whitespace and returns a normalized id string.
    E.g. "第一条  工程概况" -> "第一条", "三、" -> "三、"
    """
    title_stripped = title_line.strip()
    m = re.match(r'(第\s*[一二三四五六七八九十\d]+\s*[条章])', title_stripped)
    if m:
        return m.group(1).strip()
    m = re.match(r'^([一二三四五六七八九十]+)、', title_stripped)
    if m:
        return m.group(1) + "、"
    m = re.match(r'^[（(][一二三四五六七八九十]+[）)]', title_stripped)
    if m:
        return m.group(0)
    m = re.match(r'^(\d+)[.、]', title_stripped)
    if m:
        return m.group(1) + "."
    # Fallback: use the first 20 chars as pseudo-id
    return title_stripped[:20]


def _is_clause_boundary(line: str) -> bool:
    """Check if a line starts a new clause or section.

    Returns True for lines matching article-level markers like "第X条"
    or section-level markers like "一、".
    """
    line = line.strip()
    if not line:
        return False
    # Check article-level marker: 第N条 or 第N章
    if _RE_CN_ARTICLE.match(line):
        return True
    # Check Chinese section marker: 一、 二、 etc. at line start
    if _RE_CN_SECTION.match(line):
        return True
    # Check parenthesized marker: （一） （二） etc.
    # Only at sub-section level, but we still treat as a boundary
    if _RE_CN_PAREN.match(line):
        return True
    return False


def _infer_clause_type(clause: ClauseBlock) -> tuple[str, float]:
    """Infer the type of a clause — PLACEHOLDER now that keywords are removed.

    Always returns ("其他", 0.0).  Real classification happens via the
    classifier parameter of split().
    """
    return _DEFAULT_TYPE, 0.0


def _trim_content(content_lines: list[str]) -> list[str]:
    """Trim blank lines from both ends of a content block.

    Also removes trailing whitespace on each line for clean output.
    """
    if not content_lines:
        return content_lines
    # Strip trailing whitespace
    stripped = [line.rstrip() for line in content_lines]
    # Remove leading blank lines
    start = 0
    while start < len(stripped) and not stripped[start].strip():
        start += 1
    # Remove trailing blank lines
    end = len(stripped)
    while end > start and not stripped[end - 1].strip():
        end -= 1
    return stripped[start:end]


class ClauseSplitter:
    """Contract text clause splitter.

    Splits a full contract document into a list of ClauseBlock objects,
    each representing a structural section of the document.

    Example:
        blocks = ClauseSplitter.split(text)
        for b in blocks:
            print(f"{b.clause_id}: {b.clause_type}")
    """

    @classmethod
    def split(
        cls, text: str, classifier: Optional[object] = None,
    ) -> list[ClauseBlock]:
        """Split contract text into clause blocks.

        Args:
            text: Full contract text
            classifier: Optional HybridClauseClassifier for dynamic feature-space
                        type inference.  If None, falls back to keyword table.

        Returns:
            List of ClauseBlock objects, in document order.  The first block
            (before the first numbered clause) is type "头部".
        """
        if not text or not text.strip():
            return []

        defer = classifier is not None

        lines = text.splitlines()
        blocks: list[ClauseBlock] = []
        current_lines: list[str] = []
        current_title = ""
        current_id = ""
        has_header = False

        for i, line in enumerate(lines):
            if _is_clause_boundary(line):
                # Save the previous block if we have one
                if current_lines and current_id:
                    trimmed = _trim_content(current_lines)
                    if trimmed:
                        content = "\n".join(trimmed)
                        block = cls._make_block(current_id, current_title, content,
                                                defer_classification=defer)
                        blocks.append(block)
                elif current_lines and not current_id and not has_header:
                    # This is the header section (before first clause)
                    trimmed = _trim_content(current_lines)
                    if trimmed:
                        header_block = ClauseBlock(
                            clause_id="头部",
                            clause_title="合同头部",
                            clause_type="头部",
                            type_confidence=1.0,
                            content="\n".join(trimmed),
                            raw_lines=trimmed,
                            level=0,
                        )
                        blocks.append(header_block)
                        has_header = True

                # Start new clause
                current_title = line.strip()
                current_id = _extract_clause_id(line)
                current_lines = [line]
            else:
                if not current_lines:
                    current_lines = [line]
                else:
                    current_lines.append(line)

        # Save the last block
        if current_lines:
            trimmed = _trim_content(current_lines)
            if trimmed:
                content = "\n".join(trimmed)
                if current_id:
                    block = cls._make_block(current_id, current_title, content,
                                            defer_classification=defer)
                    blocks.append(block)
                else:
                    # Trailing content after last clause (footer)
                    if not has_header:
                        # If no header was found, this might be the only block
                        block = cls._make_block(current_id, current_title, content,
                                                defer_classification=defer)
                        blocks.append(block)
                    else:
                        # Footer content after the last clause
                        # Still create a block for completeness
                        footer_block = ClauseBlock(
                            clause_id="尾部",
                            clause_title="合同尾部",
                            clause_type="其他",
                            type_confidence=0.0,
                            content=content,
                            raw_lines=trimmed,
                            level=0,
                        )
                        blocks.append(footer_block)

        # If only one block and no id, it's the entire document (no struct found)
        if len(blocks) == 1 and not blocks[0].clause_id:
            blocks[0].clause_id = "全文"
            blocks[0].clause_title = "全文"
            blocks[0].level = 0

        # ── Apply dynamic classifier to all blocks (if provided) ──
        if classifier is not None:
            cls._apply_classifier(blocks, classifier, text)

        return blocks

    @classmethod
    def _make_block(
        cls, clause_id: str, clause_title: str, content: str,
        defer_classification: bool = False,
    ) -> ClauseBlock:
        """Create a ClauseBlock.

        When defer_classification=True, the block's type is left as "其他"
        (to be filled in later by _apply_classifier).  Since keyword-based
        classification has been removed, all blocks are always deferred;
        the defer parameter is kept for backward compatibility.
        """
        block = ClauseBlock(
            clause_id=clause_id,
            clause_title=clause_title,
            content=content,
            raw_lines=content.splitlines(),
            level=1,
        )
        return block
    @classmethod
    def _apply_classifier(
        cls, blocks: "list[ClauseBlock]", classifier: object, full_text: str,
    ) -> None:
        """Apply dynamic classifier to all blocks in a batch."""
        from app.engine.feature_extractor import FeatureExtractor
        try:
            from app.engine.dynamic_classifier import HybridClauseClassifier as HCC
        except ImportError:
            return
        if not isinstance(classifier, HCC):
            return
        features = FeatureExtractor.extract(blocks, full_text)
        if len(features) != len(blocks):
            return
        for block, fv in zip(blocks, features):
            content_head = block.content[:150] if block.content else ""
            type_name, confidence = classifier.classify(
                fv, clause_title=block.clause_title, content_head=content_head,
            )
            block.clause_type = type_name
            block.type_confidence = confidence

    # ── Cluster-based classifier builder ──

    @classmethod
    def build_cluster_prototypes(
        cls, clause_blocks: "list[ClauseBlock]",
    ) -> object:
        """Build a HybridClauseClassifier from unsupervised clustering.

        Feeds the given clause blocks through AutoClusterer, converts each
        discovered cluster to a TypePrototype, seeds a DynamicClauseClassifier,
        and wraps it in a HybridClauseClassifier.  The returned classifier
        can be passed to split().

        No keyword fallback is used — classification relies purely on
        feature-space prototypes.

        Args:
            clause_blocks: A list of ClauseBlock objects spanning multiple
                           contracts (a "cross-contract pool").

        Returns:
            HybridClauseClassifier instance, or None if clustering fails.
        """
        try:
            from app.engine.auto_clusterer import AutoClusterer
            from app.engine.dynamic_classifier import (
                DynamicClauseClassifier,
                TypePrototype,
                HybridClauseClassifier,
            )
            from app.engine.feature_extractor import FeatureExtractor
        except ImportError:
            return None

        if not clause_blocks:
            return None

        # Step 1: Cluster the input blocks
        clusterer = AutoClusterer()
        clusters = clusterer.cluster(clause_blocks)
        if not clusters:
            return None

        # Step 2: Extract feature vectors once (AutoClusterer already cached them)
        feature_vectors = clusterer._last_fvs
        if not feature_vectors:
            extractor = FeatureExtractor()
            feature_vectors = extractor.extract(clause_blocks)

        # Step 3: Build a DynamicClauseClassifier and seed from clusters
        dc = DynamicClauseClassifier()

        for cluster in clusters:
            auto_label = cluster.auto_label or f"cluster_{cluster.cluster_id}"
            proto = TypePrototype(
                type_name=auto_label,
                centroid=cluster.centroid,
                  example_count=cluster.member_count,
                structural_centroid=cluster.centroid[:6],
                numeric_centroid=cluster.centroid[6:10],
                ngram_centroid={},
                ngram_total_weight=0.0,
            )
            dc._prototypes[auto_label] = proto

        # Step 4: Wrap in HybridClauseClassifier (no keyword fallback)
        return HybridClauseClassifier(dynamic=dc, keyword_table={})
