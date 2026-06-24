"""
wo41-core-split: _eval_ handler extracted from core.py PythonMatcher.
This function is a class method of PythonMatcher — it accesses `self` for
_make_evidence, _check_negation_context, _all_occurrences_negated,
_parse_cn_number, _extract_numbers, _extract_fragment,
_build_entity_term_lookup, _expand_required_terms, _resolve_direction,
_tokeniser, and class-level constants.
"""
import re
from typing import Optional

# Types referenced from core.py (required for type annotation resolution at definition time)
from app.engine.core import CompiledRule, EvidenceItem, Verdict

def _eval_topic_coverage(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """topic_coverage: check whether the response text covers the user's key terms.
    Condition params:
      - source_keywords: list of keywords from user input
      - min_coverage_ratio: float 0-1, minimum fraction of keywords that must appear (default 0.5)
    Returns PASSED if coverage >= min_coverage_ratio, FAILED otherwise.
    Does NOT perform semantic analysis - only surface token matching.
    """
    source_keywords = rule.condition_params.get("source_keywords", [])
    min_coverage_ratio = float(rule.condition_params.get("min_coverage_ratio", 0.5))
    if not source_keywords:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, text[:200], [],
            "No source_keywords provided for topic_coverage check.",
            "", None,
        )
    text_lower = text.lower()
    matched = []
    missing = []
    for kw in source_keywords:
        if kw.lower() in text_lower:
            matched.append(kw)
        else:
            missing.append(kw)
    coverage = len(matched) / len(source_keywords) if source_keywords else 0.0
    if coverage >= min_coverage_ratio:
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], matched,
            f"Topic coverage: {len(matched)}/{len(source_keywords)} keywords matched ({coverage:.0%}), threshold {min_coverage_ratio:.0%}. "
            f"Matched: {matched}. Missing: {missing}.",
            "", None,
        )
    else:
        return self._make_evidence(
            rule, Verdict.FAILED, text[:200], matched,
            f"Topic coverage: {len(matched)}/{len(source_keywords)} keywords matched ({coverage:.0%}), below threshold {min_coverage_ratio:.0%}. "
            f"Missing: {missing}. The response may have drifted away from the user's core question.",
            f"Consider addressing the missing keywords: {missing}.",
            None,
        )


from app.engine.handlers._registry import register_handler
register_handler("topic_coverage", _eval_topic_coverage)
