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

def _eval_definition_contains(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """definition_contains: when a concept is mentioned, its definition context
    must contain certain required sub-terms.  This checks STRUCTURED co-occurrence
    rather than pattern matching — it first locates the concept, then checks the
    surrounding text for required sub-terms.

    Condition params:
      - concept: str — the main concept to locate (e.g. "认知失调")
      - required_terms: list[str] — terms that must appear near the concept (e.g. ["态度", "不一致"])
      - window_chars: int — context window around concept (default 400)
      - min_ratio: float — fraction of required_terms that must appear (default 0.5)
    """
    concept = rule.condition_params.get("concept", "").lower()
    required_terms = rule.condition_params.get("required_terms", [])
    window_chars = int(rule.condition_params.get("window_chars", 400))
    min_ratio = float(rule.condition_params.get("min_ratio", 0.5))

    if not concept or not required_terms:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, "", [],
            "Missing concept or required_terms for definition_contains.", "", None,
        )

    text_lower = text.lower()

    # Find all occurrences of the concept
    positions = []
    idx = 0
    while True:
        pos = text_lower.find(concept, idx)
        if pos == -1:
            break
        positions.append(pos)
        idx = pos + 1

    if not positions:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, text[:200], [],
            f"Concept '{concept}' not found in text.", "", None,
        )

    # For each occurrence, check which required_terms appear nearby
    best_matched = []
    best_missing = []
    best_ratio = 0.0

    for pos in positions:
        window_start = max(0, pos - window_chars)
        window_end = min(len(text_lower), pos + len(concept) + window_chars)
        window = text_lower[window_start:window_end]

        matched = [t for t in required_terms if t.lower() in window]
        missing = [t for t in required_terms if t.lower() not in window]
        ratio = len(matched) / len(required_terms) if required_terms else 0.0

        if ratio > best_ratio:
            best_matched = matched
            best_missing = missing
            best_ratio = ratio

    if best_ratio >= min_ratio:
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], [concept] + best_matched,
            f"'{concept}' definition context contains {len(best_matched)}/{len(required_terms)} "
            f"required terms ({best_ratio:.0%}, threshold {min_ratio:.0%}). "
            f"Matched: {best_matched}. Missing: {best_missing}.",
            "", None,
        )
    else:
        return self._make_evidence(
            rule, Verdict.FAILED, text[:200], [concept] + best_matched,
            f"'{concept}' found but only {len(best_matched)}/{len(required_terms)} "
            f"required terms appear in its context ({best_ratio:.0%}, need {min_ratio:.0%}). "
            f"Matched: {best_matched}. Missing: {best_missing}. "
            f"The text may discuss '{concept}' without properly defining it.",
            f"Include the missing terms {best_missing} when discussing '{concept}'.",
            None,
        )


from app.engine.handlers._registry import register_handler
register_handler("definition_contains", _eval_definition_contains)
