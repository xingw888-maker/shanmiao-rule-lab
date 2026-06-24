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

def _eval_contextual_co_occurrence(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """contextual_co_occurrence: term A and term B must appear within N characters
    of each other in the text.  Unlike regular co_occurrence (which only checks
    global presence), this checks LOCAL adjacency — essential for conceptual
    domains where two terms are only meaningfully related if they appear in
    the same paragraph/context.

    Condition params:
      - term_a: str — first concept
      - term_b: str — second concept (must appear near term_a)
      - window_chars: int — max distance between the two terms (default 500)
      - min_occurrences: int — minimum number of co-occurrence windows (default 1)
    """
    term_a = rule.condition_params.get("term_a", "").lower()
    term_b = rule.condition_params.get("term_b", "").lower()
    window_chars = int(rule.condition_params.get("window_chars", 500))
    min_occurrences = int(rule.condition_params.get("min_occurrences", 1))

    if not term_a or not term_b:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, "", [],
            "Missing term_a or term_b for contextual_co_occurrence.", "", None,
        )

    text_lower = text.lower()

    # Find all positions of term_a
    positions_a = []
    idx = 0
    while True:
        pos = text_lower.find(term_a, idx)
        if pos == -1:
            break
        positions_a.append(pos)
        idx = pos + 1

    if not positions_a:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, text[:200], [],
            f"Term A '{term_a}' not found in text — contextual check not applicable.",
            "", None,
        )

    # For each occurrence of term_a, check if term_b appears within window_chars
    co_occurrences = 0
    for pos_a in positions_a:
        window_start = max(0, pos_a - window_chars)
        window_end = min(len(text_lower), pos_a + len(term_a) + window_chars)
        window = text_lower[window_start:window_end]
        if term_b in window:
            co_occurrences += 1

    if co_occurrences >= min_occurrences:
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], [term_a, term_b],
            f"'{term_a}' and '{term_b}' co-occur within {window_chars} chars "
            f"in {co_occurrences} instance(s) (min required: {min_occurrences}).",
            "", None,
        )
    else:
        return self._make_evidence(
            rule, Verdict.FAILED, text[:200], [term_a],
            f"'{term_a}' found but '{term_b}' does not appear within {window_chars} chars "
            f"in any of {len(positions_a)} occurrence(s) (found {co_occurrences}, need {min_occurrences}). "
            f"This may indicate the text mentions '{term_a}' without connecting it to '{term_b}'.",
            f"Ensure '{term_b}' is discussed near each occurrence of '{term_a}'.",
            None,
        )

# ── definition_contains ──


from app.engine.handlers._registry import register_handler
register_handler("contextual_co_occurrence", _eval_contextual_co_occurrence)
