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

def _eval_mutual_exclusion(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """mutual_exclusion: N terms cannot co-occur above a threshold."""
    terms = rule.condition_params.get("terms", [])
    threshold = rule.condition_params.get("threshold", 2)
    matched_terms = [t for t in terms if t.lower() in ngrams]
    if len(matched_terms) >= threshold:
        fragment = self._extract_fragment(text, matched_terms)
        msg = rule.message.format(matched=", ".join(matched_terms), threshold=threshold)
        return self._make_evidence(
            rule, Verdict.FAILED, fragment, matched_terms,
            f"Two mutually-exclusive terms co-occur. Threshold is {threshold - 1}, found {len(matched_terms)}.",
            f"Remove or qualify at least one of: {', '.join(matched_terms)}.",
            None,
        )
    return self._make_evidence(
        rule, Verdict.PASSED, text[:200], matched_terms,
        f"No mutual exclusion violation. Found {len(matched_terms)} of threshold {threshold}.",
            "", None,
    )


from app.engine.handlers._registry import register_handler
register_handler("mutual_exclusion", _eval_mutual_exclusion)
