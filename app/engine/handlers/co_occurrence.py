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

def _eval_co_occurrence(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """co_occurrence: If antecedent appears, consequent must also appear."""
    antecedent = rule.condition_params.get("antecedent", "").lower()
    consequent = rule.condition_params.get("consequent", "").lower()
    antecedent_present = antecedent in ngrams
    consequent_present = consequent in ngrams
    if antecedent_present and not consequent_present:
        fragment = self._extract_fragment(text, [antecedent])
        return self._make_evidence(
            rule, Verdict.FAILED, fragment, [antecedent],
            f"Antecedent '{antecedent}' present but consequent '{consequent}' absent.",
            f"Add '{consequent}' to satisfy co-occurrence requirement.",
            None,
        )
    return self._make_evidence(
        rule, Verdict.PASSED, text[:200], [antecedent, consequent],
        f"Co-occurrence condition met: antecedent '{antecedent}' and consequent '{consequent}'.",
        "", None,
    )


from app.engine.handlers._registry import register_handler
register_handler("co_occurrence", _eval_co_occurrence)
