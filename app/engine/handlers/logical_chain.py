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

def _eval_logical_chain(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """logical_chain: All premises present but conclusion absent -> FAILED."""
    premises = rule.condition_params.get("premises", [])
    conclusion = rule.condition_params.get("conclusion", "").lower()
    premise_terms = [p.lower() for p in premises]
    premises_present = [p for p in premise_terms if p in ngrams]
    conclusion_present = conclusion in ngrams
    if len(premises_present) == len(premise_terms) and not conclusion_present:
        fragment = self._extract_fragment(text, premise_terms)
        return self._make_evidence(
            rule, Verdict.FAILED, fragment, premise_terms,
            f"All premises present ({', '.join(premises)}) but conclusion '{conclusion}' absent.",
            f"Add '{conclusion}' to complete the logical chain.",
            None,
        )
    return self._make_evidence(
        rule, Verdict.PASSED, text[:200], premise_terms + ([conclusion] if conclusion_present else []),
        "Logical chain condition satisfied.",
        "", None,
    )


from app.engine.handlers._registry import register_handler
register_handler("logical_chain", _eval_logical_chain)
