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

def _eval_sum_numeric_comparison(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """Sum all numbers matching a context, compare total against threshold."""
    label = rule.condition_params.get("label", rule.name)
    context_pat = rule.condition_params.get("context_pattern", "%%")
    unit = rule.condition_params.get("unit", "%%")
    operator = rule.condition_params.get("operator", "<=")
    expected = rule.condition_params.get("expected", 100)
    legal_ref = rule.condition_params.get("legal_ref", "")
    try: expected = float(expected)
    except (ValueError, TypeError): expected = 100
    search_text = text
    # WO-28: clause_blocks context window
    clause_blocks = rule.condition_params.get("_clause_blocks", None)
    if not clause_blocks:
        clause_blocks = getattr(self, "_clause_blocks", None)
    clause_type = rule.condition_params.get("clause_type") or getattr(rule, "clause_type", None)
    if clause_type and clause_blocks:
        clause_text_parts = []
        for cb in clause_blocks:
            if cb.get("clause_type") == clause_type:
                clause_text_parts.append(cb.get("content", ""))
        if clause_text_parts:
            search_text = "\n".join(clause_text_parts)
    if context_pat:
        try:
            all_ctx = list(re.finditer(context_pat, text))
            if all_ctx:
                s = max(0, all_ctx[0].start() - 300)
                e = min(len(text), all_ctx[-1].end() + 300)
                search_text = text[s:e]
        except re.error: pass
    numbers = self._extract_numbers(search_text)
    relevant = [n for n in numbers if n["unit"] == unit and n.get("kind") != "date"]
    if not relevant:
        numbers = self._extract_numbers(text)
        relevant = [n for n in numbers if n["unit"] == unit and n.get("kind") != "date"]
    if not relevant:
        return self._make_evidence(rule, Verdict.NOT_APPLICABLE, search_text[:200], [],
            "Cannot find any value with unit '%s' for sum." % unit, "", None)
    total = sum(n["value"] for n in relevant)
    raw_terms = [n["raw"] for n in relevant]
    op = {">=": lambda a,b: a>=b, "<=": lambda a,b: a<=b}
    ok = op.get(operator, lambda a,b: a<=b)(total, expected)
    if ok:
        return self._make_evidence(rule, Verdict.PASSED, ", ".join(raw_terms[:5]), raw_terms[:5],
            "Sum of %d values = %s%s, meets (%s %s%s)" % (len(relevant), total, unit, operator, expected, unit)
            + (". Ref: %s" % legal_ref if legal_ref else ""), "", None)
    else:
        return self._make_evidence(rule, Verdict.FAILED, ", ".join(raw_terms[:5]), raw_terms[:5],
            "Sum of %d values = %s%s, FAILS (%s %s%s)" % (len(relevant), total, unit, operator, expected, unit)
            + (". Ref: %s" % legal_ref if legal_ref else ""),
            "Total should be %s %s%s, is %s%s." % (operator, expected, unit, total, unit), None)

# ── contextual_co_occurrence ──


from app.engine.handlers._registry import register_handler
register_handler("sum_numeric_comparison", _eval_sum_numeric_comparison)
