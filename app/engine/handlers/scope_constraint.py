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

def _eval_scope_constraint(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """scope_constraint: Evaluate delegate rule only within scoped segments."""
    scope = rule.condition_params.get("scope", "").lower()
    delegate = rule.condition_params.get("delegate")
    if delegate is None:
        return self._make_evidence(rule, Verdict.NOT_APPLICABLE, "", [], "No delegate rule defined.", "", None)
    # Tokenise scope for matching
    scope_tokens = self._tokeniser.tokenise(scope)
    # Find text portions matching the scope
    # Simple approach: look at the full text for scope indicators
    scope_matches = [t for t in scope_tokens if t in ngrams]
    if not scope_matches:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, "", [],
            f"Scope '{scope}' not applicable to this input.", "", None,
        )
    # Compile delegate rule on the fly
    try:
        if isinstance(delegate, dict):
            delegate_compiled = PythonRuleCompiler._compile_rule(
                delegate, rule.package_id, rule.package_version
            )
        else:
            return self._make_evidence(rule, Verdict.NOT_APPLICABLE, "", [], "Invalid delegate rule format.", "", None)
    except Exception as e:
        return self._make_evidence(rule, Verdict.NOT_APPLICABLE, "", [], f"Delegate compilation failed: {e}", "", None)
    # Evaluate delegate against text containing scope
    return self.evaluate(delegate_compiled, text, tokens, ngrams,
                         contract_broad_type="", estimated_value=0.0)
# ── numeric_comparison ──
# Chinese number parser (一=1, 二=2, ..., 十=10, 百=100, 十五=15, 三百=300, etc.)
_CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100,
    "千": 1000, "万": 10000,
}
_CN_UNIT_RE = re.compile(r'[万亿千百十]')
_NUM_VALUE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(个月|[年月日天元万%％])')
_CN_NUM_RE = re.compile(r'[零一二两三四五六七八九十百千万亿]+')
# Chinese percentage/permille patterns: 百分之三 → 3%, 万分之五 → 0.05%
_CN_PERCENT_RE = re.compile(r'百分之([零一二两三四五六七八九十百千万]+)')
_CN_PERMYRIAD_RE = re.compile(r'万分之([零一二两三四五六七八九十百千万]+)')


from app.engine.handlers._registry import register_handler
register_handler("scope_constraint", _eval_scope_constraint)
