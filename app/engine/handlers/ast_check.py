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

def _eval_ast_check(
    self, rule, text, tokens, ngrams, full_text
):
    """ast_check: Tree-sitter AST structure patterns for code review."""
    lang = rule.condition_params.get("language", "javascript")
    check = {
        "search_type": rule.condition_params.get("search_type", "forbidden_node"),
        "node_type": rule.condition_params.get("node_type", ""),
        "node_pattern": rule.condition_params.get("node_pattern", ""),
        "min_count": int(rule.condition_params.get("min_count", 1)),
    }
    try:
        from app.engine.ast_checker import check_ast
        hits = check_ast(text, lang, check)
    except Exception as e:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, text[:200], [],
            f"AST check error: {e}", "", None,
        )
    if not hits:
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], [],
            f"AST check '{check['search_type']}' passed: no violations.", "", None,
        )
    first = hits[0]
    return self._make_evidence(
        rule, Verdict.FAILED, first.node_text, [],
        f"AST check '{check['search_type']}' found {len(hits)} violation(s). "
        f"First: line {first.line} [{first.node_type}] {first.node_text[:120]}",
        f"Review line {first.line}: {first.node_text[:100]}",
        None,
    )



from app.engine.handlers._registry import register_handler
register_handler("ast_check", _eval_ast_check)
