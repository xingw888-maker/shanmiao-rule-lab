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

def _eval_required_pattern(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """required_pattern: required term NOT found → FAILED.

    Uses ngrams (CJK sliding-window tokens) for matching, so entity-alias
    expansion and Tokeniser segmentation both apply.

    Entity group expansion: if entity_groups were loaded from domain ontology,
    each term in the rule's terms list is expanded to all synonyms in its
    entity group before matching.  This catches real-world variants that
    rules.json could never exhaustively enumerate.
    """
    terms = rule.condition_params.get("terms", [])
    # If no explicit term list, extract from compiled pattern for backward compat
    if not terms and rule.compiled_pattern is not None:
        raw = rule.compiled_pattern.pattern
        inner = re.sub(r'^\(\?[imsx]*-?[imsx]*\)', '', raw)
        inner = inner.strip('()')
        terms = [t.strip() for t in inner.split('|') if t.strip()]

    # Save original terms BEFORE entity expansion — the negation check
    # (below) operates on these so that entity-expanded variants do not
    # override a negation of the actual required term.
    original_terms = list(terms)

    # ── Ontology-based term expansion ──
    # Build flat lookup once per matcher session (entity_groups don't change)
    if self._entity_groups:
        entity_lookup = self._build_entity_term_lookup()
        expanded = self._expand_required_terms(terms, entity_lookup)
        if expanded != terms:
            import logging as _log
            _log.getLogger(__name__).debug("Entity expansion[%s]: %s -> %s", rule.id, terms, expanded)
            terms = expanded
        else:
            # Debug: even when no change, log what groups are loaded
            import logging as _log
            _log.getLogger(__name__).debug("Entity expansion[%s]: no change (groups=%d, terms=%s)",
                                           rule.id, len(self._entity_groups), terms)

    matched = [t for t in terms if t.lower() in ngrams]
    if matched:
        # \u2500\u2500 Negation-context check for required_pattern \u2500\u2500
        # Check ORIGINAL terms (from rules.json, before entity expansion).
        # If all original terms appear only in negation contexts (e.g.
        # "\u4e0d\u53e6\u884c\u7b7e\u8ba2\u8d28\u91cf\u4fdd\u4fee\u4e66"), the contract explicitly refuses to
        # provide what's required => FAILED.  Entity-expanded terms alone
        # (e.g. "\u4fdd\u4fee" from "\u4fdd\u4fee\u4e8b\u5b9c") do NOT override a negation of
        # the original term.
        src = full_text or text
        original_matched = [t for t in original_terms if t.lower() in ngrams]
        if original_matched:
            all_negated = all(self._all_occurrences_negated(src, t)
                              for t in original_matched)
            if all_negated:
                return self._make_evidence(
                    rule, Verdict.FAILED, text[:200], matched,
                    f"Required term(s) negated in contract: {', '.join(original_matched)}. "
                    f"The contract explicitly says it will NOT provide these.",
                    f"Ensure '{', '.join(original_matched)}' is positively stated in the contract.",
                    None,
                )
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], matched,
            f"Required term(s) found: {', '.join(matched)}.",
            "", None,
        )
    if rule.compiled_pattern is not None and not terms:
        # Legacy fallback
        match = rule.compiled_pattern.search(text)
        if match:
            return self._make_evidence(
                rule, Verdict.PASSED, text[:200], [match.group()],
                f"Required pattern found: '{match.group()}'.",
                "", None,
            )
    missing_desc = ', '.join(terms) if terms else rule.condition_params.get('pattern', 'unknown')
    return self._make_evidence(
        rule, Verdict.FAILED, text[:200], [],
        f"Required term(s) NOT found: {missing_desc}.",
        f"Add '{missing_desc}' to the text.",
        None,
    )


from app.engine.handlers._registry import register_handler
register_handler("required_pattern", _eval_required_pattern)
