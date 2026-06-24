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

def _eval_forbidden_pattern(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    """forbidden_pattern: any forbidden term found → FAILED.

    Uses ngrams (CJK sliding-window tokens) for matching rather than raw regex
    on the original text.  This means the Tokeniser's Unicode segmentation and
    entity-alias expansion both reach this handler.
    """
    terms = rule.condition_params.get("terms", [])
    # If no explicit term list, extract from compiled pattern for backward compat
    if not terms and rule.compiled_pattern is not None:
        raw = rule.compiled_pattern.pattern
        # Strip (?i) prefix and split OR groups
        inner = re.sub(r'^\(\?[imsx]*-?[imsx]*\)', '', raw)
        inner = inner.strip('()')
        terms = [t.strip() for t in inner.split('|') if t.strip()]

    matched = [t for t in terms if t.lower() in ngrams]
    # Fallback: check raw text for multi-char Chinese terms not in ngrams
    if not matched:
        for t in terms:
            if t in full_text or t in text:
                matched.append(t)
    if matched:
        # Negation context filter
        filtered = [t for t in matched
                    if not self._check_negation_context(full_text or text, t)]
        if not filtered:
            return self._make_evidence(
                rule, Verdict.PASSED, text[:200], matched,
                'All forbidden terms found in negation context — contract is compliant.',
                '', None,
            )
        matched = filtered
        fragment = self._extract_fragment(text, matched)
        return self._make_evidence(
            rule, Verdict.FAILED, fragment, matched,
            f"Forbidden term(s) found: {', '.join(matched)}.",
            f"Remove or rephrase: {', '.join(matched)}.",
            None,
        )
    if rule.compiled_pattern is not None and not terms:
        # Legacy fallback: no term extraction possible, use regex
        match = rule.compiled_pattern.search(text)
        if match:
            fragment = text[max(0, match.start()-20):min(len(text), match.end()+20)]
            return self._make_evidence(
                rule, Verdict.FAILED, fragment, [match.group()],
                f"Forbidden pattern matched: '{match.group()}'.",
                "Remove or rephrase the matched content.",
                None,
            )
    return self._make_evidence(
        rule, Verdict.PASSED, text[:200], [],
        f"No forbidden terms found.",
        "", None,
    )


from app.engine.handlers._registry import register_handler
register_handler("forbidden_pattern", _eval_forbidden_pattern)
