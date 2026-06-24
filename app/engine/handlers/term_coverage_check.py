# Term Coverage Check: Handler for candidate domain validation
#
# When a candidate domain has ontology terms but no rules, this handler
# provides a minimal validation by checking ontology term coverage in the
# input text.  Think of it as "does this text even talk about what this
# domain claims to cover?"
#
# Condition type: "term_coverage_check"
# Condition params:
#   - entity_groups: dict[str, list[str]]  (e.g. {"保修": ["保修","防水","防渗漏"]})
#   - min_entity_groups_present: int       (minimum number of entity groups needed to PASS, default 1)
#
# Returns: PASSED if entity_groups_present >= min_entity_groups_present, else FAILED
#
# Constitution 1.2: New handler — no modification to frozen handlers.

from __future__ import annotations

import re as _re
from typing import Optional

from app.engine.core import CompiledRule, EvidenceItem, Verdict


def _eval_term_coverage_check(
    self, rule: CompiledRule, text: str,
    tokens: list[str], ngrams: set[str], full_text: str,
) -> EvidenceItem:
    """Check whether input text contains terms from the candidate domain's ontology.

    Args:
        rule: Compiled rule whose condition_params must include `entity_groups`
               and `min_entity_groups_present`.
        text: The text to check.
        tokens, ngrams, full_text: standard handler signature (may be unused).

    Returns:
        EvidenceItem with PASSED/FAILED status.
    """
    entity_groups = rule.condition_params.get("entity_groups", {})
    min_entity_groups_present = int(
        rule.condition_params.get("min_entity_groups_present", 1)
    )

    if not entity_groups:
        return self._make_evidence(
            rule, Verdict.NOT_APPLICABLE, text[:200], [],
            "No entity_groups defined for term_coverage_check.",
            "", None,
        )

    total_groups = len(entity_groups)
    groups_present = 0
    groups_absent = 0
    matched_terms: list[str] = []
    missing_groups: list[str] = []

    for group_name, terms in entity_groups.items():
        found_any = False
        for term in terms:
            if term in text:
                matched_terms.append(term)
                found_any = True
        if found_any:
            groups_present += 1
        else:
            groups_absent += 1
            missing_groups.append(group_name)

    coverage_ratio = groups_present / max(total_groups, 1)

    if groups_present >= min_entity_groups_present:
        return self._make_evidence(
            rule, Verdict.PASSED, text[:200], matched_terms,
            (
                f"Term coverage: {groups_present}/{total_groups} entity groups "
                f"matched ({coverage_ratio:.0%}). "
                f"Matched terms: {matched_terms}. "
                f"Missing groups: {missing_groups if missing_groups else 'none'}."
            ),
            "", None,
        )
    else:
        return self._make_evidence(
            rule, Verdict.FAILED, text[:200], matched_terms,
            (
                f"Term coverage: only {groups_present}/{total_groups} entity groups "
                f"matched ({coverage_ratio:.0%}), required at least "
                f"{min_entity_groups_present}. "
                f"Missing groups: {missing_groups}."
            ),
            (
                f"Consider adding content about: {', '.join(missing_groups)}."
            ),
            None,
        )


from app.engine.handlers._registry import register_handler
register_handler("term_coverage_check", _eval_term_coverage_check)
