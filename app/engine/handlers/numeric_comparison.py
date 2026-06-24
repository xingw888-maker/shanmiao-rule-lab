"""
wo41-core-split: _eval_ handler extracted from core.py PythonMatcher.
Road 2: reads self._structured_inputs[rule.id] (sidecar) before falling back to regex.
R2.10: null-value signal — when model returns value=null, handler returns NOT_APPLICABLE
       instead of falling back to regex (which would produce false positives).
"""
import re
from typing import Optional

from app.engine.core import CompiledRule, EvidenceItem, Verdict

def _eval_numeric_comparison(
    self, rule: CompiledRule, text: str, tokens: list[str], ngrams: set[str], full_text: str
) -> EvidenceItem:
    label = rule.condition_params.get("label", rule.name)
    context_pat = rule.condition_params.get("context_pattern", "")
    unit = rule.condition_params.get("unit", "年")
    operator = rule.condition_params.get("operator", ">=")
    expected = rule.condition_params.get("expected", 0)
    legal_ref = rule.condition_params.get("legal_ref", "")
    try:
        expected = float(expected)
    except (ValueError, TypeError):
        expected = 0

    # ── Road 2 + R2.10: Sidecar structured input ──
    si = getattr(self, "_structured_inputs", {}).get(rule.id)
    if si:
        confidence = float(si.get("confidence", 0.5))
        # R2.10: null-value signal — model explicitly says "no such value"
        if si.get("value") is None:
            if confidence >= 0.3:
                return self._make_evidence(
                    rule, Verdict.NOT_APPLICABLE, "", [],
                    f"Field '{label}': model extraction returned no numeric value "
                    f"(confidence={confidence:.2f}). Skipping regex fallback to avoid false positive.",
                    f"Manual review may be required. Source: {si.get('source_text', 'N/A')[:120]}",
                    None,
                )
            # confidence too low, fall through to regex as safety net
        else:
            # Normal path: model extracted a value
            val = float(si["value"])
            raw_str = str(si.get("value", ""))
            raw_pos = 0
            contract_value = {"value": val, "raw": raw_str, "position": 0}
            if si.get("unit"):
                unit = si["unit"]
            direction_op = si.get("operator_hint") or None
            direction_note = f" [structured: {direction_op}]" if direction_op else ""

            effective_op = direction_op if direction_op else operator
            op_map = {
                ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b, "<": lambda a, b: a < b,
                "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
            }
            compare = op_map.get(effective_op, op_map.get(operator, lambda a, b: a >= b))
            ok = compare(val, expected)
            ref_suffix = (". Ref: " + legal_ref) if legal_ref else ""
            if ok:
                return self._make_evidence(
                    rule, Verdict.PASSED, contract_value["raw"], [contract_value["raw"]],
                    f"Field '{label}': contract value {val}{unit} meets legal threshold "
                    f"({effective_op} {expected}{unit}){ref_suffix}.{direction_note}",
                    "", None,
                )
            else:
                gap = abs(expected - val)
                if direction_op:
                    suggest = (f"'{label}' is {val}{unit} ({direction_op}), "
                               f"does not meet legal threshold {operator} {expected}{unit}.")
                else:
                    suggest = f"Increase '{label}' from {val}{unit} to at least {expected}{unit}."
                return self._make_evidence(
                    rule, Verdict.FAILED, contract_value["raw"], [contract_value["raw"]],
                    f"Field '{label}': contract value {val}{unit} FAILS legal threshold "
                    f"({effective_op} {expected}{unit}){ref_suffix}. Gap: {gap}{unit}.{direction_note}",
                    suggest, None,
                )

    # ── Fallback: regex path (si absent or low-confidence null) ──
    # WO-28: clause_blocks context window — if rule has clause_type, restrict
    # regex search to matching clause blocks to prevent cross-clause false positives.
    search_text = text
    clause_blocks = rule.condition_params.get("_clause_blocks", None)
    if not clause_blocks:
        # Try to get clause_blocks from the matcher's sidecar (set by core.py evaluate())
        clause_blocks = getattr(self, "_clause_blocks", None)
    clause_type = rule.condition_params.get("clause_type") or getattr(rule, "clause_type", None)
    if clause_type and clause_blocks:
        clause_text_parts = []
        for cb in clause_blocks:
            if cb.get("clause_type") == clause_type:
                clause_text_parts.append(cb.get("content", ""))
        if clause_text_parts:
            search_text = "\n".join(clause_text_parts)
    best_start, best_end, best_ctx_pos = 0, 0, 0
    best_count = -1
    all_matches = []
    if context_pat:
        try:
            all_matches = list(re.finditer(context_pat, text))
            if not all_matches:
                return self._make_evidence(
                    rule, Verdict.NOT_APPLICABLE, text[:200], [],
                    f"Rule '{rule.name}' not applicable: context_pattern '{context_pat}' did not match.",
                    "", None,
                )
            for ctx_match in all_matches:
                start = max(0, ctx_match.start() - 200)
                end = min(len(text), ctx_match.end() + 200)
                region = text[start:end]
                nums = self._extract_numbers(region)
                count = sum(1 for n in nums if n["unit"] == unit)
                if count == 0 and unit == "月":
                    accept_units = rule.condition_params.get("accept_units", None)
                    if accept_units and "年" in accept_units:
                        count = sum(1 for n in nums if n["unit"] == "年")
                if count > best_count:
                    best_count = count
                    best_start, best_end = start, end
                    best_ctx_pos = ctx_match.start()
                elif count == best_count and count > 0:
                    region_nums = [n for n in nums if n["unit"] == unit]
                    if region_nums:
                        ctx_rel = ctx_match.start() - start
                        min_dist = min(abs(n["position"] - ctx_rel) for n in region_nums)
                        curr_region = text[best_start:best_end]
                        curr_nums_ = self._extract_numbers(curr_region)
                        curr_rel = [n for n in curr_nums_ if n["unit"] == unit]
                        curr_min_dist = min(abs(n["position"] - (best_ctx_pos - best_start)) for n in curr_rel) if curr_rel else float('inf')
                        if min_dist < curr_min_dist:
                            best_count = count
                            best_start, best_end = start, end
                            best_ctx_pos = ctx_match.start()
            if best_count > 0:
                search_text = text[best_start:best_end]
        except re.error:
            pass
    numbers = self._extract_numbers(search_text)
    if not numbers:
        numbers = self._extract_numbers(text)
    relevant = [n for n in numbers if n["unit"] == unit]
    if not relevant:
        accept_units = rule.condition_params.get("accept_units", None)
        if accept_units and unit == "月" and "年" in accept_units:
            year_nums = [n for n in numbers if n["unit"] == "年"]
            if year_nums:
                for yn in year_nums:
                    yn["value"] = yn["value"] * 12
                    yn["unit"] = "月"
                    yn["raw"] = str(int(yn["value"])) + "月"
                relevant = year_nums
        if not relevant:
            return self._make_evidence(
                rule, Verdict.NOT_APPLICABLE, search_text[:200], [],
                f"Cannot find any numeric value with unit '{unit}' for field '{label}'.",
                f"Ensure the contract specifies {label} as a number with unit '{unit}'.",
                None,
            )
    contract_value = relevant[0]
    if context_pat and len(relevant) > 1:
        best_dist = float('inf')
        best_after = float('inf')
        best_after_val = None
        try:
            ctx_positions = [m.start() for m in re.finditer(context_pat, text)]
            for n in relevant:
                full_pos = n.get("position", 0)
                for ctx_pos in ctx_positions:
                    if full_pos >= ctx_pos:
                        dist = full_pos - ctx_pos
                        if dist < best_after:
                            best_after = dist
                            best_after_val = n
                if best_after_val is not None:
                    continue
                for ctx_pos in ctx_positions:
                    dist = abs(full_pos - ctx_pos)
                    if dist < best_dist:
                        best_dist = dist
                        contract_value = n
            if best_after_val is not None:
                contract_value = best_after_val
        except re.error:
            pass
    val = contract_value["value"]
    raw_str = contract_value["raw"]
    raw_pos = contract_value.get("position", 0)

    # Direction word check (deterministic keywords, zero LLM)
    direction_op: str | None = None
    direction_note: str = ""
    if operator in (">=", ">", "<=", "<"):
        snippet_start = max(0, raw_pos - 60)
        snippet_end = min(len(text), raw_pos + len(raw_str) + 60)
        snippet = text[snippet_start:snippet_end].strip()
        direction_op = self._resolve_direction(snippet, raw_str, label)
        direction_note = f" [direction: {direction_op}]" if direction_op else ""

    effective_op = direction_op if direction_op else operator
    op_map = {
        ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    }
    compare = op_map.get(effective_op, op_map.get(operator, lambda a, b: a >= b))
    ok = compare(val, expected)
    ref_suffix = (". Ref: " + legal_ref) if legal_ref else ""
    if ok:
        return self._make_evidence(
            rule, Verdict.PASSED, contract_value["raw"], [contract_value["raw"]],
            f"Field '{label}': contract value {val}{unit} meets legal threshold ({effective_op} {expected}{unit}){ref_suffix}.{direction_note}",
            "", None,
        )
    else:
        gap = abs(expected - val)
        if direction_op:
            suggest = (f"'{label}' is {val}{unit} ({direction_op}), "
                       f"does not meet legal threshold {operator} {expected}{unit}.")
        else:
            suggest = f"Increase '{label}' from {val}{unit} to at least {expected}{unit}."
        return self._make_evidence(
            rule, Verdict.FAILED, contract_value["raw"], [contract_value["raw"]],
            f"Field '{label}': contract value {val}{unit} FAILS legal threshold ({effective_op} {expected}{unit}){ref_suffix}. Gap: {gap}{unit}.{direction_note}",
            suggest,
            None,
        )


from app.engine.handlers._registry import register_handler
register_handler("numeric_comparison", _eval_numeric_comparison)
