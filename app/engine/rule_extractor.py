"""StructuredRuleExtractor — deterministic rule candidate extraction from Chinese legal texts.

Ring 3 of the self-bootstrap pipeline. Zero LLM. Pure regex + sentence-structure templates.
Extracts structured RuleCandidate objects using 10+ syntactic templates covering common
Chinese legal sentence patterns (numeric comparisons, required/forbidden patterns,
mutual exclusions, consistency requirements).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RuleCandidate:
    """A structured rule candidate extracted from legal text via template matching.

    Attributes:
        condition_type: Type of rule condition.
            "numeric_comparison" | "required_pattern" | "forbidden_pattern"
            | "mutual_exclusion" | "co_occurrence"
        subject: The entity/concept being regulated (e.g. "屋面防水工程保修期限").
        operator: Comparison operator or directive type.
            ">=", "<=", ">", "<", "requires", "forbids", "mutually_excludes", "consistent_with"
        expected_value: Numeric threshold (for numeric_comparison).
        unit: Unit of measurement ("年", "月", "日", "%").
        required_terms: Key terms for required/forbidden/mutual exclusion patterns.
        source_text: Original sentence that produced this candidate.
        confidence: Match quality score 0-1.
    """
    condition_type: str  # "numeric_comparison" | "required_pattern" | "forbidden_pattern" | "mutual_exclusion" | "co_occurrence"
    subject: str
    operator: str  # ">=" | "<=" | ">" | "<" | "requires" | "forbids" | "mutually_excludes" | "consistent_with"
    expected_value: Optional[float] = None
    unit: Optional[str] = None
    required_terms: Optional[list[str]] = None
    source_text: str = ""
    confidence: float = 0.5
    source_type: str = ""  # "义务性条款" | "禁止性条款" | "补缺条款" | "担保物权条款" | "时效条款" | "约定优先条款"


# ═══════════════════════════════════════════════════════════════════════════
# Chinese numeral conversion
# ═══════════════════════════════════════════════════════════════════════════

_CN_DIGITS: dict[str, int] = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100, "千": 1000, "万": 10000,
}
_CN_MULTIPLIERS: dict[str, int] = {
    "十": 10, "百": 100, "千": 1000, "万": 10000,
}


def _chinese_numeral_to_int(text: str) -> Optional[int]:
    """Convert a Chinese numeral string to an integer.

    Handles:
        "一" -> 1, "二十" -> 20, "二十五" -> 25,
        "五十" -> 50, "一百" -> 100, "一千" -> 1000
    """
    text = text.strip()
    if not text:
        return None

    # Check if it is already an ASCII digit string
    if text.isdigit():
        return int(text)

    total = 0
    current = 0
    for ch in text:
        if ch in _CN_MULTIPLIERS:
            m = _CN_MULTIPLIERS[ch]
            if current == 0:
                current = 1
            current *= m
            total += current
            current = 0
        elif ch in _CN_DIGITS:
            current += _CN_DIGITS[ch]
        else:
            return None
    total += current
    return total if total > 0 else None


def _cn_num(text: str) -> Optional[int]:
    """Short alias for Chinese numeral conversion."""
    return _chinese_numeral_to_int(text)


# ═══════════════════════════════════════════════════════════════════════════
# Template definitions
# ═══════════════════════════════════════════════════════════════════════════

def _cn_num_re() -> str:
    """Regex character class for Chinese numerals (plus ASCII digits)."""
    return r"[一二两三四五六七八九十百千万零\d]+"


class _Template:
    """Internal template descriptor binding a regex pattern to a parse handler."""

    def __init__(self, name: str, pattern: str, confidence: float):
        self.name = name
        self.pattern = re.compile(pattern)
        self.confidence = confidence


def _compile_templates() -> list[_Template]:
    """Build and return the list of extraction templates.

    Each template captures structured data from named groups:
        subject, value, unit, terms, etc.

    NOTE: subject groups use `[^。；]` (allow Chinese commas) rather than
    `[^，。；]` so that subjects can span comma-separated enumerations — a
    common pattern in Chinese legal texts (e.g. "屋面防水工程、有防水要求的外墙面").
    """
    # ── Helpers —─
    # Character that can appear in a subject (anything except sentence-ending punct)
    # NOTE: cannot use +? inside ? (multiple repeat error), so use {2,50} as fixed range
    ANY = r"[^。；\n]"                       # any char except sentence-ending punct
    G = r"[^，。；\n]{2,60}"               # a non-comma token (2-60 chars)
    CN = _cn_num_re()

    templates: list[_Template] = []

    # ── Template 1: numeric_comparison — "XX不少于YY年" / "XX最低不少于YY年" ──
    # Match: subject-gap-不少于-value-unit
    templates.append(_Template(
        name="numeric_min_not_less_than",
        pattern=(
            rf"(?P<subject>{ANY}{{2,50}}?(?:保修|期限|寿命|使用年限|工程))?"
            r"[^。；\n]{0,15}?"
            r"(?:最低|至少)?"
            r"不少于\s*"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期)"
        ),
        confidence=0.85,
    ))

    # ── Template 2: numeric_comparison — "XX不超过YY%" / "XX不得超过YY" ──
    templates.append(_Template(
        name="numeric_max_not_exceed",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,15}?"
            r"(?:不超过|不得超过|不高于|不得高于)"
            r"\s*"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期|元|万)"
        ),
        confidence=0.85,
    ))

    # ── Template 3: numeric_comparison — "XX不得低于YY" / "XX不得少于YY" ──
    templates.append(_Template(
        name="numeric_min_not_below",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,15}?"
            r"(?:不得低于|不得少于|不应低于|不少于)"
            r"\s*"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期|元|万)"
        ),
        confidence=0.85,
    ))

    # ── Template 4: numeric_comparison — "XX不得高于YY" / "XX不得超过YY" ──
    templates.append(_Template(
        name="numeric_max_not_above",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,15}?"
            r"(?:不得高于|不得大于|不得超过|不应超过)"
            r"\s*"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期|元|万)"
        ),
        confidence=0.85,
    ))

    # ── Template 5: numeric_comparison — "XX，为YY" / subject is "YY年"
    # This is the pattern found in 279 decree:
    # "屋面防水工程、有防水要求的卫生间、房间和外墙面的防渗漏，为5年"
    # Strategy: collect everything up to the "，为" as subject
    templates.append(_Template(
        name="numeric_prefix_is",
        pattern=(
            rf"(?P<subject>[^。；\n]{{2,50}}?)"
            r"[，；]?\s*"
            r"(?:为|是)\s*"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期)"
        ),
        confidence=0.65,
    ))

    # ── Template 5b: "XX，为YY" with warranty/waterproof keywords (higher confidence) ──
    templates.append(_Template(
        name="numeric_warranty_is",
        pattern=(
            rf"(?P<subject>[^。；\n]{{2,50}}?(?:保修|防水|防渗漏|管道|管线|结构|外墙))"
            r"[^。；\n]{0,5}?"
            r"[，；]?\s*"
            r"(?:为|是)\s*"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期)"
        ),
        confidence=0.80,
    ))

    # ── Template 6: numeric_comparison — "XX应当不低于/不少于YY" ──
    templates.append(_Template(
        name="numeric_min_should_not_less",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,10}?"
            r"(?:应当|应|须)"
            r"[^。；\n]{0,5}?"
            r"(?:不少于|不低于|不小于)"
            r"\s*"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期)"
        ),
        confidence=0.80,
    ))

    # ── Template 7: required_pattern — "应当/必须 + 明确/具备/出具/约定/包含" ──
    templates.append(_Template(
        name="required_shall_have",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,5}?"
            r"(?:应当|必须|应|须)"
            r"[^。；\n]{0,3}?"
            r"(?:明确|具备|出具|约定|包含|提供|遵守|执行|符合|按照|依照)"
            rf"(?P<terms>{G})"
        ),
        confidence=0.75,
    ))

    # ── Template 8: required_pattern — "应当有/必须有/需有 XX" ──
    templates.append(_Template(
        name="required_must_have",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,5}?"
            r"(?:应当|必须|须|应)"
            r"[有需]"
            rf"(?P<terms>{G})"
        ),
        confidence=0.70,
    ))

    # ── Template 9: forbidden_pattern — "禁止XX" ──
    templates.append(_Template(
        name="forbidden_ban",
        pattern=(
            r"禁止"
            rf"(?P<terms>{G})"
        ),
        confidence=0.80,
    ))

    # ── Template 10: forbidden_pattern — "不得XX" ──
    templates.append(_Template(
        name="forbidden_shall_not",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,8}?"
            r"不得"
            rf"(?P<terms>{G})"
        ),
        confidence=0.75,
    ))

    # ── Template 11: mutual_exclusion — "XX不得同时YY和ZZ" ──
    templates.append(_Template(
        name="mutual_exclusion_not_simultaneous",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,8}?"
            r"(?:不得|不能|不可以)"
            r"[^。；\n]{0,4}?"
            r"(?:同时)"
            r"[^。；\n]{0,3}?"
            rf"(?P<terms>{G})"
        ),
        confidence=0.80,
    ))

    # ── Template 12: mutual_exclusion — "XX和YY不能同时" ──
    templates.append(_Template(
        name="mutual_exclusion_not_together",
        pattern=(
            rf"(?P<subject>{ANY}{{2,50}}?)"
            r"(?:和|与|、)"
            r"[^。；\n]{2,20}?"
            r"(?:不能|不得)"
            r"[^。；\n]{0,4}?"
            r"(?:同时)"
        ),
        confidence=0.80,
    ))

    # ── Template 13: co_occurrence / consistency — "XX应当与YY一致/相符" ──
    templates.append(_Template(
        name="co_occurrence_consistent_with",
        pattern=(
            rf"(?P<subject>{G})?"
            r"(?:应当|必须|应|须)"
            r"[^。；\n]{0,6}?"
            r"(?:与|和|同)"
            rf"(?P<terms>{G})"
            r"[^。；\n]{0,6}?"
            r"(?:一致|相符|相同|匹配|对应)"
        ),
        confidence=0.75,
    ))

    # ── Template 14: numeric_comparison — "XX不得同时出现和ZZ" ──
    templates.append(_Template(
        name="numeric_only_years",
        pattern=(
            r"(?P<value>\d+)\s*(?P<unit>年)"
        ),
        confidence=0.35,  # Low confidence - just a fallback for bare numbers
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # Second set: Obligation/Prohibition templates (added 2026-06-20)
    # for purchase/sales, labor, company law, and other general regulations.
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Template 15: required_pattern — "出卖人应当|买受人应当|XX应当|XX必须" ──
    templates.append(_Template(
        name="required_obligation_shall",
        pattern=(
            rf"(?P<subject>{G})"
            r"[^。；\n]{0,3}?"
            r"(?:应当|必须)"
            r"(?:履行|承担|负责|保证|支付|交付|提供|出具|办理|通知|按照)"
            rf"(?P<terms>{G})"
        ),
        confidence=0.80,
    ))

    # ── Template 16: required_pattern — "XX有权|XX可以行使" ──
    templates.append(_Template(
        name="required_entitled_right",
        pattern=(
            rf"(?P<subject>{G})"
            r"[^。；\n]{0,3}?"
            r"(?:有权|可以行使)"
            rf"(?P<terms>{G})"
        ),
        confidence=0.70,
    ))

    # ── Template 17: forbidden_pattern — "不得|禁止|严禁|不可" broader match ──
    # Captures "XX不得...", "严禁XX", "XX不可XX" patterns
    templates.append(_Template(
        name="forbidden_broad_prohibition",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,5}?"
            r"(?:不得|禁止|严禁|不可)"
            rf"(?P<terms>{G})"
        ),
        confidence=0.80,
    ))

    # ── Template 18: required_pattern — "XX承担|XX负责" ──
    templates.append(_Template(
        name="required_bears_responsibility",
        pattern=(
            rf"(?P<subject>{G})"
            r"(?:承担|负责)"
            rf"(?P<terms>{G})"
        ),
        confidence=0.65,
    ))

    # ── Template 19: required_pattern — "自XX之日起|自XX起" (时效条款) ──
    templates.append(_Template(
        name="required_time_limit_from",
        pattern=(
            rf"自(?P<subject>{G})"
            r"(?:之日起|起|开始)"
            r"[^。；\n]{0,8}?"
            rf"(?P<terms>{G})"
        ),
        confidence=0.65,
    ))

    # ── Template 20: required_pattern — "按照约定|依照约定|根据约定" (约定优先) ──
    templates.append(_Template(
        name="required_per_agreement",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,5}?"
            r"(?:按照|依照|根据)"
            r"(?:约定|合同)"
            rf"(?P<terms>{G})"
        ),
        confidence=0.65,
    ))

    # ── Template 21: required_pattern — "XX不明确的|XX没有约定|约定不明" (补缺条款) ──
    templates.append(_Template(
        name="required_gap_filling",
        pattern=(
            rf"(?P<subject>{G})"
            r"(?:不明确|没有约定|约定不明|未约定)"
            r"[^。；\n]{0,10}?"
            rf"(?P<terms>{G})"
        ),
        confidence=0.70,
    ))

    # ── Template 22: required_pattern — "质权|抵押权|留置权" (担保物权条款) ──
    templates.append(_Template(
        name="required_security_right",
        pattern=(
            rf"(?P<subject>{G})?"
            r"[^。；\n]{0,5}?"
            r"(?:质权|抵押权|留置权|担保物权)"
            rf"(?P<terms>{G})"
        ),
        confidence=0.75,
    ))

    # ── Template 23: sum_numeric_comparison — "付款比例合计不超过X%" ──
    # Matches: "预付款、进度款、结算款等各项付款比例合计不得超过合同总价的105%"
    #          "各项付款的合计比例不超过XX%"
    # Note: allows qualifying phrases (e.g. "合同总价的") between operator and value
    # via [^。；\n]{0,20}? gap — real legal texts rarely put the number directly
    # after the operator for sum clauses.
    # Captures the subject (payment scope), the aggregate threshold, and operator.
    # Produces condition_type="sum_numeric_comparison" for multi-value sum rules.
    templates.append(_Template(
        name="sum_numeric_not_exceed",
        pattern=(
            rf"(?P<subject>{ANY}{{2,80}}?)"
            r"(?:合计|之和|总和|总计)"
            r"\s*"
            r"(?:不超过|不得超过|不高于|不得高于)"
            r"[^。；\n]{0,20}?"
            rf"(?P<value>{CN})\s*"
            r"(?P<unit>％|%)"
        ),
        confidence=0.80,
    ))

    # ── Template 24: numeric_payment_deadline — payment deadline extraction ──
    # Matches: 后X日内付款, 之日起X个工作日内支付, etc.
    # Purpose: cn-027 structured rewrite — recognize non-standard payment deadline wording
    _PAYMENT_DEADLINE_PATTERN = (
        r"(?P<subject>[^。；\n]{0,40}?)?"
        r"(?:入账|到货|结算|验收|收到发票|完工|竣工|验收合格|竣工结算|交付完成)"
        r"[^。；\n]{0,8}?"
        r"(?:后|之日起)"
        r"[^。；\n]{0,5}?"
        rf"(?P<value>{CN})\s*"
        r"(?P<unit>个工作日内|日内|天内|个月内)"
    )
    templates.append(_Template(
        name="numeric_payment_deadline",
        pattern=_PAYMENT_DEADLINE_PATTERN,
        confidence=0.85,
    ))

    # ── Template 25: payment_deadline_broad — broad payment period pattern ──
    _PAYMENT_BROAD_PATTERN = (
        rf"(?P<value>{CN})\s*"
        r"(?P<unit>个工作日内|日内|天内|个月内)"
        r"[^。；\n]{0,5}?"
        r"(?:付款|支付|付清|结清|结算|到款)"
    )
    templates.append(_Template(
        name="numeric_payment_deadline_broad",
        pattern=_PAYMENT_BROAD_PATTERN,
        confidence=0.65,
    ))

    return templates


# ═══════════════════════════════════════════════════════════════════════════
# StructuredRuleExtractor
# ═══════════════════════════════════════════════════════════════════════════

class StructuredRuleExtractor:
    """Extract structured rule candidates from Chinese legal/regulatory texts.

    Uses 23 regex-based syntactic templates (no LLM, no domain-specific knowledge)
    to identify numeric constraints, required patterns, forbidden patterns,
    mutual exclusions, and co-occurrence requirements.
    """

    def __init__(self):
        self._templates = _compile_templates()
        logger.info(
            "StructuredRuleExtractor initialized with %d templates",
            len(self._templates),
        )

    # ── Public API ──

    def extract(self, text: str) -> list[RuleCandidate]:
        """Extract rule candidates from the given text.

        Args:
            text: Raw Chinese legal/regulatory text.

        Returns:
            A list of de-duplicated RuleCandidate objects, sorted by
            confidence descending.
        """
        # Split into sentences for clean matching
        sentences = self._split_sentences(text)

        candidates: list[RuleCandidate] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for candidate in self._match_templates(sentence):
                candidates.append(candidate)

        # Deduplicate by (subject, condition_type)
        candidates = self._deduplicate(candidates)

        # Sort by confidence descending
        candidates.sort(key=lambda c: -c.confidence)

        logger.info("Extracted %d rule candidates", len(candidates))
        return candidates

    # ── Sentence splitting ──

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on Chinese punctuation."""
        # Normalize line breaks
        text = re.sub(r"[\n\r]+", "", text)
        # Split on Chinese sentence-ending punctuation
        parts = re.split(r"(?<=[。；！？\n])", text)
        # Also split on "XX条" breaks
        refined: list[str] = []
        for part in parts:
            # Handle remaining "；" separators within clauses
            sub = re.split(r"(?<=[；；])", part)
            refined.extend(sub)
        # Clean whitespace
        return [s.strip() for s in refined if s.strip()]

    # ── Template matching ──

    def _match_templates(self, sentence: str) -> list[RuleCandidate]:
        """Try all templates against a single sentence.

        Returns list of RuleCandidate objects from matching templates.
        """
        results: list[RuleCandidate] = []

        for tmpl in self._templates:
            for m in tmpl.pattern.finditer(sentence):
                candidate = self._build_candidate(m, tmpl, sentence)
                if candidate is not None:
                    results.append(candidate)

        return results

    def _build_candidate(
        self,
        match: re.Match,
        tmpl: _Template,
        sentence: str,
    ) -> Optional[RuleCandidate]:
        """Convert a regex match into a RuleCandidate."""
        groups = match.groupdict()

        # Extract subject
        subject = (groups.get("subject") or "").strip()
        if not subject and tmpl.name in ("forbidden_ban",):
            # For bare "禁止XX", try to take context before the match
            pass  # subject remains empty; will be inferred from terms

        # Determine condition type and operator from template name
        ci_type, operator = self._classify_template(tmpl.name)

        # Extract value and unit
        value_str = groups.get("value")
        unit = groups.get("unit", "").strip()
        expected_value: Optional[float] = None

        if value_str is not None:
            # Try Chinese numeral first
            cn_val = _cn_num(value_str)
            if cn_val is not None:
                expected_value = float(cn_val)
            else:
                try:
                    expected_value = float(value_str)
                except (ValueError, TypeError):
                    pass

        # Clean unit
        if unit == "％":
            unit = "%"
        if unit.startswith("个"):
            unit = unit[1:]

        # Handle "采暖期" variant in unit detection (template 5 or generic)
        if unit == "" and expected_value is not None:
            # Look for non-standard units after the value
            unit_match = re.search(
                r"(?:年|月|日|天|％|%|个[年月日]|个[一-鿿]{2,4}|[一-鿿]{2,4}期|采暖期|供冷期)",
                match.group(0)[match.end() - match.start():],
            )
            if unit_match:
                unit = unit_match.group(0)
                if unit.startswith("个"):
                    unit = unit[1:]

        # Extract terms for required/forbidden/mutual_exclusion
        terms_str = groups.get("terms", "").strip()
        required_terms: Optional[list[str]] = None
        if terms_str:
            # Split on common delimiters
            term_parts = re.split(r"[、，,和与及以及]", terms_str)
            term_parts = [t.strip() for t in term_parts if t.strip()]
            if term_parts:
                required_terms = term_parts

        # For bare "禁止XX" without subject, use the first term as the subject
        if not subject and tmpl.name == "forbidden_ban":
            if required_terms:
                subject = required_terms[0]

        # If subject is still empty, try prefix before match
        if not subject:
            before = sentence[: match.start()].strip()
            # Take up to 20 chars before the match
            if before:
                subj_match = re.search(r"([^，。；]{2,20})$", before)
                if subj_match:
                    subject = subj_match.group(1)

        # If subject is still empty, use term
        if not subject and required_terms:
            subject = required_terms[0]

        # Skip if we have nothing meaningful
        if not subject and not required_terms:
            return None

        # Compute confidence adjustments
        confidence = tmpl.confidence

        # Penalise very short subjects
        if len(subject) < 2:
            confidence *= 0.9
        # Boost if subject contains key domain terms (warranty, construction, etc.)
        if any(kw in subject for kw in ["保修", "防水", "管线", "管道", "结构", "期限"]):
            confidence = min(1.0, confidence + 0.05)
        # Penalise if expected_value is suspiciously high/low
        if expected_value is not None and unit in ("年", "月", "日", "天", "%"):
            if expected_value > 1000:
                confidence *= 0.8
            if expected_value == 0:
                confidence *= 0.5

        # Determine source_type from template name (second-set templates)
        source_type = ""
        if tmpl.name in ("forbidden_broad_prohibition", "forbidden_ban", "forbidden_shall_not"):
            source_type = "禁止性条款"
        elif tmpl.name in ("required_obligation_shall", "required_bears_responsibility"):
            source_type = "义务性条款"
        elif tmpl.name == "required_time_limit_from":
            source_type = "时效条款"
        elif tmpl.name == "required_per_agreement":
            source_type = "约定优先条款"
        elif tmpl.name == "required_gap_filling":
            source_type = "补缺条款"
        elif tmpl.name == "required_security_right":
            source_type = "担保物权条款"
        elif tmpl.name == "required_entitled_right":
            source_type = "权利性条款"

        return RuleCandidate(
            condition_type=ci_type,
            subject=subject,
            operator=operator,
            expected_value=expected_value,
            unit=unit or None,
            required_terms=required_terms,
            source_text=sentence,
            confidence=round(confidence, 4),
            source_type=source_type,
        )

    @staticmethod
    def _classify_template(name: str) -> tuple[str, str]:
        """Map template name to (condition_type, operator)."""
        mapping: dict[str, tuple[str, str]] = {
            # Numeric comparisons
            "numeric_min_not_less_than": ("numeric_comparison", ">="),
            "numeric_max_not_exceed": ("numeric_comparison", "<="),
            "numeric_min_not_below": ("numeric_comparison", ">="),
            "numeric_max_not_above": ("numeric_comparison", "<="),
            "numeric_prefix_is": ("numeric_comparison", ">="),
            "numeric_warranty_is": ("numeric_comparison", ">="),
            "numeric_min_should_not_less": ("numeric_comparison", ">="),
            "numeric_only_years": ("numeric_comparison", ">="),
            # Required patterns
            "required_shall_have": ("required_pattern", "requires"),
            "required_must_have": ("required_pattern", "requires"),
            # Forbidden patterns
            "forbidden_ban": ("forbidden_pattern", "forbids"),
            "forbidden_shall_not": ("forbidden_pattern", "forbids"),
            # Mutual exclusions
            "mutual_exclusion_not_simultaneous": ("mutual_exclusion", "mutually_excludes"),
            "mutual_exclusion_not_together": ("mutual_exclusion", "mutually_excludes"),
            # Co-occurrence
            "co_occurrence_consistent_with": ("co_occurrence", "consistent_with"),
            # ── Second set (obligation/prohibition templates) ──
            "required_obligation_shall": ("required_pattern", "requires"),
            "required_entitled_right": ("required_pattern", "requires"),
            "forbidden_broad_prohibition": ("forbidden_pattern", "forbids"),
            "required_bears_responsibility": ("required_pattern", "requires"),
            "required_time_limit_from": ("required_pattern", "requires"),
            "required_per_agreement": ("required_pattern", "requires"),
            "required_gap_filling": ("required_pattern", "requires"),
            "required_security_right": ("required_pattern", "requires"),
            # ── Sum numeric comparison ──
            "sum_numeric_not_exceed": ("sum_numeric_comparison", "<="),
            # ── Payment deadline ──
            "numeric_payment_deadline": ("numeric_comparison", "<="),
            "numeric_payment_deadline_broad": ("numeric_comparison", "<="),
        }
        return mapping.get(name, ("required_pattern", "requires"))

    # ── Deduplication ──

    @staticmethod
    def _deduplicate(candidates: list[RuleCandidate]) -> list[RuleCandidate]:
        """Remove duplicates based on (subject, condition_type).

        When duplicates exist, the one with higher confidence is kept.
        """
        seen: dict[tuple[str, str], RuleCandidate] = {}
        for c in candidates:
            key = (c.subject, c.condition_type)
            existing = seen.get(key)
            if existing is None or c.confidence > existing.confidence:
                seen[key] = c
        return list(seen.values())


# ===========================================================================
# Convenience functions
# ===========================================================================

def extract_from_text(text: str) -> list[RuleCandidate]:
    """One-shot extraction helper."""
    extractor = StructuredRuleExtractor()
    return extractor.extract(text)


def candidates_to_dicts(candidates: list[RuleCandidate]) -> list[dict]:
    """Convert RuleCandidate objects to plain dicts for serialization."""
    import dataclasses
    return [dataclasses.asdict(c) for c in candidates]
