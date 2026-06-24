"""LLMRuleExtractor — LLM-based rule candidate extraction from Chinese legal texts.

Ring 3 (supplement) of the self-bootstrap pipeline. Uses an LLM API call to
extract structured RuleCandidate objects from authoritative legal texts where
template-based extraction (StructuredRuleExtractor) has low or zero coverage.

Key design points:
  - Output schema matches RuleCandidate from rule_extractor.py exactly, so
    Ring 4 (AutoValidator) treats both template and LLM sources identically.
  - Prompt is structured: system role definition, condition_type definitions,
    domain-specific guidance, then the full legal text.
  - Post-LLM validation: JSON schema check, origin-text verification for terms
    and numeric values, duplicate rule_id avoidance.
  - MockRuleExtractor provides offline testing without an API key.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Re-export RuleCandidate for unified consumption by Ring 4
# ═══════════════════════════════════════════════════════════════════════════

# Import the canonical RuleCandidate from rule_extractor so Ring 4
# sees exactly one type regardless of source (template or LLM).
try:
    from app.engine.rule_extractor import RuleCandidate
except ImportError:
    # Fallback definition when running standalone / in tests
    @dataclass
    class RuleCandidate:
        condition_type: str
        subject: str
        operator: str
        expected_value: Optional[float] = None
        unit: Optional[str] = None
        required_terms: Optional[list[str]] = None
        source_text: str = ""
        confidence: float = 0.5
        source_type: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# JSON schema for LLM output validation
# ═══════════════════════════════════════════════════════════════════════════

LLM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["condition_type", "subject", "operator"],
        "properties": {
            "condition_type": {
                "type": "string",
                "enum": [
                    "numeric_comparison",
                    "required_pattern",
                    "forbidden_pattern",
                    "mutual_exclusion",
                    "co_occurrence",
                ],
            },
            "subject": {"type": "string", "minLength": 1},
            "operator": {
                "type": "string",
                "enum": [
                    ">=", "<=", ">", "<",
                    "requires", "forbids",
                    "mutually_excludes", "consistent_with",
                ],
            },
            "expected_value": {"type": "number", "minimum": 0},
            "unit": {"type": "string"},
            "required_terms": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "source_text": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "source_type": {"type": "string"},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Condition type definitions for prompt construction
# ═══════════════════════════════════════════════════════════════════════════

CONDITION_TYPE_DEFINITIONS: dict[str, str] = {
    "numeric_comparison": (
        "数值比较规则。法律/规范对某个数值指标设定了明确的上下限约束。"
        "例如：保修期限不少于5年，违约金不超过合同总价的30%。"
        "字段: subject=被约束的对象, operator=比较符(>=/<=/>/<), "
        "expected_value=数值阈值, unit=单位(年月日%)"
    ),
    "required_pattern": (
        "必备内容规则。法律要求合同中必须包含某类条款或表述。"
        "例如：出卖人应当履行交付义务。"
        "字段: subject=义务主体, operator=requires, "
        "required_terms=必须出现的关键词列表"
    ),
    "forbidden_pattern": (
        "禁止性规则。法律明确禁止的行为或条款。"
        "例如：不得转让、禁止转包。"
        "字段: subject=约束主体, operator=forbids, "
        "required_terms=禁止出现的关键词列表"
    ),
    "mutual_exclusion": (
        "互斥规则。两个或多个条款/情形不能同时存在。"
        "例如：不得同时担任两个以上职务。"
        "字段: subject=约束主体, operator=mutually_excludes, "
        "required_terms=互斥的关键词列表"
    ),
    "co_occurrence": (
        "协同出现规则。某个条款出现时，另一个条款必须同时出现。"
        "例如：约定分期付款的，应当约定违约责任。"
        "字段: subject=触发条件, operator=consistent_with, "
        "required_terms=必须一起出现的关键词列表"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Domain-specific condition_type whitelist
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_CONDITION_TYPES: dict[str, list[str]] = {
    "purchase": [
        "numeric_comparison",
        "required_pattern",
        "forbidden_pattern",
    ],
    "construction": [
        "numeric_comparison",
        "required_pattern",
        "forbidden_pattern",
        "co_occurrence",
    ],
    "labor": [
        "numeric_comparison",
        "required_pattern",
        "forbidden_pattern",
    ],
}

# All condition types used when no domain-specific whitelist exists
ALL_CONDITION_TYPES: list[str] = [
    "numeric_comparison",
    "required_pattern",
    "forbidden_pattern",
    "mutual_exclusion",
    "co_occurrence",
]

# Legal source type labels mapped from condition types
SOURCE_TYPE_MAP: dict[str, str] = {
    "numeric_comparison": "义务性条款",
    "required_pattern": "义务性条款",
    "forbidden_pattern": "禁止性条款",
    "mutual_exclusion": "互斥性条款",
    "co_occurrence": "一致性条款",
}

# Domain suggestions for prompt context
DOMAIN_GUIDANCE: dict[str, str] = {
    "purchase": (
        "本领域为买卖合同审查。重点关注："
        "交付义务（标的物、单证）、质量标准、检验期限、"
        "价款支付（数额、方式、时间）、风险转移、"
        "所有权保留、分期付款的加速到期/解除权。"
    ),
    "construction": (
        "本领域为建设工程合同审查。重点关注："
        "施工资质、工期、质量标准、安全生产、"
        "工程款支付、验收、保修期限、违约责任、"
        "转包/分包限制。"
    ),
    "labor": (
        "本领域为劳动合同审查。重点关注："
        "试用期期限、合同期限、社会保险、"
        "工资支付、工作时间、加班费、解除合同条件。"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Schema validation
# ═══════════════════════════════════════════════════════════════════════════


def _validate_json_schema(data: Any) -> bool:
    """Validate LLM output against the expected JSON schema.

    This is a lightweight validation without external schema libraries.
    Returns True if data conforms, False otherwise.
    """
    if not isinstance(data, list):
        logger.warning("LLM output is not a list (got %s)", type(data).__name__)
        return False

    for item in data:
        if not isinstance(item, dict):
            logger.warning("LLM output item is not a dict")
            return False
        for required_key in ("condition_type", "subject", "operator"):
            if required_key not in item:
                logger.warning("LLM output missing required key: %s", required_key)
                return False
        ct = item.get("condition_type", "")
        if ct not in ALL_CONDITION_TYPES:
            logger.warning("LLM output has invalid condition_type: %s", ct)
            return False
        operator = item.get("operator", "")
        valid_ops = [">=", "<=", ">", "<", "requires", "forbids",
                     "mutually_excludes", "consistent_with"]
        if operator not in valid_ops:
            logger.warning("LLM output has invalid operator: %s", operator)
            return False
        ev = item.get("expected_value")
        if ev is not None and (not isinstance(ev, (int, float)) or ev < 0):
            logger.warning("LLM output has invalid expected_value: %s", ev)
            return False
        conf = item.get("confidence", 0.5)
        if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
            logger.warning("LLM output has invalid confidence: %s", conf)
            return False
        terms = item.get("required_terms")
        if terms is not None:
            if not isinstance(terms, list):
                logger.warning("LLM output required_terms is not a list")
                return False
            for t in terms:
                if not isinstance(t, str) or not t.strip():
                    logger.warning("LLM output has empty required_term")
                    return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Origin-text verification
# ═══════════════════════════════════════════════════════════════════════════


def _verify_terms_in_source(
    terms: list[str], source_text: str
) -> tuple[list[str], list[str]]:
    """Verify that each term in the list appears in the source text.

    Returns (valid_terms, invalid_terms) where valid_terms are those
    found in source_text via string.find().
    """
    valid: list[str] = []
    invalid: list[str] = []
    for term in terms:
        if source_text.find(term) != -1:
            valid.append(term)
        else:
            invalid.append(term)
    return valid, invalid


def _verify_numeric_value_in_source(
    expected_value: float, unit: Optional[str], source_text: str
) -> bool:
    """Verify that the expected numeric value appears in the source text near its unit.

    Searches for patterns like '5年', '5 年', '五 年', '五 年', etc.
    """
    # Convert expected_value to int if it's a whole number
    int_val = int(expected_value) if expected_value == int(expected_value) else expected_value

    # Try to find the number as Arabic digits near a unit
    patterns = [
        rf"{int_val}\s*{unit}" if unit else str(int_val),
        rf"{int_val}",  # bare number
    ]
    for pat in patterns:
        if re.search(pat, source_text):
            return True

    # Try Chinese numeral form (e.g. "五" for 5)
    _CN_DIGITS = "零一二三四五六七八九十"
    if isinstance(int_val, int) and 0 <= int_val <= 10:
        cn_char = _CN_DIGITS[int_val]
        patterns_cn = [
            rf"{cn_char}\s*{unit}" if unit else cn_char,
            cn_char,
        ]
        for pat in patterns_cn:
            if re.search(pat, source_text):
                return True

    return False


def _verify_candidate_origin(
    candidate: dict, source_text: str
) -> dict:
    """Run origin-text verification on a single candidate dict.

    Marks the candidate with a 'noisy' flag if verification fails,
    and reduces confidence accordingly.

    Returns the candidate dict (mutated in-place).
    """
    noisy = False
    reasons: list[str] = []

    # Verify required_terms
    terms = candidate.get("required_terms")
    if terms:
        valid, invalid = _verify_terms_in_source(terms, source_text)
        if invalid:
            noisy = True
            reasons.append(f"terms not in source: {invalid}")
            # Keep only valid terms
            candidate["required_terms"] = valid if valid else None

    # Verify numeric value
    ev = candidate.get("expected_value")
    unit = candidate.get("unit")
    if ev is not None:
        if not _verify_numeric_value_in_source(ev, unit, source_text):
            noisy = True
            reasons.append(f"value {ev}{unit or ''} not found in source")

    # Mark candidate
    if noisy:
        candidate["_noisy"] = True
        candidate["_noise_reasons"] = "; ".join(reasons)
        # Reduce confidence for noisy candidates
        current_conf = candidate.get("confidence", 0.5)
        candidate["confidence"] = round(current_conf * 0.5, 4)

    return candidate


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════


def build_extraction_prompt(
    legal_text: str,
    domain_id: str = "",
    existing_rule_ids: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Build the system and user prompts for LLM-based rule extraction.

    Args:
        legal_text: The full text of the authority source (law, regulation, etc.).
        domain_id: Optional domain identifier for domain-specific guidance.
        existing_rule_ids: Optional list of existing rule IDs to avoid duplicates.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    # ── System prompt ──
    system_lines: list[str] = [
        "你是规则提取器。从法律/规范文本中提取结构化合规检查规则。",
        "",
        "你的任务是阅读给定的法律条文，提取其中可以自动化为合同审查规则的条款。",
        "提取的规则将用于自动审查合同是否符合法律规定。",
        "",
        "## 规则类型",
        "",
    ]

    # Determine which condition types are relevant for this domain
    if domain_id and domain_id in DOMAIN_CONDITION_TYPES:
        allowed_types = DOMAIN_CONDITION_TYPES[domain_id]
    else:
        allowed_types = ALL_CONDITION_TYPES

    for ct in allowed_types:
        if ct in CONDITION_TYPE_DEFINITIONS:
            system_lines.append(f"### {ct}")
            system_lines.append(CONDITION_TYPE_DEFINITIONS[ct])
            system_lines.append("")

    # Domain-specific guidance
    if domain_id and domain_id in DOMAIN_GUIDANCE:
        system_lines.append("## 领域指引")
        system_lines.append(DOMAIN_GUIDANCE[domain_id])
        system_lines.append("")

    # Existing rules guidance
    if existing_rule_ids:
        system_lines.append(
            "## 注意：已有规则 ID 列表（不要生成重复的规则）"
        )
        system_lines.append(", ".join(existing_rule_ids))
        system_lines.append("")

    system_lines.append(
        "输出格式：纯 JSON 数组，每个元素包含以下字段：\n"
        "  - condition_type: 规则类型（见上）\n"
        "  - subject: 规则约束的对象/概念\n"
        "  - operator: 操作符(>=/<=/>/</requires/forbids/mutually_excludes/consistent_with)\n"
        "  - expected_value: 数值（数字类型，仅numeric_comparison）\n"
        "  - unit: 单位（年月日%等）\n"
        "  - required_terms: 关键词列表（字符串数组）\n"
        "  - source_text: 原文片段标记你从哪里提取的\n"
        "  - confidence: 确信度 0-1\n"
        "\n"
        "返回纯 JSON，不要附带任何额外文字说明。"
    )

    system_prompt = "\n".join(system_lines)

    # ── User prompt ──
    user_prompt = (
        "请从以下法律条文中提取合规检查规则：\n\n"
        f"{legal_text}"
    )

    return system_prompt, user_prompt


# ═══════════════════════════════════════════════════════════════════════════
# LLM API call
# ═══════════════════════════════════════════════════════════════════════════


def _call_anthropic_api(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    base_url: str = "https://api.anthropic.com",
    max_tokens: int = 4096,
) -> Optional[str]:
    """Call the Anthropic Messages API.

    Args:
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        api_key: Anthropic API key.
        model: Model identifier string.
        base_url: API base URL.
        max_tokens: Maximum tokens in the response.

    Returns:
        The response text content, or None if the API call failed.
    """
    try:
        import httpx
    except ImportError:
        logger.error(
            "httpx is required for LLM API calls. Install with: pip install httpx"
        )
        return None

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/v1/messages",
                headers=headers,
                json=body,
            )
        if response.status_code != 200:
            logger.error(
                "Anthropic API error: %d %s",
                response.status_code,
                response.text[:500],
            )
            return None

        data = response.json()
        content_blocks = data.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text":
                return block.get("text", "")

        logger.warning("No text content block in API response")
        return None

    except httpx.TimeoutException:
        logger.error("Anthropic API call timed out after 120s")
        return None
    except httpx.RequestError as e:
        logger.error("Anthropic API request failed: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error calling Anthropic API: %s", e)
        return None


def _parse_llm_response(response_text: str) -> Optional[list[dict]]:
    """Parse the LLM response text into a list of candidate dicts.

    Tries to extract a JSON array from the response text using several
    strategies: direct JSON parse, markdown code block extraction,
    and first-bracket-to-last-bracket extraction.
    """
    if not response_text or not response_text.strip():
        logger.warning("Empty LLM response")
        return None

    text = response_text.strip()

    # Strategy 1: Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown code block
    code_block_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)```", text
    )
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1).strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from LLM response")
    logger.debug("Raw response: %s", text[:500])
    return None


# ═══════════════════════════════════════════════════════════════════════════
# LLMRuleExtractor
# ═══════════════════════════════════════════════════════════════════════════


class LLMRuleExtractor:
    """Extract rule candidates from legal text using an LLM API call.

    Complements StructuredRuleExtractor (template-based) by covering
    patterns the templates miss, especially for domains with nuanced or
    non-standard regulatory language.

    Usage:
        extractor = LLMRuleExtractor()
        candidates = extractor.extract(legal_text, domain_id="purchase")
        # Returns list[RuleCandidate]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = base_url

    def extract(
        self,
        legal_text: str,
        domain_id: str = "",
        condition_types: Optional[list[str]] = None,
        existing_rule_ids: Optional[list[str]] = None,
        max_retries: int = 1,
    ) -> list[RuleCandidate]:
        """Extract rule candidates from legal text via LLM API.

        Pipeline:
          1. Build system + user prompts
          2. Call LLM API
          3. Parse JSON response
          4. Validate against schema
          5. Verify origin text (terms and values must be in source)
          6. Convert to RuleCandidate objects (mark noisy ones with reduced confidence)

        Args:
            legal_text: Full text of the authority source (law/regulation).
            domain_id: Domain identifier for domain-specific guidance.
            condition_types: Override allowed condition types. If None, uses
                             domain whitelist or all types.
            existing_rule_ids: IDs of already-extracted rules to avoid duplicates.
            max_retries: Number of retries on parse/validation failure.

        Returns:
            List of RuleCandidate objects. Candidates failing origin verification
            are marked with low confidence (halved) and a `_noisy` attribute.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not configured.
        """
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not configured. "
                "Set the ANTHROPIC_API_KEY environment variable or pass api_key=..."
            )

        # Build prompts
        system_prompt, user_prompt = build_extraction_prompt(
            legal_text=legal_text,
            domain_id=domain_id,
            existing_rule_ids=existing_rule_ids,
        )

        # Call API with retries
        response_text: Optional[str] = None
        for attempt in range(max_retries + 1):
            response_text = _call_anthropic_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
            )
            if response_text:
                break
            if attempt < max_retries:
                logger.info("Retrying API call (attempt %d/%d)", attempt + 1, max_retries)

        if not response_text:
            logger.error("LLM API returned no response after %d attempts", max_retries + 1)
            return []

        # Parse response
        raw_candidates = _parse_llm_response(response_text)
        if raw_candidates is None:
            logger.warning("Failed to parse LLM response as JSON")
            return []

        # Validate schema
        if not _validate_json_schema(raw_candidates):
            logger.warning(
                "LLM output failed schema validation (%d items rejected)",
                len(raw_candidates),
            )
            return []

        # Filter condition types if specified
        if condition_types:
            raw_candidates = [
                c for c in raw_candidates
                if c.get("condition_type") in condition_types
            ]

        # Filter existing rule IDs (if LLM generated duplicate subjects)
        if existing_rule_ids:
            pass  # Rule IDs are assigned later; dedup is by subject+condition_type

        # Convert to RuleCandidate with origin verification
        candidates: list[RuleCandidate] = []
        for raw in raw_candidates:
            # Run origin-text verification
            raw = _verify_candidate_origin(raw, legal_text)

            subject = raw.get("subject", "").strip()
            condition_type = raw.get("condition_type", "")
            operator = raw.get("operator", "")

            # Build RuleCandidate
            candidate = RuleCandidate(
                condition_type=condition_type,
                subject=subject,
                operator=operator,
                expected_value=raw.get("expected_value"),
                unit=raw.get("unit"),
                required_terms=raw.get("required_terms"),
                source_text=raw.get("source_text", ""),
                confidence=raw.get("confidence", 0.5),
                source_type=raw.get("source_type", SOURCE_TYPE_MAP.get(condition_type, "")),
            )

            # Attach noise flags for downstream filtering
            if raw.get("_noisy"):
                candidate.confidence *= 0.5
                logger.debug(
                    "Noisy candidate: %s (%s) reason=%s",
                    subject, condition_type, raw.get("_noise_reasons", ""),
                )

            candidates.append(candidate)

        # Deduplicate by (subject, condition_type) — keep highest confidence
        candidates = _deduplicate_candidates(candidates)

        # Sort by confidence descending
        candidates.sort(key=lambda c: -c.confidence)

        logger.info(
            "LLMRuleExtractor: %d candidates after validation/dedup",
            len(candidates),
        )
        return candidates


# ═══════════════════════════════════════════════════════════════════════════
# MockRuleExtractor — offline testing without API key
# ═══════════════════════════════════════════════════════════════════════════


class MockRuleExtractor:
    """Heuristic-based rule candidate extractor for offline testing.

    Extracts RuleCandidate objects from legal text using regex patterns
    that approximate what an LLM would produce. This is NOT a replacement
    for LLMRuleExtractor — it exists solely to let the full pipeline
    (Ring 3 → Ring 4) be tested offline without an API key.

    Coverage is intentionally broader than StructuredRuleExtractor but
    with lower precision, mimicking the LLM's ability to catch patterns
    the templates miss.
    """

    def __init__(self):
        self._patterns: list[tuple[str, str, str, float, str]] = self._compile_patterns()
        # Common subject entities for context extraction
        self._subject_keywords = [
            "出卖人", "买受人", "承包人", "发包人", "甲方", "乙方",
            "标的物", "买卖合同", "屋面防水", "地下室", "防渗漏", "电气管线",
            "给排水管道", "设备安装", "装修工程", "供热", "供冷",
            "主体结构", "地基基础", "基础设施", "试用期",
            "分期付款", "所有权", "质量", "检验", "价款",
            "包装方式", "交付期限", "交付地点",
        ]
        # Chinese numeral mapping
        self._cn_number_map = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        # Chinese fraction mapping: "五分之一" -> 1/5 = 0.2
        self._cn_fraction_map = {
            "五分之一": 0.2, "三分之一": 0.333, "四分之一": 0.25,
            "十分之一": 0.1, "五分之二": 0.4, "五分之三": 0.6,
            "五分之四": 0.8, "三分之二": 0.667, "四分之三": 0.75,
        }

    def _compile_patterns(self) -> list[tuple[str, str, str, float, str]]:
        """Build heuristic patterns.

        Each entry: (condition_type, operator, regex_pattern, base_confidence, handler_name)
        """
        patterns: list[tuple[str, str, str, float, str]] = [
            # ═══════════════════════════════════════════════════════════════
            # numeric_comparison patterns
            # ═══════════════════════════════════════════════════════════════

            # "XX不少于YY年" / "不少于 YY 年"
            ("numeric_comparison", ">=",
             r"不少于\s*(\d+)\s*(年|月|日|天|％|%)", 0.70, "n_min"),

            # "不超过YY年" / "不得超过YY"
            ("numeric_comparison", "<=",
             r"(?:不超过|不得超过|不高于)\s*(\d+)\s*(年|月|日|天|％|%)", 0.70, "n_max"),

            # "不得低于YY" / "不得少于YY"
            ("numeric_comparison", ">=",
             r"(?:不得低于|不得少于|不应低于)\s*(\d+)\s*(年|月|日|天|％|%)", 0.70, "n_low"),

            # "不得高于YY" / "不得大于YY"
            ("numeric_comparison", "<=",
             r"(?:不得高于|不得大于)\s*(\d+)\s*(年|月|日|天|％|%)", 0.70, "n_high"),

            # ",为YY年" pattern from 279 decree: "屋面防水...，为5年"
            ("numeric_comparison", ">=",
             r"[，,]\s*为\s*(\d+)\s*(年|月|日|天|％|%)", 0.60, "n_is"),

            # "须...YY年" / "应...YY年"
            ("numeric_comparison", ">=",
             r"(?:须|应)\s*.{0,8}?(\d+)\s*(年|月|日|天)", 0.50, "n_must"),

            # X-digit-year bare patterns like "5年" near warranty keywords
            ("numeric_comparison", ">=",
             r"(?:保修|防水|供冷|供热|装修|电气|管道|管线|结构)"
             r".{0,6}?(\d+)\s*(年|月|日|天)", 0.55, "n_warranty"),

            # "最低保修期限为N年" from 279 decree
            ("numeric_comparison", ">=",
             r"(?:最低)?保修期限\s*为\s*(\d+)\s*(年|月|日|天|％|%|采暖期|供冷期)",
             0.65, "n_warranty_period"),

            # "保修期限: N年" or similar subject-value patterns

            # "达到五分之一" from purchase domain (分期付款)
            # This captures "达到全部价款的五分之一" as value=0.2 (20%)
            ("numeric_comparison", ">=",
             r"达到\s*.{0,10}?(五分之[一二三四五六七八九十])", 0.65, "n_fraction"),

            # "达到XX%" pattern
            ("numeric_comparison", ">=",
             r"达到\s*.{0,10}?(\d+)\s*[%％]", 0.60, "n_pct"),

            # ═══════════════════════════════════════════════════════════════
            # required_pattern patterns
            # ═══════════════════════════════════════════════════════════════

            # "出卖人应当履行/承担/交付" — obligation pattern
            ("required_pattern", "requires",
             r"(出卖人|买受人|承包人|发包人).{0,6}?(?:应当|必须|应)\s*(履行|承担|负责|保证|支付|交付|提供|出具|办理|通知|按照|转移)",
             0.75, "r_obligation"),

            # "XX应当包含/明确/具备" — content requirement
            ("required_pattern", "requires",
             r"应当.{0,6}?(包含|明确|具备|约定|遵守|执行|符合|依照|包括)",
             0.70, "r_shall"),

            # "应当在XX时间做XX" — timeline requirement
            ("required_pattern", "requires",
             r"应当在.{0,12}?(检验|交付|支付|验收|通知|告知)",
             0.70, "r_timeline"),

            # "按照约定/依照约定" — agreement-based
            ("required_pattern", "requires",
             r"(?:按照|依照|根据)(?:约定|合同)",
             0.55, "r_agreement"),

            # "可以...请求..." — right to claim
            ("required_pattern", "requires",
             r"可以.{0,12}?(请求).{0,10}?(支付|解除|承担|交付)",
             0.60, "r_claim"),

            # "有权..." — entitlement
            ("required_pattern", "requires",
             r"有权.{2,30}?",
             0.55, "r_right"),

            # Broad obligation: "XX应当..." + action verb (wider coverage)
            ("required_pattern", "requires",
             r"(出卖人|买受人|当事人)\s*(?:应当|必须|应)\s*(履行|承担|负责|保证|支付|交付|提供|出具|办理|通知|按照|转移|交付)",
             0.70, "r_obligation_broad"),

            # "XX应当在检验期限内检验" — specific inspection duty
            ("required_pattern", "requires",
             r"应当在.{0,16}?检验",
             0.65, "r_inspect"),

            # "XX的，出卖人/买受人可以" — conditional right
            ("required_pattern", "requires",
             r"[。；][^。；]{0,20}?可以.{0,12}?(请求)",
             0.55, "r_conditional"),

            # ═══════════════════════════════════════════════════════════════
            # forbidden_pattern patterns
            # ═══════════════════════════════════════════════════════════════

            # "XX不得YY" — general prohibition
            ("forbidden_pattern", "forbids",
             r"(?:出卖人|买受人|承包人|发包人|当事人).{0,6}?不得.{2,15}?",
             0.65, "f_shall_not"),

            # Bare "不得XX"
            ("forbidden_pattern", "forbids",
             r"[。；，]不得.{2,20}?",
             0.55, "f_shall_not_bare"),

            # "禁止XX"
            ("forbidden_pattern", "forbids",
             r"禁止.{2,20}?",
             0.70, "f_ban"),

            # "严禁XX"
            ("forbidden_pattern", "forbids",
             r"严禁.{2,20}?",
             0.70, "f_strict"),

            # ═══════════════════════════════════════════════════════════════
            # mutual_exclusion patterns
            # ═══════════════════════════════════════════════════════════════

            ("mutual_exclusion", "mutually_excludes",
             r"不得.{0,8}?同时.{2,20}?", 0.65, "m_simul"),
            ("mutual_exclusion", "mutually_excludes",
             r"和.{2,20}?不能同时", 0.65, "m_together"),

            # ═══════════════════════════════════════════════════════════════
            # co_occurrence patterns
            # ═══════════════════════════════════════════════════════════════

            ("co_occurrence", "consistent_with",
             r"应当.{0,6}?与.{2,30}?(一致|相符|相同|匹配)", 0.65, "c_consistent"),
        ]
        return patterns

    def extract(self, text: str, domain_id: str = "") -> list[RuleCandidate]:
        """Extract rule candidates using heuristic patterns.

        Args:
            text: Raw legal text.
            domain_id: Optional domain identifier (used for subject extraction).

        Returns:
            List of deduplicated RuleCandidate objects.
        """
        candidates: list[RuleCandidate] = []

        for ct, op, pat, conf, handler in self._patterns:
            for m in re.finditer(pat, text):
                groups = m.groups()
                # Determine subject: look for keywords before the match
                before = text[:m.start()].strip()
                subject = ""
                # Try to extract subject from the sentence/context
                # Last 80 chars before match (up to sentence boundary)
                ctx = before[-80:] if len(before) > 80 else before
                sentence_break = max(ctx.rfind("。"), ctx.rfind("；"), ctx.rfind("\n"))
                if sentence_break != -1:
                    ctx = ctx[sentence_break + 1:]
                ctx = ctx.strip()

                # Heuristic subject extraction from matched context
                subject = self._extract_subject(ctx, text, m, pat)

                # Extract expected_value and unit
                expected_value: Optional[float] = None
                unit: Optional[str] = None

                # Handle fraction patterns specially
                if handler == "n_fraction" and groups:
                    cn_fraction = groups[0]
                    expected_value = self._cn_fraction_map.get(cn_fraction, 0.2)
                    unit = "%"
                elif groups:
                    try:
                        expected_value = float(groups[0])
                    except (ValueError, TypeError):
                        pass
                    if len(groups) > 1:
                        unit = groups[1]
                # Extract terms
                required_terms = self._extract_terms(ct, m, text)

                # Source text: up to 120 chars around match
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 60)
                source_text = text[start:end].replace("\n", " ").strip()

                # Assign source_type
                source_type = SOURCE_TYPE_MAP.get(ct, "")

                candidate = RuleCandidate(
                    condition_type=ct,
                    subject=subject,
                    operator=op,
                    expected_value=expected_value,
                    unit=unit,
                    required_terms=required_terms,
                    source_text=source_text,
                    confidence=conf,
                    source_type=source_type,
                )
                candidates.append(candidate)

        # Deduplicate
        candidates = _deduplicate_candidates(candidates)

        # Sort by confidence descending
        candidates.sort(key=lambda c: -c.confidence)

        return candidates

    def _extract_subject(self, ctx, full_text, match, pattern):
        """Try to determine the subject entity from the context."""
        for kw in self._subject_keywords:
            if kw in ctx[-40:]:
                return kw
            if kw in full_text[max(0, match.start() - 60):match.start()]:
                return kw

        article_match = re.search(r'第[一二三四五六七八九十百千]+条\s+([^，。；\n]{2,10})', ctx)
        if article_match:
            entity = article_match.group(1).strip()
            if entity and len(entity) >= 2:
                return entity

        ctx_clean = re.sub(r'第[一二三四五六七八九十百千]+条\s*', '', ctx)
        entity_match = re.search(r'([^，。；、\d\s\n]{2,10})$', ctx_clean)
        if entity_match:
            candidate = entity_match.group(1).strip()
            if candidate and len(candidate) >= 2:
                return candidate

        words = re.findall(r'[^，。；、\s\n]{2,6}', ctx)
        if words:
            return words[-1]

        return ctx[-8:] if ctx and len(ctx) >= 3 else ctx or ""

    def _extract_terms(self, condition_type, match, full_text):
        """Extract key terms from the matched text."""
        matched_text = match.group(0)
        parts = re.split(r"[、，,和与及以及。；]", matched_text)
        terms = [t.strip() for t in parts if t.strip() and len(t.strip()) >= 2]
        return terms if terms else None


def _deduplicate_candidates(candidates):
    """Remove duplicates by (subject, condition_type), keeping highest confidence."""
    seen = {}
    for c in candidates:
        key = (c.subject, c.condition_type)
        existing = seen.get(key)
        if existing is None or c.confidence > existing.confidence:
            seen[key] = c
    return list(seen.values())


def extract_with_llm(legal_text, domain_id="", api_key=None, model="claude-sonnet-4-20250514"):
    """One-shot LLM extraction helper."""
    from app.engine.llm_rule_extractor import LLMRuleExtractor
    extractor = LLMRuleExtractor(api_key=api_key, model=model)
    return extractor.extract(legal_text, domain_id=domain_id)


def extract_mock(legal_text, domain_id=""):
    """One-shot mock extraction helper."""
    from app.engine.llm_rule_extractor import MockRuleExtractor
    extractor = MockRuleExtractor()
    return extractor.extract(legal_text, domain_id=domain_id)


def merge_candidates(template_candidates, llm_candidates):
    """Merge two lists of candidates with deduplication."""
    all_candidates = list(template_candidates) + list(llm_candidates)
    return _deduplicate_candidates(all_candidates)
