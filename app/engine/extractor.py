"""Rule extractor — keyword scanner + optional LLM."""
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
logger = logging.getLogger(__name__)

# Patterns ordered by specificity — most specific first
RULE_PATTERNS = [
    # mutual_exclusion — "X和Y不得同时"
    (r"(.+?(?:和|与|、).+?)\s*(?:不得|不能|不可|禁止|严禁|不应)\s*(?:同时|并存|共存|并现)",
     "mutual_exclusion", "error"),
    # mutual_exclusion — "不得同时出现X和Y"
    (r"(?:不得|不能|不可|禁止|严禁|不应)\s*(?:同时\s*)?(?:出现|提交|存在|使用|采用|包含|含有)?\s*(.+?(?:和|与|、).+?)(?:[。；;]|$)",
     "mutual_exclusion", "error"),
    (r"cannot\s+(?:co[- ]?exist|both|simultaneously).{0,30}(.+?\band\b.+)",
     "mutual_exclusion", "error"),

    # co_occurrence — "如果X，则必须Y"
    (r"(?:如果|如|若|当)"
     r"\s*(.+?)" r"(?:时)?\s*" r"[，,]\s*"
     r"(?:则\s*)?" r"(?:必须|应当|应|须|必需|务必)?\s*"
     r"(?:包含|含有|载明|具备|约定|写明|列明|注明|附|支付|赔偿|返还|退还|提供|提交|发出|送达)\s*"
     r"(.+?)" r"(?:[。；;]|$)",
     "co_occurrence", "warning"),
    (r"(?:if|when)\s+(.+?)\s*[,.]\s*(?:then\s+)?(?:must|shall|should)\s+(?:include|contain|have)\s+(.+)",
     "co_occurrence", "warning"),

    # logical_chain — "由于A且B，因此C"
    (r"(?:由于|因为|基于|according to)\s*(.+?(?:和|与|、|且).+?)\s*[，,]\s*(?:所以|因此|故而|则|故)\s*(.+?)(?:[。；;]|$)",
     "logical_chain", "info"),
    (r"(?:because|since)\s+(.+?\band\b.+?)\s*[,.]\s*(?:therefore|hence|thus|so)\s+(.+)",
     "logical_chain", "info"),

    # required_pattern — "必须附X" / "必须包含X"
    (r"(?:必须|应当|应|须|必需|务必|must|shall)\s*(?:包含|含有|载明|具备|注明|写明|列明|约定|包括|涵盖|附)\s*(.+?)(?:[。；;]|$)",
     "required_pattern", "warning"),
    (r"(?:must|shall|should)\s+(?:include|contain|specify|state|provide)\s+(.+)",
     "required_pattern", "warning"),

    # forbidden_pattern — catch-all AFTER more specific patterns
    (r"(?:禁止|严禁|不得|不许|不应|切勿|不能|prohibit(?:ed)?|must\s+not|shall\s+not|may\s+not)\s*(.+?)(?:[。；;]|$)",
     "forbidden_pattern", "error"),
]

@dataclass
class ExtractedRule:
    condition_type: str
    severity: str
    terms: list[str] = field(default_factory=list)
    source_text: str = ""
    confidence: float = 1.0
    suggested_name: str = ""
    suggestion: str = ""


class KeywordRuleScanner:
    def scan(self, text: str) -> list[ExtractedRule]:
        results: list[ExtractedRule] = []
        seen_spans: set[tuple[int, int]] = set()
        for pattern, rule_type, severity in RULE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                span = match.span()
                # Only filter out if same region already matched (5-char tolerance)
                s0 = span[0]
                too_close = any(
                    s0 >= s - 5 and s0 <= e + 5
                    for s, e in seen_spans
                )
                if too_close:
                    continue
                seen_spans.add(span)
                source = match.group(0).strip()[:200]
                groups = match.groups()
                terms = self._extract_terms(rule_type, groups)
                if not terms:
                    continue
                rule = ExtractedRule(
                    condition_type=rule_type, severity=severity,
                    terms=terms, source_text=source,
                    confidence=0.9,
                    suggested_name=self._auto_name(rule_type, terms),
                    suggestion=self._auto_suggestion(rule_type, terms, source),
                )
                results.append(rule)
        return results

    def _extract_terms(self, rule_type: str, groups: tuple) -> list[str]:
        terms = []
        strip_words = {
            '不得', '不能', '不可', '禁止', '严禁', '同时', '出现', '提交',
            '存在', '使用', '采用', '包含', '含有', '附', '必须', '应当',
            '应', '须', '必需', '务必', '不许', '不应', '切勿', '如果',
            '如', '若', '当', '由于', '因为', '基于', '则', '约定了',
            '写明', '注明', '列明', '载明', '具备', '约定', '包括', '涵盖',
            '支付', '赔偿', '返还', '退还', '提供', '提交', '发出', '送达',
        }
        for g in groups:
            if not g:
                continue
            g = g.strip("，,；;。. ")
            parts = re.split(r"[和与、且,;，；]", g)
            for p in parts:
                p = p.strip(" '\"（(）)\"\'。.")
                while len(p) >= 2:
                    stripped = False
                    for w in sorted(strip_words, key=len, reverse=True):
                        if p.startswith(w) and p != w:
                            p = p[len(w):].strip()
                            stripped = True
                            break
                    if not stripped:
                        break
                if len(p) >= 2 and not p.isspace():
                    terms.append(p)
        return terms

    def _auto_name(self, rule_type: str, terms: list[str]) -> str:
        joined = "、".join(terms[:3])
        names = {
            "mutual_exclusion": f"{joined} 互斥校验",
            "co_occurrence": f"{joined} 共存校验",
            "forbidden_pattern": f"禁止 {joined}",
            "required_pattern": f"必须包含 {joined}",
            "logical_chain": f"{joined} 逻辑链校验",
        }
        return names.get(rule_type, f"规则 - {joined}")

    def _auto_suggestion(self, rule_type: str, terms: list[str], source: str) -> str:
        joined = "、".join(terms[:3])
        suggestions = {
            "mutual_exclusion": f"以下术语不能同时出现: {joined}。请选择其一或增加限定条件。",
            "co_occurrence": f"当 {terms[0] if terms else '前件'} 出现时，必须同时出现 {terms[1] if len(terms) > 1 else '后件'}。",
            "forbidden_pattern": f"应删除或改写包含 '{joined}' 的内容。",
            "required_pattern": f"应添加包含 '{joined}' 的条款或内容。",
            "logical_chain": f"前提({joined})成立时，结论必须明确。",
        }
        return suggestions.get(rule_type, "")


# ── CJK stopwords used by segment_long_terms ──
_CJK_STOPWORDS: frozenset[str] = frozenset({
    '不得', '不能', '不可', '禁止', '严禁', '必须', '应当', '可以',
    '但是', '因为', '所以', '如果', '虽然', '而且', '或者', '并且',
    '同时', '没有', '不是', '就是', '这个', '那个', '什么', '怎么',
    '自己', '他们', '我们', '你们', '一个', '一种', '的', '了', '在',
    '有', '和', '是', '就', '也', '都', '而', '及', '或', '与',
    '于', '之', '其', '所', '被', '把', '将', '从', '对', '上',
    '下', '中', '内', '外', '前', '后', '左', '右', '以', '为',
})


def segment_long_terms(terms: list[str], min_len: int = 4) -> list[str]:
    """Break terms longer than min_len into CJK sliding-window sub-tokens.

    Uses the same bigram+trigram sliding-window approach as the engine Tokeniser
    (core.py), applied to each term that exceeds min_len.  Sub-tokens that are
    shorter than 2 characters or are pure stop words are discarded.
    Short terms pass through unchanged.

    For mixed-content terms (CJK + punctuation), CJK substrings are extracted
    and each is independently segmented.  If any CJK run was > min_len (triggering
    sliding-window splitting), only sub-tokens are kept; otherwise short CJK runs
    are kept as individual terms alongside the original.

    Args:
        terms: List of extracted term strings.
        min_len: Terms longer than this many characters are segmented.

    Returns:
        Deduplicated list of terms (short terms preserved, long terms replaced
        by their constituent sub-tokens).
    """
    _CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]+')
    result: list[str] = []
    seen: set[str] = set()

    for term in terms:
        if len(term) <= min_len:
            if term not in seen:
                seen.add(term)
                result.append(term)
            continue

        # Find all CJK runs in the term
        cjk_runs = _CJK_RE.findall(term)
        any_long_run = any(len(r) > min_len for r in cjk_runs)

        for run in cjk_runs:
            if len(run) <= min_len:
                if run not in seen:
                    seen.add(run)
                    result.append(run)
                continue
            # Sliding window: 2-char and 3-char substrings
            n = len(run)
            for length in (2, 3):
                for i in range(n - length + 1):
                    sub = run[i:i + length]
                    if sub not in _CJK_STOPWORDS and sub not in seen:
                        seen.add(sub)
                        result.append(sub)

        if not any_long_run:
            # No CJK run was long enough to split; keep original term
            if term not in seen:
                seen.add(term)
                result.append(term)

    return result


class RulePackageBuilder:
    def build(self, extracted: list[ExtractedRule], domain: str = "custom",
              package_name: str = "用户自定义规则包", source_filename: str = "") -> dict:
        pkg_id = f"custom-{uuid.uuid4().hex[:8]}"
        rules = []
        for i, ex in enumerate(extracted):
            rule = self._build_single(ex, i)
            rules.append(rule)
        return {
            "id": pkg_id, "name": package_name, "version": "0.1.0",
            "domain": domain,
            "description": f"从 '{source_filename or '上传文档'}' 自动提取的规则包 · 包含 {len(rules)} 条规则 · 请人工审核后使用",
            "maintainer": "auto-extracted", "rules": rules,
        }

    def _build_single(self, ex: ExtractedRule, index: int) -> dict:
        rule_id = f"AUTO_{index + 1:03d}"
        condition = self._build_condition(ex)
        return {
            "id": rule_id, "name": ex.suggested_name or f"自动规则 {index + 1}",
            "condition": condition, "severity": ex.severity,
            "message": ex.source_text[:200], "category": f"auto.{ex.condition_type}",
        }

    def _build_condition(self, ex: ExtractedRule) -> dict:
        if ex.condition_type == "mutual_exclusion":
            return {"type": "mutual_exclusion", "terms": ex.terms, "threshold": 2}
        elif ex.condition_type == "co_occurrence":
            return {"type": "co_occurrence", "antecedent": ex.terms[0] if ex.terms else "", "consequent": ex.terms[1] if len(ex.terms) > 1 else ""}
        elif ex.condition_type == "forbidden_pattern":
            return {"type": "forbidden_pattern", "pattern": f"(?i)({'|'.join(re.escape(t) for t in ex.terms)})"}
        elif ex.condition_type == "required_pattern":
            return {"type": "required_pattern", "pattern": f"(?i)({'|'.join(re.escape(t) for t in ex.terms)})"}
        elif ex.condition_type == "logical_chain":
            return {"type": "logical_chain", "premises": ex.terms[:-1] if len(ex.terms) > 1 else ex.terms, "conclusion": ex.terms[-1] if ex.terms else ""}
        return {"type": "forbidden_pattern", "pattern": ""}




def normalize_extracted_rule(rule: dict, method: str = "keyword_scan") -> dict:
    """Normalize an extracted rule dict to match rules.json format.

    Adds standard fields: source, source_credibility, extraction_method, category.
    Converts from ExtractedRule dataclass dict representation to the
    format expected by rules.json (for candidates/ directory ingest).

    Args:
        rule: Rule dict (may be from ExtractedRule dataclass or plain dict).
        method: "llm_extract" or "keyword_scan".

    Returns:
        Normalized rule dict matching rules.json rule schema.
    """
    source_map = {
        "llm_extract": ("LLM提取", 0.5),
        "keyword_scan": ("关键词扫描", 0.7),
    }
    source, credibility = source_map.get(method, ("关键词扫描", 0.7))

    # Determine condition type (could be nested or flat depending on source)
    if "condition" in rule:
        # Already in rules.json-like format
        cond_type = rule["condition"].get("type", "forbidden_pattern")
        label = rule["condition"].get("label", "")
        context_pattern = rule["condition"].get("context_pattern", "")
        # Preserve existing condition fields
        condition = dict(rule["condition"])
    else:
        # From ExtractedRule dataclass fields
        cond_type = rule.get("condition_type", "forbidden_pattern")
        label = rule.get("suggested_name", "")
        context_pattern = rule.get("source_text", "")[:80]
        # Build condition dict matching rules.json schema
        condition = _build_rulesjson_condition(
            cond_type=cond_type,
            terms=rule.get("terms", []),
            source_text=rule.get("source_text", ""),
            label=label,
            context_pattern=context_pattern,
        )

    normalized = {
        "id": rule.get("id", f"auto_{abs(hash(str(rule))) % 100000:05d}"),
        "name": rule.get("name") or rule.get("suggested_name", f"自动提取规则"),
        "condition": condition,
        "severity": rule.get("severity", "warning"),
        "message": rule.get("message") or rule.get("source_text", "")[:200] or rule.get("suggestion", ""),
        "category": rule.get("category", ""),
        "source": source,
        "source_credibility": credibility,
        "extraction_method": method,
    }

    return normalized


def _build_rulesjson_condition(cond_type: str, terms: list, source_text: str,
                                label: str = "", context_pattern: str = "") -> dict:
    """Build a condition dict matching rules.json schema from raw extracted fields."""
    if cond_type in ("forbidden_pattern", "required_pattern"):
        if terms:
            pattern = f"(?i)({'|'.join(re.escape(t) for t in terms)})"
        else:
            pattern = ""
        return {
            "type": cond_type,
            "label": label,
            "context_pattern": context_pattern,
            "pattern": pattern,
        }
    elif cond_type == "mutual_exclusion":
        return {
            "type": cond_type,
            "label": label,
            "context_pattern": context_pattern,
            "terms": terms,
            "threshold": 2,
        }
    elif cond_type == "co_occurrence":
        return {
            "type": cond_type,
            "label": label,
            "context_pattern": context_pattern,
            "antecedent": terms[0] if terms else "",
            "consequent": terms[1] if len(terms) > 1 else "",
        }
    elif cond_type == "logical_chain":
        return {
            "type": cond_type,
            "label": label,
            "context_pattern": context_pattern,
            "premises": terms[:-1] if len(terms) > 1 else terms,
            "conclusion": terms[-1] if terms else "",
        }
    elif cond_type == "topic_coverage":
        return {
            "type": cond_type,
            "label": label,
            "context_pattern": context_pattern,
            "source_keywords": terms,
            "min_coverage_ratio": 0.4,
        }
    else:
        # fallback: forbidden_pattern
        pattern = f"(?i)({'|'.join(re.escape(t) for t in terms)})" if terms else ""
        return {
            "type": "forbidden_pattern",
            "label": label,
            "context_pattern": context_pattern,
            "pattern": pattern,
        }


def normalize_extracted_rules(rules: list, method: str = "keyword_scan") -> list[dict]:
    """Normalize a list of extracted rules into rules.json-compatible format.

    Args:
        rules: List of ExtractedRule dataclass instances or dicts.
        method: "llm_extract" or "keyword_scan".

    Returns:
        List of normalized rule dicts ready for candidates/ directory.
    """
    normalized = []
    for r in rules:
        # Convert dataclass to dict if needed
        if hasattr(r, '__dataclass_fields__'):
            rule_dict = {
                "condition_type": r.condition_type,
                "severity": r.severity,
                "terms": r.terms,
                "source_text": r.source_text,
                "confidence": r.confidence,
                "suggested_name": r.suggested_name,
                "suggestion": r.suggestion,
            }
        elif isinstance(r, dict):
            rule_dict = r
        else:
            continue
        normalized.append(normalize_extracted_rule(rule_dict, method=method))
    return normalized


class LegacyRuleExtractor:
    SYSTEM_PROMPT = """你是一个规则提取引擎。给定一份文档，提取其中的逻辑校验规则。
只输出 JSON 数组，不要任何解释。

每条规则格式：
{
  "condition_type": "mutual_exclusion" | "co_occurrence" | "forbidden_pattern" | "required_pattern" | "logical_chain",
  "severity": "error" | "warning" | "info",
  "terms": ["术语1", "术语2"],
  "source_text": "触发此规则的原句",
  "suggested_name": "简短规则名",
  "suggestion": "违反时的修改建议",
  "confidence": 0.0-1.0
}

规则类型：
- mutual_exclusion: X和Y不能同时出现 / X与Y互斥
- co_occurrence: X出现时必须有Y / X需要配套Y
- forbidden_pattern: 禁止X / 不得X
- required_pattern: 必须包含X / 应载明X
- logical_chain: 由于A和B成立故C必须成立

只提取文档中明确的规则，不要编造。没有规则返回 []。最多 20 条。"""
    def __init__(self, api_url: str = "", api_key: str = "", model: str = ""):
        self.api_url = api_url; self.api_key = api_key; self.model = model
    @property
    def enabled(self) -> bool:
        return bool(self.api_url and self.api_key)
    async def extract(self, text: str, max_chars: int = 8000) -> list[ExtractedRule]:
        if not self.enabled:
            return []
        truncated = text[:max_chars]
        if len(text) > max_chars:
            truncated += f"\n\n[... 原文共 {len(text)} 字符，已截取前 {max_chars} 字符]"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url.rstrip('/')}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model or "deepseek-chat", "max_tokens": 4096, "temperature": 0.1,
                          "messages": [{"role": "system", "content": self.SYSTEM_PROMPT},
                                       {"role": "user", "content": f"从以下文档提取规则：\n\n{truncated}"}]},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("LLM HTTP %s: %s", resp.status, body[:200])
                        return []
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return self._parse_llm_response(content)
        except Exception as e:
            logger.warning("LLM extractor error: %s", e)
            return []

    def _parse_llm_response(self, content: str) -> list[ExtractedRule]:
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            return []
        try:
            items = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return []
        rules = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rules.append(ExtractedRule(
                condition_type=item.get("condition_type", "forbidden_pattern"),
                severity=item.get("severity", "error"), terms=item.get("terms", []),
                source_text=item.get("source_text", ""), confidence=item.get("confidence", 0.7),
                suggested_name=item.get("suggested_name", ""), suggestion=item.get("suggestion", ""),
            ))
        return rules


async def extract_rules_from_text(text: str, domain: str = "custom",
    package_name: str = "用户自定义规则包", source_filename: str = "",
    use_llm: bool = False, llm_url: str = "", llm_key: str = "", llm_model: str = "") -> dict:
    scanner = KeywordRuleScanner()
    builder = RulePackageBuilder()
    keyword_rules = scanner.scan(text)
    all_rules = list(keyword_rules)
    if use_llm:
        llm = LLMRuleExtractor(api_url=llm_url, api_key=llm_key, model=llm_model)
        llm_rules = await llm.extract(text)
        keyword_sources = {r.source_text[:60] for r in keyword_rules}
        for lr in llm_rules:
            if lr.source_text[:60] not in keyword_sources:
                all_rules.append(lr)
    pkg = builder.build(extracted=all_rules, domain=domain, package_name=package_name, source_filename=source_filename)
    return pkglist(keyword_rules)
    if use_llm:
        llm = LLMRuleExtractor(api_url=llm_url, api_key=llm_key, model=llm_model)
        llm_rules = await llm.extract(text)
        keyword_sources = {r.source_text[:60] for r in keyword_rules}
        for lr in llm_rules:
            if lr.source_text[:60] not in keyword_sources:
                all_rules.append(lr)
    pkg = builder.build(extracted=all_rules, domain=domain, package_name=package_name, source_filename=source_filename)
    return pkg
