"""Non-numeric candidate proposition extraction pipeline.
LLM-only extraction — no regex keyword fallback.
Works for any domain: legal, philosophy, psychology, abhidhamma, etc.
"""

from __future__ import annotations

import json, logging, re
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)
SUPPORTED_PREDICATES = ["REQUIRES", "FORBIDS", "IMPLIES", "MUTUALLY_EXCLUSIVE_WITH"]

SYSTEM_PROMPT = """你是一个命题提取引擎。从任何领域文本中提取逻辑命题。只输出 JSON 数组。

每条命题格式：
{"subject":"主体","predicate":"REQUIRES|FORBIDS|IMPLIES|MUTUALLY_EXCLUSIVE_WITH","object":"客体","source_text":"原句","confidence":0.5,"rationale":"为什么这样提取"}

谓词语义（按文本实际含义选，不套法律框）：
- REQUIRES: A 出现或成立时必须有 B（必要条件、依处、基础）
- FORBIDS: A 被禁止、不应出现（禁忌、错误见解、不允许的事物）
- IMPLIES: A 出现意味着 B 也应出现或成立（推论、缘起、因果关系）
- MUTUALLY_EXCLUSIVE_WITH: A 和 B 不能同时成立（矛盾、对立、互斥）

覆盖文本中的所有命题类型，不要只盯着法律/规范用语。佛法文本的缘起关系、心理学文本的概念关联、哲学文本的逻辑推论——都提取。只看命题逻辑结构，不限于特定领域词汇。

不要数值判断。不编造。最多20条。没有返回[]。"""


@dataclass
class CandidateProposition:
    subject: str
    predicate: str
    object: str
    source_text: str
    source_ref: str = ""
    confidence: float = 0.5
    domain: str = ""
    extraction_method: str = "llm_extract"
    layer: str = "L2_SOURCE_UNCERTAIN"
    requires_human_review: bool = True
    rationale: str = ""
    rule_name: str = ""


def candidate_to_rule(c: CandidateProposition) -> dict:
    name = c.rule_name or f"{c.subject} {_pred_label(c.predicate)} {c.object}"
    cond_type = _pred_to_cond_type(c.predicate)
    condition = _build_condition(cond_type, c)
    return {
        "id": f"acc_{c.predicate.lower()}_{abs(hash(c.subject + c.object)) % 100000:05d}",
        "name": name,
        "condition": condition,
        "severity": "warning",
        "message": f"{c.subject} {_pred_label(c.predicate)} {c.object}",
        "category": f"auto.{cond_type}",
        "source": f"LLM提取+人工确认: {c.source_text[:80]}",
        "source_credibility": 0.8,
        "extraction_method": "llm_extract+human_accepted",
    }


def _pred_label(p: str) -> str:
    m = {"REQUIRES": "必须包含", "FORBIDS": "禁止", "IMPLIES": "意味着需要",
         "MUTUALLY_EXCLUSIVE_WITH": "与...互斥"}
    return m.get(p, p)


def _pred_to_cond_type(p: str) -> str:
    m = {"REQUIRES": "required_pattern", "FORBIDS": "forbidden_pattern",
         "IMPLIES": "contextual_co_occurrence", "MUTUALLY_EXCLUSIVE_WITH": "mutual_exclusion"}
    return m.get(p, "required_pattern")


def _build_condition(cond_type: str, c: CandidateProposition) -> dict:
    if cond_type == "mutual_exclusion":
        return {"type": "mutual_exclusion", "terms": [c.subject, c.object], "threshold": 2}
    elif cond_type == "co_occurrence":
        return {"type": "co_occurrence", "antecedent": c.subject, "consequent": c.object}
    elif cond_type == "contextual_co_occurrence":
        return {"type": "contextual_co_occurrence", "term_a": c.subject, "term_b": c.object, "window_chars": 500}
    elif cond_type == "forbidden_pattern":
        return {"type": "forbidden_pattern", "pattern": f"(?i)({re.escape(c.subject)}|{re.escape(c.object)})"}
    elif cond_type == "required_pattern":
        return {"type": "required_pattern", "pattern": f"(?i)({re.escape(c.subject)}|{re.escape(c.object)})"}
    return {"type": cond_type}


class CandidateExtractor:
    def __init__(self, llm_url="", llm_key="", llm_model=""):
        self.llm_url = llm_url
        self.llm_key = llm_key
        self.llm_model = llm_model or "deepseek-chat"

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_url and self.llm_key)

    async def extract(self, text: str, domain: str = "") -> list[CandidateProposition]:
        if not self.has_llm:
            return []
        return await self._extract_llm(text, domain)

    async def _extract_llm(self, text: str, domain: str) -> list[CandidateProposition]:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.llm_url.rstrip('/')}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.llm_key}", "Content-Type": "application/json"},
                    json={"model": self.llm_model, "max_tokens": 4096, "temperature": 0.1,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": f"领域: {domain}\n\n从以下文本提取命题：\n\n{text[:6000]}"}]},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    return self._parse_response(data["choices"][0]["message"]["content"], domain)
        except Exception as e:
            logger.warning("Candidate LLM error: %s", e)
            return []

    def _parse_response(self, content: str, domain: str) -> list[CandidateProposition]:
        m = re.search(r'\[[\s\S]*\]', content)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        candidates = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pred = item.get("predicate", "")
            if pred not in SUPPORTED_PREDICATES:
                continue
            candidates.append(CandidateProposition(
                subject=item.get("subject", ""), predicate=pred,
                object=item.get("object", ""), source_text=item.get("source_text", ""),
                confidence=item.get("confidence", 0.5), domain=domain,
                extraction_method="llm_extract", layer="L2_SOURCE_UNCERTAIN",
                rationale=item.get("rationale", ""),
            ))
        return candidates
