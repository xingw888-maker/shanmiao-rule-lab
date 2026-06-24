"""Chapter classifier — uses LLM to classify book chapters into domains.

Domain taxonomy:
  - contract      — 合同条款, contract clauses
  - regulation    — 制度规范, internal policies, compliance rules
  - definition    — 定义解释, terminology
  - conceptual    — 学术概念体系, academic knowledge
  - irrelevant    — 与规则无关的叙述/背景/前言
  - unknown       — LLM 无法判断
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.engine.book import Chapter

logger = logging.getLogger(__name__)


DOMAIN_TAXONOMY = {
    "contract":      "合同条款 / contract clauses",
    "regulation":    "制度规范 / internal policies / compliance rules",
    "definition":    "定义解释 / terminology / definitions",
    "conceptual":     "概念体系 / conceptual system / academic knowledge",
    "irrelevant":    "与规则校验无关 / narrative / preface",
    "unknown":       "无法判断 / cannot determine",
}

SYSTEM_PROMPT = """你是一个文档分类引擎。判断章节属于以下哪个类别：
- contract: 合同条款，涉及双方权利义务、违约责任、赔偿等
- regulation: 制度规范，涉及公司管理、合规要求、操作流程等
- definition: 定义解释，对术语或概念进行定义说明
- conceptual: 学术概念体系，涉及学科知识、理论概念、原理说明
- irrelevant: 与规则校验无关，如前言、目录、纯叙述
- unknown: 无法判断"""


@dataclass
class ClassifiedChapter:
    chapter: Chapter
    domain: str


@dataclass
class DomainGroup:
    domain: str
    label: str
    chapters: list[ClassifiedChapter] = field(default_factory=list)

    @property
    def merged_body(self) -> str:
        return "\n\n".join(cc.chapter.body for cc in self.chapters)


class LLMChapterClassifier:
    def __init__(self, api_url, api_key, model=""):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model or "deepseek-chat"

    async def classify(self, chapters):
        import aiohttp
        results = []
        for ch in chapters:
            text = f"标题：{ch.title}\n\n正文：{ch.body[:2000]}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url.rstrip('/')}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={"model": self.model, "max_tokens": 16, "temperature": 0.0,
                              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                           {"role": "user", "content": text}]},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status != 200:
                            results.append(ClassifiedChapter(chapter=ch, domain="unknown"))
                            continue
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip().lower()
                        results.append(ClassifiedChapter(chapter=ch, domain=self._normalize(content)))
            except Exception as e:
                logger.warning("Classifier error: %s", e)
                results.append(ClassifiedChapter(chapter=ch, domain="unknown"))
        return results

    def _normalize(self, raw):
        raw = raw.strip().lower().rstrip(".")
        mapping = {
            "contract": "contract", "regulation": "regulation", "regulatory": "regulation",
            "compliance": "regulation", "policy": "regulation",
            "definition": "definition", "definitions": "definition", "terminology": "definition",
            "conceptual": "conceptual", "concept": "conceptual", "academic": "conceptual",
            "knowledge": "conceptual", "theory": "conceptual", "psychology": "conceptual",
            "philosophy": "conceptual",
            "irrelevant": "irrelevant", "narrative": "irrelevant", "preface": "irrelevant",
            "unknown": "unknown",
        }
        for key, value in mapping.items():
            if key in raw:
                return value
        return "unknown"

    _DOMAIN_KEYWORDS = {
        "contract": ["甲方", "乙方", "违约责任", "赔偿", "交付", "验收", "付款", "保密"],
        "regulation": ["制度", "规定", "管理", "考勤", "流程", "审批", "合规", "岗位"],
        "definition": ["定义", "是指", "所称", "术语", "简称"],
        "conceptual": ["理论", "概念", "原理", "学派", "认为", "提出", "研究"],
    }

    def classify_heuristic(self, chapters):
        results = []
        for ch in chapters:
            combined = ch.title + " " + ch.body[:500]
            scores = {}
            for domain, keywords in self._DOMAIN_KEYWORDS.items():
                scores[domain] = sum(1 for kw in keywords if kw.lower() in combined.lower())
            best = max(scores, key=scores.get)
            domain = best if scores[best] > 0 else "unknown"
            results.append(ClassifiedChapter(chapter=ch, domain=domain))
        return results


def group_by_domain(classified, skip_domains=("irrelevant", "unknown")):
    """Regroup classified chapters by domain."""
    groups = defaultdict(list)
    for cc in classified:
        if cc.domain in skip_domains:
            continue
        groups[cc.domain].append(cc)
    result = []
    for domain, chapters in groups.items():
        label = DOMAIN_TAXONOMY.get(domain, domain)
        result.append(DomainGroup(domain=domain, label=label, chapters=chapters))
    return result
