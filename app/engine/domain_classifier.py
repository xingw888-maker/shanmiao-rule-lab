# -*- coding: utf-8 -*-
"""Document-level domain classifier — determines which domain a document belongs to.

Uses DOMAIN_TERM_SETS: each known domain has a curated set of MUST-HAVE terms
that are strong signals. This is a rule-based classifier with confidence scoring,
NOT a statistical model — it works with zero training data and handles
out-of-domain texts via negative scoring.

Pipeline:
  1. Score each domain by term set hit ratio
  2. If best score < threshold → no known domain → "未知"
  3. If best score >= threshold AND second-best is far behind → classified
  4. If score gap is narrow → needs manual review

Zero statistical training.  Works on 1 document.  100% deterministic.
"""
from __future__ import annotations

import re
from typing import Optional

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]+")

# ── Domain term sets: human-curated strong signals ──
# Each domain has POSITIVE terms (must appear for it to be relevant)
# and NEGATIVE_OVERRIDE terms (if these appear, domain is likely wrong)
DOMAIN_TERM_SETS: dict[str, dict] = {
    "建设工程": {
        "positive": [
            "建设工程", "施工合同", "发包人", "承包人", "工程质量",
            "保修期", "竣工验收", "质量保证金", "缺陷责任期",
            "屋面防水", "主体结构", "地基基础", "分包",
            "工程价款", "监理单位", "设计变更", "开工日期",
            "竣工日期", "保修期限", "防水工程",
        ],
        "negative_override": None,  # no override needed for construction
        "strength_weight": 1.0,
    },
    "购销合同": {
        "positive": [
            "供方", "需方", "买受方", "出卖方", "供货方",
            "购销合同", "工矿产品", "交货", "验收",
            "产品名称", "规格型号", "数量", "单价", "总价款",
        ],
        "negative_override": [
            "建设工程", "施工合同", "发包人", "承包人",
        ],
        "strength_weight": 0.9,
    },
    "设计服务": {
        "positive": [
            "设计合同", "设计服务", "施工图设计", "方案设计",
            "设计单位", "设计变更", "初步设计", "技术设计",
        ],
        "negative_override": [
            "购销合同", "工矿产品", "供方", "需方",
        ],
        "strength_weight": 0.8,
    },
    "学术论文": {
        "positive": [
            "研究", "本文", "摘要", "关键词", "参考文献",
            "数据来源", "研究方法", "结论", "建议",
            "定义", "定理", "证明", "推论", "命题",
            "分析", "讨论", "结果表明", "提出",
            "因变量", "自变量", "假设检验", "回归分析",
            "问卷", "访谈", "参与式观察",
            "理论", "概念", "范畴", "本真", "此在",
        ],
        "negative_override": [
            "发包人", "承包人", "建设工程", "施工", "供方", "需方",
        ],
        "strength_weight": 0.6,
    },
    "文学作品": {
        "positive": [
            # Fiction markers — no single word defines fiction,
            # but clusters of dialogue, description, emotional words
        ],
        "negative_override": [
            "建设工程", "发包人", "承包人", "施工合同",
            "供方", "需方", "购销合同",
        ],
        "strength_weight": 0.3,
    },
}

# ── Generic CJK noise filter ──
_GENERIC_NOISE = {
    "一个", "这种", "那种", "可以", "进行", "或者",
    "没有", "他们", "我们", "什么", "怎么", "因为",
    "所以", "但是", "如果", "虽然", "而且", "然后",
}


class DomainClassifier:
    """Rule-based domain classifier with confidence scoring.

    NOT a statistical model.  Scores each domain against curated term sets.
    Out-of-domain texts get low scores → classified as "未知".
    """

    def __init__(self):
        self._term_sets = DOMAIN_TERM_SETS

    def predict(self, text: str) -> str:
        label, _conf = self.predict_with_confidence(text)
        return label

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        """Return (domain_label, confidence 0-1)."""
        scores = self._score_all(text)

        if not scores:
            return ("未知", 0.0)

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        best_domain, best_score = sorted_scores[0]

        # Threshold: must have at least some hits
        if best_score < 0.15:
            return ("未知", best_score)

        # If second-best is close, confidence drops
        if len(sorted_scores) > 1:
            second_score = sorted_scores[1][1]
            gap = best_score - second_score
            if gap < 0.10:
                confidence = 0.3 + gap * 2
                return (best_domain if best_score > 0.20 else "未知", min(confidence, 0.65))

        return (best_domain, min(best_score + 0.15, 0.95))

    def predict_proba(self, text: str) -> dict[str, float]:
        scores = self._score_all(text)
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def _score_all(self, text: str) -> dict[str, float]:
        """Score every domain against the text."""
        scores: dict[str, float] = {}
        for domain, cfg in self._term_sets.items():
            score = self._score_one(text, cfg)
            if score > 0:
                scores[domain] = score
        return scores

    def _score_one(self, text: str, cfg: dict) -> float:
        """Score a single domain's term set against text.

        Returns 0.0 – 1.0.
        """
        positive = cfg.get("positive", [])
        negative = cfg.get("negative_override") or []
        weight = cfg.get("strength_weight", 1.0)

        if not positive:
            return 0.0

        # Count positive term hits
        hits = 0
        matched = []
        for term in positive:
            if term in text:
                hits += 1
                matched.append(term)

        hit_ratio = hits / len(positive)

        # Negative override: if any negative term appears, reduce score
        neg_hits = sum(1 for t in negative if t in text)
        neg_penalty = min(neg_hits / max(len(negative), 1) * 0.5, 0.5)

        # Score formula: hit_ratio weighted by domain strength, minus negative penalty
        score = max(0.0, hit_ratio * weight - neg_penalty)

        # Bonus: consecutive hits (terms appearing close together)
        if hits >= 2:
            score += 0.05

        return min(score, 1.0)
