"""Domain classification pipeline — unified multi-layer classifier.

Orchestrates 6 classification layers in priority order:
  Layer 1:  LexicalPrototypeStore (validated CJK bigram coverage)
  Layer 1b: CandidatePrototypeStore (candidate fallback)
  Layer 2:  DomainClassifier (rule-based term-set hit ratio)
  Layer 3:  DomainPrototypeStore (14-dim Mahalanobis distance)
  Layer 4:  Keyword fallback (hardcoded domain keywords)
  Layer 5:  Rejection gate (vocab + keyword gate)

Classification flow: validated lexical -> candidate lexical -> domain
classifier -> prototype -> keyword -> rejection gate.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Guarded imports ──
try:
    from app.engine.lexical_prototype import LexicalPrototypeStore
    _HAS_LEXICAL_STORE = True
except (ImportError, Exception):
    LexicalPrototypeStore = None
    _HAS_LEXICAL_STORE = False

try:
    from app.engine.candidate_store import CandidatePrototypeStore
    _HAS_CANDIDATE_STORE = True
except (ImportError, Exception):
    CandidatePrototypeStore = None
    _HAS_CANDIDATE_STORE = False

try:
    from app.engine.domain_prototype import DomainPrototypeStore
    _HAS_PROTOTYPE_STORE = True
except (ImportError, Exception):
    DomainPrototypeStore = None
    _HAS_PROTOTYPE_STORE = False

try:
    from app.engine.domain_classifier import DomainClassifier
    _HAS_DOMAIN_CLASSIFIER = True
except (ImportError, Exception):
    DomainClassifier = None
    _HAS_DOMAIN_CLASSIFIER = False


# ═══════════════════════════════════════════════════════════════════════
# Domain constants (moved out of kernel.py _infer_contract_profile)
# ═══════════════════════════════════════════════════════════════════════

# Internal domain ID → Chinese broad type display name
BT_MAP: dict[str, str] = {
    "construction": "建设工程",
    "purchase": "购销合同",
    "service": "服务合同",
    "lease": "租赁合同",
}

# Lexical internal domain ID → Chinese broad type
LEXICAL_TO_BROAD_TYPE: dict[str, str] = {
    "construction": "建设工程",
    "buddhism": "佛学",
    "purchase": "购销合同",
}

# Candidate source string for pipeline result
CANDIDATE_SOURCE = "candidate"

# Domain keyword gates (P0 fallback & confirmation)
# Each domain must hit ≥2 keywords to pass the gate.
DOMAIN_KEYWORD_GATES: dict[str, list[str]] = {
    "建设工程": [
        "建设", "施工", "工程", "承包", "发包", "保修",
        "质保金", "竣工验收", "工程质量", "防水", "缺陷责任",
    ],
    "购销合同": [
        "供方", "需方", "供货", "购销", "买受方", "出卖方",
        "工矿产品", "交货", "验收",
    ],
}

# Non-construction keyword overrides (wo-43)
# Even if vector says construction, explicit non-construction keywords win.
# Each entry: ([keywords...], target_broad_type)
NON_CONSTRUCTION_OVERRIDES: list[tuple[list[str], str]] = [
    (["设计合同", "设计服务", "施工图设计", "方案设计"], "设计服务"),
    (["审计约定", "审计报告", "审计服务"], "审计服务"),
    (["融资租赁", "承租方", "出租方", "租赁物"], "融资租赁"),
    (["技术开发", "技术服务", "技术标准", "技术协议"], "技术服务"),
    (["资产负债表", "利润表", "现金流量表", "审计意见"], "财务报表"),
    (["买受方", "出卖方", "供货方"], "购销合同"),
]

# Hardcoded keyword fallback (used when no classifier has an answer)
HARDCODED_KEYWORD_FALLBACKS: list[tuple[list[str], str]] = [
    (["建设工程", "施工合同", "承包方式", "工程概况", "竣工验收", "工程质量"], "建设工程"),
    (["购销合同", "工矿产品", "供方", "需方", "供货"], "购销合同"),
    (["物业服务", "物业管理", "保洁", "保安"], "服务合同"),
    (["租赁合同", "出租", "承租", "租金"], "租赁合同"),
    # Non-construction signals (wo-43)
    (["设计合同", "设计服务", "施工图设计", "方案设计"], "设计服务"),
    (["审计约定", "审计报告", "审计服务"], "审计服务"),
    (["融资租赁", "承租方", "出租方", "租赁物"], "融资租赁"),
    (["技术开发", "技术服务", "技术标准", "技术协议"], "技术服务"),
    (["资产负债表", "利润表", "现金流量表", "审计意见"], "财务报表"),
    (["买受方", "出卖方", "供货方"], "购销合同"),
]

# Prototype distance threshold — below this is considered a match
PROTOTYPE_DISTANCE_THRESHOLD = 3.0

# Domain classifier confidence threshold — above this, skip vector
DOMAIN_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.40


# ═══════════════════════════════════════════════════════════════════════
# PipelineResult
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Unified result from the classification pipeline."""
    matched_domains: list[str] = field(default_factory=list)
    primary_domain: str = ""
    broad_type: str = "未知"
    confidence: float = 0.0
    reject_reason: Optional[str] = None
    source: str = "none"  # "lexical" | "domain_classifier" | "prototype" | "keyword" | "none"
    lexical_scores: dict = field(default_factory=dict)
    proto_distance: float = float('inf')


# ═══════════════════════════════════════════════════════════════════════
# ClassifierPipeline
# ═══════════════════════════════════════════════════════════════════════

class ClassifierPipeline:
    """Unified 5-layer domain classifier.

    Layers execute in priority order; the first layer with a confident
    answer short-circuits the rest.
    """

    def __init__(
        self,
        lexical_store=None,
        domain_path: str = "",
        proto_store=None,
        candidate_store=None,
    ):
        self._lexical_store = lexical_store
        self._domain_path = domain_path
        self._proto_store = proto_store
        self._candidate_store = candidate_store

    # ── Layer 1: LexicalPrototypeStore ──

    def _layer1_lexical(self, text: str) -> tuple[list[str], dict]:
        """CJK bigram coverage classification.

        Returns (domain_ids, scores_dict). Empty list if no match.
        """
        if not _HAS_LEXICAL_STORE or self._lexical_store is None:
            return [], {}
        try:
            domains = self._lexical_store.classify(text)
            scores = self._lexical_store.match_scores(text)
            return domains or [], scores or {}
        except Exception:
            logger.debug("Layer 1 (lexical) failed", exc_info=True)
            return [], {}

    # ── Layer 1b: CandidatePrototypeStore (candidate fallback) ──

    def _layer1b_candidate(self, text: str) -> tuple[list[tuple[str, float, float]], dict]:
        """Candidate domain prototype classification.

        Probed only after Layer 1 (validated lexical) finds no match.
        Returns (domains_with_scores, scores_dict).
        Empty lists if no match or candidate store unavailable.
        """
        if not _HAS_CANDIDATE_STORE or self._candidate_store is None:
            return [], {}
        try:
            candidates = self._candidate_store.classify(text)
            scores = self._candidate_store.match_scores(text)
            return candidates or [], scores or {}
        except Exception:
            logger.debug("Layer 1b (candidate) failed", exc_info=True)
            return [], []

    # ── Layer 2: DomainClassifier ──

    def _layer2_domain_classifier(self, text: str) -> tuple[Optional[str], float]:
        """Rule-based term-set classification.

        Returns (broad_type, confidence). ("未知", 0.0) if no match.
        """
        if not _HAS_DOMAIN_CLASSIFIER:
            return None, 0.0
        try:
            clf = DomainClassifier()
            label, conf = clf.predict_with_confidence(text)
            if label and label != "未知" and conf >= DOMAIN_CLASSIFIER_CONFIDENCE_THRESHOLD:
                return label, conf
            return None, conf
        except Exception:
            logger.debug("Layer 2 (domain classifier) failed", exc_info=True)
            return None, 0.0

    # ── Layer 3: DomainPrototypeStore ──

    def _layer3_prototype(
        self, text: str, domain_id: Optional[str]
    ) -> tuple[Optional[str], float]:
        """14-dim Mahalanobis distance classification.

        Returns (matched_domain_id_or_None, best_distance).
        """
        if not _HAS_PROTOTYPE_STORE or self._proto_store is None:
            return None, float('inf')
        if not domain_id:
            return None, float('inf')
        try:
            if not self._proto_store.get_prototype(domain_id):
                return None, float('inf')
            domain_scores = self._proto_store.classify(text)
            if not domain_scores:
                return None, float('inf')
            best_dist = float('inf')
            current_dist = None
            for dom, dist in domain_scores:
                if dist < best_dist:
                    best_dist = dist
                if dom == domain_id:
                    current_dist = dist
            if current_dist is not None and current_dist < PROTOTYPE_DISTANCE_THRESHOLD:
                return domain_id, best_dist
            return None, best_dist
        except Exception:
            logger.debug("Layer 3 (prototype) failed", exc_info=True)
            return None, float('inf')

    # ── Layer 3.5: Non-construction keyword override ──

    @staticmethod
    def _check_non_construction_override(text: str) -> Optional[str]:
        """If text contains non-construction keywords, return override type."""
        for keywords, bt in NON_CONSTRUCTION_OVERRIDES:
            if any(kw in text for kw in keywords):
                return bt
        return None

    # ── Layer 4: Keyword fallback ──

    def _layer4_keyword_fallback(
        self, text: str, classification: Optional[dict]
    ) -> tuple[str, list[dict]]:
        """Keyword-based classification.

        1. Try domain.json contract_classification types first.
        2. Fall back to hardcoded keywords.
        """
        scored = []

        # Try domain.json contract_classification first
        if classification and "types" in classification:
            type_configs = classification["types"]
            threshold_cfg = classification.get("threshold", {})
            min_confidence = threshold_cfg.get("min_confidence", 0.8)
            max_second_confidence = threshold_cfg.get("max_second_confidence", 0.3)

            for type_name, cfg in type_configs.items():
                keywords = cfg.get("keywords", [])
                weight = cfg.get("weight", 1.0)
                if not keywords:
                    scored.append({"type": type_name, "score": 0.0})
                    continue
                hits = sum(1 for kw in keywords if kw in text)
                score = (hits / len(keywords)) * weight
                scored.append({"type": type_name, "score": round(score, 4)})

            scored.sort(key=lambda x: x["score"], reverse=True)
            top_score = scored[0]["score"] if scored else 0.0
            second_score = scored[1]["score"] if len(scored) > 1 else 0.0

            if top_score > min_confidence and second_score < max_second_confidence:
                return scored[0]["type"], scored

        # Hardcoded keyword fallback
        for keywords, bt in HARDCODED_KEYWORD_FALLBACKS:
            if any(kw in text for kw in keywords):
                return bt, scored

        return "未知", scored

    # ── Layer 5: Rejection gate ──

    def _layer5_rejection_gate(
        self,
        text: str,
        broad_type: str,
        lexical_domains: list[str],
        proto_distance: float,
    ) -> Optional[str]:
        """Determine if the classification result should be rejected.

        Returns reject_reason string, or None if accepted.
        """
        if lexical_domains:
            # Lexical match found — high confidence, bypass rejection
            return None

        # No lexical match — apply keyword gate
        if broad_type in DOMAIN_KEYWORD_GATES:
            keywords = DOMAIN_KEYWORD_GATES[broad_type]
            hits = [kw for kw in keywords if kw in text]
            if len(hits) < 2:
                return (
                    f"语义分类未匹配已知域，关键词门控也不通过"
                    f"(命中{len(hits)}/{len(keywords)}: {hits})。"
                    f"该文本可能不属于任何已加载领域。"
                )

        # Check if broad_type is completely unknown
        all_known_types = set(LEXICAL_TO_BROAD_TYPE.values()) | set(DOMAIN_KEYWORD_GATES.keys())
        if broad_type not in all_known_types:
            proto_conf = round(max(0.0, 1.0 / (1.0 + proto_distance)), 3) if proto_distance < float('inf') else 0.0
            return (
                f"文本不属于任何已加载领域"
                f"(向量={proto_conf:.2f})。"
            )

        return None

    # ── Main classify entry point ──

    def classify(
        self,
        text: str,
        domain_id: Optional[str] = None,
        classification: Optional[dict] = None,
    ) -> PipelineResult:
        """Run the full 5-layer classification pipeline.

        Args:
            text: The input contract text.
            domain_id: Current domain ID (e.g. "construction").
            classification: contract_classification from domain.json.

        Returns:
            PipelineResult with matched_domains, broad_type, confidence, etc.
        """
        result = PipelineResult()

        # ── Layer 1: Lexical prototype ──
        lexical_domains, lexical_scores = self._layer1_lexical(text)
        result.lexical_scores = lexical_scores

        if lexical_domains:
            result.matched_domains = lexical_domains
            matched_types = [
                LEXICAL_TO_BROAD_TYPE.get(d, d) for d in lexical_domains
            ]
            result.broad_type = (
                matched_types[0] if len(matched_types) == 1
                else "+".join(matched_types)
            )
            result.source = "lexical"
            result.confidence = 0.9
            result.primary_domain = lexical_domains[0]
            return result

        # ── Layer 1b: Candidate prototype (fallback after validated lexical) ──
        candidate_matches, candidate_scores = self._layer1b_candidate(text)
        if candidate_matches:
            # Merge candidate scores into lexical_scores for downstream reporting
            lexical_scores.update(candidate_scores)
            result.lexical_scores = lexical_scores

            candidate_domains = [d for d, _, _ in candidate_matches]
            result.matched_domains = candidate_domains
            matched_types = [
                LEXICAL_TO_BROAD_TYPE.get(d, d) for d in candidate_domains
            ]
            result.broad_type = (
                matched_types[0] if len(matched_types) == 1
                else "+".join(matched_types)
            )
            result.source = CANDIDATE_SOURCE  # "candidate"
            result.confidence = 0.6  # Lower confidence than validated lexical
            result.primary_domain = candidate_domains[0]
            return result

        # ── Layer 2: Domain classifier ──
        dc_label, dc_conf = self._layer2_domain_classifier(text)
        if dc_label is not None:
            result.broad_type = dc_label
            result.confidence = dc_conf
            result.source = "domain_classifier"
            # DomainClassifier returns Chinese broad type — no proto step needed
            # But we still need to compute proto_distance for rejection gate
            _, best_dist = self._layer3_prototype(text, domain_id)
            result.proto_distance = best_dist
            reject_reason = self._layer5_rejection_gate(
                text, result.broad_type, lexical_domains, best_dist,
            )
            result.reject_reason = reject_reason
            return result

        # ── Layer 3: Prototype store ──
        proto_match, best_dist = self._layer3_prototype(text, domain_id)
        result.proto_distance = best_dist

        if proto_match is not None:
            # Proto matched — check non-construction override first
            override = self._check_non_construction_override(text)
            if override:
                result.broad_type = override
                result.source = "keyword"
            else:
                # Map internal ID to Chinese display name
                result.broad_type = BT_MAP.get(proto_match, proto_match)
                result.primary_domain = proto_match
                result.source = "prototype"
        else:
            # ── Layer 4: Keyword fallback ──
            broad_type, scored = self._layer4_keyword_fallback(text, classification)
            result.broad_type = broad_type
            result.source = "keyword"

        # ── Layer 5: Rejection gate ──
        result.confidence = round(
            max(0.0, 1.0 / (1.0 + best_dist)), 3
        ) if best_dist < float('inf') else 0.0
        result.reject_reason = self._layer5_rejection_gate(
            text, result.broad_type, lexical_domains, best_dist,
        )

        return result


# ═══════════════════════════════════════════════════════════════════════
# Contract type inference (extracted from kernel.py 597-614)
# ═══════════════════════════════════════════════════════════════════════

_CONTRACT_TYPE_NEGATION_RE = re.compile(
    r'(?:不得|禁止|不可|严禁|不应|不许)\S{0,20}分[包包]'
)


def infer_contract_types(text: str, estimated_value: int) -> list[str]:
    """Pure data logic — infer contract types from text and value.

    Independent of classification; uses only text + estimated value.
    """
    types: list[str] = []

    if estimated_value > 50_000_000:
        types.append("大型工程")

    if "总承包" in text or "施工总承包" in text or "总包" in text:
        types.append("施工总承包")
    elif "分包" in text:
        has_negative = bool(_CONTRACT_TYPE_NEGATION_RE.search(text))
        if not has_negative:
            types.append("专业分包")

    if "屋面防水" in text or "外墙保温" in text:
        if estimated_value < 100_000:
            types.append("小型维修")

    if not types:
        if estimated_value > 10_000_000:
            types.append("大型工程")
        else:
            types.append("中小型工程")

    return types
