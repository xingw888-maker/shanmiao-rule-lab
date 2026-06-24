"""知识自动分层引擎 — Knowledge Auto-Layering Engine.

按照用户架构：
  底层把知识自动分层。
  规则结果不正确 → 追查来源是否专业统一
  → 依条件设立是否有可能
  → 规则外分析交叉可能性
  → 一部分验证回归到规则内，一部分产生规则外结果。

四层知识结构：
  L0_VALIDATED       — 来源专业统一、经过确认的正式规则
  L1_CONJECTURE      — 统计发现，标记为猜想，等人确认
  L2_SOURCE_UNCERTAIN — 规则存在但来源存疑（非权威源、版本不一致等）
  L3_OUTER_POSSIBILITY — 规则外交叉分析发现的可能性（不在任何规则覆盖内）

流水线：
  输入: 校验结果 + 文本 + 规则包元信息
  1. 来源审查 (Source Audit) — 每条触发的规则，来源是否专业统一？
  2. 条件成立性 (Conditional Feasibility) — 规则结果是否在条件下合理？
  3. 规则外交叉 (Outer-Region Cross Analysis) — 未覆盖区域有什么模式？
  4. 回归判定 (Regression Resolution) — 哪些回规则内，哪些留在规则外？

该模块独立于现有 core.py / conjecture.py，通过函数式接口接入。
"""

import hashlib
import logging
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Knowledge Layers
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeLayer(str, Enum):
    """知识四层分类。"""
    L0_VALIDATED = "VALIDATED"              # 来源专业统一，已确认
    L1_CONJECTURE = "CONJECTURE"            # 统计猜想，待确认
    L2_SOURCE_UNCERTAIN = "SOURCE_UNCERTAIN" # 来源存疑，待追查
    L3_OUTER_POSSIBILITY = "OUTER_POSSIBILITY"  # 规则外交叉发现


# ═══════════════════════════════════════════════════════════════════════
# Source Credibility 来源可信度
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SourceProfile:
    """规则来源的可信度画像。"""
    source_id: str                    # 来源标识（URL / 机构 / 作者）
    source_type: str                  # "官方机构" | "学术期刊" | "行业标准" | "个人编写" | "LLM提取" | "未知"
    version: str = "0.0.0"
    domain_authority: float = 0.5     # 0-1, 该来源在领域内的权威程度
    consistency_score: float = 1.0    # 0-1, 与同类来源的一致性
    has_citations: bool = False       # 是否有引用依据
    extraction_method: str = ""       # "manual" | "keyword_scan" | "llm_extract" | "conjecture_mine"
    extraction_confidence: float = 0.0  # 提取时的置信度
    cross_verified: bool = False      # 是否经过交叉验证
    last_verified_at: str = ""        # 最后验证时间

    @property
    def credibility(self) -> float:
        """综合可信度评分 0-1。"""
        scores = []
        # 来源类型基础分
        type_scores = {
            "官方机构": 1.0, "学术期刊": 0.95, "行业标准": 0.9,
            "个人编写": 0.4, "LLM提取": 0.35, "未知": 0.2,
        }
        scores.append(type_scores.get(self.source_type, 0.3))
        # 提取方式
        method_scores = {"manual": 1.0, "keyword_scan": 0.65, "llm_extract": 0.45, "conjecture_mine": 0.3}
        scores.append(method_scores.get(self.extraction_method, 0.3))
        # 领域权威 × 一致性
        scores.append(self.domain_authority)
        scores.append(self.consistency_score)
        # 引用加分
        if self.has_citations:
            scores.append(0.2)
        # 交叉验证加分
        if self.cross_verified:
            scores.append(0.25)
        # 加权平均
        weights = [0.30, 0.20, 0.20, 0.15, 0.10, 0.10]
        weighted = sum(s * w for s, w in zip(scores[:len(weights)], weights[:len(scores)]))
        return min(1.0, max(0.0, weighted))


# ═══════════════════════════════════════════════════════════════════════
# Knowledge Node
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeNode:
    """知识图谱中的一个节点——可能是一条规则、一个猜想、一个发现。"""
    node_id: str
    layer: KnowledgeLayer
    # 核心内容
    term: str = ""                     # 关键术语
    rule_id: str = ""                  # 关联规则 ID（如果在规则内）
    description: str = ""              # 人类可读描述
    # 来源链条
    source: Optional[SourceProfile] = None
    source_chain: list[str] = field(default_factory=list)  # 追溯链 ["原始文档A → LLM提取 → 关键词确认"]
    # 条件判定
    condition_feasible: bool = True    # 在当前条件下是否可能成立
    condition_notes: str = ""          # 条件分析备注
    # 交叉分析
    cross_references: list[str] = field(default_factory=list)  # 交叉引用的其他节点 ID
    cross_confidence: float = 0.0      # 交叉验证置信度
    # 回归标记
    regression_target: str = ""        # 如果回归规则内，目标规则 ID
    regression_action: str = ""        # "upgrade_to_validated" | "downgrade_to_conjecture" | "correct_rule" | "new_rule" | "keep_outer"
    # 元信息
    created_from: str = ""             # 从哪个流水线阶段产生
    evidence_snippets: list[str] = field(default_factory=list)
    created_at: str = ""
    confidence: float = 0.0


@dataclass
class LayerReport:
    """一次分层分析的完整报告。"""
    # 各层节点
    l0_validated: list[KnowledgeNode] = field(default_factory=list)
    l1_conjecture: list[KnowledgeNode] = field(default_factory=list)
    l2_source_uncertain: list[KnowledgeNode] = field(default_factory=list)
    l3_outer_possibility: list[KnowledgeNode] = field(default_factory=list)
    # 统计
    total_rules_evaluated: int = 0
    sources_audited: int = 0
    sources_flagged: int = 0           # 来源存疑的数量
    outer_discoveries: int = 0         # 规则外发现数
    regressions_to_inner: int = 0      # 回归规则内的数量
    regressions_kept_outer: int = 0    # 保留在规则外的数量
    # 摘要
    summary: str = ""

    @property
    def all_nodes(self) -> list[KnowledgeNode]:
        return self.l0_validated + self.l1_conjecture + self.l2_source_uncertain + self.l3_outer_possibility

    def to_dict(self) -> dict:
        def node_dict(n: KnowledgeNode) -> dict:
            return {
                "node_id": n.node_id,
                "layer": n.layer.value,
                "term": n.term,
                "rule_id": n.rule_id,
                "description": n.description,
                "source_credibility": n.source.credibility if n.source else 0,
                "source_type": n.source.source_type if n.source else "未知",
                "source_chain": n.source_chain,
                "condition_feasible": n.condition_feasible,
                "condition_notes": n.condition_notes,
                "cross_confidence": n.cross_confidence,
                "regression_target": n.regression_target,
                "regression_action": n.regression_action,
                "evidence_snippets": n.evidence_snippets[:5],
                "confidence": n.confidence,
            }
        return {
            "summary": self.summary,
            "stats": {
                "total_rules_evaluated": self.total_rules_evaluated,
                "sources_audited": self.sources_audited,
                "sources_flagged": self.sources_flagged,
                "outer_discoveries": self.outer_discoveries,
                "regressions_to_inner": self.regressions_to_inner,
                "regressions_kept_outer": self.regressions_kept_outer,
            },
            "layers": {
                "L0_VALIDATED": [node_dict(n) for n in self.l0_validated],
                "L1_CONJECTURE": [node_dict(n) for n in self.l1_conjecture],
                "L2_SOURCE_UNCERTAIN": [node_dict(n) for n in self.l2_source_uncertain],
                "L3_OUTER_POSSIBILITY": [node_dict(n) for n in self.l3_outer_possibility],
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Source Auditor — 来源审查
# ═══════════════════════════════════════════════════════════════════════

class SourceAuditor:
    """审查每条规则的来源是否专业统一。

    问题：规则结果不正确 → 首先问：这条规则是谁写的？来源可靠吗？
    比如从一篇非权威博客提取的规则判了 FAILED，这种结果的权重应该降低。

    判断维度：
    1. 来源类型（官方/学术/个人/LLM/未知）
    2. 提取方式（人工编写 > 关键词扫描 > LLM 提取 > 猜想挖掘）
    3. 领域一致性（与同类来源是否一致）
    4. 是否有引用依据
    5. 交叉验证状态
    """

    # 常见来源模式识别
    SOURCE_PATTERNS = [
        (r"(?:国标|GB[/\s]|ISO[/\s]|IEEE[/\s])", "行业标准"),
        (r"(?:最高人民法院|国务院|部令|法规|条例)", "官方机构"),
        (r"(?:学术期刊|大学学报|doi:|DOI)", "学术期刊"),
        (r"(?:LLM提取|AI生成|auto[-_]extract|conjecture)", "LLM提取"),
        (r"(?:manual|人工|hand[-_]written)", "个人编写"),
    ]

    def audit_rule_source(
        self,
        rule_data: dict,
        package_meta: dict,
        extraction_info: dict | None = None,
    ) -> SourceProfile:
        """审查单条规则的来源。

        Args:
            rule_data: 规则的完整 dict
            package_meta: 规则包元信息 {domain, maintainer, description, ...}
            extraction_info: 提取方式信息 {method, confidence, ...}
        """
        rule_id = rule_data.get("id", "unknown")
        # 从规则包元信息推断来源类型
        maintainer = package_meta.get("maintainer", "")
        domain = package_meta.get("domain", "")
        description = package_meta.get("description", "")

        source_type = "未知"
        combined_text = f"{maintainer} {description} {rule_data.get('category', '')}"
        for pattern, stype in self.SOURCE_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                source_type = stype
                break

        # 特殊判断：maintainer 字段直接指定
        if maintainer.lower() in ("auto-extracted", "auto_extracted", "llm", "conjecture"):
            source_type = "LLM提取"

        # 提取方式
        ext_info = extraction_info or {}
        ext_method = ext_info.get("method", "")
        ext_confidence = ext_info.get("confidence", 0.0)

        # 类别判断提取方式
        category = rule_data.get("category", "")
        if "conjecture" in category.lower():
            ext_method = ext_method or "conjecture_mine"
        elif "auto" in category.lower():
            ext_method = ext_method or "keyword_scan"
        elif not ext_method:
            ext_method = "manual" if maintainer and maintainer not in ("auto-extracted", "conjecture") else "keyword_scan"

        # 领域一致性：规则包的 domain 和规则 category 是否匹配
        consistency = 1.0
        if domain and category:
            if domain not in category and category not in domain:
                consistency = 0.3

        # 是否有引用依据
        has_citations = bool(
            re.search(r'(?:引用|参考|依据|参见|ref|cite|source)', description + combined_text, re.IGNORECASE)
        )

        profile = SourceProfile(
            source_id=maintainer or f"source_{rule_id[:8]}",
            source_type=source_type,
            version=package_meta.get("version", "0.0.0"),
            domain_authority=0.8 if source_type in ("官方机构", "行业标准") else 0.5,
            consistency_score=consistency,
            has_citations=has_citations,
            extraction_method=ext_method,
            extraction_confidence=ext_confidence,
        )
        return profile

    def flag_source_issues(self, profile: SourceProfile) -> list[str]:
        """返回来源存在的问题列表，空列表表示无问题。"""
        issues = []
        if profile.credibility < 0.4:
            issues.append(f"来源可信度低（{profile.credibility:.2f}），类型={profile.source_type}")
        if profile.source_type == "LLM提取":
            issues.append("规则由 LLM 自动提取，未经人工审核确认")
        if profile.extraction_method == "conjecture_mine":
            issues.append("规则由统计猜想生成，仅概率关联，非逻辑证明")
        if profile.consistency_score < 0.5:
            issues.append("规则与所在领域的其他来源存在不一致")
        if not profile.has_citations and profile.source_type not in ("官方机构", "行业标准"):
            issues.append("规则缺少引用依据")
        return issues

    def audit_package_sources(
        self,
        evidence_chain: list[dict],
        package_registry: dict[str, dict],
    ) -> tuple[list[KnowledgeNode], dict[str, SourceProfile]]:
        """对证据链中所有涉及规则进行来源审计。

        返回 (source_uncertain_nodes, all_source_profiles)
        """
        source_uncertain_nodes: list[KnowledgeNode] = []
        all_profiles: dict[str, SourceProfile] = {}

        seen_rules: set[str] = set()
        for ev in evidence_chain:
            rule_id = ev.get("rule_id", "")
            pkg_id = ev.get("package_id", "")
            if rule_id in seen_rules:
                continue
            seen_rules.add(rule_id)

            pkg_meta = package_registry.get(pkg_id, {})
            # 重建规则数据（从 evidence 中提取）
            rule_data = {
                "id": rule_id,
                "name": ev.get("rule_name", rule_id),
                "category": ev.get("category", ""),
            }
            profile = self.audit_rule_source(rule_data, pkg_meta)
            all_profiles[rule_id] = profile

            issues = self.flag_source_issues(profile)
            if issues or profile.credibility < 0.6:
                # 这条规则来源存疑 → L2
                node = KnowledgeNode(
                    node_id=f"SRC_{rule_id}",
                    layer=KnowledgeLayer.L2_SOURCE_UNCERTAIN,
                    rule_id=rule_id,
                    term=ev.get("matched_terms", [""])[0] if ev.get("matched_terms") else ev.get("rule_name", ""),
                    description=f"来源审查：{'; '.join(issues)}" if issues else f"来源可信度偏低（{profile.credibility:.2f}）",
                    source=profile,
                    source_chain=[f"{profile.source_type} → {profile.extraction_method}"],
                    condition_feasible=True,
                    condition_notes="来源存疑，需追查原始出处",
                    confidence=profile.credibility,
                    evidence_snippets=[ev.get("input_fragment", "")[:200]],
                )
                source_uncertain_nodes.append(node)

        return source_uncertain_nodes, all_profiles


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Conditional Feasibility — 条件成立性分析
# ═══════════════════════════════════════════════════════════════════════

class ConditionalFeasibilityAnalyzer:
    """分析规则结果在给定条件下是否可能成立。

    问题：规则判定 FAILED，但在当前文本语境下，这个 FAILED 合理吗？
    比如"保密条款"和"违约赔偿"互斥规则触发，但文档是采购合同模板，
    这两个条款本来就该同时存在——规则本身可能设错了。

    分析维度：
    1. 文本语境是否匹配规则设计的应用场景
    2. 规则阈值是否过于严格
    3. 是否存在例外条件未被捕获
    """

    def analyze(
        self,
        evidence_item: dict,
        text_context: str,
        rule_meta: dict | None = None,
    ) -> tuple[bool, str]:
        """分析一条 evidence 在当前条件下的成立性。

        Returns:
            (feasible, explanation) — feasible=False 表示规则结果可能不成立/可疑
        """
        status = evidence_item.get("status", "")
        rule_type = ""  # 从 rule_meta 获取
        severity = evidence_item.get("severity", "info")

        # 只分析 FAILED 的证据（PASSED 的不需要质疑）
        if status != "FAILED":
            return True, "规则通过，无需条件审查"

        # 提取匹配术语
        matched_terms = evidence_item.get("matched_terms", [])
        fragment = evidence_item.get("input_fragment", "")

        reasons: list[str] = []

        # 1. 检查是否是 warning 级别 → 可能本来就是软约束
        if severity == "warning":
            reasons.append("规则严重度为 warning，可能是软约束而非硬条件")

        # 2. 检查是否有否定/例外词在上下文中
        exception_markers = ["除非", "例外", "除.*外", "不适用", "免责", "unless", "except", "not applicable"]
        for marker in exception_markers:
            if re.search(marker, fragment, re.IGNORECASE):
                reasons.append(f"文本包含例外标记「{marker}」，规则可能不适用于此段")
                break

        # 3. 检查匹配术语是否在上下文中被限定/修饰
        qualifiers = ["部分", "有限", "经.*同意", "在.*范围内", "partially", "limited to", "subject to"]
        for qualifier in qualifiers:
            if re.search(qualifier, fragment, re.IGNORECASE):
                reasons.append(f"匹配术语存在限定词，可能并非完全违反规则")
                break

        # 4. 检查是否是阈值边界情况
        if len(matched_terms) == 2 and evidence_item.get("rationale", ""):
            rationale = evidence_item.get("rationale", "")
            if "threshold is" in rationale.lower():
                reasons.append("恰好处在阈值边界，可能需要人工判断")

        feasible = len(reasons) <= 1  # 有 2 个以上可疑信号 → 标记为不可行
        explanation = "条件分析：" + "；".join(reasons) if reasons else "在当前条件下规则判定合理"

        return feasible, explanation


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Outer-Region Cross Analyzer — 规则外交叉分析
# ═══════════════════════════════════════════════════════════════════════

class OuterRegionAnalyzer:
    """分析规则覆盖不到的文本区域，发现可能的新知识。

    思路：
    1. 找出所有已触发规则覆盖的文本片段（covered regions）
    2. 找出未覆盖区域（uncovered regions）→ 从未触发任何规则的文本
    3. 在未覆盖区域中提取高频术语组合
    4. 与已覆盖区域的术语进行交叉对比
    5. 产出 L3_OUTER_POSSIBILITY 节点

    核心洞察：
    - 如果未覆盖区域频繁出现某组术语，而已有规则从未涉及 → 可能存在知识盲区
    - 如果未覆盖区域的术语与某条规则的术语高度互补 → 可能该规则应该扩展
    """

    # English stopwords to filter noise from L3
    _EN_STOPWORDS = {
        'this', 'that', 'the', 'and', 'for', 'has', 'its', 'not',
        'are', 'was', 'were', 'will', 'would', 'have', 'been', 'being',
        'can', 'could', 'may', 'might', 'shall', 'should', 'must',
        'with', 'from', 'they', 'them', 'their', 'there', 'here',
        'also', 'then', 'than', 'only', 'just', 'very', 'any', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'some', 'such',
        'other', 'into', 'over', 'under', 'after', 'before', 'between',
        'about', 'which', 'what', 'when', 'where', 'said', 'does', 'did',
        'well', 'now', 'new', 'one', 'two', 'per', 'use', 'used', 'using',
    }

    TERM_PATTERN = re.compile(
        r'[一-鿿]{2,8}|'      # 中文 2-8 字词
        r'[a-zA-Z_]{3,20}'            # 英文 3-20 字符
    )

    def extract_regions(
        self,
        text: str,
        evidence_chain: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Split text into covered and uncovered regions using sentence-level precision.

        Instead of fuzzy fragment matching (which leaves gaps),
        split the full text into sentences. Any sentence that contains
        a matched term from the evidence chain is 'covered'; all others
        are 'uncovered'. This guarantees L3 always has candidate regions.
        """
        if not text:
            return [], []

        # Collect all matched terms from evidence chain
        all_terms: set[str] = set()
        for ev in evidence_chain:
            for t in ev.get("matched_terms", []):
                if t:
                    all_terms.add(t.lower())

        # Split into sentences
        sentences = re.split(r'(?<=[。；;])\s*', text)
        covered: list[dict] = []
        uncovered: list[dict] = []

        pos = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 3:
                pos += len(sent) + 1
                continue
            terms = self.TERM_PATTERN.findall(sent)

            # A sentence is covered if ANY matched term appears in it
            sent_lower = sent.lower()
            is_covered = any(t in sent_lower for t in all_terms) if all_terms else False

            region = {"start": pos, "end": pos + len(sent), "text": sent, "terms": terms}
            if is_covered:
                covered.append(region)
            else:
                uncovered.append(region)
            pos += len(sent) + 1

        return covered, uncovered

    def cross_analyze(
        self,
        covered_regions: list[dict],
        uncovered_regions: list[dict],
        known_rule_ids: set[str],
    ) -> list[KnowledgeNode]:
        """交叉分析已覆盖和未覆盖区域，发现规则外可能的新知识。

        分析策略：
        1. 高频术语对比 — 未覆盖区域的高频词 vs 已覆盖区域
        2. 互补关系 — 未覆盖术语是否与已知规则术语形成互补
        3. 聚集检测 — 同一未覆盖区域内多术语共现
        """
        nodes: list[KnowledgeNode] = []

        # 收集术语统计
        covered_terms: Counter = Counter()
        for r in covered_regions:
            covered_terms.update(r.get("terms", []))

        uncovered_terms: Counter = Counter()
        for r in uncovered_regions:
            uncovered_terms.update(r.get("terms", []))

        # 去重
        unique_covered = set(covered_terms.keys())
        unique_uncovered = set(uncovered_terms.keys())

        # ── 分析 1: 未覆盖区域的高频独有术语 ──
        only_in_uncovered = unique_uncovered - unique_covered
        for term in only_in_uncovered:
            # Skip English stopwords and single chars
            if term.lower() in self._EN_STOPWORDS:
                continue
            if len(term) < 3:
                continue
            freq = uncovered_terms.get(term, 0)
            if freq >= 2:  # 至少出现 2 次才关注
                node = KnowledgeNode(
                    node_id=f"OUT_{uuid.uuid4().hex[:8]}",
                    layer=KnowledgeLayer.L3_OUTER_POSSIBILITY,
                    term=term,
                    description=f"规则外发现：术语「{term}」在未覆盖区域出现 {freq} 次，"
                                f"但未被任何已知规则覆盖，可能存在知识盲区",
                    cross_references=[],
                    cross_confidence=min(0.5, freq / 10),  # 频率归一化
                    regression_action="keep_outer",
                    condition_notes=f"该术语完全不在任何规则覆盖范围内，需人工确认其是否应纳入规则体系",
                    evidence_snippets=[
                        r["text"][:200] for r in uncovered_regions
                        if term in " ".join(r.get("terms", []))
                    ][:3],
                    confidence=min(0.6, freq / 8),
                )
                nodes.append(node)

        # ── 分析 2: 未覆盖区域的术语共现（可能形成新的规则集群） ──
        for region in uncovered_regions:
            region_terms = region.get("terms", [])
            # Filter stopwords
            region_terms = [t for t in region_terms if t.lower() not in self._EN_STOPWORDS and len(t) >= 3]
            if len(region_terms) >= 2:
                # 取高频的 2-gram 组合
                for i in range(len(region_terms)):
                    for j in range(i + 1, min(i + 5, len(region_terms))):
                        a, b = region_terms[i], region_terms[j]
                        if a in only_in_uncovered or b in only_in_uncovered:
                            node = KnowledgeNode(
                                node_id=f"CLU_{uuid.uuid4().hex[:8]}",
                                layer=KnowledgeLayer.L3_OUTER_POSSIBILITY,
                                term=f"{a} ⟷ {b}",
                                description=f"未覆盖区域中发现术语共现：「{a}」与「{b}」"
                                            f"在同一文本片段中共同出现，但无规则覆盖此关系",
                                cross_references=[a, b],
                                cross_confidence=0.3,
                                regression_action="keep_outer",
                                condition_notes="可能是一条新的共存/互斥规则的基础",
                                evidence_snippets=[region["text"][:200]],
                                confidence=0.25,
                            )
                            nodes.append(node)
                            break  # 每个区域最多产生一条共现发现

        return nodes


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Regression Resolver — 回归判定
# ═══════════════════════════════════════════════════════════════════════

class RegressionResolver:
    """判定知识节点应该回归到规则内，还是保持在规则外。

    回归条件：
    - 与已有规则高度互补（术语相似度 > 0.8）→ 回归为规则修正
    - 统计置信度足够高（cross_confidence > 0.7）→ 回归为猜想
    - 与已有规则无关 + 置信度低 → 保持在规则外
    """

    def resolve(
        self,
        outer_nodes: list[KnowledgeNode],
        existing_rule_terms: set[str],
        existing_rule_ids: set[str],
    ) -> tuple[list[KnowledgeNode], list[KnowledgeNode]]:
        """将规则外节点分类：回归规则内 / 保持规则外。

        Returns:
            (regressed_to_inner, kept_outer)
        """
        regressed: list[KnowledgeNode] = []
        kept: list[KnowledgeNode] = []

        for node in outer_nodes:
            # 与已有规则术语的重叠度
            node_terms = set(node.term.split(" ⟷ ")) if " ⟷ " in node.term else {node.term}
            overlap = node_terms & existing_rule_terms
            overlap_ratio = len(overlap) / max(len(node_terms), 1)

            if overlap_ratio >= 0.8 and node.cross_confidence >= 0.3:
                # 高度重叠 → 可能应该修正已有规则
                node.regression_action = "correct_rule"
                node.regression_target = next(iter(overlap), "")
                node.layer = KnowledgeLayer.L1_CONJECTURE
                node.description += " [回归：建议修正已有规则以覆盖此术语]"
                regressed.append(node)

            elif node.confidence >= 0.5 and overlap_ratio >= 0.5:
                # 中度相关 → 升级为猜想
                node.regression_action = "upgrade_to_conjecture"
                node.layer = KnowledgeLayer.L1_CONJECTURE
                node.description += " [回归：统计置信度足够，升级为猜想等待确认]"
                regressed.append(node)

            elif node.cross_confidence >= 0.6:
                # 交叉验证通过但无直接规则关联 → 建议新建规则
                node.regression_action = "new_rule"
                node.layer = KnowledgeLayer.L1_CONJECTURE
                node.description += " [回归：建议新建规则覆盖此发现]"
                regressed.append(node)

            else:
                # 其余 → 保持在规则外
                node.regression_action = "keep_outer"
                kept.append(node)

        return regressed, kept


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline: KnowledgeLayeringEngine
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeLayeringEngine:
    """知识自动分层引擎 — 主流水线。

    用法:
        engine = KnowledgeLayeringEngine()
        report = engine.layered_validate(
            text="待校验文本...",
            validation_result=engine.validate(...),
            package_registry={...},
            conjectures=[...],
        )
        # report.to_dict() 输出四层知识结构
    """

    def __init__(self):
        self.source_auditor = SourceAuditor()
        self.feasibility_analyzer = ConditionalFeasibilityAnalyzer()
        self.outer_analyzer = OuterRegionAnalyzer()
        self.regression_resolver = RegressionResolver()
        self._confirmed_rules: set[str] = set()  # user-confirmed rule IDs

    def confirm_rule(self, rule_id: str) -> None:
        """Mark a rule as L0 validated (user confirmed)."""
        self._confirmed_rules.add(rule_id)

    def layered_validate(
        self,
        text: str,
        validation_result: dict,
        package_registry: dict[str, dict],
        conjectures: list | None = None,
    ) -> LayerReport:
        """运行完整的知识分层流水线。

        Args:
            text: 原始输入文本
            validation_result: 引擎 validate() 的输出结果
            package_registry: {package_id: package_metadata} 所有已加载规则包的元信息
            conjectures: 可选的猜想列表

        Returns:
            LayerReport 完整四层报告
        """
        evidence_chain = validation_result.get("evidence_chain", [])
        conflicts = validation_result.get("conflicts", [])

        # ── Stage 1: 来源审查 ──
        src_nodes, source_profiles = self.source_auditor.audit_package_sources(
            evidence_chain, package_registry
        )

        # ── Stage 2: 条件成立性分析 ──
        condition_nodes: list[KnowledgeNode] = []
        for ev in evidence_chain:
            if ev.get("status") != "FAILED":
                continue
            feasible, explanation = self.feasibility_analyzer.analyze(ev, text)
            if not feasible:
                node = KnowledgeNode(
                    node_id=f"CND_{ev.get('rule_id', 'unknown')}",
                    layer=KnowledgeLayer.L2_SOURCE_UNCERTAIN,
                    rule_id=ev.get("rule_id", ""),
                    term=ev.get("matched_terms", [""])[0] if ev.get("matched_terms") else "",
                    description=f"条件存疑：{explanation}",
                    condition_feasible=False,
                    condition_notes=explanation,
                    source=source_profiles.get(ev.get("rule_id", "")),
                    confidence=0.4,
                    evidence_snippets=[ev.get("input_fragment", "")[:200]],
                )
                condition_nodes.append(node)

        # ── Stage 3: 规则外交叉分析 ──
        covered, uncovered = self.outer_analyzer.extract_regions(text, evidence_chain)
        outer_cross_nodes = self.outer_analyzer.cross_analyze(
            covered, uncovered,
            known_rule_ids=set(source_profiles.keys()),
        )

        # ── Stage 4: 回归判定 ──
        existing_terms = set()
        for ev in evidence_chain:
            for t in ev.get("matched_terms", []):
                existing_terms.add(t.lower())

        regressed, kept_outer = self.regression_resolver.resolve(
            outer_cross_nodes,
            existing_rule_terms=existing_terms,
            existing_rule_ids=set(source_profiles.keys()),
        )

        # ── 组装报告 ──
        # L0: 已通过的证据项（来源可信的 PASSED 规则）
        # ALSO include FAILED rules that have been explicitly confirmed
        l0_nodes = []
        for ev in evidence_chain:
            rid = ev.get("rule_id", "")
            # User-confirmed rules always go to L0
            if rid in self._confirmed_rules:
                l0_nodes.append(KnowledgeNode(
                    node_id=ev.get("trace_id", f"L0_{uuid.uuid4().hex[:6]}"),
                    layer=KnowledgeLayer.L0_VALIDATED,
                    rule_id=rid,
                    term=ev.get("matched_terms", [""])[0] if ev.get("matched_terms") else ev.get("rule_name", ""),
                    description=f"[已确认] {ev.get('rationale', '')[:100]}",
                    confidence=1.0,
                ))
                continue
            if ev.get("status") == "PASSED":
                profile = source_profiles.get(ev.get("rule_id", ""))
                # Be more generous: seed packages and manual rules get trusted
                trust = profile.credibility if profile else 0.9
                if profile and profile.extraction_method in ("manual", ""):
                    trust = 1.0  # manual rules are always trusted
                if trust >= 0.5:  # Lower bar to surface more L0 results
                    node = KnowledgeNode(
                        node_id=ev.get("trace_id", f"L0_{uuid.uuid4().hex[:6]}"),
                        layer=KnowledgeLayer.L0_VALIDATED,
                        rule_id=ev.get("rule_id", ""),
                        term=ev.get("matched_terms", [""])[0] if ev.get("matched_terms") else ev.get("rule_name", ""),
                        description=f"已验证通过：{ev.get('rationale', '')[:100]}",
                        source=profile,
                        condition_feasible=True,
                        confidence=trust,
                    )
                    l0_nodes.append(node)

        # L1: 回归的猜想 + 外部猜想
        l1_nodes = list(regressed)

        # L2: 来源存疑 + 条件存疑
        l2_nodes = src_nodes + condition_nodes

        # L3: 规则外发现
        l3_nodes = kept_outer

        # 生成摘要
        total_failures = sum(1 for e in evidence_chain if e.get("status") == "FAILED")
        summary_lines = [
            f"知识分层完成：{len(l0_nodes)} 条已验证规则通过",
            f"{len(l1_nodes)} 条回归猜想待确认",
            f"{len(l2_nodes)} 条规则来源或条件存疑",
            f"{len(l3_nodes)} 条规则外新发现",
        ]
        if total_failures > 0:
            summary_lines.append(f"注意：{total_failures} 条规则判定未通过，建议逐条审查来源和条件")

        report = LayerReport(
            l0_validated=l0_nodes,
            l1_conjecture=l1_nodes,
            l2_source_uncertain=l2_nodes,
            l3_outer_possibility=l3_nodes,
            total_rules_evaluated=len(evidence_chain),
            sources_audited=len(source_profiles),
            sources_flagged=len(src_nodes),
            outer_discoveries=len(outer_cross_nodes),
            regressions_to_inner=len(regressed),
            regressions_kept_outer=len(kept_outer),
            summary="\n".join(summary_lines),
        )

        logger.info(
            "Knowledge layering done: %d L0, %d L1, %d L2, %d L3",
            len(l0_nodes), len(l1_nodes), len(l2_nodes), len(l3_nodes),
        )
        return report


# ═══════════════════════════════════════════════════════════════════════
# Convenience: run layering on top of existing engine
# ═══════════════════════════════════════════════════════════════════════

def run_knowledge_layering(
    engine,  # PythonValidationEngine or RustEngineAdapter
    text: str,
    packages: list[str],
    package_registry: dict[str, dict],
    options: dict | None = None,
) -> dict:
    """在现有引擎校验结果之上运行知识分层。

    这是一个便捷函数——不破坏现有 validate() 流程，
    在拿到校验结果后额外调用分层引擎。

    Args:
        engine: 现有引擎实例（PythonValidationEngine 或 RustEngineAdapter）
        text: 待校验文本
        packages: 包 ID 列表
        package_registry: {pkg_id: pkg_metadata}
        options: validate() 选项

    Returns:
        包含 validation_result 和 layer_report 的完整 dict
    """
    result = engine.validate(
        input_data={"text": text},
        packages=packages,
        options=options or {},
    )

    layering = KnowledgeLayeringEngine()
    layer_report = layering.layered_validate(
        text=text,
        validation_result=result,
        package_registry=package_registry,
    )

    return {
        "validation": result,
        "knowledge_layers": layer_report.to_dict(),
    }
