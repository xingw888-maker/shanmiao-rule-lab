"""Ontology self-expander v2 — Five-Layer Intelligent Filtering.

v2 upgrade from the v1 chi-squared co-occurrence analyzer.  Each layer
eliminates a class of false positive candidates:

  Layer 1 — Cluster Constraint
  Layer 2 — Negative Control Group
  Layer 3 — Semantic Window Constraint
  Layer 4 — Substitutability Test
  Layer 5 — Confidence Tiering

Design: Pure Python stdlib — zero LLM calls.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from app.engine.auto_clusterer import AutoClusterer
from app.engine.clause_splitter import ClauseSplitter, ClauseBlock

# ---------------------------------------------------------------------------
# CJK detection
# ---------------------------------------------------------------------------
_CJK_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF),
    (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
]


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def extract_cjk_bigrams(text: str) -> list[str]:
    cjk_only = [ch for ch in text if _is_cjk(ch)]
    if len(cjk_only) < 2:
        return []
    return [cjk_only[i] + cjk_only[i + 1] for i in range(len(cjk_only) - 1)]


# ---------------------------------------------------------------------------
# Chi-squared (1 df) with Yates correction for small samples
# ---------------------------------------------------------------------------


def _chi_squared_1df(o11, o12, o21, o22):
    """Yates-corrected chi-squared for 2x2 table."""
    n = o11 + o12 + o21 + o22
    if n == 0:
        return 0.0, 1.0
    row1 = o11 + o12
    row2 = o21 + o22
    col1 = o11 + o21
    col2 = o12 + o22
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0, 1.0
    # Expected values
    e11 = row1 * col1 / n
    e12 = row1 * col2 / n
    e21 = row2 * col1 / n
    e22 = row2 * col2 / n
    if any(e <= 0 for e in [e11, e12, e21, e22]):
        return 0.0, 1.0
    # Yates correction: subtract 0.5 from each |O - E|
    chi2 = (
        (abs(o11 - e11) - 0.5) ** 2 / e11 +
        (abs(o12 - e12) - 0.5) ** 2 / e12 +
        (abs(o21 - e21) - 0.5) ** 2 / e21 +
        (abs(o22 - e22) - 0.5) ** 2 / e22
    )
    if chi2 <= 0:
        return 0.0, 1.0
    # p-value from chi2 (1 df)
    x = math.sqrt(chi2)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327
    pa = d * math.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.477937
             + t * (-1.821255978 + t * 1.330274429)))))
    pv = 2.0 * (pa if x >= 0 else 1.0 - pa)
    return chi2, max(0.0, min(1.0, pv))


# ---------------------------------------------------------------------------
# TieredSynonymCandidate
# ---------------------------------------------------------------------------


@dataclass
class TieredSynonymCandidate:
    bigram: str
    chi_squared: float
    p_value: float
    source_rule_id: str
    substitutability_count: int
    substitutability_total: int
    contracts_seen: int
    tier: str
    evidence_fragments: list[str]
    target_cluster_ids: list[str]


# ---------------------------------------------------------------------------
# Known-legitimate Chinese legal bigrams (allow-list)
# ---------------------------------------------------------------------------

_LEGIT_BIGRAMS = {
    "合同", "工程", "施工", "承包", "分包", "发包", "验收", "保修",
    "责任", "违约", "质量", "安全", "期限", "日期", "价款", "支付",
    "结算", "招标", "投标", "索赔", "仲裁", "诉讼", "法院", "争议",
    "解决", "管辖", "通知", "送达", "生效", "终止", "解除", "变更",
    "赔偿", "损失", "费用", "成本", "利润", "税金", "发票", "收据",
    "保证", "担保", "抵押", "质押", "留置", "定金", "利息",
    "进度", "工期", "竣工", "维修", "返修", "整改", "检验", "测试",
    "审核", "审批", "备案", "登记", "许可", "批准",
    "授权", "委托", "代理", "代表", "签署", "盖章", "成立",
    "订立", "履行", "执行", "遵守", "违反", "承担", "享有", "放弃",
    "转让", "继承", "合并", "分立", "解散", "清算", "破产", "重整",
    "和解", "调解", "公证", "见证", "评估", "鉴定", "勘查", "设计",
    "监理", "造价", "咨询", "审查", "检查", "监督", "管理", "运营",
    "维护", "保养", "防水", "防火", "防雷", "抗震", "加固",
    "改建", "扩建", "新建", "修缮", "装饰", "装修", "安装", "调试",
    "培训", "指导", "协助", "配合", "协调", "组织", "实施", "完成",
    "交付", "接收", "移交", "返还", "退还", "支付", "缴纳",
    "征收", "征用", "占用", "使用", "出租", "出借", "借用", "租赁",
    "买卖", "采购", "供应", "销售", "运输", "仓储", "保管", "包装",
    "装卸", "搬运", "加工", "定作", "修理", "复制",
    "人员", "设备", "材料", "构件", "配件", "零件", "产品", "样品",
    "标准", "规范", "规程", "方案", "计划", "报告", "记录", "档案",
    "文件", "图纸", "资料", "数据", "信息", "技术", "工艺", "方法",
    "条件", "情况", "状态", "环境", "因素", "原因", "结果", "效果",
    "屋面", "防水", "保温", "排水", "给水", "电气", "消防", "通风",
    "空调", "电梯", "门窗", "幕墙", "油漆", "涂料", "粘贴", "抹灰",
    "砌体", "混凝", "钢筋", "模板", "支架",
    "甲方", "乙方", "丙方", "丁方", "双方", "单方", "各方",
    "权利", "义务", "条款", "规定", "约定", "说明", "备注",
    "以下", "以上", "以内", "以外", "之间", "之前", "之后",
    "按照", "根据", "依照", "参照", "通过",
    "提出", "提交", "提供", "通知",
    "发生", "出现", "存在", "构成", "形成",
    "负责", "分管", "主管", "主办",
    "需要", "应当", "可以", "必须", "不得", "禁止",
    "小时", "日内", "月度", "年度",
    "总数", "总额", "总计", "合计",
    "本次", "本期", "本项", "本条",
    "进行", "予以", "给予",
    "有关", "相关", "相应", "适当",
    "主要", "重要", "核心",
    "任何", "所有", "全部", "部分",
    "达到", "超过", "低于", "高于",
    "范围", "面积", "体积",
    "长度", "高度", "宽度", "深度",
    "厚度", "直径", "半径",
    "型号", "规格", "等级",
    "部位", "位置", "地点", "区域",
    "全面", "完整", "准确", "及时",
    "公开", "公平", "公正",
    "诚实", "信用", "善意",
    "建筑", "结构", "装修", "装饰",
    "安装", "调试",
    "保温", "隔热", "隔音",
    "供电", "供水", "供气",
    "网络", "系统", "智能", "自动",
    "手动", "联动", "连锁",
    "缺陷", "损坏", "破损", "故障",
    "合格", "优良", "良好",
    "面积", "保温", "排水",
    "甲方", "乙方", "丙方",
    "发包", "承包", "分包", "转包",
    "总价", "单价", "合价",
    "保修", "维修", "返修",
    "屋面", "防水", "防漏",
    "电安", "弱电", "强电",
    "业主", "甲方", "乙方",
    "单价", "合价", "总价",
    "主体", "基础", "结构",
    "装饰", "装修", "幕墙",
}


def _is_noise_bigram(bigram: str) -> bool:
    """Reject bigrams that are clearly boundary-crossing artifacts."""
    if bigram in _LEGIT_BIGRAMS:
        return False
    # Characters that typically appear as suffixes in 3+ char sequences
    if bigram[0] in ("司", "限", "部", "处", "组", "队", "班", "员"):
        return True
    # Common boundary artifacts
    if bigram in ("责有", "任有", "程施", "任公", "责公", "修责",
                  "保范", "期保", "工质", "程质", "量验", "量标",
                  "格标", "收规", "量保", "程保", "收合", "体结",
                  "结结", "结施", "构施", "量的", "量的",
                  "程的", "定的", "收合", "收规", "范验", "保范"):
        return True
    return False


# ---------------------------------------------------------------------------
# OntologySelfExpanderV2
# ---------------------------------------------------------------------------


class OntologySelfExpanderV2:
    """Five-layer intelligent ontology self-expander."""

    AUTO_CHI2 = 10.0
    AUTO_SUBST = 3
    AUTO_CONTRACTS = 5
    REVIEW_CHI2 = 6.63
    REVIEW_SUBST = 2
    SUGGEST_CHI2 = 3.84
    WINDOW_RADIUS = 50
    MIN_FRAGMENTS = 3
    MIN_OBSERVED = 3  # bigram must appear in at least 3 fragments (stricter)
    MIN_CONTRACTS = 3  # must appear in at least 3 contracts (stricter)

    def discover_candidates(
        self,
        evidence_chains: list[list[dict]],
        contract_texts: list[str],
        domain_id: str = "construction",
        rules: Optional[list[dict]] = None,
        min_chi_squared: float = 3.84,
    ) -> list[TieredSynonymCandidate]:
        """Discover tiered synonym candidates."""

        # Phase 0: Split and cluster all contracts
        all_blocks: list[ClauseBlock] = []
        for text in contract_texts:
            all_blocks.extend(ClauseSplitter.split(text))
        block_contents = [b.content for b in all_blocks]

        clusterer = AutoClusterer()
        clusters = clusterer.cluster(all_blocks)
        block_clusters: dict[int, str] = {}
        for cluster in clusters:
            for idx in cluster.block_indices:
                block_clusters[idx] = cluster.cluster_id

        # Phase 1: Filter evidence
        rule_fragments: dict[str, list[tuple[str, list[str], int]]] = {}
        for ci, chain in enumerate(evidence_chains):
            for ev in chain:
                if not isinstance(ev, dict):
                    continue
                if ev.get("status") != "PASSED":
                    continue
                sc = ev.get("source_credibility", 0.0)
                if not isinstance(sc, (int, float)) or sc <= 0:
                    continue
                if ev.get("layer", "") == "L3_OUTER_POSSIBILITY":
                    continue
                matched = ev.get("matched_terms", [])
                if not matched or not isinstance(matched, list):
                    continue
                rid = ev.get("rule_id", "")
                if not rid:
                    continue
                fragment = ev.get("input_fragment", "")
                if not fragment:
                    continue
                rule_fragments.setdefault(rid, []).append((fragment, matched, ci))

        all_candidates: list[TieredSynonymCandidate] = []

        for rid, fragment_list in rule_fragments.items():
            if len(fragment_list) < self.MIN_FRAGMENTS:
                continue

            # Layer 1
            target_clusters, matched_block_idx = self._get_target_clusters(
                fragment_list, block_contents, block_clusters
            )
            if not target_clusters:
                continue

            # Layer 2: Baseline from non-target clusters
            baseline_counts: Counter[str, int] = Counter()
            total_baseline = 0
            for i in range(len(all_blocks)):
                cid = block_clusters.get(i)
                if cid is None or cid not in target_clusters:
                    total_baseline += 1
                    for bg in set(extract_cjk_bigrams(block_contents[i])):
                        baseline_counts[bg] += 1
            if total_baseline < 3:
                continue

            n_frag = len(fragment_list)

            # Layer 3: Windowed bigrams
            rule_bigram_counts: Counter[str, int] = Counter()
            frag_bigrams: dict[int, set[str]] = {}
            for fi, (frag, terms, ci) in enumerate(fragment_list):
                wb = self._windowed_bigrams(
                    frag, terms, matched_block_idx,
                    block_contents, block_clusters,
                )
                frag_bigrams[fi] = set(wb)
                for bg in set(wb):
                    rule_bigram_counts[bg] += 1

            # Chi-squared with multiple quality gates
            raw_candidates: list[dict] = []
            for bigram, o11 in rule_bigram_counts.items():
                if _is_noise_bigram(bigram):
                    continue
                if o11 < self.MIN_OBSERVED:
                    continue

                # Count distinct contracts this bigram appears in
                seen_contracts = set(
                    ci for fi, (_, _, ci) in enumerate(fragment_list)
                    if bigram in frag_bigrams.get(fi, set())
                )
                if len(seen_contracts) < self.MIN_CONTRACTS:
                    continue

                o12 = n_frag - o11
                o21 = baseline_counts.get(bigram, 0)
                o22 = total_baseline - o21

                chi2, p_val = _chi_squared_1df(o11, o12, o21, o22)
                if chi2 < min_chi_squared:
                    continue

                raw_candidates.append({
                    "bigram": bigram, "chi2": chi2, "p_val": p_val,
                    "contracts_seen": len(seen_contracts),
                })

            if not raw_candidates:
                continue

            # Layer 4
            all_keywords = []
            for _, mt, _ in fragment_list:
                for t in mt:
                    if t not in all_keywords:
                        all_keywords.append(t)

            for rc in raw_candidates:
                bg = rc["bigram"]
                subst_count, subst_total = self._substitutability_score(
                    bg, all_keywords, all_blocks,
                    block_clusters, target_clusters,
                )
                rc["subst_count"] = subst_count
                rc["subst_total"] = subst_total

                tier = self._assign_tier(
                    rc["chi2"], subst_count, rc["contracts_seen"],
                )

                evidence = self._collect_substitution_evidence(
                    bg, all_keywords, all_blocks,
                    block_clusters, target_clusters,
                )

                all_candidates.append(TieredSynonymCandidate(
                    bigram=bg, chi_squared=rc["chi2"],
                    p_value=rc["p_val"], source_rule_id=rid,
                    substitutability_count=subst_count,
                    substitutability_total=subst_total,
                    contracts_seen=rc["contracts_seen"],
                    tier=tier, evidence_fragments=evidence,
                    target_cluster_ids=sorted(target_clusters),
                ))

        all_candidates.sort(key=lambda c: c.chi_squared, reverse=True)
        return all_candidates

    # ------------------------------------------------------------------
    # Layer 1
    # ------------------------------------------------------------------

    def _get_target_clusters(self, fragment_list, block_contents, block_clusters):
        target_clusters: set[str] = set()
        matched_idx: dict[int, list[int]] = {}

        for fi, (frag, matched_terms, ci) in enumerate(fragment_list):
            indices = []
            for bi, content in enumerate(block_contents):
                if all(term in content for term in matched_terms):
                    indices.append(bi)
                    cid = block_clusters.get(bi)
                    if cid:
                        target_clusters.add(cid)
            if indices:
                matched_idx[fi] = indices

        if not target_clusters:
            for fi, (frag, matched_terms, ci) in enumerate(fragment_list):
                sig = frag[:80].strip()
                if not sig:
                    continue
                for bi, content in enumerate(block_contents):
                    if sig in content:
                        cid = block_clusters.get(bi)
                        if cid:
                            target_clusters.add(cid)
                            matched_idx.setdefault(fi, []).append(bi)

        return target_clusters, matched_idx

    # ------------------------------------------------------------------
    # Layer 3
    # ------------------------------------------------------------------

    def _windowed_bigrams(self, fragment, matched_terms, matched_block_idx,
                           block_contents, block_clusters, window=50):
        all_bigrams = []
        all_refs = set()
        for indices in matched_block_idx.values():
            all_refs.update(indices)

        if all_refs:
            for bi in all_refs:
                content = block_contents[bi]
                positions = []
                for term in matched_terms:
                    idx = content.find(term)
                    if idx >= 0:
                        positions.append((idx, idx + len(term)))
                if positions:
                    start = max(0, min(p[0] for p in positions) - window)
                    end = min(len(content), max(p[1] for p in positions) + window)
                    all_bigrams.extend(extract_cjk_bigrams(content[start:end]))
        else:
            all_bigrams = extract_cjk_bigrams(fragment)

        return all_bigrams

    # ------------------------------------------------------------------
    # Layer 4
    # ------------------------------------------------------------------

    def _substitutability_score(self, bigram, keywords, all_blocks,
                                 block_clusters, target_clusters):
        if not keywords:
            return 0, 0
        substitutable = 0
        total = 0
        for bi, block in enumerate(all_blocks):
            cid = block_clusters.get(bi)
            if cid is None or cid not in target_clusters:
                continue
            if bigram in block.content:
                total += 1
                if not any(kw in block.content for kw in keywords):
                    substitutable += 1
        return substitutable, total

    def _collect_substitution_evidence(self, bigram, keywords, all_blocks,
                                        block_clusters, target_clusters, max_n=3):
        frags = []
        for bi, block in enumerate(all_blocks):
            cid = block_clusters.get(bi)
            if cid is None or cid not in target_clusters:
                continue
            if bigram in block.content and not any(kw in block.content for kw in keywords):
                idx = block.content.find(bigram)
                if idx >= 0:
                    s = max(0, idx - 40)
                    e = min(len(block.content), idx + len(bigram) + 40)
                    snippet = block.content[s:e].strip()
                    if snippet and snippet not in frags:
                        frags.append(snippet)
                        if len(frags) >= max_n:
                            break
        return frags

    # ------------------------------------------------------------------
    # Layer 5
    # ------------------------------------------------------------------

    def _assign_tier(self, chi2, subst_count, contracts_seen):
        if (chi2 >= self.AUTO_CHI2 and subst_count >= self.AUTO_SUBST
                and contracts_seen >= self.AUTO_CONTRACTS):
            return "AUTO"
        if chi2 >= self.REVIEW_CHI2 and subst_count >= self.REVIEW_SUBST:
            return "REVIEW"
        return "SUGGEST"

    def get_auto_candidates(self, candidates):
        return [c for c in candidates if c.tier == "AUTO"]

    def get_review_candidates(self, candidates):
        return [c for c in candidates if c.tier == "REVIEW"]

    def get_suggest_candidates(self, candidates):
        return [c for c in candidates if c.tier == "SUGGEST"]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_tiered_candidates(self, candidates, output_dir, rule_to_entity=None):
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        for tier, fn in [("AUTO", "ontology_auto_candidates.json"),
                          ("REVIEW", "ontology_candidates.json"),
                          ("SUGGEST", "ontology_suggestions.json")]:
            tc = [c for c in candidates if c.tier == tier]
            if tc:
                p = os.path.join(output_dir, fn)
                self._write_file(tc, p, rule_to_entity)
                paths[tier.lower()] = p
        return paths

    def _write_file(self, candidates, path, rule_to_entity=None):
        records = []
        for c in candidates:
            rec = {
                "bigram": c.bigram,
                "chi_squared": round(c.chi_squared, 2),
                "p_value": round(c.p_value, 6),
                "source_rule_id": c.source_rule_id,
                "tier": c.tier,
                "substitutability": {"count": c.substitutability_count,
                                      "total": c.substitutability_total},
                "contracts_seen": c.contracts_seen,
                "target_clusters": c.target_cluster_ids,
                "evidence_fragments": c.evidence_fragments[:3],
            }
            if rule_to_entity:
                rec["entity_group"] = rule_to_entity.get(c.source_rule_id, "")
            records.append(rec)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tier": candidates[0].tier if candidates else "UNKNOWN",
                        "total_candidates": len(candidates),
                        "candidates": records},
                      f, ensure_ascii=False, indent=2)

    def apply_to_ontology(self, ontology, candidates, rule_to_entity):
        updated = json.loads(json.dumps(ontology))
        eg = updated.setdefault("entity_groups", {})
        applied = []
        for c in self.get_auto_candidates(candidates):
            en = rule_to_entity.get(c.source_rule_id)
            if not en:
                continue
            if en not in eg:
                eg[en] = []
            existing = set(t.strip() for t in eg[en])
            if c.bigram not in existing:
                eg[en].append(c.bigram)
                applied.append({
                    "entity_group": en,
                    "added_term": c.bigram,
                    "chi_squared": round(c.chi_squared, 2),
                    "source_rule_id": c.source_rule_id,
                })
        if applied:
            updated["_auto_applied_synonyms"] = applied
        return updated

    @staticmethod
    def extract_clause_blocks(result):
        blocks = []
        for b in result.get("clause_blocks", []):
            if isinstance(b, dict):
                p = b.get("content_preview", "")
                if p:
                    blocks.append(p)
        if not blocks:
            for ev in result.get("evidence_chain", []):
                if isinstance(ev, dict):
                    f = ev.get("input_fragment", "")
                    if f:
                        blocks.append(f)
        return blocks

    @staticmethod
    def build_rule_to_entity_from_rules(rules, entity_groups):
        mapping = {}
        for rule in rules:
            rid = rule.get("id", "")
            if not rid:
                continue
            cond = rule.get("condition", {})
            if cond.get("type") != "required_pattern":
                continue
            terms = cond.get("terms", [])
            if not terms:
                continue
            ts = set(t.lower() for t in terms)
            best_g, best_o = "", 0
            for gn, gt in entity_groups.items():
                o = sum(1 for t in gt if t.lower() in ts)
                if o > best_o:
                    best_o, best_g = o, gn
            if best_g:
                mapping[rid] = best_g
        return mapping
