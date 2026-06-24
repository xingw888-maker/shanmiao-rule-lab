"""Query engine — natural language questions → Z3 reasoning → answer + proof.

Usage:
    qe = QueryEngine()
    qe.load_knowledge(bootstrapped_knowledge)
    answer = qe.ask("屋面防水保修期需要多少年？")
    print(answer.proof)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.engine.solver import (
    CittaZ3Solver,
    LegalConstraint,
    NumericProposition,
    SolverResult,
    SolverVerdict,
    ViolationSeverity,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QueryAnswer:
    """Answer to a natural language query."""
    query: str
    answer_type: str               # "numeric", "boolean", "list", "comparison", "definition"
    answer: str                    # concise answer
    proof: str                     # step-by-step proof trace
    confidence: float
    source_rules: list[str]        # which rules were used
    numeric_value: Optional[float] = None
    numeric_unit: Optional[str] = None


@dataclass
class KnowledgeBase:
    """Structured knowledge ready for querying."""
    concepts: dict[str, dict]            # term → {parent, surfaces, freq}
    taxonomy: dict[str, list[str]]       # parent → children
    entity_groups: dict[str, list[str]]  # canonical → surfaces
    rules: list[dict]                    # rule entries
    constraints: list[LegalConstraint]   # Z3-ready constraints
    source_name: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Query Engine
# ═══════════════════════════════════════════════════════════════════════

class QueryEngine:
    """Natural language → structured query → Z3 → answer.

    Understands query types:
    - "X 需要多少年？" → numeric lookup + threshold comparison
    - "X 是否合规？" → boolean satisfiability check
    - "合同中哪些条款不满足法规？" → list all violations
    - "如果 X 改为 Y，是否合规？" → what-if analysis
    - "X 是什么？" → definition from concept registry
    """

    def __init__(self):
        self.kb: Optional[KnowledgeBase] = None
        self.solver = CittaZ3Solver()

    def load_knowledge(self, bootstrapped: "BootstrappedKnowledge") -> None:
        """Load bootstrapped knowledge into the query engine.

        Args:
            bootstrapped: BootstrappedKnowledge from AutoBootstrapper.
        """
        from app.engine.bootstrapper import BootstrappedKnowledge as BK

        concepts_dict = {}
        for c in bootstrapped.concepts:
            concepts_dict[c.term] = {
                "parent": c.parent_term,
                "surfaces": c.surface_forms,
                "freq": c.frequency,
            }

        # Convert numeric rules to LegalConstraints
        constraints = []
        for nr in bootstrapped.numeric_rules:
            severity = ViolationSeverity.MAJOR
            if any(w in nr.get("legal_ref", "") for w in ["法律", "条例", "国务院", "强制性"]):
                severity = ViolationSeverity.FATAL
            constraints.append(LegalConstraint(
                field=nr.get("label", ""),
                operator=nr.get("operator", ">="),
                threshold=nr.get("expected", 0),
                unit=nr.get("unit", ""),
                severity=severity,
                legal_ref=nr.get("legal_ref", ""),
            ))

        self.kb = KnowledgeBase(
            concepts=concepts_dict,
            taxonomy=bootstrapped.concepts_taxonomy,
            entity_groups=bootstrapped.entity_groups,
            rules=bootstrapped.numeric_rules + bootstrapped.other_rules,
            constraints=constraints,
            source_name=bootstrapped.source_title,
        )

    def ask(self, query: str, contract_propositions: list[NumericProposition] | None = None) -> QueryAnswer:
        """Answer a natural language question.

        Args:
            query: Natural language question in Chinese or English.
            contract_propositions: Optional contract facts for comparison.

        Returns:
            QueryAnswer with proof trace.
        """
        if not self.kb:
            return QueryAnswer(
                query=query, answer_type="error",
                answer="No knowledge loaded. Run load_knowledge() first.",
                proof="Knowledge base is empty.",
                confidence=0.0, source_rules=[],
            )

        # Classify query type and dispatch
        query_lower = query.lower()

        if any(w in query for w in ["多少", "几", "阈值", "下限", "上限", "最少", "最多", "至少", "至多"]):
            return self._answer_numeric_lookup(query)

        elif any(w in query for w in ["是否合规", "是否合法", "符合", "满足", "违反", "违规"]):
            return self._answer_compliance_check(query, contract_propositions)

        elif any(w in query for w in ["如果", "假如", "假设", "改为", "改成", "what if"]):
            return self._answer_what_if(query)

        elif any(w in query for w in ["列出", "全部", "所有", "哪些", "list"]):
            return self._answer_list(query)

        elif any(w in query for w in ["是什么", "什么是", "定义", "含义", "define"]):
            return self._answer_definition(query)

        else:
            # Default: numeric lookup
            return self._answer_numeric_lookup(query)

    # ── Query type handlers ──

    def _answer_numeric_lookup(self, query: str) -> QueryAnswer:
        """Look up the numeric threshold for a field.

        E.g. "屋面防水保修期需要多少年？" → "≥5年 (国务院令第279号)"
        """
        # Find matching concept/constraint
        best_match = None
        best_score = 0

        for constraint in self.kb.constraints:
            # Score by term overlap with query
            field = constraint.field
            score = self._term_overlap(query, field)
            # Also check surface forms
            for surfaces in self.kb.entity_groups.values():
                for s in surfaces:
                    score = max(score, self._term_overlap(query, s) * 0.7)

            if score > best_score:
                best_score = score
                best_match = constraint

        if best_match and best_score > 0.3:
            proof_lines = [
                f"1. 查询: {query}",
                f"2. 匹配字段: {best_match.field}",
                f"3. 法规约束: {best_match.field} {best_match.operator} {best_match.threshold} {best_match.unit}",
            ]
            if best_match.legal_ref:
                proof_lines.append(f"4. 法规依据: {best_match.legal_ref}")

            return QueryAnswer(
                query=query,
                answer_type="numeric",
                answer=f"{best_match.field} 必须 {best_match.operator} {best_match.threshold} {best_match.unit}",
                proof="\n".join(proof_lines),
                confidence=best_score,
                source_rules=[best_match.legal_ref],
                numeric_value=best_match.threshold,
                numeric_unit=best_match.unit,
            )

        return QueryAnswer(
            query=query, answer_type="numeric",
            answer="未找到匹配的法规阈值。",
            proof=f"在已加载的 {len(self.kb.constraints)} 条约束中未找到与 '{query}' 匹配的字段。",
            confidence=0.0, source_rules=[],
        )

    def _answer_compliance_check(
        self, query: str, propositions: list[NumericProposition] | None = None
    ) -> QueryAnswer:
        """Check if contract propositions satisfy all constraints.

        E.g. "屋面防水保修是否合规？"
        """
        if not propositions:
            return QueryAnswer(
                query=query, answer_type="boolean",
                answer="无法判定：未提供合同条款数据。",
                proof="需要 NumericProposition 列表作为合同事实输入。",
                confidence=0.0, source_rules=[],
            )

        self.solver.load_propositions(propositions)
        self.solver.load_constraints(self.kb.constraints)
        result = self.solver.check()

        if result.verdict == SolverVerdict.SATISFIABLE:
            answer = "合规：所有条款均满足法规要求。"
        elif result.verdict == SolverVerdict.UNSATISFIABLE:
            v_desc = "; ".join(
                f"{v.field} ({v.actual_value} vs {v.expected})"
                for v in result.violations[:5]
            )
            answer = f"不合规：{len(result.violations)} 条违反。{v_desc}"
        else:
            answer = "无法判定。"

        return QueryAnswer(
            query=query,
            answer_type="boolean",
            answer=answer,
            proof=result.proof_summary,
            confidence=0.95,
            source_rules=[v.legal_ref for v in result.violations if v.legal_ref],
        )

    def _answer_what_if(self, query: str) -> QueryAnswer:
        """What-if analysis: change a value and re-check.

        E.g. "如果保修期改为5年是否合规？"
        """
        # Parse: field name + proposed value
        field_match = re.search(r'(?:如果|假如|假设|改为|改成)\s*(.{2,10}?)\s*(?:改为|改成|调整为|变为|为)\s*([\d.]+)\s*(年|月|日|天|元|%|％)?', query)
        if not field_match:
            return QueryAnswer(
                query=query, answer_type="comparison",
                answer="无法解析 what-if 格式。",
                proof="格式: '如果[字段]改为[数值][单位]是否合规？'",
                confidence=0.0, source_rules=[],
            )

        field = field_match.group(1)
        value = float(field_match.group(2))
        unit = field_match.group(3) or ""

        # Find matching constraint
        constraint = None
        for c in self.kb.constraints:
            if self._term_overlap(field, c.field) > 0.5:
                constraint = c
                break

        if not constraint:
            return QueryAnswer(
                query=query, answer_type="comparison",
                answer=f"未找到与 '{field}' 匹配的法规约束。",
                proof=f"在 {len(self.kb.constraints)} 条约束中寻找 '{field}' 未果。",
                confidence=0.0, source_rules=[],
            )

        # Run what-if
        result = self.solver.what_if(constraint.field, value)

        proof = [
            f"1. 假设: {constraint.field} = {value} {unit}",
            f"2. 法规: {constraint.field} {constraint.operator} {constraint.threshold} {constraint.unit}",
            f"3. 判定: {result.proof_summary}",
        ]

        return QueryAnswer(
            query=query,
            answer_type="comparison",
            answer=result.proof_summary,
            proof="\n".join(proof),
            confidence=0.9,
            source_rules=[constraint.legal_ref],
            numeric_value=value,
            numeric_unit=unit,
        )

    def _answer_list(self, query: str) -> QueryAnswer:
        """List all constraints or rules of a certain type.

        E.g. "列出所有防水相关的法规要求"
        """
        # Filter by topic keywords in query
        topic_terms = re.findall(r'[一-鿿]{2,8}', query)
        filtered = self.kb.constraints
        if topic_terms:
            filtered = [
                c for c in self.kb.constraints
                if any(t in c.field for t in topic_terms)
            ]

        if not filtered:
            return QueryAnswer(
                query=query, answer_type="list",
                answer="未找到匹配的约束。",
                proof=f"在 {len(self.kb.constraints)} 条约束中筛选未果。",
                confidence=0.0, source_rules=[],
            )

        items = [
            f"{i+1}. {c.field}: {c.operator} {c.threshold} {c.unit} [{c.severity.value}]"
            for i, c in enumerate(filtered)
        ]

        return QueryAnswer(
            query=query,
            answer_type="list",
            answer=f"找到 {len(filtered)} 条约束:\n" + "\n".join(items),
            proof=f"从 {len(self.kb.constraints)} 条总约束中筛选。关键词: {', '.join(topic_terms)}",
            confidence=0.8,
            source_rules=[c.legal_ref for c in filtered if c.legal_ref],
        )

    def _answer_definition(self, query: str) -> QueryAnswer:
        """Look up a concept definition.

        E.g. "质量保证金是什么？"
        """
        # Find the concept being asked about
        target = ""
        for term in self.kb.concepts:
            if term in query:
                target = term
                break
        if not target:
            # Try surface forms
            for canonical, surfaces in self.kb.entity_groups.items():
                for s in surfaces:
                    if s in query:
                        target = canonical
                        break
                if target:
                    break

        if not target:
            return QueryAnswer(
                query=query, answer_type="definition",
                answer="未找到该概念的定义。",
                proof=f"在已加载的 {len(self.kb.concepts)} 个概念中未找到匹配。",
                confidence=0.0, source_rules=[],
            )

        concept = self.kb.concepts.get(target, {})
        surfaces = self.kb.entity_groups.get(target, [target])
        parent = concept.get("parent", "")
        children = self.kb.taxonomy.get(target, [])

        # Find rules about this concept
        related_rules = [
            r for r in self.kb.rules
            if target in r.get("label", "") or target in r.get("name", "")
        ]

        proof_lines = [
            f"1. 概念: {target}",
        ]
        if surfaces:
            proof_lines.append(f"2. 同义: {', '.join(surfaces)}")
        if parent:
            proof_lines.append(f"3. 上位概念: {parent}")
        if children:
            proof_lines.append(f"4. 下位概念: {', '.join(children[:5])}")
        if related_rules:
            proof_lines.append(f"5. 相关规则: {len(related_rules)} 条")

        return QueryAnswer(
            query=query,
            answer_type="definition",
            answer=f"{target}: {', '.join(surfaces)}" + (f"，属于 {parent}" if parent else ""),
            proof="\n".join(proof_lines),
            confidence=0.8,
            source_rules=[],
        )

    def _term_overlap(self, query: str, field: str) -> float:
        """Compute term overlap score between query and field name."""
        query_chars = set(query)
        field_chars = set(field)
        if not field_chars:
            return 0.0

        # Character-level Jaccard for CJK
        intersection = query_chars & field_chars
        union = query_chars | field_chars
        char_score = len(intersection) / max(len(union), 1)

        # Bigram overlap
        query_bigrams = {query[i:i+2] for i in range(len(query)-1)}
        field_bigrams = {field[i:i+2] for i in range(len(field)-1)}
        bigram_intersection = query_bigrams & field_bigrams
        bigram_union = query_bigrams | field_bigrams
        bigram_score = len(bigram_intersection) / max(len(bigram_union), 1)

        return char_score * 0.4 + bigram_score * 0.6

    # ── Counterfactual reasoning ──

    def counterfactual(
        self,
        remove_field: str,
        contract_propositions,
    ) -> dict:
        from app.engine.solver import CittaZ3Solver
        baseline_solver = CittaZ3Solver()
        baseline_solver.load_propositions(contract_propositions)
        baseline_solver.load_constraints(self.kb.constraints)
        baseline = baseline_solver.check()
        remaining = [p for p in contract_propositions if p.field != remove_field]
        cf_solver = CittaZ3Solver()
        cf_solver.load_propositions(remaining)
        cf_solver.load_constraints(self.kb.constraints)
        cf_result = cf_solver.check()
        baseline_viol = {v.field for v in baseline.violations}
        cf_viol = {v.field for v in cf_result.violations}
        newly_broken = list(cf_viol - baseline_viol - {remove_field})
        newly_fixed = list(baseline_viol - cf_viol)
        delta_lines = []
        if newly_broken:
            delta_lines.append(f"Removing '{remove_field}' BREAKS: {', '.join(newly_broken)}")
        if newly_fixed:
            delta_lines.append(f"Removing '{remove_field}' FIXES: {', '.join(newly_fixed)}")
        if not delta_lines:
            delta_lines.append(f"Removing '{remove_field}' has no effect.")
        return {
            "remove_field": remove_field,
            "baseline_verdict": baseline.verdict.value,
            "baseline_violations": len(baseline.violations),
            "counterfactual_verdict": cf_result.verdict.value,
            "counterfactual_violations": len(cf_result.violations),
            "newly_broken": newly_broken,
            "newly_fixed": newly_fixed,
            "delta_summary": "\n".join(delta_lines),
        }

    def counterfactual_chain(self, contract_propositions) -> list[dict]:
        results = []
        for prop in contract_propositions:
            cf = self.counterfactual(prop.field, contract_propositions)
            impact = len(cf["newly_broken"]) + len(cf["newly_fixed"]) * 0.5
            results.append({"field": prop.field, "impact_score": impact, **cf})
        results.sort(key=lambda r: -r["impact_score"])
        return results
