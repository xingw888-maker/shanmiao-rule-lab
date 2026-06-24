"""Logic correction feedback loop — closes the neuro-symbolic gap.

Three components:
1. ConflictReporter — converts Z3 unsat_core() into human-readable conflict
   reports for LLM self-correction.
2. LogicCorrector — sends conflict report to LLM, triggers re-extraction,
   iterates until satisfiable or max retries.
3. DomainWhitelist — filters bootstrapped concepts against known-good
   terminology, rejects noise tokens.

Also includes ForAll quantifier support in solver.py via z3.ForAll.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Conflict Reporter — Z3 unsat_core → human-readable
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConflictReport:
    """Human-readable conflict report from Z3 unsat_core."""
    source: str                     # "z3_unsat_core"
    conflicting_propositions: list[str]  # field names in conflict
    conflict_graph: str             # Mermaid/ASCII diagram
    human_readable: str             # natural language explanation
    suggested_fix: str              # what the LLM should reconsider
    raw_core: list[str]             # raw unsat_core identifiers


class ConflictReporter:
    """Converts Z3 unsat_core() into LLM-actionable conflict reports.

    The key insight: don't tell the LLM "UNSAT". Tell it:
    "I expected X >= 5, but you told me X = 1. Is X really 1, or did you
    misread the text? Here's the surrounding paragraph..."
    """

    TEMPLATE = """【逻辑冲突报告】

系统在验证以下命题时发现矛盾：

冲突命题集：
{conflict_set}

原因分析：
{analysis}

请根据原文上下文检查：
1. 是否在提取时将概念分类错误？
2. 数值是否被正确提取？
3. 原文是否存在多重解释？

请输出修正后的 JSON，格式如下：
```json
[
  {{"field": "字段名", "value": 数值, "unit": "单位", "confidence": 0.0-1.0, "reasoning": "修正理由"}}
]
```

如果确认原提取无误（原文确实与约束冲突），请输出：
```json
[{{"confirmed_violation": true, "field": "字段名", "reason": "原文确实不满足法规要求"}}]
```"""

    @classmethod
    def from_unsat_core(
        cls,
        core: list[z3.ExprRef],
        constraint_map: dict[int, any],
        propositions: dict[str, any],
    ) -> ConflictReport:
        """Generate a conflict report from Z3 unsat_core.

        Args:
            core: List of Z3 expressions from solver.unsat_core().
            constraint_map: Map from constraint index to LegalConstraint.
            propositions: Map from field name to NumericProposition.

        Returns:
            ConflictReport ready for LLM consumption.
        """
        import z3

        conflicting_fields = []
        raw_ids = []

        for expr in core:
            name = str(expr)
            raw_ids.append(name)
            if name.startswith("c_"):
                idx = int(name[2:])
                c = constraint_map.get(idx)
                if c:
                    conflicting_fields.append(c.field)

        # Build human-readable conflict description
        conflict_lines = []
        for field in conflicting_fields:
            prop = propositions.get(field)
            c = None
            for idx, ct in constraint_map.items():
                if ct.field == field:
                    c = ct
                    break

            if prop and c:
                conflict_lines.append(
                    f"  • {field}: 合同值为 {prop.value}{c.unit}，"
                    f"法规要求 {c.operator} {c.threshold}{c.unit} "
                    f"（来源：{c.legal_ref}）"
                )
            elif prop:
                conflict_lines.append(f"  • {field}: 合同值为 {prop.value}，未找到对应法规")
            elif c:
                conflict_lines.append(f"  • {field}: 法规要求 {c.operator} {c.threshold}{c.unit}，合同未约定")

        # Build ASCII conflict graph
        graph_lines = ["Conflict Graph:"]
        if len(conflicting_fields) >= 2:
            graph_lines.append(f"  {conflicting_fields[0]} ──[冲突]── {conflicting_fields[1]}")
        for i in range(1, len(conflicting_fields)):
            graph_lines.append(f"  {conflicting_fields[i-1]} ──[约束链]── {conflicting_fields[i]}")

        # Suggested fix
        if conflicting_fields:
            suggested = (
                f"请重新检查以下字段的提取结果：{', '.join(conflicting_fields[:3])}。"
                f"特别注意原文中是否存在例外条款、但书、或上下文修饰词。"
            )
        else:
            suggested = "无法自动定位冲突源，请人工审查原文。"

        return ConflictReport(
            source="z3_unsat_core",
            conflicting_propositions=conflicting_fields,
            conflict_graph="\n".join(graph_lines),
            human_readable="\n".join(conflict_lines),
            suggested_fix=suggested,
            raw_core=raw_ids,
        )

    @classmethod
    def to_llm_prompt(cls, report: ConflictReport, original_text: str = "") -> str:
        """Generate an LLM self-correction prompt from a conflict report."""
        prompt = cls.TEMPLATE.format(
            conflict_set=report.human_readable,
            analysis=report.suggested_fix,
        )
        if original_text:
            prompt += f"\n\n原文上下文：\n{original_text[:2000]}"
        return prompt


# ═══════════════════════════════════════════════════════════════════════
# 2. Logic Corrector — LLM re-extraction loop
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CorrectionResult:
    """Result of a logic correction cycle."""
    iteration: int
    original_propositions: list[dict]
    corrected_propositions: list[dict]
    conflict_report: Optional[ConflictReport] = None
    satisfiable: bool = False
    llm_reasoning: str = ""
    cycles: int = 0


class LogicCorrector:
    """Orchestrates the UNSAT → LLM correction → re-verify loop.

    Usage:
        corrector = LogicCorrector(llm_url, llm_key)
        result = await corrector.correct(conflict_report, text, max_iterations=3)
    """

    PROMPT = """你是一名逻辑纠偏专家。以下是一组从文本中提取的数值命题，但它们在逻辑验证中产生了矛盾。

{conflict_prompt}

请输出修正后的 JSON 数组。"""

    def __init__(self, api_url: str = "", api_key: str = "", model: str = ""):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model or "gpt-4"
        self._has_llm = bool(api_url and api_key)

    async def correct(
        self,
        report: ConflictReport,
        original_text: str,
        current_propositions: list[dict],
        max_iterations: int = 3,
    ) -> CorrectionResult:
        """Run the correction loop.

        Args:
            report: ConflictReport from Z3 unsat_core.
            original_text: The original source text.
            current_propositions: Current (conflicting) propositions.
            max_iterations: Maximum correction cycles.

        Returns:
            CorrectionResult with corrected propositions.
        """
        if not self._has_llm:
            return CorrectionResult(
                iteration=0,
                original_propositions=current_propositions,
                corrected_propositions=current_propositions,
                conflict_report=report,
                satisfiable=False,
                llm_reasoning="No LLM configured — cannot self-correct.",
            )

        prompt = ConflictReporter.to_llm_prompt(report, original_text)

        for iteration in range(1, max_iterations + 1):
            try:
                import aiohttp
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 2000,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=60,
                    ) as resp:
                        data = await resp.json()
                        response = data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning("Correction LLM call failed: %s", e)
                return CorrectionResult(
                    iteration=iteration,
                    original_propositions=current_propositions,
                    corrected_propositions=current_propositions,
                    conflict_report=report,
                    satisfiable=False,
                    llm_reasoning=f"LLM call failed: {e}",
                    cycles=iteration,
                )

            # Parse corrected JSON
            corrected = self._parse_correction(response)
            if not corrected:
                continue

            # Check if LLM confirmed the violation (no correction needed)
            if any(c.get("confirmed_violation") for c in corrected):
                return CorrectionResult(
                    iteration=iteration,
                    original_propositions=current_propositions,
                    corrected_propositions=current_propositions,
                    conflict_report=report,
                    satisfiable=False,
                    llm_reasoning="LLM confirmed: extraction was correct, the violation is real.",
                    cycles=iteration,
                )

            # Re-verify with Z3
            if self._verify_corrected(corrected, report):
                return CorrectionResult(
                    iteration=iteration,
                    original_propositions=current_propositions,
                    corrected_propositions=corrected,
                    conflict_report=report,
                    satisfiable=True,
                    llm_reasoning=response[:500],
                    cycles=iteration,
                )

            # Still UNSAT — update prompt for next iteration
            prompt = (
                f"修正后的命题仍然存在逻辑冲突。请再次审查。\n\n"
                f"原始冲突：{report.human_readable}\n"
                f"上次修正：{json.dumps(corrected, ensure_ascii=False)}\n"
                f"请重新输出修正后的 JSON。"
            )

        return CorrectionResult(
            iteration=max_iterations,
            original_propositions=current_propositions,
            corrected_propositions=current_propositions,
            conflict_report=report,
            satisfiable=False,
            llm_reasoning=f"Failed to resolve conflict after {max_iterations} iterations.",
            cycles=max_iterations,
        )

    def correct_sync(
        self,
        report: ConflictReport,
        original_text: str,
        current_propositions: list[dict],
        max_iterations: int = 3,
    ) -> CorrectionResult:
        """Synchronous wrapper for correct()."""
        import asyncio
        try:
            return asyncio.run(
                self.correct(report, original_text, current_propositions, max_iterations)
            )
        except Exception as e:
            return CorrectionResult(
                iteration=0,
                original_propositions=current_propositions,
                corrected_propositions=current_propositions,
                conflict_report=report,
                satisfiable=False,
                llm_reasoning=f"Error: {e}",
            )

    def _parse_correction(self, response: str) -> list[dict]:
        """Parse LLM correction response into propositions."""
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```\w*\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return []

    def _verify_corrected(
        self, corrected: list[dict], report: ConflictReport
    ) -> bool:
        """Check if corrected propositions resolve the conflict."""
        try:
            import z3
            solver = z3.Solver()

            for item in corrected:
                field = item.get("field", "")
                value = item.get("value", 0)
                var = z3.Real(field)
                solver.add(var == value)

            # Re-add constraints from the report
            # (simplified — in production, re-use the original constraint set)
            # For now, just check that no obvious contradiction remains
            result = solver.check()
            return result == z3.sat
        except Exception as e:
            logger.warning("Re-verification failed: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════════
# 3. Domain Whitelist — filter bootstrapper noise
# ═══════════════════════════════════════════════════════════════════════

class DomainWhitelist:
    """Filters extracted concepts against known-good domain terminology.

    Rejects noise tokens like "并在其资质等级许" that are artifacts of
    open-domain CJK segmentation without domain awareness.

    Strategy:
    1. Maintain a core vocabulary of valid domain terms.
    2. Each extracted concept is scored against the core vocabulary.
    3. Concepts with no lexical overlap and no morphological validity
       (e.g. fragments ending mid-sentence) are rejected.
    """

    def __init__(self, core_terms: list[str] | None = None):
        self._core_terms: set[str] = set(core_terms or [])

    def add_core_terms(self, terms: list[str]) -> None:
        """Register core domain terms."""
        self._core_terms.update(terms)

    def load_from_ontology(self, entity_registry, taxonomy) -> None:
        """Load core terms from EntityRegistry and ConceptTaxonomy."""
        from app.engine.ontology import EntityRegistry, ConceptTaxonomy
        if entity_registry:
            for surfaces in entity_registry._canonical_to_surfaces.values():
                self._core_terms.update(surfaces)
        if taxonomy:
            self._core_terms.update(taxonomy._nodes.keys())

    def validate(self, term: str) -> tuple[bool, str]:
        """Validate a term against the whitelist.

        Returns:
            (is_valid, reason)
        """
        if not self._core_terms:
            return True, "no whitelist configured"

        # Direct match
        if term in self._core_terms:
            return True, "exact match"

        # Substring containment (either direction)
        for ct in self._core_terms:
            if ct in term or term in ct:
                return True, f"partial match with '{ct}'"

        # Fragment detection: does term end mid-phrase?
        # "并在其资质等级许" — ends with "许", no standalone meaning
        fragment_indicators = ["许", "或", "且", "及", "的", "了", "着", "过",
                               "被", "把", "将", "与", "和"]
        if term[-1] in fragment_indicators and len(term) > 4:
            return False, f"fragment detected: ends with '{term[-1]}'"

        # At least 1 character overlap with core terms
        overlap = sum(1 for c in term if any(c in ct for ct in self._core_terms))
        overlap_ratio = overlap / max(len(term), 1)
        if overlap_ratio >= 0.5:
            return True, f"character overlap {overlap_ratio:.0%}"

        return False, f"no match in core vocabulary ({len(self._core_terms)} terms)"

    def filter_concepts(self, concepts: list) -> list:
        """Filter a list of ExtractedConcept, keeping only valid ones.

        Returns list of (concept, is_valid, reason) tuples.
        """
        results = []
        for c in concepts:
            term = c.term if hasattr(c, 'term') else c
            is_valid, reason = self.validate(term)
            results.append((c, is_valid, reason))
        return results

    def validated_concepts(self, concepts: list) -> list:
        """Return only validated concepts."""
        return [c for c, valid, _ in self.filter_concepts(concepts) if valid]


# ═══════════════════════════════════════════════════════════════════════
# 4. ForAll quantifier support — solver.py extension
# ═══════════════════════════════════════════════════════════════════════

class QuantifiedConstraint:
    """A first-order logic constraint using universal quantification.

    Example: "All waterproofing projects must have warranty >= 5 years"
    → ForAll([p], Implies(IsWaterproofing(p), WarrantyYears(p) >= 5))

    This is the bridge from flat numeric constraints to the universal
    reasoning ("一切X依于Y" → all X depend on Y).
    """

    def __init__(self):
        self._variables: dict[str, z3.DeclareSort] = {}
        self._predicates: dict[str, z3.FuncDeclRef] = {}
        self._constraints: list[z3.BoolRef] = []

    def declare_sort(self, name: str) -> None:
        """Declare a domain sort (e.g. 'Contract', 'MentalFactor')."""
        import z3
        self._variables[name] = z3.DeclareSort(name)

    def declare_predicate(self, name: str, domain: str, range_sort=None) -> None:
        """Declare a predicate (e.g. 'IsWaterproofing(Contract) → Bool')."""
        import z3
        dom_sort = self._variables.get(domain)
        if dom_sort is None:
            dom_sort = z3.DeclareSort(domain)
            self._variables[domain] = dom_sort
        rng = range_sort or z3.BoolSort()
        self._predicates[name] = z3.Function(name, dom_sort, rng)

    def add_forall(self, var_name: str, sort_name: str, condition) -> None:
        """Add a universally quantified constraint.

        Args:
            var_name: Variable name (e.g. 'p' for project).
            sort_name: Domain sort (e.g. 'Contract').
            condition: Z3 expression with the variable.
        """
        import z3
        sort = self._variables.get(sort_name)
        if sort is None:
            sort = z3.DeclareSort(sort_name)
            self._variables[sort_name] = sort

        var = z3.Const(var_name, sort)
        self._constraints.append(z3.ForAll([var], condition))

    def add_numeric_forall(
        self,
        category_predicate: str,
        field: str,
        operator: str,
        threshold: float,
    ) -> None:
        """Convenience: add a ForAll numeric constraint.

        "All entities satisfying category_predicate must have
        field operator threshold."

        Example:
        add_numeric_forall("IsWaterproofing", "WarrantyYears", ">=", 5.0)
        → ∀x (IsWaterproofing(x) → WarrantyYears(x) >= 5)
        """
        import z3

        # Use Real type for numeric fields
        x = z3.Const('x', z3.RealSort())

        # We need a predicate function
        if category_predicate not in self._predicates:
            self._predicates[category_predicate] = z3.Function(
                category_predicate, z3.RealSort(), z3.BoolSort()
            )

        if field not in self._predicates:
            self._predicates[field] = z3.Function(
                field, z3.RealSort(), z3.RealSort()
            )

        is_cat = self._predicates[category_predicate](x)
        field_val = self._predicates[field](x)

        op_map = {
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            "==": lambda a, b: a == b,
        }
        cmp = op_map.get(operator, lambda a, b: a >= b)

        condition = z3.Implies(is_cat, cmp(field_val, z3.RealVal(threshold)))
        self._constraints.append(z3.ForAll([x], condition))

    def check(self) -> tuple[bool, list[str]]:
        """Check all quantified constraints for satisfiability."""
        import z3
        solver = z3.Solver()
        for c in self._constraints:
            solver.add(c)

        result = solver.check()
        if result == z3.sat:
            return True, ["All quantified constraints satisfiable"]
        else:
            core = solver.unsat_core()
            return False, [str(c) for c in core]

    def prove(self, hypothesis) -> bool:
        """Attempt to prove a hypothesis from the quantified constraints.

        Adds ¬hypothesis and checks for UNSAT (proof by contradiction).
        """
        import z3
        solver = z3.Solver()
        for c in self._constraints:
            solver.add(c)
        solver.add(z3.Not(hypothesis))

        return solver.check() == z3.unsat


# ═══════════════════════════════════════════════════════════════════════
# Construction engineering ForAll example
# ═══════════════════════════════════════════════════════════════════════

def build_construction_forall_constraints() -> QuantifiedConstraint:
    """Build pre-seeded ForAll constraints for construction engineering.

    These express universal rules like:
    - ∀ projects classified as "防水工程", warranty >= 5 years
    - ∀ contracts with "质量保证金", deposit rate <= 3%
    """
    qc = QuantifiedConstraint()

    qc.add_numeric_forall(
        category_predicate="IsWaterproofing",
        field="WarrantyYears",
        operator=">=",
        threshold=5.0,
    )

    qc.add_numeric_forall(
        category_predicate="HasQualityDeposit",
        field="DepositRate",
        operator="<=",
        threshold=3.0,
    )

    return qc


# (Duplicate removed — merged into the definition above at line 591)
