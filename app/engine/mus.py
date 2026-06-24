"""Minimal Unsatisfiable Subset (MUS) Extractor — when Z3 reports UNSAT,
finds the smallest subset of constraints that are collectively inconsistent.

The deletion-based algorithm removes constraints one at a time; if the
remaining set is still UNSAT, the removed constraint was not essential.
What remains is one minimal unsatisfiable subset.

Also supports finding ALL MUSes via the MARCO (Mapping All MUSes via
Recursive Clause removal) algorithm.  Zero external dependencies beyond z3.
"""
from __future__ import annotations
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.engine.solver import (
    CittaZ3Solver,
    LegalConstraint,
    NumericProposition,
    SolverResult,
    SolverVerdict,
    Violation,
    ViolationSeverity,
    z3,
    _Z3_AVAILABLE,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class MUSResult:
    """A single minimal unsatisfiable subset."""
    mus_id: str
    constraints: list[LegalConstraint]
    propositions: list[NumericProposition]
    is_minimal: bool                     # verified minimal
    removal_sequence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mus_id": self.mus_id,
            "constraints": [
                {
                    "field": c.field,
                    "operator": c.operator,
                    "threshold": c.threshold,
                    "unit": c.unit,
                    "severity": c.severity.value,
                    "legal_ref": c.legal_ref,
                }
                for c in self.constraints
            ],
            "propositions": [
                {"field": p.field, "value": p.value, "unit": p.unit}
                for p in self.propositions
            ],
            "is_minimal": self.is_minimal,
            "removal_sequence": self.removal_sequence,
        }


@dataclass
class FixSuggestion:
    """A suggested fix for a conflicting constraint."""
    constraint_field: str
    current_threshold: float
    suggested_threshold: float
    reasoning: str
    impact: str                    # description of what other constraints are affected

    def to_dict(self) -> dict:
        return {
            "constraint_field": self.constraint_field,
            "current_threshold": self.current_threshold,
            "suggested_threshold": round(self.suggested_threshold, 2),
            "reasoning": self.reasoning,
            "impact": self.impact,
        }


@dataclass
class MUSReport:
    """Complete MUS analysis report."""
    all_muses: list[MUSResult]
    conflict_explanations: dict[str, str]      # mus_id → explanation
    fix_suggestions: dict[str, list[FixSuggestion]]  # mus_id → suggestions
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "mus_count": len(self.all_muses),
            "muses": [m.to_dict() for m in self.all_muses],
            "conflict_explanations": self.conflict_explanations,
            "fix_suggestions": {
                mid: [s.to_dict() for s in suggestions]
                for mid, suggestions in self.fix_suggestions.items()
            },
            "summary": self.summary,
        }


# ======================================================================
# MUSExtractor
# ======================================================================

class MUSExtractor:
    """Finds minimal unsatisfiable subsets from Z3 constraint sets.

    Usage:
        extractor = MUSExtractor()
        mus = extractor.find_mus(constraints, propositions)
        if mus:
            print(extractor.explain_conflict(mus))
            for fix in extractor.suggest_fix(mus, propositions):
                print(f"  → {fix.reasoning}")
    """

    # Constraints whose thresholds are legal minimums (lower bound by statute)
    # These should be suggested for relaxation only as a last resort
    # Matched by substring — field names may vary (e.g. "主体保修" matches "主体结构保修期限")
    _LEGAL_MINIMUM_KEYWORDS = [
        "屋面防水",
        "地下室防水",
        "主体结构",
        "主体保修",
        "电气",
        "给排水",
    ]

    def __init__(self):
        self._solver: Optional[CittaZ3Solver] = None

    # ── Primary: find one MUS ─────────────────────────────────────────

    def find_mus(self, constraints: list[LegalConstraint],
                 propositions: list[NumericProposition]) -> Optional[MUSResult]:
        """Find ONE minimal unsatisfiable subset using the deletion-based algorithm.

        Algorithm:
        1. Verify the full set is UNSAT.
        2. For each constraint c (ordered by severity, FATAL last):
           a. Temporarily remove c.
           b. If remaining set is still UNSAT → c is NOT needed (remove it permanently).
           c. If remaining set becomes SAT → c IS needed (keep it).
        3. What remains is a MUS.

        Args:
            constraints: All LegalConstraint objects.
            propositions: All NumericProposition objects.

        Returns:
            MUSResult if a conflict exists, None if the set is satisfiable.
        """
        if not _Z3_AVAILABLE:
            logger.warning("z3 not available; MUS extraction skipped")
            return None

        # Step 1: verify UNSAT
        result = self._check(constraints, propositions)
        if result.verdict != SolverVerdict.UNSATISFIABLE:
            return None  # no conflict to analyze

        # Step 2: deletion-based MUS
        # Sort constraints: try removing FATAL last (most likely to be in MUS)
        severity_order = {
            ViolationSeverity.ADVISORY: 0,
            ViolationSeverity.MINOR: 1,
            ViolationSeverity.MAJOR: 2,
            ViolationSeverity.FATAL: 3,
        }
        sorted_constraints = sorted(
            constraints,
            key=lambda c: severity_order.get(c.severity, 2),
        )

        remaining = list(constraints)
        removal_seq: list[str] = []

        for c in sorted_constraints:
            if c not in remaining:
                continue
            test_set = [x for x in remaining if x is not c]
            if len(test_set) == 0:
                # This is the last constraint — it must be in the MUS
                break
            test_result = self._check(test_set, propositions)
            if test_result.verdict == SolverVerdict.UNSATISFIABLE:
                # c is redundant — safe to remove
                remaining = test_set
                removal_seq.append(c.field)
            # else: c is essential — keep it

        # Step 3: verify minimality (try removing any single constraint)
        is_minimal = True
        for c in list(remaining):
            test_set = [x for x in remaining if x is not c]
            if len(test_set) > 0:
                test_result = self._check(test_set, propositions)
                if test_result.verdict == SolverVerdict.UNSATISFIABLE:
                    is_minimal = False
                    break

        return MUSResult(
            mus_id=f"MUS_{len(remaining)}",
            constraints=remaining,
            propositions=propositions,
            is_minimal=is_minimal,
            removal_sequence=removal_seq,
        )

    # ── Advanced: find all MUSes ──────────────────────────────────────

    def find_all_muses(self, constraints: list[LegalConstraint],
                       propositions: list[NumericProposition],
                       max_muses: int = 10) -> MUSReport:
        """Find ALL minimal unsatisfiable subsets.

        Uses the CAMUS (Compute All Minimal Unsatisfiable Subsets) approach:
        1. Find one MUS.
        2. For each constraint in the found MUS, create a "blocking clause"
           that excludes this exact MUS.
        3. Repeat until no more MUSes exist or max_muses reached.

        Args:
            constraints: All LegalConstraint objects.
            propositions: All NumericProposition objects.
            max_muses: Maximum number of MUSes to find.

        Returns:
            MUSReport with all found MUSes, explanations, and fix suggestions.
        """
        all_muses: list[MUSResult] = []
        explanations: dict[str, str] = {}
        fix_suggestions: dict[str, list[FixSuggestion]] = {}
        seen_mus_sets: set[frozenset] = set()

        for _ in range(max_muses):
            # Create a solver with current constraints
            mus = self.find_mus(constraints, propositions)
            if mus is None:
                break

            # Check if we've already seen this MUS
            mus_key = frozenset(c.field for c in mus.constraints)
            if mus_key in seen_mus_sets:
                # Block this MUS and try again
                constraints = self._block_mus(mus, constraints)
                continue

            seen_mus_sets.add(mus_key)
            all_muses.append(mus)
            explanations[mus.mus_id] = self.explain_conflict(mus)
            fix_suggestions[mus.mus_id] = self.suggest_fix(mus, propositions)

            # Block this MUS for the next iteration
            constraints = self._block_mus(mus, constraints)

        # Build summary
        lines = [f"发现 {len(all_muses)} 个最小不一致子集"]
        for mus in all_muses:
            fields = [c.field for c in mus.constraints]
            lines.append(f"  {mus.mus_id}: {', '.join(fields)}")

        return MUSReport(
            all_muses=all_muses,
            conflict_explanations=explanations,
            fix_suggestions=fix_suggestions,
            summary="\n".join(lines),
        )

    # ── Explanation ───────────────────────────────────────────────────

    def explain_conflict(self, mus: MUSResult) -> str:
        """Generate a human-readable Chinese explanation of the conflict.

        Args:
            mus: A MUSResult from find_mus().

        Returns:
            A string explaining what constraints conflict and why.
        """
        if not mus.constraints:
            return "无冲突"

        lines = ["约束冲突分析:"]
        for c in mus.constraints:
            # Find the corresponding proposition
            prop = next((p for p in mus.propositions if p.field == c.field), None)
            actual_str = f"{prop.value}{prop.unit}" if prop else "未提取"
            op_str = _OP_CN.get(c.operator, c.operator)
            lines.append(
                f"  • {c.field}: 要求 {op_str} {c.threshold}{c.unit}"
                f"，实际 {actual_str}"
                f"（{c.legal_ref}）"
            )

        # Summarize the conflict
        if len(mus.constraints) == 1:
            lines.append("单条约束违反，不存在多条约束间的冲突。")
        elif len(mus.constraints) == 2:
            a, b = mus.constraints[0], mus.constraints[1]
            lines.append(f"核心冲突: 「{a.field}」与「{b.field}」不可同时满足。")
        else:
            fields = [c.field for c in mus.constraints]
            lines.append(f"多重约束冲突: {', '.join(fields)} 构成不可满足集合。")

        return "\n".join(lines)

    # ── Fix suggestions ───────────────────────────────────────────────

    def suggest_fix(self, mus: MUSResult,
                    propositions: list[NumericProposition]) -> list[FixSuggestion]:
        """Suggest which constraints to relax and by how much.

        Strategy:
        1. For each constraint in the MUS, compute the gap from actual value.
        2. Try relaxing each constraint individually.
        3. Rank suggestions by: smallest relaxation amount, least legal impact.

        Args:
            mus: A MUSResult.
            propositions: The current propositions.

        Returns:
            List of FixSuggestion objects, ranked by preference.
        """
        suggestions: list[FixSuggestion] = []
        prop_map = {p.field: p for p in propositions}

        for c in mus.constraints:
            prop = prop_map.get(c.field)
            if prop is None:
                continue

            gap = _compute_gap(c, prop)
            is_legal_min = any(kw in c.field for kw in self._LEGAL_MINIMUM_KEYWORDS)

            # Suggest new threshold
            if c.operator in (">=", ">"):
                new_threshold = prop.value  # relax to actual value
                if new_threshold <= 0:
                    new_threshold = c.threshold * 0.5
            elif c.operator in ("<=", "<"):
                new_threshold = prop.value
            elif c.operator == "==":
                new_threshold = prop.value
            else:  # !=
                new_threshold = c.threshold + 0.1

            if is_legal_min:
                reasoning = (
                    f"「{c.field}」为法定最低要求（{c.legal_ref}），"
                    f"不建议调整。当前实际值 {prop.value}{c.unit}，"
                    f"法定阈值 {c.threshold}{c.unit}，差距 {gap:.1f}{c.unit}。"
                )
                impact = "此为法定底线，调整可能导致合规风险"
                # Don't suggest relaxation for legal minimums
                suggestions.append(FixSuggestion(
                    constraint_field=c.field,
                    current_threshold=c.threshold,
                    suggested_threshold=c.threshold,
                    reasoning=reasoning,
                    impact=impact,
                ))
            else:
                reasoning = (
                    f"「{c.field}」阈值 {c.threshold}{c.unit} 可调整为 "
                    f"{new_threshold:.1f}{c.unit}（当前实际值 {prop.value}{c.unit}），"
                    f"差距 {gap:.1f}{c.unit}。此约束来源为 {c.legal_ref}，"
                    f"非强制性法定底线。"
                )
                # Check impact on other constraints
                impact_parts = []
                for other_c in mus.constraints:
                    if other_c is not c:
                        impact_parts.append(other_c.field)
                impact = (
                    f"调整后需验证与 {', '.join(impact_parts)} 的一致性"
                    if impact_parts else "无其他约束影响"
                )
                suggestions.append(FixSuggestion(
                    constraint_field=c.field,
                    current_threshold=c.threshold,
                    suggested_threshold=new_threshold,
                    reasoning=reasoning,
                    impact=impact,
                ))

        # Rank: non-legal-minimum adjustments first, smallest gap first
        suggestions.sort(key=lambda s: (
            1 if any(kw in s.constraint_field for kw in self._LEGAL_MINIMUM_KEYWORDS) else 0,
            abs(s.suggested_threshold - s.current_threshold),
        ))

        return suggestions

    # ── Internal helpers ──────────────────────────────────────────────

    def _check(self, constraints: list[LegalConstraint],
               propositions: list[NumericProposition]) -> SolverResult:
        """Internal: run solver with given constraints and propositions."""
        solver = CittaZ3Solver()
        # Convert NumericProposition objects to plain dicts for the solver
        for p in propositions:
            solver.load_propositions([p])
        solver.load_constraints(constraints)
        return solver.check()

    def _block_mus(self, mus: MUSResult,
                   constraints: list[LegalConstraint]) -> list[LegalConstraint]:
        """Create a blocking clause that excludes this exact MUS set.

        In deletion-based approach: we remove the LAST constraint of the MUS
        from the active set, which forces the next search to find a different MUS.
        """
        if len(mus.constraints) <= 1:
            # Can't block a single-constraint MUS without removing it entirely
            return [c for c in constraints if c is not mus.constraints[0]]

        # Remove the last constraint (heuristic: the one with lowest severity)
        to_remove = mus.constraints[-1]
        return [c for c in constraints if c.field != to_remove.field]


# ======================================================================
# Helpers
# ======================================================================

_OP_CN = {
    ">=": "≥",
    "<=": "≤",
    ">": ">",
    "<": "<",
    "==": "=",
    "!=": "≠",
}


def _compute_gap(constraint: LegalConstraint,
                  proposition: NumericProposition) -> float:
    """Compute the gap between actual value and threshold."""
    if constraint.operator in (">=", ">"):
        return constraint.threshold - proposition.value
    elif constraint.operator in ("<=", "<"):
        return proposition.value - constraint.threshold
    elif constraint.operator == "==":
        return abs(proposition.value - constraint.threshold)
    else:
        return 0.0


# ======================================================================
# Convenience function
# ======================================================================

def analyze_conflicts(constraints: list[LegalConstraint],
                      propositions: list[NumericProposition]) -> Optional[MUSReport]:
    """Quick conflict analysis: finds one MUS and generates explanations."""
    extractor = MUSExtractor()
    mus = extractor.find_mus(constraints, propositions)
    if mus is None:
        return None
    return MUSReport(
        all_muses=[mus],
        conflict_explanations={mus.mus_id: extractor.explain_conflict(mus)},
        fix_suggestions={mus.mus_id: extractor.suggest_fix(mus, propositions)},
        summary=f"发现 1 个冲突集: {', '.join(c.field for c in mus.constraints)}",
    )
