"""Z3 constraint solver — structural constraint layer for Citta Engine.

Position in the architecture:
  Text → Engine (numeric_comparison, etc.) → Propositions → Z3 Solver → Verdict

Z3 receives CLEAN propositions (numbers, booleans) not raw text.
The engine's numeric_comparison has ~0% translation noise for numbers;
Z3 works with that clean data. No regex, no LLM, no ambiguity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    z3 = None
    _Z3_AVAILABLE = False

logger = logging.getLogger(__name__)


# Data types

class SolverVerdict(str, Enum):
    SATISFIABLE = "SATISFIABLE"
    UNSATISFIABLE = "UNSATISFIABLE"
    UNDEFINED = "UNDEFINED"


class ViolationSeverity(str, Enum):
    FATAL = "fatal"
    MAJOR = "major"
    MINOR = "minor"
    ADVISORY = "advisory"


@dataclass
class NumericProposition:
    field: str
    value: float
    unit: str
    legal_ref: str = ""
    severity: str = "error"
    source_rule_id: str = ""


@dataclass
class LegalConstraint:
    field: str
    operator: str
    threshold: float
    unit: str
    severity: ViolationSeverity
    legal_ref: str = ""
    weight: float = 1.0


@dataclass
class SolverResult:
    verdict: SolverVerdict
    total_constraints: int
    satisfied: int
    violated: int
    violations: list = field(default_factory=list)
    model: dict = field(default_factory=dict)
    proof_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "total_constraints": self.total_constraints,
            "satisfied": self.satisfied,
            "violated": self.violated,
            "violations": [v.to_dict() for v in self.violations],
            "model": self.model,
            "proof_summary": self.proof_summary,
        }


@dataclass
class Violation:
    field: str
    actual_value: float
    expected: str
    gap: float
    severity: ViolationSeverity
    legal_ref: str
    proof: str

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "actual_value": self.actual_value,
            "expected": self.expected,
            "gap": self.gap,
            "severity": self.severity.value,
            "legal_ref": self.legal_ref,
            "proof": self.proof,
        }


# Z3 Solver Adapter

class CittaZ3Solver:
    """Converts Citta propositions and constraints into Z3 and solves.

    When z3 is not installed, check() returns UNDEFINED.

    # Note: Currently used for advanced constraint research.
    # Not invoked in standard validation flow.
    """

    def __init__(self):
        self._propositions = {}
        self._constraints = []
        if _Z3_AVAILABLE:
            self._z3_solver = z3.Solver()
        self._z3_vars = {}

    def load_propositions(self, propositions):
        for p in propositions:
            key = p.field
            self._propositions[key] = p
            if _Z3_AVAILABLE and key not in self._z3_vars:
                self._z3_vars[key] = z3.Real(key)

    def load_constraints(self, constraints):
        self._constraints.extend(constraints)

    def load_from_evidence_chain(self, evidence_chain):
        propositions = []
        for ev in evidence_chain:
            rid = ev.get("rule_id", "")
            matched = ev.get("matched_terms", [])
            if not matched:
                continue
            raw = matched[0]
            try:
                import re
                num_match = re.match(r'([\d.]+)', raw)
                if num_match:
                    value = float(num_match.group(1))
                    unit = raw[len(num_match.group(0)):]
                    field = ev.get("rule_name", rid)
                    propositions.append(NumericProposition(
                        field=field, value=value, unit=unit or "",
                        severity=ev.get("severity", "error"),
                        source_rule_id=rid,
                    ))
            except (ValueError, TypeError):
                pass
        self.load_propositions(propositions)
        return propositions

    def check(self) -> SolverResult:
        if not _Z3_AVAILABLE:
            return SolverResult(
                verdict=SolverVerdict.UNDEFINED,
                total_constraints=0, satisfied=0, violated=0,
                proof_summary="Z3 solver not installed.",
            )
        if not self._constraints:
            return SolverResult(
                verdict=SolverVerdict.UNDEFINED,
                total_constraints=0, satisfied=0, violated=0,
            )

        solver = z3.Solver()
        constraint_map = {}

        for field, prop in self._propositions.items():
            var = self._z3_vars.get(field)
            if var is None:
                var = z3.Real(field)
                self._z3_vars[field] = var
            solver.add(var == z3.RealVal(prop.value))

        for i, c in enumerate(self._constraints):
            var = self._z3_vars.get(c.field)
            if var is None:
                var = z3.Real(c.field)
                self._z3_vars[c.field] = var
            threshold = z3.RealVal(c.threshold)
            op = c.operator
            if op == ">=":
                cond = var >= threshold
            elif op == "<=":
                cond = var <= threshold
            elif op == ">":
                cond = var > threshold
            elif op == "<":
                cond = var < threshold
            elif op == "==":
                cond = var == threshold
            elif op == "!=":
                cond = var != threshold
            else:
                continue
            constraint_map[i] = c
            solver.assert_and_track(cond, f"c_{i}")

        result = solver.check()
        violations = []
        violated = 0

        if result == z3.sat:
            return SolverResult(
                verdict=SolverVerdict.SATISFIABLE,
                total_constraints=len(self._constraints),
                satisfied=len(self._constraints), violated=0,
                proof_summary=f"All {len(self._constraints)} constraints satisfied.",
            )

        core = solver.unsat_core()
        for core_expr in core:
            name = str(core_expr)
            if not name.startswith("c_"):
                continue
            idx = int(name[2:])
            c = constraint_map.get(idx)
            if c is None:
                continue
            prop = self._propositions.get(c.field)
            actual_value = prop.value if prop else None
            if actual_value is not None:
                if c.operator in (">=", ">"):
                    gap = c.threshold - actual_value
                elif c.operator in ("<=", "<"):
                    gap = actual_value - c.threshold
                else:
                    gap = abs(actual_value - c.threshold)
            else:
                gap = 0.0
            expected_str = f"{c.operator} {c.threshold} {c.unit}"
            proof = f"Legal: {c.field} {c.operator} {c.threshold} {c.unit} | Source: {c.legal_ref}"
            violations.append(Violation(
                field=c.field, actual_value=actual_value or 0,
                expected=expected_str, gap=gap,
                severity=c.severity, legal_ref=c.legal_ref, proof=proof,
            ))
            violated += 1

        satisfied = len(self._constraints) - violated
        sev_order = {ViolationSeverity.FATAL: 0, ViolationSeverity.MAJOR: 1,
                     ViolationSeverity.MINOR: 2, ViolationSeverity.ADVISORY: 3}
        violations.sort(key=lambda v: sev_order.get(v.severity, 9))

        lines = [f"Contract violates {violated}/{len(self._constraints)} constraints."]
        for v in violations:
            lines.append(f"  [{v.severity.value.upper()}] {v.field}: {v.actual_value} vs {v.expected}")
        return SolverResult(
            verdict=SolverVerdict.UNSATISFIABLE,
            total_constraints=len(self._constraints),
            satisfied=satisfied, violated=violated,
            violations=violations,
            proof_summary="\n".join(lines),
        )

    def what_if(self, field, proposed_value):
        if not _Z3_AVAILABLE:
            return SolverResult(verdict=SolverVerdict.UNDEFINED, total_constraints=0, satisfied=0, violated=0)
        var = self._z3_vars.get(field)
        if var is None:
            var = z3.Real(field)
            self._z3_vars[field] = var
        solver = z3.Solver()
        for c in self._constraints:
            threshold = z3.RealVal(c.threshold)
            if c.field != field:
                target_var = self._z3_vars.get(c.field, z3.Real(c.field))
                op_map = {">=": lambda a,b: a>=b, "<=": lambda a,b: a<=b,
                          ">": lambda a,b: a>b, "<": lambda a,b: a<b,
                          "==": lambda a,b: a==b, "!=": lambda a,b: a!=b}
                cond = op_map.get(c.operator, lambda a,b: a>=b)(target_var, threshold)
                solver.add(cond)
        solver.add(var == z3.RealVal(proposed_value))
        result = solver.check()
        return SolverResult(
            verdict=SolverVerdict.SATISFIABLE if result == z3.sat else SolverVerdict.UNSATISFIABLE,
            total_constraints=0, satisfied=0, violated=0,
            proof_summary=f"What-if {field}={proposed_value}: {'COMPLIANT' if result == z3.sat else 'STILL VIOLATES'}",
        )

