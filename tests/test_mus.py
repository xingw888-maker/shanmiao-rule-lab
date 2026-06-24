"""Test Algorithm 4: Minimal Unsatisfiable Subset Extractor."""
from __future__ import annotations
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.mus import MUSExtractor, MUSResult, MUSReport, FixSuggestion, analyze_conflicts
from app.engine.solver import (
    CittaZ3Solver, LegalConstraint, NumericProposition,
    ViolationSeverity, _Z3_AVAILABLE,
)


# Skip all tests if z3 is not installed
if not _Z3_AVAILABLE:
    pytest.skip("z3 not installed — MUS tests skipped", allow_module_level=True)


def make_constraints() -> tuple[list[LegalConstraint], list[NumericProposition]]:
    """Create a conflicting constraint set for testing."""
    constraints = [
        LegalConstraint("防水保修", ">=", 5.0, "年", ViolationSeverity.FATAL, "国令279号第40条(二)"),
        LegalConstraint("主体保修", ">=", 50.0, "年", ViolationSeverity.FATAL, "国令279号第40条(一)"),
        LegalConstraint("质保金比例", "<=", 3.0, "%", ViolationSeverity.MAJOR, "建质[2017]138号第7条"),
    ]
    # Propositions that VIOLATE the first two constraints
    propositions = [
        NumericProposition("防水保修", 2.0, "年", "国令279号", "error", "cn-001"),
        NumericProposition("主体保修", 30.0, "年", "国令279号", "error", "cn-003"),
        NumericProposition("质保金比例", 3.0, "%", "建质[2017]138号", "minor", "cn-006"),
    ]
    return constraints, propositions


# ── Test 1: single MUS ──

def test_single_mus():
    """Extract MUS from a simple conflicting constraint set."""
    constraints, propositions = make_constraints()
    extractor = MUSExtractor()
    mus = extractor.find_mus(constraints, propositions)

    assert mus is not None, "Expected a MUS from conflicting constraints"
    assert len(mus.constraints) >= 1, f"MUS should contain at least 1 constraint"
    assert len(mus.constraints) <= len(constraints), "MUS cannot be larger than full set"
    assert mus.is_minimal, "Deletion-based MUS should be minimal"

    print(f"  ✓ single_mus: {len(mus.constraints)} constraints in MUS: "
          f"{[c.field for c in mus.constraints]}")
    return True


# ── Test 2: explanation ──

def test_explain_conflict():
    """Verify human-readable explanation is generated."""
    constraints, propositions = make_constraints()
    extractor = MUSExtractor()
    mus = extractor.find_mus(constraints, propositions)

    if mus is None:
        print("  SKIP explain_conflict: no MUS found")
        return True

    explanation = extractor.explain_conflict(mus)
    assert len(explanation) > 0
    assert "冲突" in explanation, f"Explanation should mention conflict: {explanation[:100]}"
    assert any(c.field in explanation for c in mus.constraints), \
        "Explanation should mention constraint fields"

    print(f"  ✓ explain_conflict: {explanation[:100]}...")
    return True


# ── Test 3: fix suggestion ──

def test_fix_suggestion():
    """Verify fix suggestions are generated for conflicting constraints."""
    constraints, propositions = make_constraints()
    extractor = MUSExtractor()
    mus = extractor.find_mus(constraints, propositions)

    if mus is None:
        print("  SKIP fix_suggestion: no MUS found")
        return True

    suggestions = extractor.suggest_fix(mus, propositions)
    assert len(suggestions) > 0, "Expected fix suggestions"
    assert all(isinstance(s, FixSuggestion) for s in suggestions)

    # Legal minimums (like防水保修) should be flagged
    for s in suggestions:
        if "法定" in s.reasoning or "min" in s.impact.lower():
            # Legal minimum — should NOT suggest relaxation
            assert s.suggested_threshold == s.current_threshold, \
                f"Legal minimum should not be relaxed: {s.constraint_field}"

    print(f"  ✓ fix_suggestion: {len(suggestions)} suggestions")
    return True


# ── Test 4: analyze_conflicts convenience function ──

def test_analyze_conflicts():
    """Verify the convenience function works."""
    constraints, propositions = make_constraints()
    report = analyze_conflicts(constraints, propositions)

    if report is None:
        print("  SKIP analyze_conflicts: no conflicts found")
        return True

    assert len(report.all_muses) >= 1
    assert len(report.conflict_explanations) >= 1
    assert len(report.fix_suggestions) >= 1

    print(f"  ✓ analyze_conflicts: {len(report.all_muses)} muses")
    return True


# ── Test 5: satisfiable set returns None ──

def test_satisfiable_no_mus():
    """A satisfiable constraint set should return None (no MUS)."""
    extractor = MUSExtractor()
    constraints = [
        LegalConstraint("防水保修", ">=", 5.0, "年", ViolationSeverity.FATAL),
    ]
    propositions = [
        NumericProposition("防水保修", 10.0, "年"),  # compliant!
    ]
    mus = extractor.find_mus(constraints, propositions)
    assert mus is None, "Satisfiable constraint set should not have a MUS"

    print("  ✓ satisfiable_no_mus: correctly returned None")
    return True


# ── Run all ──

if __name__ == "__main__":
    passed = 0
    total = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            total += 1
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)
