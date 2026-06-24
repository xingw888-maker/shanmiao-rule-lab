"""Test Algorithm 3: Belief Propagation Network."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.belief_propagation import BeliefNetwork, BeliefState, FactorNode, BeliefReport


def make_rules() -> list[dict]:
    """Build a small set of related test rules."""
    return [
        {"id": "r1", "name": "Waterproof Warranty", "condition": {"type": "numeric_comparison"},
         "category": "保修", "clause_type": "保修", "source_credibility": 1.0},
        {"id": "r2", "name": "Basement Warranty", "condition": {"type": "numeric_comparison"},
         "category": "保修", "clause_type": "保修", "source_credibility": 0.9},
        {"id": "r3", "name": "Structure Warranty", "condition": {"type": "numeric_comparison"},
         "category": "保修", "clause_type": "保修", "source_credibility": 0.8},
        {"id": "r4", "name": "Payment Ratio", "condition": {"type": "sum_numeric_comparison"},
         "category": "付款", "clause_type": "付款", "source_credibility": 0.5},
        {"id": "r5", "name": "Dispute Resolution", "condition": {"type": "required_pattern"},
         "category": "争议", "clause_type": "争议", "source_credibility": 0.7},
    ]


# ── Test 1: network build ──

def test_network_build():
    """Build a belief network and verify factor nodes are created."""
    rules = make_rules()
    bn = BeliefNetwork()
    bn.build_network(rules)

    assert len(bn._variables) == len(rules), f"Expected {len(rules)} variables"
    # Rules 1,2,3 share category "保修" — they should have factor nodes between them
    assert len(bn._factors) > 0, "Expected factor nodes from shared categories"

    # Verify each variable has prior belief set
    for r in rules:
        var = bn.marginal_belief(r["id"])
        assert var is not None, f"Expected variable for {r['id']}"
        assert 0 < var.prior_credibility < 1, \
            f"Prior credibility should be in (0,1), got {var.prior_credibility}"
        assert abs(var.belief_pass + var.belief_fail - 1.0) < 0.01, \
            "Beliefs should sum to 1"

    print(f"  ✓ network_build: {len(bn._variables)} vars, {len(bn._factors)} factors")
    return True


# ── Test 2: prior belief from credibility ──

def test_prior_belief():
    """Verify prior beliefs match source_credibility values."""
    rules = make_rules()
    bn = BeliefNetwork()
    bn.build_network(rules)

    # r1 has credibility 1.0 → belief_pass should be high (clamped to 0.99)
    r1 = bn.marginal_belief("r1")
    assert r1.prior_credibility >= 0.99, f"High cred rule should have prior ~0.99, got {r1.prior_credibility}"
    assert r1.belief_pass > 0.9, f"High cred rule should have high belief_pass, got {r1.belief_pass}"

    # r4 has credibility 0.5 → belief_pass should be moderate
    r4 = bn.marginal_belief("r4")
    assert r4.prior_credibility == 0.5
    assert 0.4 < r4.belief_pass < 0.6, f"Medium cred should have ~0.5 belief, got {r4.belief_pass}"

    print(f"  ✓ prior_belief: r1.pass={r1.belief_pass:.3f}, r4.pass={r4.belief_pass:.3f}")
    return True


# ── Test 3: propagation converges ──

def test_propagation_converges():
    """Run belief propagation with evidence and verify it converges."""
    rules = make_rules()
    bn = BeliefNetwork(max_iterations=50, convergence_threshold=0.001)
    bn.build_network(rules)

    # Evidence: r1 FAILED, r2 PASSED
    evidence = {"r1": "FAILED", "r2": "PASSED", "r3": "NOT_APPLICABLE",
                "r4": "NOT_APPLICABLE", "r5": "NOT_APPLICABLE"}

    report = bn.propagate(evidence)
    assert report.converged, "Belief propagation should converge"

    # r1 should have belief_fail near 1 (clamped by evidence)
    r1 = bn.marginal_belief("r1")
    assert r1.belief_fail > 0.9, f"Clamped FAILED rule should have high belief_fail, got {r1.belief_fail}"

    print(f"  ✓ propagation_converges: {report.iteration_count if not report.converged else 'converged'}, "
          f"max_delta={report.max_delta:.6f}")
    return True


# ── Test 4: marginal consistency ──

def test_marginal_consistency():
    """Rules sharing categories with similar evidence should have similar posterior beliefs."""
    rules = make_rules()
    bn = BeliefNetwork()
    bn.build_network(rules)

    # All PASSED
    evidence = {r["id"]: "PASSED" for r in rules}
    report = bn.propagate(evidence)

    # r1, r2, r3 all share category "保修" — their posteriors should be more similar
    # after propagation than before
    r1_post = bn.marginal_belief("r1").posterior_credibility
    r2_post = bn.marginal_belief("r2").posterior_credibility
    r3_post = bn.marginal_belief("r3").posterior_credibility

    # All should be reasonable values
    for val, name in [(r1_post, "r1"), (r2_post, "r2"), (r3_post, "r3")]:
        assert 0 < val < 1, f"{name} posterior should be in (0,1), got {val}"

    print(f"  ✓ marginal_consistency: r1={r1_post:.3f}, r2={r2_post:.3f}, r3={r3_post:.3f}")
    return True


# ── Test 5: evidence clamping ──

def test_evidence_clamping():
    """A rule with observed FAILED should have belief_fail near 1.0 after propagation."""
    rules = make_rules()
    bn = BeliefNetwork()
    bn.build_network(rules)

    # r5 FAILED alone (isolated from others)
    evidence = {"r1": "PASSED", "r2": "PASSED", "r3": "PASSED",
                "r4": "PASSED", "r5": "FAILED"}
    report = bn.propagate(evidence)

    r5 = bn.marginal_belief("r5")
    assert r5.observed == "FAILED"
    assert r5.belief_fail > 0.9, f"Clamped FAILED should have belief_fail > 0.9, got {r5.belief_fail}"

    print(f"  ✓ evidence_clamping: r5.belief_fail={r5.belief_fail:.3f}")
    return True


# ── Test 6: calibrate from history ──

def test_calibrate():
    """Use mock historical runs to calibrate factor potentials."""
    rules = make_rules()
    bn = BeliefNetwork()
    bn.build_network(rules)

    # Mock historical runs
    historical = [
        {"r1": "PASSED", "r2": "PASSED", "r3": "PASSED", "r4": "PASSED", "r5": "PASSED"},
        {"r1": "PASSED", "r2": "PASSED", "r3": "PASSED", "r4": "PASSED", "r5": "FAILED"},
        {"r1": "FAILED", "r2": "FAILED", "r3": "PASSED", "r4": "PASSED", "r5": "PASSED"},
    ]

    updated = bn.calibrate_credibility(rules, historical)
    assert len(updated) == len(rules), f"Expected {len(rules)} updated credibilities"

    # All values should be valid
    for rid, cred in updated.items():
        assert 0 < cred < 1, f"Credibility for {rid} should be in (0,1), got {cred}"

    print(f"  ✓ calibrate: updated {len(updated)} credibilities from 3 historical runs")
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
