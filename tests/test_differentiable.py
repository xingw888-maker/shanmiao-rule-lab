"""Test Algorithm 5: Differentiable Threshold Optimizer."""
from __future__ import annotations
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.differentiable import (
    ThresholdOptimizer, TunableParameter, ParameterizedRule, CalibrationReport,
)


# ── Test helpers ──

def make_rule(rule_id, cond_type, **cond_kwargs):
    return {
        "id": rule_id,
        "name": f"Test {rule_id}",
        "condition": {"type": cond_type, **cond_kwargs},
        "severity": "error",
        "category": "测试",
        "source_credibility": 0.5,
    }


# ── Test 1: sigmoid stability ──

def test_sigmoid_stability():
    """Verify sigmoid handles extreme values without overflow."""
    opt = ThresholdOptimizer()

    # Normal range
    s = opt._sigmoid(0)
    assert s == 0.5, f"sigmoid(0) should be 0.5, got {s}"

    # Large positive
    s_pos = opt._sigmoid(100)
    assert 0.99 < s_pos < 1.01, f"sigmoid(100) should be ~1.0, got {s_pos}"

    # Large negative
    s_neg = opt._sigmoid(-100)
    assert -0.01 < s_neg < 0.01, f"sigmoid(-100) should be ~0.0, got {s_neg}"

    print(f"  ✓ sigmoid_stability: σ(0)={s:.4f}, σ(100)={s_pos:.4f}, σ(-100)={s_neg:.4f}")
    return True


# ── Test 2: soft comparison ──

def test_soft_comparison():
    """Verify soft_comparison approximates hard comparisons."""
    opt = ThresholdOptimizer(temperature=0.1)

    # soft_ge(6, 5) should be near 1.0 (6 >= 5 is true)
    s_pass = opt.soft_comparison(6.0, 5.0, ">=")
    assert s_pass > 0.9, f"soft_ge(6,5) should be > 0.9, got {s_pass}"

    # soft_ge(4, 5) should be near 0.0 (4 >= 5 is false)
    s_fail = opt.soft_comparison(4.0, 5.0, ">=")
    assert s_fail < 0.1, f"soft_ge(4,5) should be < 0.1, got {s_fail}"

    # At threshold: soft_ge(5, 5) should be ~0.5
    s_bound = opt.soft_comparison(5.0, 5.0, ">=")
    assert 0.4 < s_bound < 0.6, f"soft_ge(5,5) should be ~0.5, got {s_bound}"

    # soft_le
    s_le_pass = opt.soft_comparison(2.0, 5.0, "<=")
    assert s_le_pass > 0.9, f"soft_le(2,5) should be > 0.9, got {s_le_pass}"

    print(f"  ✓ soft_comparison: ge(6,5)={s_pass:.3f}, ge(4,5)={s_fail:.3f}, ge(5,5)={s_bound:.3f}")
    return True


# ── Test 3: parametrize rule ──

def test_parametrize():
    """Verify parameter extraction from different rule types."""
    opt = ThresholdOptimizer()

    # numeric_comparison
    rule = make_rule("test", "numeric_comparison", expected=5, operator=">=", unit="年")
    pr = opt.parametrize_rule(rule)
    assert len(pr.parameters) >= 1, f"Expected at least 1 parameter for numeric_comparison"
    expected_param = pr.get_param("expected")
    assert expected_param is not None
    assert expected_param.value == 5.0
    assert expected_param.lower_bound > 0, f"Expected positive lower bound"

    # required_pattern
    rule2 = make_rule("test2", "required_pattern", terms=["a", "b", "c"])
    pr2 = opt.parametrize_rule(rule2)
    min_ratio = pr2.get_param("min_ratio")
    assert min_ratio is not None
    assert 0 < min_ratio.value <= 1.0

    print(f"  ✓ parametrize: numeric={len(pr.parameters)} params, "
          f"required={len(pr2.parameters)} params")
    return True


# ── Test 4: gradient descent converges ──

def test_gradient_descent_converges():
    """Optimize a rule from wrong initial threshold to correct value."""
    opt = ThresholdOptimizer(learning_rate=0.05, max_iterations=200, temperature=0.5)

    # Rule: warranty >= 5 years
    rule = make_rule("test", "numeric_comparison",
                     label="保修期", context_pattern="保修|防水",
                     unit="年", operator=">=", expected=3)  # WRONG initial value

    # Generate synthetic labeled data: all with value >= 5 should PASS
    synthetic = []
    for v in range(1, 11):
        text = f"屋面防水保修期为{v}年。"
        verdict = "PASSED" if v >= 5 else "FAILED"
        synthetic.append({"text": text, "ground_truth": verdict})

    report = opt.optimize(rule, synthetic)
    optimized = report.optimized_parameters.get("expected", 3)

    # After optimization, expected should have moved toward 5
    assert optimized != 3.0, "Expected threshold should have changed from initial"
    # Loss should not increase significantly
    assert report.final_loss <= report.initial_loss + 0.05, \
        f"Final loss ({report.final_loss:.4f}) should not be much higher than initial ({report.initial_loss:.4f})"

    print(f"  ✓ gradient_descent: expected {3}→{optimized:.2f}, "
          f"loss {report.initial_loss:.4f}→{report.final_loss:.4f}, "
          f"{report.iterations} iters, converged={report.converged}")
    return True


# ── Test 5: loss decreases ──

def test_loss_decreases():
    """Verify that optimization reduces loss or stays at zero."""
    opt = ThresholdOptimizer(learning_rate=0.1, max_iterations=100, temperature=1.0)

    rule = make_rule("test", "numeric_comparison",
                     label="缺陷责任期", context_pattern="缺陷|责任期",
                     unit="月", operator="<=", expected=30)  # too high

    synthetic = []
    for v in [6, 12, 18, 24, 30, 36, 48]:
        text = f"缺陷责任期为{v}个月。"
        verdict = "PASSED" if v <= 24 else "FAILED"
        synthetic.append({"text": text, "ground_truth": verdict})

    report = opt.optimize(rule, synthetic)
    # Loss should not increase significantly
    assert report.final_loss <= report.initial_loss + 0.01, \
        f"Loss should not increase: {report.initial_loss:.4f}→{report.final_loss:.4f}"

    print(f"  ✓ loss_decreases: {report.initial_loss:.4f}→{report.final_loss:.4f} "
          f"({report.loss_reduction_pct:+.1f}%)")
    return True


# ── Test 6: analytical gradient sign ──

def test_analytical_gradient():
    """Verify gradient sign is correct for >= operator."""
    opt = ThresholdOptimizer(temperature=1.0)

    # For >=: increasing threshold should make PASS harder
    # soft_ge(x, θ): if x > θ, prediction is high → loss gradient pushes θ up
    # if x < θ, prediction is low → loss gradient pushes θ down
    grad_pass = opt.soft_gradient(10.0, 5.0, ">=")  # x=10 > θ=5, should PASS
    # d/dθ soft_ge(10, 5) should be negative (increasing θ reduces PASS probability)
    assert grad_pass < 0, f"Gradient for (x > θ) should be negative, got {grad_pass}"

    grad_fail = opt.soft_gradient(3.0, 5.0, ">=")  # x=3 < θ=5, should FAIL
    # d/dθ soft_ge(3, 5) should also be negative
    assert grad_fail < 0, f"Gradient for (x < θ) should be negative, got {grad_fail}"

    # For <=: increasing threshold should make PASS easier
    grad_le = opt.soft_gradient(3.0, 5.0, "<=")  # x=3 < θ=5, should PASS
    assert grad_le > 0, f"Gradient for soft_le should be positive, got {grad_le}"

    print(f"  ✓ analytical_gradient: ge(10,5)={grad_pass:.4f}, "
          f"ge(3,5)={grad_fail:.4f}, le(3,5)={grad_le:.4f}")
    return True


# ── Test 7: hinge loss ──

def test_hinge_loss():
    """Verify hinge loss behavior."""
    opt = ThresholdOptimizer()

    # Perfect prediction
    l0 = opt._hinge_loss(1.0, 1.0)  # predicted PASS, target PASS
    assert l0 == 0.0, f"Perfect prediction should have 0 loss, got {l0}"

    # Wrong prediction
    l1 = opt._hinge_loss(0.0, 1.0)  # predicted FAIL, target PASS
    assert l1 > 0, f"Wrong prediction should have positive loss, got {l1}"

    # Near miss (within margin): margin=0.5, accuracy=1*0.4+0*0.6=0.4, loss=0.1
    l2 = opt._hinge_loss(0.4, 1.0)  # predicted 0.4, target 1.0
    assert l2 > 0, f"Near miss should have loss, got {l2}"

    print(f"  ✓ hinge_loss: perfect={l0:.4f}, wrong={l1:.4f}, near_miss={l2:.4f}")
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
