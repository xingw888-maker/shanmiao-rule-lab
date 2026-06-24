"""Handler unit test skeleton — minimum smoke tests for core condition types.

Covers:
  - numeric_comparison: sidecar path + fallback path
  - required_pattern: match + negation context
  - forbidden_pattern: hit + negation filter

Run: python tests/unit/test_handlers.py  (or pytest if installed)
"""

# Optional pytest import
try:
    import pytest  # noqa: F401
except ImportError:
    pass

from tests.conftest import make_matcher, make_rule


# ═══════════════════════════════════════════════════════════════════
# numeric_comparison
# ═══════════════════════════════════════════════════════════════════

def test_numeric_comparison_sidecar(matcher):
    """Sidecar injection: mock value overrides regex extraction."""
    from app.engine.handlers.numeric_comparison import _eval_numeric_comparison

    matcher._structured_inputs["cn-test"] = {
        "value": 20.0,
        "unit": "年",
        "operator_hint": ">=",
        "source_text": "mock",
        "confidence": 1.0,
    }

    rule = make_rule(
        rule_id="cn-test",
        name="主体结构保修期",
        condition_type="numeric_comparison",
        condition_params={
            "label": "主体结构保修期",
            "expected": 50,
            "operator": ">=",
            "unit": "年",
            "context_pattern": "主体结构|地基基础",
            "legal_ref": "国务院令第279号第40条(一)",
        },
    )

    # Text says "保修50年" — regex would extract 50 → PASSED.
    # Sidecar says 20 → should FAIL (20 < 50).
    text = "主体结构保修五十年。"
    result = _eval_numeric_comparison(matcher, rule, text, [], set(text.lower().split()), text)
    assert result.status == "FAILED", f"Expected FAILED (20 < 50), got {result.status}"
    assert "[structured:" in result.rationale, "Missing [structured:] marker"


def test_numeric_comparison_fallback_no_sidecar(matcher):
    """No sidecar → fall back to regex path, behavior unchanged."""
    from app.engine.handlers.numeric_comparison import _eval_numeric_comparison

    matcher._structured_inputs = {}  # empty

    rule = make_rule(
        rule_id="cn-test-2",
        name="屋面防水保修期",
        condition_type="numeric_comparison",
        condition_params={
            "label": "屋面防水保修期",
            "expected": 5,
            "operator": ">=",
            "unit": "年",
            "context_pattern": "防水|屋面",
            "legal_ref": "国务院令第279号第40条(二)",
        },
    )

    # Text says "防水保修五年" → regex extracts 5 → should PASS
    text = "屋面防水保修五年。"
    result = _eval_numeric_comparison(matcher, rule, text, [], set(text.lower().split()), text)
    assert result.status == "PASSED", f"Expected PASSED, got {result.status}"
    assert "[structured:" not in result.rationale, "Should NOT have structured marker"


def test_numeric_comparison_fallback_fail(matcher):
    """No sidecar, text has under-threshold value → FAILED."""
    from app.engine.handlers.numeric_comparison import _eval_numeric_comparison

    matcher._structured_inputs = {}

    rule = make_rule(
        rule_id="cn-test-3",
        name="屋面防水保修期",
        condition_type="numeric_comparison",
        condition_params={
            "label": "屋面防水保修期",
            "expected": 5,
            "operator": ">=",
            "unit": "年",
            "context_pattern": "防水|屋面",
            "legal_ref": "国务院令第279号第40条(二)",
        },
    )

    text = "屋面防水保修三年。"
    result = _eval_numeric_comparison(matcher, rule, text, [], set(text.lower().split()), text)
    assert result.status == "FAILED", f"Expected FAILED (3 < 5), got {result.status}"


# ═══════════════════════════════════════════════════════════════════
# required_pattern
# ═══════════════════════════════════════════════════════════════════

def test_required_pattern_match(matcher):
    """Pattern found in text → PASSED."""
    from app.engine.handlers.required_pattern import _eval_required_pattern

    rule = make_rule(
        rule_id="req-1",
        name="必须包含安全生产条款",
        condition_type="required_pattern",
        condition_params={
            "label": "安全生产条款",
            "terms": ["安全生产"],
        },
    )

    text = "本合同 包含 安全生产 条款 ， 双方应严格遵守。"
    result = _eval_required_pattern(matcher, rule, text, [], set(text.lower().split()), text)
    assert result.status == "PASSED", f"Expected PASSED, got {result.status}"


def test_required_pattern_missing(matcher):
    """Pattern not found → FAILED."""
    from app.engine.handlers.required_pattern import _eval_required_pattern

    rule = make_rule(
        rule_id="req-2",
        name="必须包含竣工验收条款",
        condition_type="required_pattern",
        condition_params={
            "label": "竣工验收",
            "terms": ["竣工验收"],
        },
    )

    text = "本合同无相关条款。"
    result = _eval_required_pattern(matcher, rule, text, [], set(text.lower().split()), text)
    assert result.status == "FAILED", f"Expected FAILED, got {result.status}"


def test_required_pattern_negated(matcher):
    """Pattern is negated in surrounding context → FAILED or NOT_APPLICABLE."""
    from app.engine.handlers.required_pattern import _eval_required_pattern

    rule = make_rule(
        rule_id="req-3",
        name="必须包含质量保修条款",
        condition_type="required_pattern",
        condition_params={
            "label": "质量保修",
            "terms": ["质量保修"],
        },
    )

    text = "本合同不包含质量保修条款。"
    result = _eval_required_pattern(matcher, rule, text, [], set(text.lower().split()), text)
    # Negated required pattern → expected FAILED
    assert result.status == "FAILED", f"Expected FAILED, got {result.status}"


# ═══════════════════════════════════════════════════════════════════
# forbidden_pattern
# ═══════════════════════════════════════════════════════════════════

def test_forbidden_pattern_hit(matcher):
    """Forbidden pattern found → FAILED."""
    from app.engine.handlers.forbidden_pattern import _eval_forbidden_pattern

    rule = make_rule(
        rule_id="forb-1",
        name="禁止使用国家明令淘汰的材料",
        condition_type="forbidden_pattern",
        condition_params={
            "label": "淘汰材料",
            "terms": ["石棉"],
        },
    )

    text = "本工程使用石棉材料。"
    result = _eval_forbidden_pattern(matcher, rule, text, [], set(text.lower().split()), text)
    assert result.status == "FAILED", f"Expected FAILED, got {result.status}"


def test_forbidden_pattern_clean(matcher):
    """Forbidden pattern not found → PASSED."""
    from app.engine.handlers.forbidden_pattern import _eval_forbidden_pattern

    rule = make_rule(
        rule_id="forb-2",
        name="禁止使用国家明令淘汰的材料",
        condition_type="forbidden_pattern",
        condition_params={
            "label": "淘汰材料",
            "terms": ["石棉"],
        },
    )

    text = "本工程使用环保材料。"
    result = _eval_forbidden_pattern(matcher, rule, text, [], set(text.lower().split()), text)
    assert result.status == "PASSED", f"Expected PASSED, got {result.status}"


def test_forbidden_pattern_negated(matcher):
    """Forbidden pattern is negated → PASSED (not a real prohibition)."""
    from app.engine.handlers.forbidden_pattern import _eval_forbidden_pattern

    rule = make_rule(
        rule_id="forb-3",
        name="禁止使用石棉",
        condition_type="forbidden_pattern",
        condition_params={
            "label": "石棉禁止",
            "terms": ["石棉"],
        },
    )

    text = "本工程不得使用石棉材料。"
    result = _eval_forbidden_pattern(matcher, rule, text, [], set(text.lower().split()), text)
    # "不得使用石棉" — the forbidden pattern in a prohibition statement
    # means the contract is compliant → handler returns PASSED
    assert result.status == "PASSED", f"Expected FAILED, got {result.status}"


# Plain-script self-run support
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from conftest import make_matcher
    matcher = make_matcher()
    # Manual test run
    for name in dir():
        if name.startswith("test_"):
            fn = globals()[name]
            try:
                fn(matcher)
                print(f"  [PASS] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
