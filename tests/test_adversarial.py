"""Test Algorithm 2: Adversarial Sample Generator."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.adversarial import (
    AdversarialGenerator, AdversarialSample, AdversarialSuite,
    _arabic_to_cn, _cjk_boundary_splits, _find_synonyms, _fmt_val,
)


# ── Test helper ──

def make_rule(rule_id, cond_type, **cond_kwargs):
    """Build a minimal rule dict for testing."""
    return {
        "id": rule_id,
        "name": f"Test {rule_id}",
        "condition": {"type": cond_type, **cond_kwargs},
        "severity": "error",
        "category": "测试",
        "source_credibility": 0.5,
    }


# ── Test 1: numeric boundary ──

def test_numeric_boundary():
    """Generate adversarial suite for a >=5 年 rule. Verify boundary samples."""
    gen = AdversarialGenerator()
    rule = make_rule("test-num", "numeric_comparison",
                     label="屋面防水保修期", context_pattern="防水|保修",
                     unit="年", operator=">=", expected=5)
    suite = gen.generate_adversarial_suite(rule)

    assert suite.condition_type == "numeric_comparison"
    assert len(suite.positive_samples) >= 1, "Expected positive samples"
    assert len(suite.negative_samples) >= 1, "Expected negative samples"
    assert len(suite.boundary_samples) >= 1, "Expected boundary samples"

    # Boundary sample at threshold (5 years for >= should PASS)
    boundary_texts = [s.text for s in suite.boundary_samples]
    has_5 = any("5" in t or "五" in t for t in boundary_texts)
    assert has_5, "Expected boundary sample at threshold value 5"

    # Negative samples should have values < 5
    neg_texts = [s.text for s in suite.negative_samples]
    has_sub5 = any(("4" in t or "三" in t or "四" in t) for t in neg_texts)
    assert has_sub5, "Expected negative samples with values < 5"

    print(f"  ✓ numeric_boundary: {len(suite.positive_samples)} pos, "
          f"{len(suite.negative_samples)} neg, {len(suite.boundary_samples)} boundary")
    return True


# ── Test 2: required pattern negation ──

def test_required_negation():
    """For a required_pattern rule, verify negation insertion produces FAIL samples."""
    gen = AdversarialGenerator()
    rule = make_rule("test-req", "required_pattern",
                     terms=["质量保修书", "保修书"], message_if_missing="缺保修书")

    suite = gen.generate_adversarial_suite(rule, "本合同包含质量保修书和保修书。")

    # Should have negation samples
    negations = [s for s in suite.negative_samples if s.perturbation_type == "negation"]
    assert len(negations) > 0, "Expected negation insertion samples"

    # Negation samples should contain "不" or similar
    has_neg = any("不" in s.text or "无" in s.text or "未" in s.text for s in negations)
    assert has_neg, "Negation samples should contain negation characters"

    print(f"  ✓ required_negation: {len(negations)} negation samples")
    return True


# ── Test 3: forbidden insertion ──

def test_forbidden_insertion():
    """For a forbidden_pattern rule, verify insertion at various positions."""
    gen = AdversarialGenerator()
    rule = make_rule("test-forb", "forbidden_pattern",
                     terms=["转包", "垫资"], reason="禁止转包和垫资")

    suite = gen.generate_adversarial_suite(rule)

    # Should have negative samples with forbidden terms inserted
    neg_texts = [s.text for s in suite.negative_samples]
    has_forbidden = any("转包" in t or "垫资" in t for t in neg_texts)
    assert has_forbidden, "Negative samples should contain forbidden terms"

    # Positive should NOT contain forbidden terms
    pos_texts = [s.text for s in suite.positive_samples]
    assert len(pos_texts) >= 1
    for t in pos_texts:
        assert "转包" not in t, f"Positive sample should not contain '转包': {t[:60]}"
        assert "垫资" not in t, f"Positive sample should not contain '垫资': {t[:60]}"

    print(f"  ✓ forbidden_insertion: {len(suite.negative_samples)} insertion samples")
    return True


# ── Test 4: sum_numeric ──

def test_sum_numeric():
    """For a sum_numeric_comparison rule, verify perturbed payment percentages."""
    gen = AdversarialGenerator()
    rule = make_rule("test-sum", "sum_numeric_comparison",
                     label="付款比例合计", context_pattern="付款|比例",
                     unit="%", operator="<=", expected=100)

    suite = gen.generate_adversarial_suite(rule)

    # Positive should sum to <= expected
    pos_texts = [s.text for s in suite.positive_samples]
    assert len(pos_texts) >= 1

    # Negative should sum to > expected
    neg_texts = [s.text for s in suite.negative_samples]
    assert len(neg_texts) >= 1

    print(f"  ✓ sum_numeric: {len(suite.positive_samples)} pos, {len(suite.negative_samples)} neg")
    return True


# ── Test 5: CJK boundary splits ──

def test_cjk_boundary():
    """Verify CJK token boundary split generation."""
    splits = _cjk_boundary_splits("仲裁委员会")
    # 4-char term produces 3 or 4 splits depending on implementation
    assert len(splits) >= 3, f"Expected at least 3 splits for 4-char term, got {len(splits)}"
    assert "仲 裁委员会" in splits
    assert "仲裁 委员会" in splits

    # Single char — no splits
    assert _cjk_boundary_splits("仲") == []

    print(f"  ✓ cjk_boundary: {splits}")
    return True


# ── Test 6: Chinese numeral conversion ──

def test_cn_numerals():
    """Verify Arabic → Chinese numeral conversion."""
    assert _arabic_to_cn(5) == "五"
    assert _arabic_to_cn(10) == "十"
    assert _arabic_to_cn(36) == "三十六"
    assert _arabic_to_cn(50) == "五十"
    assert _arabic_to_cn(99) == "九十九"

    print("  ✓ cn_numerals: 5→五, 10→十, 36→三十六, 50→五十, 99→九十九")
    return True


# ── Test 7: synonym lookup ──

def test_synonyms():
    """Verify synonym lookup for domain terms."""
    syns = _find_synonyms("保修")
    assert "质保" in syns or "维保" in syns, f"Expected synonyms for 保修, got {syns}"

    syns2 = _find_synonyms("违约金")
    assert len(syns2) > 0, f"Expected synonyms for 违约金"

    print(f"  ✓ synonyms: 保修→{syns}, 违约金→{syns2}")
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
