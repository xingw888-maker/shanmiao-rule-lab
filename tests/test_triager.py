# -*- coding: utf-8 -*-
"""Test Triager — three-state classification layer for evidence chain triage.

Covers: default calibration, PASS items, high/low-confidence FAILED items,
boundary proximity, belief entropy, summary statistics, domain override.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.triager import Triager, TriagerCalibration, TriagerVerdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_evidence_item(**overrides) -> dict:
    """Build a minimal evidence item dict with sensible defaults."""
    base = {
        "trace_id": "evt_test001",
        "rule_id": "cn-001",
        "rule_name": "屋面防水保修期限",
        "rule_version": "1.0.0",
        "package_id": "construction",
        "package_version": "1.0.0",
        "severity": "error",
        "status": "FAILED",
        "input_fragment": "屋面防水保修期限为四年",
        "segment_id": None,
        "matched_terms": ["屋面", "防水", "保修", "四年"],
        "rationale": (
            "Field '屋面防水保修期限': contract value 4.0年 "
            "FAILS legal threshold (>= 5年). "
            "Ref: 国务院令第279号第40条(二). Gap: 1.0年."
        ),
        "suggestion": "Increase '屋面防水保修期限' from 4.0年 to at least 5.0年.",
        "category": "保修",
        "source_type": "国务院令",
        "source_credibility": 0.9,
        "extraction_method": "structured",
        "layer": "L0_VALIDATED",
    }
    base.update(overrides)
    return base


def make_belief_network(rule_beliefs: dict) -> dict:
    """Build a minimal belief network report dict."""
    return {"rule_beliefs": rule_beliefs}


# ---------------------------------------------------------------------------
# Test 1: Default calibration loads correctly
# ---------------------------------------------------------------------------


def test_default_calibration():
    """Default calibration has all six signals and valid thresholds."""
    triager = Triager()
    cal = triager.calibration
    assert cal is not None
    assert len(cal.signals) == 6, (
        "Expected 6 signals, got {}".format(len(cal.signals))
    )
    assert 0 < cal.fail_threshold < cal.pass_threshold < 1.0, (
        "Invalid thresholds: fail={}, pass={}".format(
            cal.fail_threshold, cal.pass_threshold
        )
    )
    assert cal.boundary_epsilon_pct > 0
    expected_signals = [
        "source_credibility",
        "extraction_quality",
        "boundary_proximity",
        "matched_terms_strength",
        "belief_entropy",
        "severity_gate",
    ]
    for name in expected_signals:
        assert name in cal.signals, "Missing signal: {}".format(name)
    print("  PASS: test_default_calibration")


# ---------------------------------------------------------------------------
# Test 2: PASSED items always get triager PASS
# ---------------------------------------------------------------------------


def test_passed_items_always_pass():
    """Evidence items with status PASSED or NOT_APPLICABLE always map to triager PASS."""
    triager = Triager()
    chain = [
        make_evidence_item(status="PASSED"),
        make_evidence_item(status="NOT_APPLICABLE"),
    ]
    result = triager.triage(chain)
    for ev in result:
        assert ev["triager_verdict"] == "PASS", (
            "Expected PASS for status={}, got {}".format(
                ev["status"], ev["triager_verdict"]
            )
        )
        assert "review_reason" in ev
    print("  PASS: test_passed_items_always_pass")


# ---------------------------------------------------------------------------
# Test 3: High-confidence FAILED maps to triager FAIL
# ---------------------------------------------------------------------------


def test_high_confidence_failed():
    """FAILED item with high credibility, good extraction, error severity, clear gap → FAIL."""
    triager = Triager()
    chain = [
        make_evidence_item(
            status="FAILED",
            source_credibility=0.95,
            extraction_method="manual",
            severity="error",
            matched_terms=["屋面防水", "保修期限", "五年", "建设工程"],
            rationale=(
                "contract value 3.0年 FAILS legal threshold (>= 5年). "
                "Gap: 2.0年."
            ),
        ),
    ]
    result = triager.triage(chain)
    ev = result[0]
    assert ev["triager_verdict"] == "FAIL", (
        "Expected FAIL, got {} — {}".format(
            ev["triager_verdict"], ev.get("review_reason", "")
        )
    )
    print("  PASS: test_high_confidence_failed")


# ---------------------------------------------------------------------------
# Test 4: Low-confidence FAILED maps to NEEDS_REVIEW
# ---------------------------------------------------------------------------


def test_low_confidence_needs_review():
    """FAILED item with low credibility, conjecture extraction, info severity → NEEDS_REVIEW."""
    triager = Triager()
    chain = [
        make_evidence_item(
            status="FAILED",
            source_credibility=0.2,
            extraction_method="conjecture_mine",
            severity="info",
            matched_terms=["工程"],
            rationale="Required term(s) NOT found: some_obscure_term.",
        ),
    ]
    result = triager.triage(chain)
    ev = result[0]
    assert ev["triager_verdict"] == "NEEDS_REVIEW", (
        "Expected NEEDS_REVIEW, got {} — {}".format(
            ev["triager_verdict"], ev.get("review_reason", "")
        )
    )
    print("  PASS: test_low_confidence_needs_review")


# ---------------------------------------------------------------------------
# Test 5: Boundary proximity triggers NEEDS_REVIEW
# ---------------------------------------------------------------------------


def test_boundary_proximity_needs_review():
    """FAILED with tiny gap (4.9年 vs 5年) → NEEDS_REVIEW due to boundary proximity."""
    triager = Triager()
    chain = [
        make_evidence_item(
            status="FAILED",
            source_credibility=0.85,
            extraction_method="structured",
            severity="error",
            matched_terms=["屋面", "防水", "保修", "四年九个月"],
            rationale=(
                "Field '屋面防水保修期限': contract value 4.9年 "
                "FAILS legal threshold (>= 5年). Gap: 0.1年."
            ),
        ),
    ]
    result = triager.triage(chain)
    ev = result[0]
    # With tiny gap (0.1年 vs 5年), boundary_proximity should be very low
    assert ev["triager_verdict"] == "NEEDS_REVIEW", (
        "Expected NEEDS_REVIEW for borderline value, got {} — {}".format(
            ev["triager_verdict"], ev.get("review_reason", "")
        )
    )
    scores = ev.get("_triager_scores", {})
    bp_score = scores.get("boundary_proximity", 1.0)
    assert bp_score < 0.4, (
        "Boundary proximity should be low for tiny gap (0.1年 vs 5年), "
        "got {:.3f}".format(bp_score)
    )
    print("  PASS: test_boundary_proximity_needs_review")


# ---------------------------------------------------------------------------
# Test 6: Belief entropy triggers NEEDS_REVIEW
# ---------------------------------------------------------------------------


def test_belief_entropy_needs_review():
    """FAILED item with high belief entropy → NEEDS_REVIEW."""
    triager = Triager()
    chain = [
        make_evidence_item(
            status="FAILED",
            rule_id="cn-001",
            source_credibility=0.8,
            extraction_method="llm_extract",
            severity="warning",
            matched_terms=["屋面", "防水", "保修", "五年"],
            rationale=(
                "contract value 4.5年 FAILS legal threshold (>= 5年). "
                "Gap: 0.5年."
            ),
        ),
    ]
    bn = make_belief_network({
        "cn-001": {
            "entropy": 0.85,
            "belief_pass": 0.4,
            "belief_fail": 0.6,
        },
    })
    result = triager.triage(chain, belief_network=bn)
    ev = result[0]
    assert ev["triager_verdict"] == "NEEDS_REVIEW", (
        "Expected NEEDS_REVIEW at high entropy, got {}".format(
            ev["triager_verdict"]
        )
    )
    scores = ev.get("_triager_scores", {})
    ent_score = scores.get("belief_entropy", 1.0)
    assert ent_score < 0.2, (
        "Belief entropy score should be very low (high entropy), "
        "got {:.3f}".format(ent_score)
    )
    print("  PASS: test_belief_entropy_needs_review")


# ---------------------------------------------------------------------------
# Test 7: Triager summary statistics
# ---------------------------------------------------------------------------


def test_summary_statistics():
    """Triager.summary() returns correct counts."""
    triager = Triager()
    chain = [
        make_evidence_item(status="PASSED"),
        make_evidence_item(
            status="FAILED",
            source_credibility=0.95,
            extraction_method="manual",
            severity="error",
            matched_terms=["屋面防水", "保修期限", "五年", "建设工程"],
            rationale="Gap: 2.0年.",
        ),
        make_evidence_item(
            status="FAILED",
            source_credibility=0.2,
            extraction_method="conjecture_mine",
            severity="info",
            matched_terms=["工程"],
        ),
        make_evidence_item(status="NOT_APPLICABLE"),
    ]
    result = triager.triage(chain)
    stats = triager.summary(result)
    assert stats["total"] == 4, "Expected 4 total, got {}".format(stats["total"])
    assert stats["pass"] == 2, "Expected 2 pass, got {}".format(stats["pass"])
    assert stats["fail"] == 1, "Expected 1 fail, got {}".format(stats["fail"])
    assert stats["needs_review"] == 1, (
        "Expected 1 needs_review, got {}".format(stats["needs_review"])
    )
    print("  PASS: test_summary_statistics")


# ---------------------------------------------------------------------------
# Test 8: Domain calibration overrides default
# ---------------------------------------------------------------------------


def test_domain_calibration_override():
    """Domain-specific calibration JSON overrides the default calibration."""
    with tempfile.TemporaryDirectory() as domain_dir:
        cal_path = os.path.join(domain_dir, "triager_calibration.json")
        custom_cal = {
            "version": "1.0.0-test",
            "pass_threshold": 0.90,
            "fail_threshold": 0.20,
            "boundary_epsilon_pct": 0.05,
            "signals": {
                "source_credibility": {"weight": 2.0, "description": "test"},
                "extraction_quality": {"weight": 0.5, "description": "test"},
                "boundary_proximity": {"weight": 1.5, "description": "test"},
                "matched_terms_strength": {"weight": 0.3, "description": "test"},
                "belief_entropy": {"weight": 1.0, "description": "test"},
                "severity_gate": {"weight": 0.2, "description": "test"},
            },
        }
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(custom_cal, f)

        triager = Triager()
        cal = triager.load_domain_calibration(domain_dir)
        assert cal.pass_threshold == 0.90, (
            "Expected pass_threshold 0.90, got {}".format(cal.pass_threshold)
        )
        assert cal.signals["source_credibility"].weight == 2.0, (
            "Expected weight 2.0 for source_credibility, got {}".format(
                cal.signals["source_credibility"].weight
            )
        )
        print("  PASS: test_domain_calibration_override")


# ---------------------------------------------------------------------------
# Test 9: JSON calibration file loads correctly
# ---------------------------------------------------------------------------


def test_json_calibration_file():
    """The default triager_calibration.json file loads without errors."""
    cal_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "app", "engine", "triager_calibration.json",
    )
    assert os.path.isfile(cal_path), (
        "Calibration file not found at {}".format(cal_path)
    )
    triager = Triager(calibration_path=cal_path)
    cal = triager.calibration
    assert cal.version == "1.0.0"
    assert cal.pass_threshold == 0.80
    assert len(cal.signals) == 6
    print("  PASS: test_json_calibration_file")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_default_calibration()
    test_passed_items_always_pass()
    test_high_confidence_failed()
    test_low_confidence_needs_review()
    test_boundary_proximity_needs_review()
    test_belief_entropy_needs_review()
    test_summary_statistics()
    test_domain_calibration_override()
    test_json_calibration_file()
    print("\nAll 9 tests passed.")
