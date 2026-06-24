# -*- coding: utf-8 -*-
"""Test clause classifiers: PurePython, sklearn, HybridV2, JSONL training."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.clause_classifier import (
    HybridClauseClassifierV2,
    PurePythonClauseClassifier,
    _HAS_SKLEARN,
    _cjk_ngrams,
    create_classifier,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def make_training_data():
    """Minimal training data for 5 clause types."""
    texts = [
        "屋面防水工程保修期限为五年。主体结构保修期限为五十年。",
        "地下室防水工程保修期限为五年。",
        "电气管线保修期限为三年。",
        "给排水管道保修期限为三年。",
        "本合同付款节点：签订合同后支付预付款30%，主体封顶后支付40%。",
        "工程款按月支付，每月25日前发包人向承包人支付已完工程量的80%。",
        "竣工验收合格后支付至结算价款的97%。",
        "违约金按逾期天数的万分之五计算，违约金总额不超过合同总价的10%。",
        "发包人逾期付款，按日万分之三支付违约金。",
        "承包人逾期竣工，每天罚款合同总价的千分之一。",
        "本工程计划开工日期为2025年3月1日，竣工日期为2026年9月30日。",
        "工期为540日历天，自监理工程师发出开工令之日起计算。",
    ]
    labels = [
        "保修", "保修", "保修", "保修",
        "付款", "付款", "付款",
        "违约", "违约", "违约",
        "工期", "工期",
    ]
    return texts, labels


# ── Test 1: CJK n-gram extraction ────────────────────────────────────


def test_cjk_ngrams():
    """CJK n-gram extraction produces correct n-grams."""
    text = "保修期限"
    ngrams = _cjk_ngrams(text, 1, 3)
    assert "保" in ngrams, "Missing unigram"
    assert "保修" in ngrams, "Missing bigram"
    assert "保修期" in ngrams, "Missing trigram"
    assert "期限" in ngrams, "Missing bigram at end"
    print("  PASS: test_cjk_ngrams")


# ── Test 2: PurePython classifier fits and predicts ──────────────────


def test_purepython_fit_predict():
    """PurePython classifier trains and makes correct predictions."""
    texts, labels = make_training_data()
    clf = PurePythonClauseClassifier()
    clf.fit(texts, labels)

    assert clf.fitted, "Classifier should be fitted"
    assert len(clf.classes_) == 4, (
        "Expected 4 classes, got {}".format(len(clf.classes_))
    )

    # Test on known patterns
    pred1 = clf.predict("电气管线和给排水管道保修期限为三年。")
    assert pred1 == "保修", "Expected 保修, got {}".format(pred1)

    pred2 = clf.predict("竣工验收合格后支付至已完工程量的97%。")
    assert pred2 == "付款", "Expected 付款, got {}".format(pred2)

    pred3 = clf.predict("工期延误每天罚款一万元。")
    assert pred3 == "违约", "Expected 违约, got {}".format(pred3)

    print("  PASS: test_purepython_fit_predict")


# ── Test 3: predict_proba returns valid probabilities ────────────────


def test_predict_proba():
    """predict_proba returns normalized probabilities."""
    texts, labels = make_training_data()
    clf = PurePythonClauseClassifier()
    clf.fit(texts, labels)

    proba = clf.predict_proba("屋面防水保修期限为五年。")
    assert isinstance(proba, dict), "proba should be dict"
    assert len(proba) == len(clf.classes_), (
        "proba should have same number of classes"
    )
    total = sum(proba.values())
    assert abs(total - 1.0) < 0.001, (
        "Probabilities should sum to 1, got {}".format(total)
    )
    # All probs should be between 0 and 1
    for cls, p in proba.items():
        assert 0.0 <= p <= 1.0, "Probability out of range: {}={}".format(cls, p)

    print("  PASS: test_predict_proba")


# ── Test 4: predict_top returns sorted results ───────────────────────


def test_predict_top():
    """predict_top returns top-k sorted by probability."""
    texts, labels = make_training_data()
    clf = PurePythonClauseClassifier()
    clf.fit(texts, labels)

    top = clf.predict_top("屋面防水保修期限为五年。", k=3)
    assert len(top) == 3, "Expected 3 results, got {}".format(len(top))
    # Should be sorted descending
    for i in range(len(top) - 1):
        assert top[i][1] >= top[i + 1][1], (
            "predict_top should be sorted by probability"
        )

    print("  PASS: test_predict_top")


# ── Test 5: Unfitted classifier returns safe defaults ────────────────


def test_unfitted_safe_defaults():
    """Unfitted classifier returns 'unknown' without crashing."""
    clf = PurePythonClauseClassifier()
    assert not clf.fitted
    assert clf.predict("anything") == "unknown"
    assert clf.predict_proba("anything") == {"unknown": 1.0}
    assert clf.predict_top("anything") == [("unknown", 1.0)]
    print("  PASS: test_unfitted_safe_defaults")


# ── Test 6: JSONL training from build_clause_dataset output ───────────


def test_jsonl_training():
    """HybridV2 can train from a JSONL file."""
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=False
    ) as f:
        texts, labels = make_training_data()
        for text, label in zip(texts, labels):
            rec = {
                "text": text,
                "label": label,
                "label_source": "ground_truth",
                "contract": "test",
                "features": {},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path = f.name

    try:
        hv2 = HybridClauseClassifierV2()
        hv2.fit_from_jsonl(tmp_path)
        assert hv2.fitted, "HybridV2 should be fitted"
        assert len(hv2.classes_) >= 3, (
            "Expected at least 3 classes, got {}".format(hv2.classes_)
        )

        # Test prediction
        pred = hv2.predict("屋面防水保修期限为五年。")
        assert pred in hv2.classes_, (
            "Prediction should be a known class, got {}".format(pred)
        )
    finally:
        os.unlink(tmp_path)

    print("  PASS: test_jsonl_training")


# ── Test 7: create_classifier factory ─────────────────────────────────


def test_create_classifier():
    """create_classifier factory function works."""
    clf = create_classifier()
    assert isinstance(clf, HybridClauseClassifierV2)
    assert not clf.fitted, "Should be unfitted without training data"

    # With temp JSONL
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=False
    ) as f:
        texts, labels = make_training_data()
        for text, label in zip(texts, labels):
            f.write(
                json.dumps(
                    {"text": text, "label": label}, ensure_ascii=False
                )
                + "\n"
            )
        tmp_path = f.name

    try:
        clf2 = create_classifier(training_jsonl=tmp_path)
        assert clf2.fitted, "Should be fitted with training data"
    finally:
        os.unlink(tmp_path)

    print("  PASS: test_create_classifier")


# ── Test 8: predict on unseen CJK text ───────────────────────────────


def test_unseen_text():
    """Classifier handles unseen CJK text gracefully."""
    texts, labels = make_training_data()
    clf = PurePythonClauseClassifier()
    clf.fit(texts, labels)

    # Completely unseen terms should still get a prediction (no crash)
    pred = clf.predict("仲裁委员会由三名仲裁员组成。")
    assert isinstance(pred, str)
    assert len(pred) > 0
    # Probability should be roughly uniform when nothing matches
    proba = clf.predict_proba("仲裁委员会由三名仲裁员组成。")
    assert isinstance(proba, dict)

    print("  PASS: test_unseen_text")


# ── Test 9: sklearn classifier (if available) ────────────────────────


def test_sklearn_classifier():
    """TfidfClauseClassifier works if sklearn is available."""
    if not _HAS_SKLEARN:
        print("  SKIP: test_sklearn_classifier (sklearn not installed)")
        return

    from app.engine.clause_classifier import TfidfClauseClassifier

    texts, labels = make_training_data()
    clf = TfidfClauseClassifier()
    clf.fit(texts, labels)

    assert clf.fitted
    pred = clf.predict("屋面防水保修期限为五年。")
    assert pred in clf.classes_

    proba = clf.predict_proba("屋面防水保修期限为五年。")
    total = sum(proba.values())
    assert abs(total - 1.0) < 0.01

    print("  PASS: test_sklearn_classifier")


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cjk_ngrams()
    test_purepython_fit_predict()
    test_predict_proba()
    test_predict_top()
    test_unfitted_safe_defaults()
    test_jsonl_training()
    test_create_classifier()
    test_unseen_text()
    test_sklearn_classifier()
    print("\nAll tests passed.")
