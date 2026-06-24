# -*- coding: utf-8 -*-
"""Triager — Three-state classification layer (三态分流层).

Classifies each FAILED evidence item from the engine's evidence chain into:
  - FAIL:  confident violation, clearly actionable (自动失败)
  - NEEDS_REVIEW: borderline, ambiguous, or low-confidence (需人工复核)

PASSED and NOT_APPLICABLE items always map to triager PASS (自动通过).

The triager is ADDITIVE: it adds ``triager_verdict``, ``review_reason``, and
``_triager_scores`` to each evidence dict.  The original ``status`` field is
never mutated.

Calibration is loaded from ``triager_calibration.json``, with optional
per-domain overrides from ``<domain_dir>/triager_calibration.json``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TriagerSignal:
    """Weighted confidence signal for a single classification factor."""

    name: str
    weight: float  # 0.0–2.0, relative weight in composite
    description: str = ""


@dataclass
class TriagerCalibration:
    """Calibration profile loaded from JSON."""

    version: str = "1.0.0"
    signals: dict[str, TriagerSignal] = field(default_factory=dict)
    pass_threshold: float = 0.80  # composite >= this → FAIL (confident)
    fail_threshold: float = 0.30  # reserve for future use
    boundary_epsilon_pct: float = 0.15  # e.g. 15% of threshold value
    description: str = ""

    @staticmethod
    def from_dict(data: dict) -> "TriagerCalibration":
        """Deserialize from JSON dict."""
        signals: dict[str, TriagerSignal] = {}
        for name, cfg in data.get("signals", {}).items():
            signals[name] = TriagerSignal(
                name=name,
                weight=float(cfg.get("weight", 1.0)),
                description=str(cfg.get("description", "")),
            )
        return TriagerCalibration(
            version=str(data.get("version", "1.0.0")),
            signals=signals,
            pass_threshold=float(data.get("pass_threshold", 0.80)),
            fail_threshold=float(data.get("fail_threshold", 0.30)),
            boundary_epsilon_pct=float(data.get("boundary_epsilon_pct", 0.15)),
            description=str(data.get("description", "")),
        )


@dataclass
class TriagerVerdict:
    """Output of triage for a single evidence item."""

    triager_verdict: str  # "PASS" | "FAIL" | "NEEDS_REVIEW"
    review_reason: str  # human-readable explanation
    signals_breakdown: dict[str, float]  # per-signal scores

    @staticmethod
    def from_scores(
        scores: dict[str, float],
        weights: dict[str, float],
        thresholds: tuple[float, float],
    ) -> "TriagerVerdict":
        """Compute weighted composite score and threshold it.

        Args:
            scores: per-signal confidence scores (0.0–1.0).
            weights: per-signal weights (same keys as scores).
            thresholds: (fail_threshold, pass_threshold).

        Returns:
            TriagerVerdict with the classified verdict.
        """
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight == 0:
            composite = 0.5
        else:
            weighted_sum = sum(
                scores.get(k, 0.5) * weights.get(k, 1.0) for k in scores
            )
            composite = weighted_sum / total_weight
        composite = max(0.0, min(1.0, composite))

        _fail_threshold, pass_threshold = thresholds

        if composite >= pass_threshold:
            verdict = "FAIL"
        else:
            verdict = "NEEDS_REVIEW"

        reason_parts = [
            "{}={:.2f}".format(k, scores.get(k, 0.0)) for k in sorted(scores)
        ]
        reason = "composite={:.3f} [{}]".format(composite, ", ".join(reason_parts))

        return TriagerVerdict(
            triager_verdict=verdict,
            review_reason=reason,
            signals_breakdown=dict(scores),
        )


# ---------------------------------------------------------------------------
# Main Triager class
# ---------------------------------------------------------------------------


class Triager:
    """Three-state classification layer for evidence chain triage.

    Classifies each EvidenceItem (as dict) into PASS / FAIL / NEEDS_REVIEW.
    """

    # Terms considered too generic/common to be strong signals
    _GENERIC_TERMS: ClassVar[set[str]] = {
        "合同", "双方", "约定", "责任", "义务", "权利",
        "质量", "安全", "标准", "工程", "项目", "单位",
        "款", "项", "条", "日", "年", "月", "天",
    }

    def __init__(self, calibration_path: Optional[str] = None):
        self._calibration: Optional[TriagerCalibration] = None
        if calibration_path:
            self.load_calibration(calibration_path)

    # ── Calibration management ──────────────────────────────────────────

    def load_calibration(self, path: str) -> TriagerCalibration:
        """Load calibration from a JSON file path."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._calibration = TriagerCalibration.from_dict(data)
        logger.info("Triager: loaded calibration v%s from %s", self._calibration.version, path)
        return self._calibration

    def load_domain_calibration(self, domain_dir: str) -> TriagerCalibration:
        """Load domain-specific calibration if it exists, else use default."""
        domain_cal = os.path.join(domain_dir, "triager_calibration.json")
        if os.path.isfile(domain_cal):
            return self.load_calibration(domain_cal)
        if self._calibration is None:
            self._default_calibration()
        return self._calibration

    def _default_calibration(self) -> None:
        """Load built-in default calibration from the engine directory."""
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "triager_calibration.json",
        )
        if os.path.isfile(default_path):
            self.load_calibration(default_path)
        else:
            self._calibration = self._hardcoded_default()

    @staticmethod
    def _hardcoded_default() -> TriagerCalibration:
        """Hardcoded fallback calibration (used when JSON file is missing)."""
        return TriagerCalibration(
            version="1.0.0-fallback",
            signals={
                "source_credibility": TriagerSignal(
                    "source_credibility", 1.0,
                    "Rule source trustworthiness",
                ),
                "extraction_quality": TriagerSignal(
                    "extraction_quality", 0.8,
                    "Quality of extraction method",
                ),
                "boundary_proximity": TriagerSignal(
                    "boundary_proximity", 1.2,
                    "How close numeric value is to legal threshold",
                ),
                "matched_terms_strength": TriagerSignal(
                    "matched_terms_strength", 0.6,
                    "Specificity and count of matched terms",
                ),
                "belief_entropy": TriagerSignal(
                    "belief_entropy", 0.9,
                    "Entropy from belief propagation network",
                ),
                "severity_gate": TriagerSignal(
                    "severity_gate", 0.5,
                    "Severity level gating",
                ),
            },
            pass_threshold=0.80,
            fail_threshold=0.30,
            boundary_epsilon_pct=0.15,
        )

    @property
    def calibration(self) -> TriagerCalibration:
        if self._calibration is None:
            self._default_calibration()
        return self._calibration  # type: ignore[return-value]

    # ── Main entry point ────────────────────────────────────────────────

    def triage(
        self,
        evidence_chain: list[dict],
        belief_network: Optional[dict] = None,
        contract_text: str = "",
    ) -> list[dict]:
        """Classify every evidence item into PASS / FAIL / NEEDS_REVIEW.

        Args:
            evidence_chain: List of evidence item dicts from engine.validate().
            belief_network: Optional belief network report dict.
            contract_text: Full contract text (reserved for future context).

        Returns:
            The same evidence_chain list, with each dict augmented:
              - ``triager_verdict``: "PASS" | "FAIL" | "NEEDS_REVIEW"
              - ``review_reason``: human-readable explanation string
              - ``_triager_scores``: dict of per-signal scores (debug)
        """
        cal = self.calibration

        # Pre-index belief network entropy for fast lookups
        entropy_map: dict[str, float] = {}
        if belief_network:
            rule_beliefs = belief_network.get("rule_beliefs", {})
            if isinstance(rule_beliefs, dict):
                for rid, belief_dict in rule_beliefs.items():
                    if isinstance(belief_dict, dict):
                        entropy_map[rid] = float(
                            belief_dict.get("entropy", 0.0)
                        )

        for ev in evidence_chain:
            if not isinstance(ev, dict):
                continue

            status = ev.get("status", "")
            if status != "FAILED":
                ev["triager_verdict"] = "PASS"
                ev["review_reason"] = (
                    "Original status is {} — no review needed.".format(status)
                )
                continue

            # Compute all six signals
            scores = self._compute_all_signals(ev, entropy_map, cal)

            # Build weights dict matching the scores
            weights = {
                name: cal.signals[name].weight
                for name in scores
                if name in cal.signals
            }
            # Ensure every score has a weight (use 1.0 for missing)
            for name in scores:
                if name not in weights:
                    weights[name] = 1.0

            verdict = TriagerVerdict.from_scores(
                scores,
                weights,
                (cal.fail_threshold, cal.pass_threshold),
            )

            ev["triager_verdict"] = verdict.triager_verdict
            ev["review_reason"] = verdict.review_reason
            ev["_triager_scores"] = verdict.signals_breakdown

        return evidence_chain

    # ── Signal computation ──────────────────────────────────────────────

    def _compute_all_signals(
        self,
        ev: dict,
        entropy_map: dict[str, float],
        cal: TriagerCalibration,
    ) -> dict[str, float]:
        """Compute all six signal scores for one evidence item.

        Each signal returns 0.0–1.0:
          1.0 = high confidence that the engine verdict is correct
          0.0 = low confidence → NEEDS_REVIEW
        """
        return {
            "source_credibility": self._signal_source_credibility(ev),
            "extraction_quality": self._signal_extraction_quality(ev),
            "boundary_proximity": self._signal_boundary_proximity(ev, cal),
            "matched_terms_strength": self._signal_matched_terms_strength(ev),
            "belief_entropy": self._signal_belief_entropy(ev, entropy_map),
            "severity_gate": self._signal_severity_gate(ev),
        }

    # ── Signal 1: Source Credibility ───────────────────────────────────

    def _signal_source_credibility(self, ev: dict) -> float:
        """Confidence from source credibility.

        Low credibility + FAILED → untrustworthy (score → 0.0).
        High credibility + FAILED → trustworthy (score → 1.0).
        """
        cred = float(ev.get("source_credibility", 0.5))
        if cred < 0.3:
            return 0.10 + (cred / 0.3) * 0.05  # 0.10 – 0.15
        elif cred < 0.6:
            return 0.15 + ((cred - 0.3) / 0.3) * 0.45  # 0.15 – 0.60
        else:
            return 0.60 + ((cred - 0.6) / 0.4) * 0.40  # 0.60 – 1.00

    # ── Signal 2: Extraction Method Quality ────────────────────────────

    def _signal_extraction_quality(self, ev: dict) -> float:
        """Confidence from extraction method quality."""
        method = ev.get("extraction_method", "")
        scores = {
            "manual": 1.00,
            "structured": 0.95,
            "llm_extract": 0.60,
            "keyword_scan": 0.40,
            "conjecture_mine": 0.20,
        }
        return scores.get(method, 0.30)

    # ── Signal 3: Boundary Proximity ───────────────────────────────────

    def _signal_boundary_proximity(
        self,
        ev: dict,
        cal: TriagerCalibration,
    ) -> float:
        """Confidence from numeric boundary proximity.

        When a FAILED verdict is due to numeric_comparison and the extracted
        value is very close to the threshold, confidence drops → NEEDS_REVIEW.

        Returns 1.0 for non-numeric rules (can't assess boundary proximity).
        """
        rationale = ev.get("rationale", "")
        # Parse gap from rationale: "Gap: 0.1年" style
        m = re.search(
            r"Gap:\s*([\d.]+)\s*(年|月|日|天|元|万|%|％|万元)",
            rationale,
        )
        if not m:
            # Not a numeric comparison → assume no boundary concern
            return 1.0

        gap = float(m.group(1))
        # Extract threshold from rationale for epsilon calculation
        thresh_m = re.search(
            r"(?:>=|<=|==|!=|>|<)\s*([\d.]+)\s*(年|月|日|天|元|万|%|％|万元)",
            rationale,
        )
        threshold = 0.0
        if thresh_m:
            threshold = float(thresh_m.group(1))

        epsilon = cal.boundary_epsilon_pct * max(threshold, 0.001)

        if epsilon <= 0:
            return 0.5  # degenerate case

        if gap <= epsilon:
            # Very close to threshold → low confidence in FAIL
            ratio = gap / epsilon if epsilon > 0 else 0.0
            return 0.0 + ratio * 0.5  # 0.0 – 0.5
        elif gap <= epsilon * 3:
            ratio = (gap - epsilon) / (2.0 * epsilon)
            return 0.5 + ratio * 0.3  # 0.5 – 0.8
        else:
            # Clear gap → high confidence
            excess = gap - 3.0 * epsilon
            ratio = min(excess / (10.0 * epsilon), 1.0) if epsilon > 0 else 1.0
            return 0.8 + ratio * 0.2  # 0.8 – 1.0

    # ── Signal 4: Matched Terms Strength ───────────────────────────────

    def _signal_matched_terms_strength(self, ev: dict) -> float:
        """Confidence from matched terms specificity and count.

        Few matched terms (1-2) → weaker signal → lower confidence.
        Many specific terms → stronger signal → higher confidence.
        Generic/common terms → penalty applied.
        """
        matched = ev.get("matched_terms", [])
        if not isinstance(matched, list):
            matched = []

        n = len(matched)
        if n == 0:
            return 0.05  # suspicious — FAILED with zero matched terms
        if n == 1:
            base = 0.25
        elif n == 2:
            base = 0.45
        elif n <= 4:
            base = 0.45 + (n - 2) * 0.15  # 0.45 → 0.75
        else:
            base = 0.75 + min((n - 5) * 0.05, 0.15)  # 0.75 → 0.90

        # Penalize generic terms
        generic_count = sum(
            1 for t in matched if self._is_generic_term(str(t))
        )
        if generic_count > 0 and n > 0:
            generic_ratio = generic_count / n
            base -= generic_ratio * 0.30  # up to 0.30 penalty

        return max(0.05, min(1.0, base))

    @classmethod
    def _is_generic_term(cls, term: str) -> bool:
        """Check if a term is too generic/common to be a strong signal."""
        return term in cls._GENERIC_TERMS or len(term) <= 1

    # ── Signal 5: Belief Entropy ───────────────────────────────────────

    def _signal_belief_entropy(
        self,
        ev: dict,
        entropy_map: dict[str, float],
    ) -> float:
        """Confidence from belief network entropy.

        High entropy → uncertain belief → NEEDS_REVIEW.
        Low entropy → confident belief → high score.

        Entropy ranges 0–1 in Shannon terms.
        """
        rid = ev.get("rule_id", "")
        entropy = entropy_map.get(rid, 0.0)

        if entropy < 0.3:
            return 1.0
        elif entropy < 0.7:
            # 1.0 → 0.2 as entropy increases
            return 1.0 - ((entropy - 0.3) / 0.4) * 0.8
        else:
            # 0.2 → 0.0 as entropy approaches 1.0
            return 0.2 - ((entropy - 0.7) / 0.3) * 0.2

    # ── Signal 6: Severity Gating ──────────────────────────────────────

    def _signal_severity_gate(self, ev: dict) -> float:
        """Confidence from severity level.

        "error" + FAILED → high confidence (clear violation).
        "warning" + FAILED → moderate confidence.
        "info" + FAILED → low confidence (might be false positive).
        """
        sev = ev.get("severity", "warning")
        scores = {"error": 1.0, "warning": 0.6, "info": 0.3}
        return scores.get(str(sev).lower() if isinstance(sev, str) else "warning", 0.5)

    # ── Summary statistics ─────────────────────────────────────────────

    def summary(self, evidence_chain: list[dict]) -> dict:
        """Compute aggregate triage statistics."""
        total = 0
        pass_count = 0
        fail_count = 0
        review_count = 0

        for ev in evidence_chain:
            if not isinstance(ev, dict):
                continue
            tv = ev.get("triager_verdict", "")
            total += 1
            if tv == "PASS":
                pass_count += 1
            elif tv == "FAIL":
                fail_count += 1
            elif tv == "NEEDS_REVIEW":
                review_count += 1

        return {
            "total": total,
            "pass": pass_count,
            "fail": fail_count,
            "needs_review": review_count,
            "review_pct": round(review_count / max(total, 1) * 100, 1),
        }
