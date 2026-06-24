"""Differentiable Rule Matching — gradient descent optimization of rule
thresholds using analytical gradients.  No autograd library required.

The core insight: the rule evaluation pipeline can be made "soft" by replacing
hard comparisons (>=, <=) with sigmoid-based smooth approximations.  This
makes the pipeline differentiable with respect to threshold parameters,
allowing gradient descent to find optimal values from labeled data.

Pure Python implementation — uses only the math standard library.
"""
from __future__ import annotations
import math
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class TunableParameter:
    """A single tunable parameter extracted from a rule."""
    name: str                 # "expected" | "window_chars" | "min_ratio" | "threshold" | "match_window"
    value: float
    lower_bound: float
    upper_bound: float
    learning_rate: float = 0.01
    gradient: float = 0.0     # accumulated gradient

    def apply_gradient(self) -> None:
        """Update value using accumulated gradient, clip to bounds."""
        self.value -= self.learning_rate * self.gradient
        self.value = max(self.lower_bound, min(self.upper_bound, self.value))
        self.gradient = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "gradient": round(self.gradient, 6),
        }


@dataclass
class ParameterizedRule:
    """A rule with extracted tunable parameters."""
    rule_id: str
    condition_type: str
    parameters: list[TunableParameter]
    original_condition: dict          # the full condition dict from rules.json

    def apply_gradients(self) -> None:
        for p in self.parameters:
            p.apply_gradient()

    def get_param(self, name: str) -> Optional[TunableParameter]:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "condition_type": self.condition_type,
            "parameters": [p.to_dict() for p in self.parameters],
        }


@dataclass
class CalibrationReport:
    """Results of threshold optimization for a single rule."""
    rule_id: str
    optimized_parameters: dict[str, float]   # param_name → optimized value
    initial_loss: float
    final_loss: float
    iterations: int
    converged: bool
    sample_results: list[dict] = field(default_factory=list)

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def loss_reduction_pct(self) -> float:
        if self.initial_loss <= 0:
            return 0.0
        return (self.loss_reduction / self.initial_loss) * 100

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "optimized_parameters": {
                k: round(v, 4) for k, v in self.optimized_parameters.items()
            },
            "initial_loss": round(self.initial_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "loss_reduction_pct": round(self.loss_reduction_pct, 2),
            "iterations": self.iterations,
            "converged": self.converged,
            "sample_count": len(self.sample_results),
        }


# ======================================================================
# ThresholdOptimizer
# ======================================================================

class ThresholdOptimizer:
    """Optimizes rule thresholds via gradient descent on labeled samples.

    For numeric_comparison rules, the "expected" threshold is the primary
    tunable parameter.  For other condition types, relevant parameters
    (window size, min ratio, etc.) are tuned.

    Usage:
        optimizer = ThresholdOptimizer(learning_rate=0.01, max_iterations=200)
        report = optimizer.optimize(rule, labeled_samples)
        print(f"Loss: {report.initial_loss:.4f} → {report.final_loss:.4f}")
    """

    def __init__(self, learning_rate: float = 0.01,
                 max_iterations: int = 200,
                 convergence_threshold: float = 1e-5,
                 temperature: float = 1.0):
        """Initialize the optimizer.

        Args:
            learning_rate: Step size for gradient descent.
            max_iterations: Maximum gradient descent iterations.
            convergence_threshold: Stop when loss change < this.
            temperature: Temperature for soft comparison (lower = sharper).
        """
        self._lr = learning_rate
        self._max_iter = max_iterations
        self._threshold = convergence_threshold
        self._temperature = temperature
        # Cache for numeric extraction results (text + context_pattern → float)
        self._extraction_cache: dict[tuple[str, str], Optional[float]] = {}

    # ── Soft comparison functions ─────────────────────────────────────

    def soft_comparison(self, x: float, theta: float,
                        operator: str) -> float:
        """Sigmoid-based smooth approximation of comparison operators.

        Returns value in [0, 1] where 1 = PASS (constraint satisfied),
        0 = FAIL (constraint violated).

        soft_ge(x, theta): σ((x - θ) / T) → 1 as x > θ, 0 as x < θ
        soft_le(x, theta): σ((θ - x) / T) → 1 as x < θ, 0 as x > θ
        soft_eq(x, theta): σ(-|x - θ| / T) → 1 as x ≈ θ
        """
        T = max(self._temperature, 0.001)
        if operator in (">=", ">"):
            return self._sigmoid((x - theta) / T)
        elif operator in ("<=", "<"):
            return self._sigmoid((theta - x) / T)
        elif operator == "==":
            return self._sigmoid(-abs(x - theta) / T)
        elif operator == "!=":
            return 1.0 - self._sigmoid(-abs(x - theta) / T)
        return 0.5

    def soft_gradient(self, x: float, theta: float,
                      operator: str) -> float:
        """Analytical gradient of soft_comparison w.r.t. theta.

        d/dθ soft_ge(x,θ) = -σ'((x-θ)/T) / T
        d/dθ soft_le(x,θ) = +σ'((θ-x)/T) / T
        where σ'(z) = σ(z) * (1 - σ(z))
        """
        T = max(self._temperature, 0.001)
        if operator in (">=", ">"):
            z = (x - theta) / T
            s = self._sigmoid(z)
            return -s * (1.0 - s) / T
        elif operator in ("<=", "<"):
            z = (theta - x) / T
            s = self._sigmoid(z)
            return s * (1.0 - s) / T
        elif operator == "==":
            z = abs(x - theta) / T
            s = self._sigmoid(-z)
            sign = 1.0 if theta >= x else -1.0
            return sign * s * (1.0 - s) / T
        elif operator == "!=":
            z = abs(x - theta) / T
            s = self._sigmoid(-z)
            sign = -1.0 if theta >= x else 1.0
            return sign * s * (1.0 - s) / T
        return 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)

    # ── Rule parametrization ──────────────────────────────────────────

    def parametrize_rule(self, rule: dict) -> ParameterizedRule:
        """Extract tunable parameters from a rule.

        Different condition types have different tunable parameters:
        - numeric_comparison: expected (value), window_chars (context window)
        - sum_numeric_comparison: expected (sum threshold)
        - required_pattern: min_ratio (minimum term coverage)
        - forbidden_pattern: match_window (negation context size)
        - mutual_exclusion: threshold (co-occurrence count to trigger)
        """
        cond = rule.get("condition", {})
        cond_type = cond.get("type", "")
        params: list[TunableParameter] = []

        if cond_type == "numeric_comparison":
            expected = float(cond.get("expected", 0))
            params.append(TunableParameter(
                name="expected",
                value=expected,
                lower_bound=max(0.1, expected * 0.3),
                upper_bound=expected * 3.0,
                learning_rate=self._lr,
            ))
            # window_chars (default 200 in engine)
            window = float(cond.get("window_chars", 200))
            params.append(TunableParameter(
                name="window_chars",
                value=window,
                lower_bound=20,
                upper_bound=1000,
                learning_rate=self._lr * 0.1,
            ))

        elif cond_type == "sum_numeric_comparison":
            expected = float(cond.get("expected", 100))
            params.append(TunableParameter(
                name="expected",
                value=expected,
                lower_bound=max(1, expected * 0.3),
                upper_bound=expected * 2.0,
                learning_rate=self._lr,
            ))

        elif cond_type == "required_pattern":
            min_ratio = float(cond.get("min_ratio", 0.5))
            params.append(TunableParameter(
                name="min_ratio",
                value=min_ratio,
                lower_bound=0.1,
                upper_bound=1.0,
                learning_rate=self._lr * 0.05,
            ))

        elif cond_type == "forbidden_pattern":
            match_window = float(cond.get("match_window", 200))
            params.append(TunableParameter(
                name="match_window",
                value=match_window,
                lower_bound=10,
                upper_bound=500,
                learning_rate=self._lr * 0.1,
            ))

        elif cond_type == "mutual_exclusion":
            threshold = float(cond.get("threshold", 2))
            params.append(TunableParameter(
                name="threshold",
                value=threshold,
                lower_bound=1,
                upper_bound=max(3, len(cond.get("terms", []))),
                learning_rate=self._lr * 0.1,
            ))

        return ParameterizedRule(
            rule_id=rule.get("id", "unknown"),
            condition_type=cond_type,
            parameters=params,
            original_condition=deepcopy(cond),
        )

    # ── Optimization ──────────────────────────────────────────────────

    def optimize(self, rule: dict,
                 labeled_samples: list[dict]) -> CalibrationReport:
        """Optimize rule thresholds using gradient descent.

        Args:
            rule: Rule dict from rules.json.
            labeled_samples: List of {text: str, ground_truth: str (PASSED|FAILED)}.

        Returns:
            CalibrationReport with optimized parameters and loss history.
        """
        param_rule = self.parametrize_rule(rule)
        if not param_rule.parameters:
            return CalibrationReport(
                rule_id=param_rule.rule_id,
                optimized_parameters={},
                initial_loss=0.0, final_loss=0.0,
                iterations=0, converged=True,
            )

        # Compute initial loss
        initial_loss = self._compute_loss(param_rule, rule, labeled_samples)
        prev_loss = initial_loss

        # Gradient descent loop
        converged = False
        for it in range(self._max_iter):
            # Zero gradients
            for p in param_rule.parameters:
                p.gradient = 0.0

            # Accumulate gradients over all samples
            for sample in labeled_samples:
                self._accumulate_gradients(param_rule, rule, sample)

            # Average gradients and update
            n = max(1, len(labeled_samples))
            for p in param_rule.parameters:
                p.gradient /= n
            param_rule.apply_gradients()

            # Compute new loss
            current_loss = self._compute_loss(param_rule, rule, labeled_samples)

            # Check convergence
            if abs(current_loss - prev_loss) < self._threshold:
                converged = True
                break
            prev_loss = current_loss

        # Build sample results
        sample_results = []
        for sample in labeled_samples:
            score = self._predict(param_rule, rule, sample)
            sample_results.append({
                "text": sample.get("text", "")[:80],
                "soft_score": round(score, 4),
                "ground_truth": sample.get("ground_truth", "?"),
            })

        return CalibrationReport(
            rule_id=param_rule.rule_id,
            optimized_parameters={
                p.name: p.value for p in param_rule.parameters
            },
            initial_loss=round(initial_loss, 6),
            final_loss=round(current_loss if 'current_loss' in dir() else prev_loss, 6),
            iterations=it + 1 if not converged else it + 1,
            converged=converged,
            sample_results=sample_results,
        )

    def batch_optimize(self, rules: list[dict],
                       calibration_data: list[dict]) -> list[CalibrationReport]:
        """Optimize a batch of rules using calibration data.

        Groups samples by rule_id and optimizes each rule independently.

        Args:
            rules: List of rule dicts.
            calibration_data: List of {rule_id, text, ground_truth}.

        Returns:
            List of CalibrationReport, one per rule.
        """
        # Group samples by rule_id
        by_rule: dict[str, list[dict]] = {}
        for item in calibration_data:
            rid = item.get("rule_id", "")
            if rid:
                by_rule.setdefault(rid, []).append(item)

        reports = []
        rule_map = {r["id"]: r for r in rules}
        for rid, samples in by_rule.items():
            if rid not in rule_map:
                continue
            if len(samples) < 3:
                logger.info(f"跳过 {rid}: 样本不足 ({len(samples)} < 3)")
                continue
            report = self.optimize(rule_map[rid], samples)
            reports.append(report)

        return reports

    # ── Internal: loss and gradient computation ───────────────────────

    def _compute_loss(self, param_rule: ParameterizedRule,
                      rule: dict,
                      samples: list[dict]) -> float:
        """Compute average hinge loss over all samples."""
        if not samples:
            return 0.0
        total_loss = 0.0
        for sample in samples:
            score = self._predict(param_rule, rule, sample)
            gt = sample.get("ground_truth", "FAILED")
            target = 1.0 if gt == "PASSED" else 0.0
            total_loss += self._hinge_loss(score, target)
        return total_loss / len(samples)

    def _predict(self, param_rule: ParameterizedRule,
                 rule: dict, sample: dict) -> float:
        """Compute soft prediction score for a sample.

        Returns a float in [0, 1] where 1 = PASS, 0 = FAIL.
        """
        cond_type = param_rule.condition_type
        text = sample.get("text", "")
        cond = rule.get("condition", {})

        if cond_type == "numeric_comparison":
            x = self._extract_numeric(text, cond)
            if x is None:
                return 0.0  # can't extract → don't penalize
            param = param_rule.get_param("expected")
            theta = param.value if param else float(cond.get("expected", 0))
            operator = cond.get("operator", ">=")
            return self.soft_comparison(x, theta, operator)

        elif cond_type == "sum_numeric_comparison":
            numbers = self._extract_all_numbers(text)
            if not numbers:
                return 0.0
            total = sum(numbers)
            param = param_rule.get_param("expected")
            theta = param.value if param else float(cond.get("expected", 100))
            operator = cond.get("operator", "<=")
            return self.soft_comparison(total, theta, operator)

        elif cond_type == "required_pattern":
            terms = cond.get("terms", [])
            if not terms:
                return 0.0
            # Soft ratio: what fraction of required terms appear?
            matched = sum(1 for t in terms if t in text)
            ratio = matched / len(terms)
            param = param_rule.get_param("min_ratio")
            theta = param.value if param else 0.5
            return self.soft_comparison(ratio, theta, ">=")

        elif cond_type == "forbidden_pattern":
            terms = cond.get("terms", [])
            if not terms:
                return 1.0
            # If any forbidden term appears → FAIL
            matched = any(t in text for t in terms)
            return 0.0 if matched else 1.0

        elif cond_type == "mutual_exclusion":
            terms = cond.get("terms", [])
            if not terms:
                return 1.0
            matched_count = sum(1 for t in terms if t in text)
            param = param_rule.get_param("threshold")
            theta = param.value if param else 2
            return self.soft_comparison(float(matched_count), theta, "<")

        return 0.5

    def _accumulate_gradients(self, param_rule: ParameterizedRule,
                               rule: dict, sample: dict) -> None:
        """Accumulate analytical gradients for one sample."""
        cond_type = param_rule.condition_type
        text = sample.get("text", "")
        cond = rule.get("condition", {})
        gt = sample.get("ground_truth", "FAILED")
        target = 1.0 if gt == "PASSED" else 0.0

        if cond_type == "numeric_comparison":
            x = self._extract_numeric(text, cond)
            if x is None:
                return
            param = param_rule.get_param("expected")
            if param is None:
                return
            operator = cond.get("operator", ">=")

            score = self.soft_comparison(x, param.value, operator)
            dL_dscore = self._hinge_gradient(score, target)
            dscore_dtheta = self.soft_gradient(x, param.value, operator)
            param.gradient += dL_dscore * dscore_dtheta

        elif cond_type == "sum_numeric_comparison":
            numbers = self._extract_all_numbers(text)
            if not numbers:
                return
            total = sum(numbers)
            param = param_rule.get_param("expected")
            if param is None:
                return
            operator = cond.get("operator", "<=")

            score = self.soft_comparison(total, param.value, operator)
            dL_dscore = self._hinge_gradient(score, target)
            dscore_dtheta = self.soft_gradient(total, param.value, operator)
            param.gradient += dL_dscore * dscore_dtheta

        elif cond_type == "required_pattern":
            terms = cond.get("terms", [])
            if not terms:
                return
            matched = sum(1 for t in terms if t in text)
            ratio = matched / len(terms)
            param = param_rule.get_param("min_ratio")
            if param is None:
                return

            score = self.soft_comparison(ratio, param.value, ">=")
            dL_dscore = self._hinge_gradient(score, target)
            dscore_dtheta = self.soft_gradient(ratio, param.value, ">=")
            param.gradient += dL_dscore * dscore_dtheta

        elif cond_type == "mutual_exclusion":
            terms = cond.get("terms", [])
            if not terms:
                return
            matched_count = sum(1 for t in terms if t in text)
            param = param_rule.get_param("threshold")
            if param is None:
                return

            score = self.soft_comparison(float(matched_count), param.value, "<")
            dL_dscore = self._hinge_gradient(score, target)
            dscore_dtheta = self.soft_gradient(float(matched_count), param.value, "<")
            param.gradient += dL_dscore * dscore_dtheta

    # ── Internal: numeric extraction ──────────────────────────────────

    def _extract_numeric(self, text: str, cond: dict) -> Optional[float]:
        """Extract a numeric value from text using the rule's context_pattern."""
        import re
        pattern = cond.get("context_pattern", "")
        if not pattern:
            return None

        cache_key = (text, pattern)
        if cache_key in self._extraction_cache:
            return self._extraction_cache[cache_key]

        try:
            ctx_re = re.compile(pattern)
        except re.error:
            return None

        matches = list(ctx_re.finditer(text))
        if not matches:
            return None

        # Search ±200 chars around the best context match
        best_val = None
        best_score = -1
        for m in matches:
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            window = text[start:end]
            numbers = list(re.finditer(r'(\d+(?:\.\d+)?)\s*(年|月|天|%|元|万元|万)', window))
            if numbers:
                if len(numbers) > best_score:
                    best_score = len(numbers)
                    # Pick the number closest to and after the context match
                    ctx_pos = m.start() - start
                    best_num = None
                    best_dist = float('inf')
                    for n in numbers:
                        dist = n.start() - ctx_pos
                        if dist >= -5 and dist < best_dist:
                            best_dist = dist
                            best_num = float(n.group(1))
                    best_val = best_num

        result = best_val
        self._extraction_cache[cache_key] = result
        return result

    def _extract_all_numbers(self, text: str) -> list[float]:
        """Extract all percentage-like numbers from text (for sum comparisons)."""
        import re
        numbers = []
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
            numbers.append(float(m.group(1)))
        return numbers

    # ── Internal: loss functions ──────────────────────────────────────

    @staticmethod
    def _hinge_loss(predicted: float, target: float, margin: float = 0.5) -> float:
        """Smooth hinge loss for binary (0/1) predictions.

        loss = max(0, margin - accuracy)
        where accuracy = target * predicted + (1 - target) * (1 - predicted)
        """
        accuracy = target * predicted + (1.0 - target) * (1.0 - predicted)
        return max(0.0, margin - accuracy)

    @staticmethod
    def _hinge_gradient(predicted: float, target: float, margin: float = 0.5) -> float:
        """Gradient of hinge loss w.r.t. predicted score.

        dL/dp = -(2*target - 1) if accuracy < margin else 0
        """
        accuracy = target * predicted + (1.0 - target) * (1.0 - predicted)
        if accuracy < margin:
            return -(2.0 * target - 1.0)
        return 0.0
