"""Adversarial Sample Generator — systematically generates text perturbations
that flip rule verdicts, producing high-quality test suites for each rule.

Much deeper than the basic AutoBadSampleGenerator: this generator understands
CJK token boundaries, Chinese numeral variants, negation patterns, and the
structural semantics of each condition type.  Zero external dependencies.
"""
from __future__ import annotations
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class AdversarialSample:
    """A single adversarial text sample."""
    text: str
    expected_verdict: str          # "PASSED" or "FAILED"
    perturbation_type: str         # "boundary" | "synonym" | "negation" | "insertion" | "deletion" | "unit_swap" | "cjk_split" | "context_shift"
    rule_id: str
    description: str               # human-readable description of the perturbation

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "expected_verdict": self.expected_verdict,
            "perturbation_type": self.perturbation_type,
            "rule_id": self.rule_id,
            "description": self.description,
        }


@dataclass
class AdversarialSuite:
    """A complete adversarial test suite for a single rule."""
    rule_id: str
    rule_name: str
    condition_type: str
    positive_samples: list[AdversarialSample] = field(default_factory=list)
    negative_samples: list[AdversarialSample] = field(default_factory=list)
    boundary_samples: list[AdversarialSample] = field(default_factory=list)
    minimal_flip: Optional[AdversarialSample] = None

    def all_samples(self) -> list[AdversarialSample]:
        return self.positive_samples + self.negative_samples + self.boundary_samples

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "condition_type": self.condition_type,
            "positive_count": len(self.positive_samples),
            "negative_count": len(self.negative_samples),
            "boundary_count": len(self.boundary_samples),
            "samples": [s.to_dict() for s in self.all_samples()],
            "minimal_flip": self.minimal_flip.to_dict() if self.minimal_flip else None,
        }


# ======================================================================
# CJK / Chinese numeral helpers
# ======================================================================

# Chinese numeral characters and their values
_CN_NUMERALS = {
    '零': 0, '〇': 0,
    '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100, '千': 1000,
    '万': 10000, '亿': 100000000,
}

_CN_DIGITS = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2,
              '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

_CN_MULTIPLIERS = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}

# Chinese numerals for 1-99
_ARABIC_TO_CN = {
    0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五',
    6: '六', 7: '七', 8: '八', 9: '九', 10: '十',
    11: '十一', 12: '十二', 13: '十三', 14: '十四', 15: '十五',
    16: '十六', 17: '十七', 18: '十八', 19: '十九', 20: '二十',
    21: '二十一', 22: '二十二', 23: '二十三', 24: '二十四', 25: '二十五',
    30: '三十', 36: '三十六', 40: '四十', 48: '四十八', 50: '五十',
    60: '六十', 70: '七十', 80: '八十', 90: '九十', 99: '九十九',
    100: '一百', 110: '一百一十', 120: '一百二十',
}

# Negation words for insertion before required terms
_NEGATIONS = ['不', '无', '未', '非', '无需', '免于', '免除']

# Domain-specific synonym pairs (can be extended)
_SYNONYM_PAIRS: dict[str, list[str]] = {
    "保修": ["质保", "维保", "维修保养"],
    "违约金": ["罚金", "罚款", "赔偿金", "违约赔偿"],
    "付款": ["支付", "付款", "结款", "打款"],
    "验收": ["竣工验收", "验收合格", "工程验收"],
    "安全生产": ["安全施工", "施工安全", "安全管理"],
    "争议": ["纠纷", "争议解决", "争端"],
    "仲裁": ["仲裁", "公断"],
    "法院": ["人民法院", "法庭"],
}

# Context templates for generating adversarial texts (condition-type-specific)
_TEMPLATES = {
    "numeric_comparison": {
        "保修": "本工程{subject}保修期限为{value}{unit}。",
        "付款": "合同约定{subject}为{value}{unit}。",
        "违约": "如逾期，{subject}按每日{value}{unit}计算。",
        "验收": "发包人收到竣工验收报告后{value}{unit}内组织验收。",
        "工程管理": "本工程{subject}为{value}{unit}。",
        "default": "合同中{subject}约定为{value}{unit}。",
    },
    "required_pattern": {
        "default": "本合同已明确约定{terms}。",
    },
    "forbidden_pattern": {
        "default": "双方同意按照本合同条款执行。",
    },
}


# ======================================================================
# AdversarialGenerator
# ======================================================================

class AdversarialGenerator:
    """Generates adversarial test suites for validation rules.

    For each condition type, produces systematic perturbations:
    - numeric_comparison: boundary values, Chinese numerals, unit swaps
    - required_pattern: term removal, synonym substitution, negation, CJK splits
    - forbidden_pattern: term insertion at various positions
    - sum_numeric_comparison: perturb individual numbers
    - mutual_exclusion: construct co-occurrence texts
    """

    def __init__(self, engine=None):
        """Initialize with an optional PythonValidationEngine for self-testing.

        Args:
            engine: Optional PythonValidationEngine instance. If None,
                    adversarial samples are generated but not self-verified.
        """
        self._engine = engine
        # Build CJK boundary split regex: match between two CJK chars
        self._RE_CJK = re.compile(r'[一-鿿]')

    def generate_adversarial_suite(self, rule: dict,
                                   positive_text: str = "") -> AdversarialSuite:
        """Generate a complete adversarial test suite for one rule.

        Args:
            rule: Rule dict (as in rules.json).
            positive_text: Optional positive sample text that should PASS.

        Returns:
            AdversarialSuite with positive, negative, and boundary samples.
        """
        cond = rule.get("condition", {})
        cond_type = cond.get("type", "")
        rule_id = rule.get("id", "unknown")
        rule_name = rule.get("name", rule_id)

        if cond_type == "numeric_comparison":
            suite = self._gen_numeric_suite(rule, positive_text)
        elif cond_type == "sum_numeric_comparison":
            suite = self._gen_sum_numeric_suite(rule, positive_text)
        elif cond_type == "required_pattern":
            suite = self._gen_required_suite(rule, positive_text)
        elif cond_type == "forbidden_pattern":
            suite = self._gen_forbidden_suite(rule, positive_text)
        elif cond_type == "mutual_exclusion":
            suite = self._gen_mutex_suite(rule, positive_text)
        else:
            suite = AdversarialSuite(rule_id=rule_id, rule_name=rule_name,
                                     condition_type=cond_type)
        return suite

    # ── numeric_comparison ────────────────────────────────────────────

    def _gen_numeric_suite(self, rule: dict, positive_text: str) -> AdversarialSuite:
        cond = rule.get("condition", {})
        expected = float(cond.get("expected", 0))
        operator = cond.get("operator", ">=")
        unit = cond.get("unit", "")
        label = cond.get("label", rule.get("name", ""))
        category = rule.get("category", "default")

        suite = AdversarialSuite(
            rule_id=rule.get("id", ""),
            rule_name=rule.get("name", ""),
            condition_type="numeric_comparison",
        )

        tmpl = _TEMPLATES["numeric_comparison"].get(category,
                    _TEMPLATES["numeric_comparison"]["default"])

        # Determine compliant and violating values
        if operator in (">=", ">"):
            compliant_val = expected + 1
            if expected == int(expected) and expected > 0:
                # For integer thresholds, use clear boundary values
                boundary_val = expected
                violating_vals = [expected - 1, max(1, expected - 3)]
            else:
                boundary_val = expected
                violating_vals = [expected - 0.1, expected / 2]
        elif operator in ("<=", "<"):
            compliant_val = expected - 1 if expected > 1 else expected / 2
            boundary_val = expected
            violating_vals = [expected + 1, expected * 2]
        elif operator == "==":
            compliant_val = expected
            boundary_val = expected
            violating_vals = [expected + 1, expected - 1]
        else:  # !=
            compliant_val = expected + 1
            boundary_val = expected
            violating_vals = [expected]

        # ── Positive samples (should PASS) ──
        pos_text = tmpl.format(subject=label, value=_fmt_val(compliant_val), unit=unit)
        suite.positive_samples.append(AdversarialSample(
            text=pos_text, expected_verdict="PASSED",
            perturbation_type="base", rule_id=suite.rule_id,
            description=f"合规值: {compliant_val}{unit}",
        ))

        # Chinese numeral positive
        cn_val = _arabic_to_cn(int(compliant_val)) if compliant_val == int(compliant_val) else None
        if cn_val and int(compliant_val) <= 99:
            cn_text = tmpl.format(subject=label, value=cn_val, unit=unit)
            suite.positive_samples.append(AdversarialSample(
                text=cn_text, expected_verdict="PASSED",
                perturbation_type="synonym", rule_id=suite.rule_id,
                description=f"中文数字合规: {cn_val}{unit}",
            ))

        # ── Boundary samples ──
        b_text = tmpl.format(subject=label, value=_fmt_val(boundary_val), unit=unit)
        boundary_verdict = "FAILED" if operator in (">=", ">") else "PASSED"
        if operator == ">=":
            boundary_verdict = "PASSED"  # >= means boundary is compliant
        elif operator == "<=":
            boundary_verdict = "PASSED"  # <= means boundary is compliant
        elif operator == ">":
            boundary_verdict = "FAILED"  # > means boundary is not compliant
        elif operator == "<":
            boundary_verdict = "FAILED"

        suite.boundary_samples.append(AdversarialSample(
            text=b_text, expected_verdict=boundary_verdict,
            perturbation_type="boundary", rule_id=suite.rule_id,
            description=f"边界值: {_fmt_val(boundary_val)}{unit} (应为{boundary_verdict})",
        ))

        # Chinese numeral boundary
        if boundary_val == int(boundary_val) and int(boundary_val) <= 99:
            cn_b = _arabic_to_cn(int(boundary_val))
            if cn_b:
                cn_b_text = tmpl.format(subject=label, value=cn_b, unit=unit)
                suite.boundary_samples.append(AdversarialSample(
                    text=cn_b_text, expected_verdict=boundary_verdict,
                    perturbation_type="boundary", rule_id=suite.rule_id,
                    description=f"中文数字边界: {cn_b}{unit}",
                ))

        # ── Negative samples (should FAIL) ──
        for v in violating_vals:
            if v <= 0:
                continue
            neg_text = tmpl.format(subject=label, value=_fmt_val(v), unit=unit)
            suite.negative_samples.append(AdversarialSample(
                text=neg_text, expected_verdict="FAILED",
                perturbation_type="boundary", rule_id=suite.rule_id,
                description=f"违规值: {_fmt_val(v)}{unit}",
            ))

        # Chinese numeral negative
        for v in violating_vals:
            if v == int(v) and 1 <= int(v) <= 99:
                cn_v = _arabic_to_cn(int(v))
                if cn_v:
                    cn_neg_text = tmpl.format(subject=label, value=cn_v, unit=unit)
                    suite.negative_samples.append(AdversarialSample(
                        text=cn_neg_text, expected_verdict="FAILED",
                        perturbation_type="synonym", rule_id=suite.rule_id,
                        description=f"中文数字违规: {cn_v}{unit}",
                    ))

        # Unit swap (if accept_units exists)
        accept_units = cond.get("accept_units", [])
        if accept_units and unit:
            alt_unit = accept_units[0] if accept_units else None
            if alt_unit and alt_unit != unit:
                # For unit conversion, generate a semantically different value
                if unit == "年" and "月" in accept_units:
                    # 1 year expressed in months — should still pass if numeric check is unit-aware
                    alt_val = compliant_val * 12 if operator in (">=", ">") else violating_vals[0] * 12
                    alt_text = tmpl.format(subject=label, value=_fmt_val(alt_val), unit=alt_unit)
                    suite.boundary_samples.append(AdversarialSample(
                        text=alt_text, expected_verdict="PASSED",
                        perturbation_type="unit_swap", rule_id=suite.rule_id,
                        description=f"单位转换: {_fmt_val(alt_val)}{alt_unit} (原{unit})",
                    ))

        # ── Context shift: move value outside context_pattern ──
        context_pattern = cond.get("context_pattern", "")
        if context_pattern and positive_text:
            # Try to create text where the number appears OUTSIDE the context pattern scope
            shifted = _shift_value_outside_context(positive_text, context_pattern, expected, unit)
            if shifted:
                suite.negative_samples.append(AdversarialSample(
                    text=shifted, expected_verdict="FAILED",
                    perturbation_type="context_shift", rule_id=suite.rule_id,
                    description="数值出现在上下文范围外",
                ))

        return suite

    # ── required_pattern ──────────────────────────────────────────────

    def _gen_required_suite(self, rule: dict, positive_text: str) -> AdversarialSuite:
        cond = rule.get("condition", {})
        terms = cond.get("terms", [])
        if not isinstance(terms, list) or not terms:
            terms = []

        suite = AdversarialSuite(
            rule_id=rule.get("id", ""),
            rule_name=rule.get("name", ""),
            condition_type="required_pattern",
        )

        # Base positive text
        base = positive_text if positive_text else f"本合同已明确约定{'、'.join(terms)}。"
        suite.positive_samples.append(AdversarialSample(
            text=base, expected_verdict="PASSED",
            perturbation_type="base", rule_id=suite.rule_id,
            description=f"包含全部必需术语: {', '.join(terms[:3])}",
        ))

        # ── Negative: remove each term individually ──
        for i, term in enumerate(terms):
            removed = base
            # Remove the term and collapse extra whitespace/punctuation
            removed = removed.replace(term, "")
            removed = re.sub(r'[、，。]{2,}', '、', removed)
            removed = re.sub(r'\s{2,}', ' ', removed).strip()
            if removed and removed != base:
                suite.negative_samples.append(AdversarialSample(
                    text=removed, expected_verdict="FAILED",
                    perturbation_type="deletion", rule_id=suite.rule_id,
                    description=f"缺少必需术语: {term}",
                ))

        # ── Negative: synonym replacement (term replaced with near-synonym not in list) ──
        for term in terms:
            synonyms = _find_synonyms(term)
            for syn in synonyms[:2]:  # limit to 2 synonyms per term
                if syn not in terms:
                    replaced = base.replace(term, syn)
                    if replaced != base:
                        suite.negative_samples.append(AdversarialSample(
                            text=replaced, expected_verdict="FAILED",
                            perturbation_type="synonym", rule_id=suite.rule_id,
                            description=f"近义词替换: {term}→{syn}",
                        ))

        # ── Negative: negation insertion ──
        for term in terms[:3]:  # limit to first 3 terms
            for neg in _NEGATIONS[:3]:
                # Insert negation before the term
                neg_text = base.replace(term, f"{neg}{term}")
                if neg_text != base:
                    suite.negative_samples.append(AdversarialSample(
                        text=neg_text, expected_verdict="FAILED",
                        perturbation_type="negation", rule_id=suite.rule_id,
                        description=f"否定插入: {neg}{term}",
                    ))

        # ── Negative: CJK boundary split ──
        for term in terms:
            splits = _cjk_boundary_splits(term)
            for sp in splits[:2]:
                split_text = base.replace(term, sp)
                if split_text != base:
                    suite.boundary_samples.append(AdversarialSample(
                        text=split_text, expected_verdict="FAILED",
                        perturbation_type="cjk_split", rule_id=suite.rule_id,
                        description=f"CJK边界分割: {sp}",
                    ))

        return suite

    # ── forbidden_pattern ─────────────────────────────────────────────

    def _gen_forbidden_suite(self, rule: dict, positive_text: str) -> AdversarialSuite:
        cond = rule.get("condition", {})
        terms = cond.get("terms", [])
        if not isinstance(terms, list) or not terms:
            terms = []

        suite = AdversarialSuite(
            rule_id=rule.get("id", ""),
            rule_name=rule.get("name", ""),
            condition_type="forbidden_pattern",
        )

        # Base positive text (no forbidden terms)
        base = positive_text if positive_text else "双方同意按照本合同条款执行。"
        suite.positive_samples.append(AdversarialSample(
            text=base, expected_verdict="PASSED",
            perturbation_type="base", rule_id=suite.rule_id,
            description="不含任何禁止术语",
        ))

        # ── Negative: insert each forbidden term at various positions ──
        for term in terms[:5]:  # limit to 5 terms
            # Insert at beginning
            suite.negative_samples.append(AdversarialSample(
                text=f"{term}。{base}", expected_verdict="FAILED",
                perturbation_type="insertion", rule_id=suite.rule_id,
                description=f"句首插入: {term}",
            ))
            # Insert in middle
            mid = len(base) // 2
            mid_text = base[:mid] + f"，{term}，" + base[mid:]
            suite.negative_samples.append(AdversarialSample(
                text=mid_text, expected_verdict="FAILED",
                perturbation_type="insertion", rule_id=suite.rule_id,
                description=f"句中插入: {term}",
            ))
            # Insert at end
            suite.negative_samples.append(AdversarialSample(
                text=f"{base}{term}。", expected_verdict="FAILED",
                perturbation_type="insertion", rule_id=suite.rule_id,
                description=f"句尾插入: {term}",
            ))

        return suite

    # ── sum_numeric_comparison ────────────────────────────────────────

    def _gen_sum_numeric_suite(self, rule: dict, positive_text: str) -> AdversarialSuite:
        cond = rule.get("condition", {})
        expected = float(cond.get("expected", 100))
        operator = cond.get("operator", "<=")
        unit = cond.get("unit", "%")
        label = cond.get("label", rule.get("name", ""))
        category = rule.get("category", "付款")

        suite = AdversarialSuite(
            rule_id=rule.get("id", ""),
            rule_name=rule.get("name", ""),
            condition_type="sum_numeric_comparison",
        )

        tmpl = _TEMPLATES["numeric_comparison"].get(category,
                    _TEMPLATES["numeric_comparison"]["default"])

        # ── Positive: numbers summing to compliant total ──
        parts = _split_into_parts(expected, operator, 3)
        pos_text = f"预付款{parts[0]}{unit}，进度款{parts[1]}{unit}，尾款{parts[2]}{unit}。"
        suite.positive_samples.append(AdversarialSample(
            text=pos_text, expected_verdict="PASSED",
            perturbation_type="base", rule_id=suite.rule_id,
            description=f"付款比例合计={sum(parts)}{unit} (合规)",
        ))

        # ── Negative: numbers summing to violating total ──
        if operator in ("<=", "<"):
            violating_parts = _split_into_parts(expected + 15, operator, 3)
        else:
            violating_parts = _split_into_parts(expected - 15, operator, 3)
        neg_text = f"预付款{violating_parts[0]}{unit}，进度款{violating_parts[1]}{unit}，尾款{violating_parts[2]}{unit}。"
        suite.negative_samples.append(AdversarialSample(
            text=neg_text, expected_verdict="FAILED",
            perturbation_type="boundary", rule_id=suite.rule_id,
            description=f"付款比例合计={sum(violating_parts)}{unit} (违规)",
        ))

        # ── Extra number outside context ──
        extra_text = f"预付款{parts[0]}{unit}，进度款{parts[1]}{unit}，尾款{parts[2]}{unit}。合同总价1000000元。"
        suite.boundary_samples.append(AdversarialSample(
            text=extra_text, expected_verdict="PASSED",
            perturbation_type="context_shift", rule_id=suite.rule_id,
            description="额外数值在上下文外（合同总价不应计入）",
        ))

        return suite

    # ── mutual_exclusion ──────────────────────────────────────────────

    def _gen_mutex_suite(self, rule: dict, positive_text: str) -> AdversarialSuite:
        cond = rule.get("condition", {})
        terms = cond.get("terms", [])
        if not isinstance(terms, list) or len(terms) < 2:
            terms = []

        suite = AdversarialSuite(
            rule_id=rule.get("id", ""),
            rule_name=rule.get("name", ""),
            condition_type="mutual_exclusion",
        )

        # Positive: only one term appears
        if terms:
            single_text = f"本合同计价方式为{terms[0]}。"
            suite.positive_samples.append(AdversarialSample(
                text=single_text, expected_verdict="PASSED",
                perturbation_type="base", rule_id=suite.rule_id,
                description=f"仅含单一定价方式: {terms[0]}",
            ))

        # Negative: multiple mutually-exclusive terms co-occur
        if len(terms) >= 2:
            multi_terms = '、'.join(terms[:3])
            multi_text = f"本合同计价方式包括{multi_terms}。"
            suite.negative_samples.append(AdversarialSample(
                text=multi_text, expected_verdict="FAILED",
                perturbation_type="insertion", rule_id=suite.rule_id,
                description=f"互斥术语共存: {multi_terms}",
            ))

        return suite

    # ── Minimal flip search ───────────────────────────────────────────

    def find_minimal_flip(self, rule: dict, text: str) -> Optional[AdversarialSample]:
        """Find the smallest text change that flips the rule verdict.

        Uses a greedy token-removal search: removes tokens one by one,
        checking if the verdict flips from its baseline.

        Args:
            rule: Rule dict.
            text: Starting text.

        Returns:
            AdversarialSample describing the minimal flip, or None if no flip found.
        """
        if not self._engine:
            return None

        cond_type = rule.get("condition", {}).get("type", "")
        if cond_type not in ("required_pattern", "numeric_comparison"):
            return None  # minimal flip only supported for these types

        baseline = self._evaluate_rule(rule, text)
        if baseline is None:
            return None

        # Tokenize into CJK runs + punctuation + spaces
        tokens = _cjk_tokenize(text)

        # Greedy: try removing each token individually, find the smallest
        # change that flips the verdict
        best_flip = None
        best_cost = len(tokens) + 1

        for i, token in enumerate(tokens):
            # Skip if token is just punctuation
            if re.match(r'^[，。、；：！？\s]+$', token):
                continue

            modified_tokens = tokens[:i] + tokens[i+1:]
            modified_text = ''.join(modified_tokens).strip()

            if not modified_text or modified_text == text:
                continue

            new_verdict = self._evaluate_rule(rule, modified_text)
            if new_verdict and new_verdict != baseline:
                cost = 1  # single token removal
                if cost < best_cost:
                    best_cost = cost
                    best_flip = AdversarialSample(
                        text=modified_text,
                        expected_verdict=new_verdict,
                        perturbation_type="deletion",
                        rule_id=rule.get("id", ""),
                        description=f"最小翻转: 删除'{token}' → {new_verdict}",
                    )

        return best_flip

    def _evaluate_rule(self, rule_dict: dict, text: str) -> Optional[str]:
        """Evaluate a single rule against text.  Returns 'PASSED', 'FAILED',
        'NOT_APPLICABLE', or None on error."""
        if not self._engine:
            return None
        try:
            from app.engine.core import CompiledRule
            compiled = CompiledRule(
                id=rule_dict.get("id", "test"),
                name=rule_dict.get("name", ""),
                condition_type=rule_dict.get("condition", {}).get("type", ""),
                condition_params=rule_dict.get("condition", {}),
                severity=rule_dict.get("severity", "warning"),
                message=rule_dict.get("message", ""),
                category=rule_dict.get("category", ""),
                version=rule_dict.get("version", "1.0.0"),
                package_id=rule_dict.get("package_id", "test"),
                package_version=rule_dict.get("package_version", "1.0.0"),
                source=rule_dict.get("source", ""),
                source_credibility=rule_dict.get("source_credibility", 0.5),
                extraction_method=rule_dict.get("extraction_method", ""),
                clause_type=rule_dict.get("clause_type", ""),
            )
            matcher = self._engine._matcher if hasattr(self._engine, '_matcher') else None
            if matcher is None:
                return None
            evidence = self._engine._matcher.evaluate(text, compiled)
            if evidence:
                return evidence.status if hasattr(evidence, 'status') else str(evidence)
            return None
        except Exception:
            return None


# ======================================================================
# Free helper functions
# ======================================================================

def _fmt_val(v: float) -> str:
    """Format a numeric value for template insertion."""
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"


def _arabic_to_cn(n: int) -> Optional[str]:
    """Convert an integer (1-9999) to a Chinese numeral string."""
    if n in _ARABIC_TO_CN:
        return _ARABIC_TO_CN[n]
    if n <= 0:
        return None
    # Compound: e.g., 三十六
    if n < 100:
        tens = n // 10
        ones = n % 10
        if tens == 1 and ones == 0:
            return '十'
        elif tens == 1:
            return f'十{_ARABIC_TO_CN.get(ones, "")}'
        elif ones == 0:
            return f'{_ARABIC_TO_CN.get(tens, "")}十'
        else:
            return f'{_ARABIC_TO_CN.get(tens, "")}十{_ARABIC_TO_CN.get(ones, "")}'
    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        base = f'{_ARABIC_TO_CN.get(hundreds, "")}百'
        if rest > 0:
            base += _arabic_to_cn(rest) or ''
        return base
    return str(n)


def _find_synonyms(term: str) -> list[str]:
    """Find known synonyms for a term."""
    for key, syns in _SYNONYM_PAIRS.items():
        if term in syns or key in term or term in key:
            return [s for s in syns if s != term]
    return []


def _cjk_boundary_splits(term: str) -> list[str]:
    """Generate CJK token boundary split variants of a multi-char term.
    E.g., "仲裁委员会" → ["仲 裁委员会", "仲裁 委员会", "仲裁委 员会"]
    """
    if len(term) < 2:
        return []
    splits = []
    for i in range(1, len(term)):
        a, b = term[:i], term[i:]
        if len(a) >= 1 and len(b) >= 1:
            splits.append(f"{a} {b}")
    return splits


def _cjk_tokenize(text: str) -> list[str]:
    """Simple CJK-aware tokenizer: split on non-CJK boundaries."""
    tokens = []
    buf = ""
    for ch in text:
        if '一' <= ch <= '鿿':
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
    if buf:
        tokens.append(buf)
    return tokens


def _split_into_parts(total: float, operator: str, n: int) -> list[float]:
    """Split a total into n parts that sum to approximately total."""
    base = total / n
    parts = []
    for i in range(n):
        if i == n - 1:
            parts.append(round(total - sum(parts), 1))
        else:
            # Add some variation
            variation = base * (0.2 * i - 0.1)
            parts.append(round(base + variation, 1))
    # Ensure sum is close to total
    diff = total - sum(parts)
    if diff != 0:
        parts[-1] = round(parts[-1] + diff, 1)
    return parts


def _shift_value_outside_context(text: str, context_pattern: str,
                                  value: float, unit: str) -> Optional[str]:
    """Create a text variant where the numeric value appears outside the
    context_pattern match scope."""
    try:
        ctx_re = re.compile(context_pattern)
    except re.error:
        return None

    match = ctx_re.search(text)
    if not match:
        return None

    # Insert a compliant value after the context match to ensure PASS,
    # then put a violating value far away (before the context).
    before = text[:match.start()]
    after = text[match.end():]

    # Add violating value before the context
    violating_val = max(1, value - 1)
    shifted = f"其他条款约定{_fmt_val(violating_val)}{unit}。{before}{match.group()}{after}"
    return shifted
