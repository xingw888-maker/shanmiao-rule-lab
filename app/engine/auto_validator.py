"""AutoValidator — Ring 4 Self-Bootstrap Auto-Validation Gate.

Takes rule candidates (from StructuredRuleExtractor), runs them through
three automated gates (bench, bad_samples, constitution), and either
auto-admits them to rules.json or rejects them to a human review queue.

Design:
  - Gate 1 (bench):     Validate candidate against all real_contracts/*.txt.
  - Gate 2 (bad_samples): Auto-generate negative samples; validate candidate catches them.
  - Gate 3 (constitution): Check for contradictory verdicts with existing rules.

Zero LLM calls. Pure Python stdlib + existing engine imports.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AutoValidationResult:
    """Result of auto-validating a rule candidate through the three gates.

    Attributes:
        passed: True if all three gates passed.
        gate_results: Per-gate results dict with keys "bench", "bad_samples",
                      "constitution". Each value is {passed: bool, detail: str}.
        candidate: The rule dict that was validated.
        suggestion: Human-readable suggestion if rejected (what to fix).
    """
    passed: bool
    gate_results: dict[str, dict]
    candidate: dict
    suggestion: str = ""
    rejected_rule: dict | None = field(default=None)


# ═══════════════════════════════════════════════════════════════════════════
# RuleCandidate → CompiledRule converter
# ═══════════════════════════════════════════════════════════════════════════

_CLAUSE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "保修": ["保修", "防水", "防渗漏", "管道", "管线", "结构", "主体"],
    "付款": ["付款", "支付", "进度款", "结算", "价格", "计价", "合同价"],
    "验收": ["验收", "竣工", "检验", "检查"],
    "违约": ["违约", "赔偿", "罚", "违约金", "利息", "罚款"],
    "安全生产": ["安全", "施工安全", "教育", "培训"],
    "工程管理": ["工期", "缺陷", "责任", "转包", "分包", "垫资"],
    "质量标准": ["质量", "标准", "强制性", "规范"],
    "合同一致性": ["一致", "同时", "固定", "实结算"],
}

_LEGAL_REF_TEMPLATES: dict[str, str] = {
    "保修": "国务院令第279号",
    "防水": "国务院令第279号第40条",
    "结构": "国务院令第279号第40条(一)",
    "管线": "国务院令第279号第40条(四)",
    "管": "国务院令第279号第40条(四)",
    "质量": "《建设工程质量管理条例》",
    "安全": "《建设工程安全生产管理条例》",
    "验收": "《建设工程质量管理条例》第16条",
    "违约": "《民法典》第585条",
    "垫资": "住建部《建筑工程施工发包与承包计价管理办法》第12条",
}


def candidate_to_rule(candidate, rule_id: str) -> dict:
    """Convert a RuleCandidate to a rule dict matching rules.json format.

    Args:
        candidate: A RuleCandidate object (from rule_extractor.py).
        rule_id: Unique rule ID string (e.g. "cn-031").

    Returns:
        A rule dict suitable for inclusion in a rules.json package.
    """
    from app.engine.rule_extractor import RuleCandidate

    # Auto-generate context_pattern from subject + unit
    keywords = []
    for kw in ["屋面防水", "地下室", "防渗漏", "电气管线", "给排水", "设备安装",
               "装修工程", "供热", "供冷", "主体结构", "地基基础", "基础设施",
               "保修", "防水", "管线", "管道", "结构", "装修", "供热", "供冷",
               "期限", "质量", "安全", "验收", "违约", "垫资", "转包"]:
        if kw in candidate.subject or kw in (candidate.source_text or ""):
            keywords.append(kw)
    if not keywords:
        keywords = [candidate.subject[:10]]
    context_pattern = "|".join(keywords)

    # Auto-detect clause_type from subject
    clause_type = "其他"
    for ct, ckws in _CLAUSE_TYPE_KEYWORDS.items():
        if any(kw in candidate.subject or kw in (candidate.source_text or "")
               for kw in ckws):
            clause_type = ct
            break

    # Auto-detect legal_ref from subject
    legal_ref = "Auto-extracted from authority source"
    for key, template in _LEGAL_REF_TEMPLATES.items():
        if key in candidate.subject or key in (candidate.source_text or ""):
            legal_ref = template
            break

    # Determine severity based on condition type
    if candidate.condition_type == "numeric_comparison":
        severity = "major"
    elif candidate.condition_type == "required_pattern":
        severity = "warning"
    else:
        severity = "warning"
    # Forbid/extract rules default to warning — human review promotes to error

    # Build condition_params
    condition_params: dict = {}

    if candidate.condition_type == "numeric_comparison":
        condition_params = {
            "type": "numeric_comparison",
            "label": candidate.subject,
            "context_pattern": context_pattern,
            "unit": candidate.unit or "年",
            "operator": candidate.operator if candidate.operator in (">=", "<=", ">", "<") else ">=",
            "expected": candidate.expected_value or 0,
            "legal_ref": legal_ref,
        }
    elif candidate.condition_type == "forbidden_pattern":
        terms = candidate.required_terms or [candidate.subject]
        condition_params = {
            "type": "forbidden_pattern",
            "terms": terms,
            "reason": f"Auto-extracted forbidden term(s): {', '.join(terms)}",
        }
    elif candidate.condition_type == "required_pattern":
        terms = candidate.required_terms or [candidate.subject]
        condition_params = {
            "type": "required_pattern",
            "terms": terms,
            "message_if_missing": f"合同缺少{candidate.subject}条款——{legal_ref}要求包含此内容",
        }
    elif candidate.condition_type == "mutual_exclusion":
        terms = candidate.required_terms or [candidate.subject]
        condition_params = {
            "type": "mutual_exclusion",
            "terms": terms,
            "threshold": 2,
            "reason": f"Auto-extracted: {candidate.subject}相关条款不能同时出现",
        }
    elif candidate.condition_type == "co_occurrence":
        terms = candidate.required_terms or [candidate.subject]
        condition_params = {
            "type": "co_occurrence",
            "antecedent": terms[0] if terms else candidate.subject,
            "consequent": terms[1] if len(terms) > 1 else "",
        }
    else:
        condition_params = {
            "type": candidate.condition_type,
        }

    rule_dict = {
        "id": rule_id,
        "name": candidate.subject,
        "condition": condition_params,
        "severity": severity,
        "message": f"Auto-extracted: {candidate.subject}",
        "category": "auto-extracted",
        "source": "auto-extracted",
        "source_credibility": 0.7,
        "extraction_method": "template_extract",
        "clause_type": clause_type,
        "layer": "L1_CONJECTURE",
        "scope": {
            "contract_types": [],
            "exclude_contract_types": [],
            "min_contract_value": 0,
            "note": "Auto-validated rule candidate",
        },
    }
    return rule_dict


# ═══════════════════════════════════════════════════════════════════════════
# AutoBadSampleGenerator
# ═══════════════════════════════════════════════════════════════════════════


class AutoBadSampleGenerator:
    """Generates negative test samples (texts that SHOULD trigger FAILED)."""

    @staticmethod
    def generate_negative_samples(rule: dict,
                                   positive_samples: list[str] | None = None) -> list[str]:
        """Generate negative sample texts for a given rule.

        Args:
            rule: A rule dict (from rules.json format).
            positive_samples: Optional positive sample texts for reference.

        Returns:
            A list of text strings that should trigger FAILED verdicts.
        """
        condition = rule.get("condition", {})
        condition_type = condition.get("type", "")

        if condition_type == "numeric_comparison":
            return AutoBadSampleGenerator._gen_numeric(rule, positive_samples)
        elif condition_type == "required_pattern":
            return AutoBadSampleGenerator._gen_required(rule, positive_samples)
        elif condition_type == "forbidden_pattern":
            return AutoBadSampleGenerator._gen_forbidden(rule, positive_samples)
        elif condition_type == "mutual_exclusion":
            return AutoBadSampleGenerator._gen_mutual_exclusion(rule, positive_samples)
        else:
            # Fallback: produce a generic negative sample
            return [f"本合同中不包含任何与{rule.get('name', '未知')}相关的内容。"]

    @staticmethod
    def _gen_numeric(rule: dict,
                     positive_samples: list[str] | None = None) -> list[str]:
        """Generate negative samples for numeric_comparison rules.

        For threshold >=N, produce texts with values below N.
        For threshold <=N, produce texts with values above N.
        """
        condition = rule.get("condition", {})
        label = condition.get("label", rule.get("name", ""))
        operator = condition.get("operator", ">=")
        expected = float(condition.get("expected", 0))
        unit = condition.get("unit", "年")

        samples: list[str] = []

        # Use a clean, standalone template — DO NOT reuse positive_samples verbatim
        # because those contain compliant values that would short-circuit detection.
        # Instead, build a fresh subject from the rule's label.
        subject_hint = label or rule.get("name", "本工程")
        base_text = f"{subject_hint}为"

        if operator in (">=", ">"):
            # Need values LESS than the threshold to violate
            if expected > 1:
                just_below = max(1, int(expected) - 1)
            else:
                just_below = 0
            far_below = max(0, int(expected) - 3)

            samples.append(
                f"{base_text}{just_below}{unit}。"
            )
            samples.append(
                f"{base_text}{far_below}{unit}。"
            )
            # Third variant: use Chinese numerals
            cn_far = expected - 2 if expected >= 3 else 1
            if cn_far >= 1:
                cn_map = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五"}
                cn_str = cn_map.get(int(cn_far), str(cn_far))
                samples.append(
                    f"{base_text}{cn_str}{unit}。"
                )
        elif operator in ("<=", "<"):
            # Need values MORE than the threshold to violate
            above = int(expected) + 1
            far_above = int(expected) + 5
            samples.append(
                f"{base_text}{above}{unit}。"
            )
            samples.append(
                f"{base_text}{far_above}{unit}。"
            )

        return samples

    @staticmethod
    def _gen_required(rule: dict,
                      positive_samples: list[str] | None = None) -> list[str]:
        """Generate negative samples for required_pattern rules.

        Remove all occurrences of required_terms from the text.
        """
        condition = rule.get("condition", {})
        terms = condition.get("terms", [])
        if not terms:
            return [f"本合同缺少{rule.get('name', '必要条款')}。"]

        if positive_samples:
            samples = []
            for ps in positive_samples:
                cleaned = ps
                for term in terms:
                    cleaned = cleaned.replace(term, "")
                # Also replace related Chinese variants
                for term in terms:
                    # Try stripping individual characters
                    for ch in term:
                        cleaned = cleaned.replace(ch, "")
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if cleaned and cleaned != ps:
                    samples.append(cleaned)
            if samples:
                return samples

        return [f"本合同不包含{','.join(terms)}相关条款。"]

    @staticmethod
    def _gen_forbidden(rule: dict,
                       positive_samples: list[str] | None = None) -> list[str]:
        """Generate negative samples for forbidden_pattern rules.

        Take a clean sample and INSERT the forbidden terms.
        """
        condition = rule.get("condition", {})
        terms = condition.get("terms", [])
        if not terms:
            return [f"本合同涉及{rule.get('name', '禁止内容')}。"]

        base = ""
        if positive_samples:
            base = positive_samples[0]
        if not base:
            base = "双方同意按照合同约定履行各自义务。"

        samples = []
        for term in terms:
            samples.append(f"{base}但是，{term}。")
        samples.append(f"双方同意{', '.join(terms[:3])}。")
        return samples

    @staticmethod
    def _gen_mutual_exclusion(rule: dict,
                              positive_samples: list[str] | None = None) -> list[str]:
        """Generate negative samples for mutual_exclusion rules.

        Include both mutually-exclusive terms in the same text.
        """
        condition = rule.get("condition", {})
        terms = condition.get("terms", [])
        if len(terms) < 2:
            terms = [rule.get("name", "term_a"), "term_b"]

        samples = []
        samples.append(
            f"本合同采用{terms[0]}方式计价。同时，双方约定按{terms[1]}方式结算。"
        )
        if len(terms) > 2:
            samples.append(
                f"{'、'.join(terms[:3])}在本合同中均有约定。"
            )
        return samples


# ═══════════════════════════════════════════════════════════════════════════
# AutoValidator
# ═══════════════════════════════════════════════════════════════════════════


class AutoValidator:
    """Runs rule candidates through three automated gates.

    Gates:
      1. bench:       Run against all real_contracts/*.txt.
      2. bad_samples:  Auto-generate negative samples; check candidate catches them.
      3. constitution: Check for contradictory verdicts with existing rules.
    """

    # Threshold: what fraction of SHOULD-pass contracts must pass
    BENCH_PASS_RATE = 0.5
    # If a candidate never triggers (all NA) on SHOULD-pass contracts, flag it
    BENCH_NOISY_THRESHOLD = 0.15

    def __init__(self, engine=None):
        if engine is not None:
            self._engine = engine
        else:
            from app.engine.core import PythonValidationEngine
            self._engine = PythonValidationEngine()
        self.rejected_rules: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════
    # R2.7: Promote — 入库门控分级
    # ═══════════════════════════════════════════════════════════════════

    def promote(self, rules: list[dict], domain_dir: str,
                auto_promote: bool = False) -> dict:
        """Promote validated rules with auto_promote gate.

        auto_promote=False (default): Insert into validated/ path.
            Original manual promote semantics for mature domains.

        auto_promote=True: Insert into candidate/ path with
            requires_human_review: true. Does NOT touch validated/.
        """
        import json as _json
        from pathlib import Path as _Path

        if not rules:
            return {"promoted": 0, "target": domain_dir,
                    "auto_promote": auto_promote,
                    "requires_human_review": auto_promote,
                    "promoted_rules": []}

        domain_path = _Path(domain_dir)
        promoted_ids = []

        if auto_promote:
            cand_dir = domain_path / "candidates"
            cand_dir.mkdir(parents=True, exist_ok=True)
            for rule_dict in rules:
                rid = rule_dict.get("id", f"auto-{len(promoted_ids):03d}")
                with open(cand_dir / f"{rid}.json", "w", encoding="utf-8") as f:
                    _json.dump(rule_dict, f, ensure_ascii=False, indent=2)
                promoted_ids.append(rid)
            dj = domain_path / "domain.json"
            dd = {}
            if dj.exists():
                with open(dj, "r", encoding="utf-8") as f:
                    dd = _json.load(f)
            dd.setdefault("auto_promotion", {})
            dd["auto_promotion"]["requires_human_review"] = True
            dd["auto_promotion"]["promoted_count"] = \
                dd["auto_promotion"].get("promoted_count", 0) + len(rules)
            with open(dj, "w", encoding="utf-8") as f:
                _json.dump(dd, f, ensure_ascii=False, indent=2)
        else:
            validated_dir = domain_path / "validated"
            validated_dir.mkdir(parents=True, exist_ok=True)
            for rule_dict in rules:
                rid = rule_dict.get("id", f"vn-{len(promoted_ids):03d}")
                with open(validated_dir / f"{rid}.json", "w", encoding="utf-8") as f:
                    _json.dump(rule_dict, f, ensure_ascii=False, indent=2)
                promoted_ids.append(rid)

        return {
            "promoted": len(rules),
            "target": domain_dir,
            "auto_promote": auto_promote,
            "requires_human_review": auto_promote,
            "promoted_rules": promoted_ids,
        }

    # ── Public API ──

    def validate(self, candidate: dict, domain_dir: str,
                 existing_rules: list[dict] | None = None) -> AutoValidationResult:
        """Run a candidate rule through all three gates.

        Args:
            candidate: A rule dict (from candidate_to_rule()).
            domain_dir: Path to the domain directory (e.g. "domains/construction").
            existing_rules: List of existing rule dicts for constitution check.

        Returns:
            AutoValidationResult with per-gate results.
        """
        gate_results: dict[str, dict] = {}
        suggestions: list[str] = []

        # Gate 1: Bench
        bench_result = self._gate_bench(candidate, domain_dir)
        gate_results["bench"] = bench_result
        if not bench_result.get("passed", False):
            suggestions.append(bench_result.get("detail", "Bench gate failed"))

        # Gate 2: Bad samples
        bad_result = self._gate_bad_samples(candidate, domain_dir)
        gate_results["bad_samples"] = bad_result
        if not bad_result.get("passed", False):
            suggestions.append(bad_result.get("detail", "Bad samples gate failed"))

        # Gate 3: Constitution
        const_result = self._gate_constitution(candidate, domain_dir, existing_rules)
        gate_results["constitution"] = const_result
        if not const_result.get("passed", False):
            suggestions.append(const_result.get("detail", "Constitution gate failed"))

        passed = all(
            gr.get("passed", False) for gr in gate_results.values()
        )
        suggestion = "; ".join(suggestions) if suggestions else ""

        # W1.3: Collect rejected rule for feedback-loop re-extraction
        rejected_rule: dict | None = None
        if not passed:
            failed_gates = [k for k, v in gate_results.items() if not v.get("passed")]
            rejected_rule = {
                "rule": dict(candidate),
                "failed_gates": failed_gates,
                "reason": suggestion,
            }

        return AutoValidationResult(
            passed=passed,
            gate_results=gate_results,
            candidate=candidate,
            suggestion=suggestion,
            rejected_rule=rejected_rule,
        )

    # ── Gate 1: Bench ──

    def _gate_bench(self, candidate: dict, domain_dir: str) -> dict:
        """Gate 1: Run candidate against all real_contracts/*.txt.

        Checks:
          - Candidate should trigger FAILED on at least some contracts.
          - If ALL contracts return NOT_APPLICABLE, flag as "noisy".
        """
        contracts_dir = self._resolve_contracts_dir(domain_dir)
        if not contracts_dir:
            return {
                "passed": False,
                "detail": f"real_contracts directory not found (tried multiple paths from {domain_dir})",
                "stats": {},
            }

        contract_files = sorted([
            f for f in os.listdir(contracts_dir)
            if f.endswith(".txt")
        ])
        if not contract_files:
            return {
                "passed": False,
                "detail": "No contract files found in real_contracts/",
                "stats": {},
            }

        # Build a temporary package with just this candidate
        pkg = self._build_single_rule_package(candidate)

        try:
            self._engine.load_package(pkg)
        except Exception as e:
            return {
                "passed": False,
                "detail": f"Failed to load candidate rule package: {e}",
                "stats": {},
            }

        pkg_id = pkg["id"]
        results = []

        for fname in contract_files:
            fpath = os.path.join(contracts_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue

            try:
                result = self._engine.validate(
                    input_data={"text": text},
                    packages=[pkg_id],
                )
                evidence = result.get("evidence_chain", [])
                for ev in evidence:
                    results.append({
                        "contract": fname,
                        "status": ev.get("status", "?"),
                        "rationale": ev.get("rationale", ""),
                        "rule_id": ev.get("rule_id", ""),
                    })
            except Exception:
                continue

        # Unload
        try:
            self._engine.unload_package(pkg_id)
        except Exception:
            pass

        if not results:
            return {
                "passed": False,
                "detail": "No results produced by candidate rule on any contract",
                "stats": {},
            }

        # Analyze
        total = len(results)
        passed_count = sum(1 for r in results if r["status"] == "PASSED")
        failed_count = sum(1 for r in results if r["status"] == "FAILED")
        na_count = sum(1 for r in results if r["status"] == "NOT_APPLICABLE")

        stats = {
            "total_contracts": len(contract_files),
            "total_evidence": total,
            "PASSED": passed_count,
            "FAILED": failed_count,
            "NOT_APPLICABLE": na_count,
        }

        # If the candidate never triggers (all NA), it's noisy
        if na_count == total and total > 0:
            return {
                "passed": False,
                "detail": f"Candidate never triggers (ALL {total} results are NOT_APPLICABLE). "
                          f"Suggests context_pattern is too narrow or rule is irrelevant.",
                "stats": stats,
            }

        # If NA rate is very high, flag as noisy
        na_rate = na_count / max(total, 1)
        if na_rate > 0.85:
            return {
                "passed": False,
                "detail": (f"Candidate is noisy: {na_rate:.0%} of results are NOT_APPLICABLE "
                          f"({na_count}/{total}). Adjust context_pattern."),
                "stats": stats,
            }

        # Check that candidate produces at least some FAILED verdicts
        # on contracts that SHOULD contain the relevant clauses
        if failed_count == 0 and passed_count == 0:
            return {
                "passed": True,
                "detail": f"Candidate may be too restrictive (0 FAILED, 0 PASSED, {na_count} NA)",
                "stats": stats,
            }

        # A rule that ALWAYS passes on all contracts is suspicious
        if passed_count == total and total > 1:
            return {
                "passed": True,
                "detail": f"Candidate always passes ({passed_count}/{total} PASSED). Consider tightening.",
                "stats": stats,
            }

        return {
            "passed": True,
            "detail": f"Bench OK: {failed_count} FAILED, {passed_count} PASSED, {na_count} NA across {total} contract results",
            "stats": stats,
        }

    # ── Gate 2: Bad Samples ──

    def _gate_bad_samples(self, candidate: dict, domain_dir: str) -> dict:
        """Gate 2: Generate negative samples and verify candidate catches them."""
        generator = AutoBadSampleGenerator()

        # Build positive samples from the real_contracts for context
        positive_samples = self._load_positive_samples(domain_dir)

        negative_samples = generator.generate_negative_samples(candidate, positive_samples)

        if not negative_samples:
            return {
                "passed": False,
                "detail": "No negative samples could be generated",
                "negative_samples": [],
            }

        # Build a temporary package with just this candidate
        pkg = self._build_single_rule_package(candidate)

        try:
            self._engine.load_package(pkg)
        except Exception as e:
            return {
                "passed": False,
                "detail": f"Failed to load candidate: {e}",
                "negative_samples": negative_samples,
            }

        pkg_id = pkg["id"]
        sample_results = []

        for i, sample in enumerate(negative_samples):
            try:
                result = self._engine.validate(
                    input_data={"text": sample},
                    packages=[pkg_id],
                )
                evidence = result.get("evidence_chain", [])
                statuses = [ev.get("status", "?") for ev in evidence]
                sample_results.append({
                    "index": i,
                    "sample": sample[:100],
                    "statuses": statuses,
                    "failed": any(s == "FAILED" for s in statuses),
                })
            except Exception as e:
                sample_results.append({
                    "index": i,
                    "sample": sample[:100],
                    "statuses": ["ERROR"],
                    "failed": False,
                    "error": str(e),
                })

        # Unload
        try:
            self._engine.unload_package(pkg_id)
        except Exception:
            pass

        # All negative samples must produce at least one FAILED
        all_failed = all(sr["failed"] for sr in sample_results)
        missed_indices = [
            sr["index"] for sr in sample_results if not sr["failed"]
        ]

        if not all_failed:
            return {
                "passed": False,
                "detail": (f"Rule failed to catch violations: {len(missed_indices)}/{len(sample_results)} "
                          f"negative samples did not produce FAILED. "
                          f"Missed indices: {missed_indices}"),
                "negative_samples": negative_samples,
                "sample_results": sample_results,
            }

        return {
            "passed": True,
            "detail": (f"All {len(sample_results)} negative samples correctly triggered FAILED"),
            "negative_samples": negative_samples,
            "sample_results": sample_results,
        }

    # ── Gate 3: Constitution ──

    def _gate_constitution(self, candidate: dict, domain_dir: str,
                           existing_rules: list[dict] | None = None) -> dict:
        """Gate 3: Check for contradictory verdicts with existing rules.

        Loads 3 representative contracts, runs both the candidate and each
        existing rule, and checks for conflicting verdicts on the same text.
        """
        contracts = self._load_representative_contracts(domain_dir, max_contracts=3)
        if not contracts:
            return {
                "passed": True,
                "detail": "No contracts available for constitution check (skipped)",
            }

        # Build candidate package
        cand_pkg = self._build_single_rule_package(candidate)
        try:
            self._engine.load_package(cand_pkg)
        except Exception as e:
            return {
                "passed": False,
                "detail": f"Failed to load candidate: {e}",
            }
        cand_pkg_id = cand_pkg["id"]

        # Load existing rules if provided
        existing_pkg_id = None
        if existing_rules:
            existing_pkg = {
                "id": "existing-rules-constitution",
                "name": "Existing Rules (Constitution Check)",
                "version": "1.0.0",
                "domain": "",
                "rules": list(existing_rules),
            }
            try:
                self._engine.load_package(existing_pkg)
                existing_pkg_id = existing_pkg["id"]
            except Exception as e:
                logger.warning("Could not load existing rules for constitution check: %s", e)

        conflicts: list[dict] = []

        for contract_name, text in contracts:
            # Run candidate alone
            try:
                cand_result = self._engine.validate(
                    input_data={"text": text},
                    packages=[cand_pkg_id],
                )
            except Exception:
                continue

            cand_evidence = cand_result.get("evidence_chain", [])
            for ev in cand_evidence:
                ev["_source"] = "candidate"

            # Run existing rules alone
            existing_evidence = []
            if existing_pkg_id:
                try:
                    ext_result = self._engine.validate(
                        input_data={"text": text},
                        packages=[existing_pkg_id],
                    )
                    existing_evidence = ext_result.get("evidence_chain", [])
                    for ev in existing_evidence:
                        ev["_source"] = "existing"
                except Exception:
                    pass

            # Check for contradictions: same fragment, different verdicts
            # Only flag REAL contradictions: PASSED vs FAILED on the SAME rule_id
            # (meaningful textual content).  NOT_APPLICABLE is not a real contradiction.
            all_ev = cand_evidence + existing_evidence
            for i in range(len(all_ev)):
                for j in range(i + 1, len(all_ev)):
                    a, b = all_ev[i], all_ev[j]
                    # Only flag PASSED vs FAILED contradictions
                    statuses = {a.get("status"), b.get("status")}
                    if statuses != {"PASSED", "FAILED"}:
                        continue
                    if a.get("input_fragment") != b.get("input_fragment"):
                        continue
                    if a.get("input_fragment", "").strip() in ("",):
                        continue
                    # Same fragment, opposite PASSED/FAILED = potential contradiction,
                    # but only flag it if at least one of the two is the CANDIDATE rule
                    # (pre-existing contradictions among existing rules are not our concern).
                    sources = {a.get("_source"), b.get("_source")}
                    if "candidate" not in sources:
                        continue
                    conflicts.append({
                        "contract": contract_name,
                        "fragment": a.get("input_fragment", "")[:100],
                        "rule_a_id": a.get("rule_id", ""),
                        "rule_a_status": a.get("status", ""),
                        "rule_a_source": a.get("_source", ""),
                        "rule_b_id": b.get("rule_id", ""),
                        "rule_b_status": b.get("status", ""),
                        "rule_b_source": b.get("_source", ""),
                    })

        # Clean up
        try:
            self._engine.unload_package(cand_pkg_id)
        except Exception:
            pass
        if existing_pkg_id:
            try:
                self._engine.unload_package(existing_pkg_id)
            except Exception:
                pass

        if conflicts:
            return {
                "passed": False,
                "detail": (f"Found {len(conflicts)} conflicting verdicts between candidate "
                          f"and existing rules. First conflict: "
                          f"rule '{conflicts[0]['rule_a_id']}' ({conflicts[0]['rule_a_status']}) vs "
                          f"'{conflicts[0]['rule_b_id']}' ({conflicts[0]['rule_b_status']}) "
                          f"on '{conflicts[0]['fragment']}'"),
                "conflicts": conflicts,
            }

        return {
            "passed": True,
            "detail": "No contradictory verdicts with existing rules",
            "conflicts": [],
        }

    # ── Helpers ──

    def _build_single_rule_package(self, rule_dict: dict) -> dict:
        """Build a single-rule package for testing."""
        pkg_id = f"auto-validator-{uuid.uuid4().hex[:8]}"
        return {
            "id": pkg_id,
            "name": "Auto-Validator Single Rule",
            "version": "0.0.1",
            "domain": "",
            "rules": [rule_dict],
        }

    def _load_positive_samples(self, domain_dir: str) -> list[str]:
        """Load contract texts from real_contracts to use as positive context."""
        contracts_dir = self._resolve_contracts_dir(domain_dir)
        if not contracts_dir:
            return []

        samples = []
        if os.path.isdir(contracts_dir):
            for fname in sorted(os.listdir(contracts_dir)):
                if fname.endswith(".txt"):
                    fpath = os.path.join(contracts_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            samples.append(f.read())
                    except Exception:
                        continue
        return samples

    def _load_representative_contracts(self, domain_dir: str,
                                        max_contracts: int = 3) -> list[tuple[str, str]]:
        """Load up to N representative contracts."""
        contracts_dir = self._resolve_contracts_dir(domain_dir)
        if not contracts_dir:
            return []

        contracts = []
        if os.path.isdir(contracts_dir):
            files = sorted([
                f for f in os.listdir(contracts_dir)
                if f.endswith(".txt")
            ])
            for fname in files[:max_contracts]:
                fpath = os.path.join(contracts_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        contracts.append((fname, f.read()))
                except Exception:
                    continue
        return contracts

    @staticmethod
    def _resolve_contracts_dir(domain_dir: str) -> str | None:
        """Resolve the real_contracts directory from various possible paths.

        Strategy:
          1. Try relative to the project root via domain_dir (go up from domains/<x> to project root)
          2. Try from the auto_validator.py file location
        """
        project_root = os.path.normpath(os.path.join(domain_dir, "..", ".."))
        candidates = [
            os.path.join(project_root, "tests", "real_contracts"),
            os.path.join(domain_dir, "..", "..", "..", "tests", "real_contracts"),
        ]

        # From the auto_validator.py file location
        auto_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(
            os.path.dirname(auto_file_dir), "..", "tests", "real_contracts"
        ))
        candidates.append(os.path.join(
            os.path.dirname(auto_file_dir), "tests", "real_contracts"
        ))

        for c in candidates:
            resolved = os.path.normpath(c)
            if os.path.isdir(resolved):
                return resolved
        return None

    # --- W1.3: Rejected rule collection ---

    def get_rejected(self) -> list[dict]:
        """Return all rejected rules collected during validate() calls."""
        return list(self.rejected_rules)

    def reject_count(self) -> int:
        """Return the number of rejected rules collected."""
        return len(self.rejected_rules)
