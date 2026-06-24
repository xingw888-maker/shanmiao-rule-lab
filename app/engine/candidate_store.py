"""CandidatePrototypeStore — manages candidate (auto-bootstrapped) domain prototypes.

Candidate domains are separate from validated ones.  They have lower coverage
thresholds and are probed only after the validated store fails to find a match.

This module reuses the `LexicalPrototype` class from `app/engine/lexical_prototype.py`
without modifying it.  Classification logic mirrors `LexicalPrototypeStore.classify()`
but uses candidate-appropriate thresholds and returns scores with source="candidate".

Usage:
    store = CandidatePrototypeStore.load("data/candidate_prototypes.json")
    matches = store.classify("some text")  # -> [(domain_id, coverage, threshold), ...]
    store.register("new_domain", bigrams, sample_count=1)
    similar = store.find_similar(some_bigrams, jaccard_threshold=0.5)
    store.merge("source_id", "target_id")
    store.remove("promoted_domain")
    store.save("data/candidate_prototypes.json")

Admission gates (three-gate control for candidate -> validated promotion):
    Gate 1 - Identity: domain.maintainer must be non-empty
    Gate 2 - Quality: candidate rules must have source_credibility >= 0.7
    Gate 3 - Source: rule legal_hierarchy must be in trusted-source whitelist
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

from app.engine.lexical_prototype import LexicalPrototype

# Trusted source whitelist for Gate 3 (source credibility)
_HIERARCHY_TRUSTED: set[str] = {
    "宪法", "law", "admin_regulation", "dept_rule",
    "GB标准", "GB_std", "行标", "trade_std",
}
_HIERARCHY_UNTRUSTED: set[str] = {
    "other", "未知", "unknown", "个人", "personal",
    "AI生成", "ai_generated", "网页", "web", "待核实", "tbc",
}


class CandidatePrototypeStore:
    DOMINANCE_RATIO = 2.0

    def __init__(self):
        self.prototypes: dict[str, LexicalPrototype] = {}
        self._usage_stats: dict[str, dict] = {}

    # --- Persistence ---

    @classmethod
    def load(cls, path: str) -> "CandidatePrototypeStore":
        store = cls()
        if not os.path.isfile(path):
            return store
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for domain_id, d in data.items():
            if "domain_id" not in d:
                d["domain_id"] = domain_id
            if "coverage_threshold" not in d:
                d["coverage_threshold"] = 0.01
            if "combined_threshold" not in d:
                d["combined_threshold"] = 0.001
            store.prototypes[domain_id] = LexicalPrototype(
                domain_id=d["domain_id"],
                bigrams=set(d.get("bigrams", [])),
                coverage_threshold=d.get("coverage_threshold", 0.01),
                combined_threshold=d.get("combined_threshold", 0.001),
                sample_count=d.get("sample_count", 0),
            )
            if "usage_stats" in d:
                store._usage_stats[domain_id] = d["usage_stats"]
        return store

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {}
        for did, p in self.prototypes.items():
            entry = p.to_dict()
            if did in self._usage_stats:
                entry["usage_stats"] = self._usage_stats[did]
            payload[did] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # --- Classification ---

    def classify(self, text: str) -> list[tuple[str, float, float]]:
        if not self.prototypes:
            return []
        from app.engine.lexical_prototype import _bigrams_of as _extract_bigrams
        text_bigrams = _extract_bigrams(text)
        scored: list[tuple[str, float, float, float]] = []
        for proto in self.prototypes.values():
            cov, comb = proto.score(text_bigrams)
            scored.append((proto.domain_id, cov, comb, proto.coverage_threshold))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[1])
        above = [(d, c, th) for d, c, _cb, th in scored if c >= th]
        if not above:
            return []
        top_dom, top_cov, top_th = above[0]
        sec_cov = above[1][1] if len(above) > 1 else 0.0
        if top_cov >= sec_cov * self.DOMINANCE_RATIO and sec_cov > 0:
            return [(top_dom, top_cov, top_th)]
        if sec_cov == 0:
            return [(top_dom, top_cov, top_th)]
        return [(d, c, th) for d, c, th in above]

    # --- Registration ---

    def register(self, domain_id: str, bigrams: set[str], sample_count: int = 1) -> LexicalPrototype:
        if domain_id in self.prototypes:
            raise ValueError(f"Candidate domain '{domain_id}' already registered")
        if sample_count > 1 and len(bigrams) > 0:
            coverage_th = self._calibrate_threshold(bigrams, sample_count)
        else:
            coverage_th = 0.01
        combined_th = max(0.001, round(coverage_th * 0.5 * coverage_th, 4))
        coverage_th = max(0.005, round(coverage_th, 4))
        proto = LexicalPrototype(
            domain_id=domain_id, bigrams=bigrams,
            coverage_threshold=coverage_th, combined_threshold=combined_th,
            sample_count=sample_count,
        )
        self.prototypes[domain_id] = proto
        return proto

    @staticmethod
    def _calibrate_threshold(bigrams: set[str], sample_count: int) -> float:
        if sample_count <= 1 or len(bigrams) == 0:
            return 0.01
        sorted_b = sorted(bigrams)
        group_size = max(1, len(sorted_b) // sample_count)
        groups = [set(sorted_b[i:i+group_size]) for i in range(0, len(sorted_b), group_size)]
        coverages: list[float] = []
        for i in range(len(groups)):
            loo = set()
            for j in range(len(groups)):
                if j != i:
                    loo |= groups[j]
            if len(groups[i]) > 0:
                cov = len(groups[i] & loo) / len(groups[i])
                coverages.append(cov)
        if len(coverages) < 2:
            return 0.01
        cov_mean = sum(coverages) / len(coverages)
        cov_var = sum((x - cov_mean) ** 2 for x in coverages) / len(coverages)
        cov_std = math.sqrt(cov_var)
        return max(0.005, round(cov_mean - 2 * cov_std, 4))

    # --- Similarity search ---

    def find_similar(self, bigrams: set[str], jaccard_threshold: float = 0.5) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for proto in self.prototypes.values():
            jac = proto.jaccard(bigrams)
            if jac >= jaccard_threshold:
                scored.append((proto.domain_id, round(jac, 4)))
        scored.sort(key=lambda x: -x[1])
        return scored

    # --- Merge ---

    def merge(self, source_id: str, target_id: str) -> None:
        if source_id not in self.prototypes:
            raise KeyError(f"Source domain '{source_id}' not found")
        if target_id not in self.prototypes:
            raise KeyError(f"Target domain '{target_id}' not found")
        source = self.prototypes[source_id]
        target = self.prototypes[target_id]
        target.bigrams |= source.bigrams
        target.sample_count += source.sample_count
        merged_th = self._calibrate_threshold(target.bigrams, target.sample_count)
        target.coverage_threshold = merged_th
        target.combined_threshold = max(0.001, round(merged_th * 0.5 * merged_th, 4))
        del self.prototypes[source_id]

    # --- Removal ---

    def remove(self, domain_id: str) -> None:
        if domain_id not in self.prototypes:
            raise KeyError(f"Candidate domain '{domain_id}' not found")
        del self.prototypes[domain_id]

    # --- Listing ---

    def list_domains(self) -> list[str]:
        return sorted(self.prototypes.keys())

    # --- Utility ---

    def match_scores(self, text: str) -> dict[str, dict]:
        from app.engine.lexical_prototype import _bigrams_of as _extract_bigrams
        text_bigrams = _extract_bigrams(text)
        return {
            p.domain_id: {
                "coverage": round(p.coverage(text_bigrams), 4),
                "combined": round(p.score(text_bigrams)[1], 4),
                "pass_": p.coverage(text_bigrams) >= p.coverage_threshold,
                "threshold": p.coverage_threshold,
            }
            for p in self.prototypes.values()
        }

    # --- Usage statistics ---

    def add_sample(self, domain_id: str, verdict: str) -> None:
        if domain_id not in self._usage_stats:
            self._usage_stats[domain_id] = {
                "count": 0, "pass_": 0, "fail": 0,
                "first_seen": None, "last_seen": None,
            }
        import datetime
        now = datetime.datetime.now().isoformat()
        stats = self._usage_stats[domain_id]
        stats["count"] += 1
        stats["last_seen"] = now
        if stats["first_seen"] is None:
            stats["first_seen"] = now
        if verdict in ("PASSED", "ALL_PASSED"):
            stats["pass_"] += 1
        elif verdict in ("FAILED", "CONFLICTED"):
            stats["fail"] += 1

    def usage_stats(self, domain_id: str) -> dict:
        stats = self._usage_stats.get(domain_id, {}).copy()
        total = stats.get("count", 0)
        stats["pass_rate"] = round(stats.get("pass_", 0) / total, 3) if total > 0 else 0.0
        return stats

    # --- Three-gate admission control ---

    @staticmethod
    def _check_gate_1_identity(domain_meta: dict) -> tuple[bool, str]:
        maintainer = domain_meta.get("maintainer", "").strip()
        if not maintainer:
            return False, "Gate 1 (identity): maintainer is empty or missing"
        return True, f"Gate 1 (identity): maintainer={maintainer}"

    @staticmethod
    def _check_gate_2_quality(rules_list: list[dict]) -> tuple[bool, str]:
        if not rules_list:
            return False, "Gate 2 (quality): candidate domain has no rules"
        credible = [r for r in rules_list if r.get("source_credibility", 0) >= 0.7]
        if not credible:
            return False, (
                f"Gate 2 (quality): no rules with source_credibility >= 0.7 "
                f"(total rules: {len(rules_list)})"
            )
        return True, (
            f"Gate 2 (quality): {len(credible)}/{len(rules_list)} "
            "rules have source_credibility >= 0.7"
        )

    @staticmethod
    def _check_gate_3_source(rules_list: list[dict]) -> tuple[bool, str]:
        trusted: list[str] = []
        untrusted: list[str] = []
        missing: list[str] = []
        for rule in rules_list:
            lh = rule.get("legal_hierarchy", None)
            rid = rule.get("id", "?")
            if lh is None:
                missing.append(rid)
            elif lh in _HIERARCHY_UNTRUSTED:
                untrusted.append(f"{rid}({lh})")
            elif lh in _HIERARCHY_TRUSTED:
                trusted.append(f"{rid}({lh})")
        if not trusted:
            detail = []
            if missing:
                detail.append(f"{len(missing)} rules missing legal_hierarchy")
            if untrusted:
                detail.append(f"{len(untrusted)} untrusted: {', '.join(untrusted[:3])}")
            return False, f"Gate 3 (source): no trusted legal_hierarchy -- {', '.join(detail)}"
        return True, (
            f"Gate 3 (source): {len(trusted)}/{len(rules_list)} "
            f"rules in trusted sources ({', '.join(trusted[:3])})"
        )

    # --- Promotion (W1.2) ---

    def promote(self, domain_id: str, domain_meta: dict, rules: list[dict]) -> dict:
        """Three-gate admission control for candidate -> validated promotion.

        Runs Gate 1 (identity), Gate 2 (quality), Gate 3 (source) in sequence.
        Returns {passed, gate_results, reason}.

        Caller is responsible for moving the domain directory from candidate/
        to validated/ on success, then calling remove_from_candidate().
        """
        g1_ok, g1_reason = self._check_gate_1_identity(domain_meta)
        g2_ok, g2_reason = self._check_gate_2_quality(rules)
        g3_ok, g3_reason = self._check_gate_3_source(rules)

        gate_results = {
            "identity": {"passed": g1_ok, "detail": g1_reason},
            "quality":  {"passed": g2_ok, "detail": g2_reason},
            "source":   {"passed": g3_ok, "detail": g3_reason},
        }
        passed = g1_ok and g2_ok and g3_ok

        reason_parts: list[str] = []
        if not g1_ok:
            reason_parts.append(f"G1: {g1_reason}")
        if not g2_ok:
            reason_parts.append(f"G2: {g2_reason}")
        if not g3_ok:
            reason_parts.append(f"G3: {g3_reason}")

        return {
            "domain_id": domain_id,
            "passed": passed,
            "gate_results": gate_results,
            "reason": "; ".join(reason_parts) if reason_parts else "All three gates passed",
        }

    def remove_from_candidate(self, domain_id: str, store_path: str) -> None:
        """Remove a promoted domain from the candidate store and persist.

        Args:
            domain_id: The domain to remove from candidate store.
            store_path: Path to the candidate store JSON file to save to.
        """
        if domain_id in self.prototypes:
            del self.prototypes[domain_id]
        self._usage_stats.pop(domain_id, None)
        self.save(store_path)

    def suggest_promotion(
        self,
        domain_id: str,
        domain_meta: Optional[dict] = None,
        rules_list: Optional[list[dict]] = None,
    ) -> dict:
        stats = self.usage_stats(domain_id)
        count = stats.get("count", 0)
        pass_rate = stats.get("pass_rate", 0.0)

        if count < 10 or pass_rate <= 0.8:
            return {
                "suggested": False,
                "pass_rate": pass_rate, "count": count,
                "reason": (
                    f"Usage threshold not met: count={count}/10, "
                    f"pass_rate={pass_rate}/0.80"
                ),
            }

        g1_ok, g1_reason = self._check_gate_1_identity(domain_meta or {})
        if not g1_ok:
            return {
                "suggested": False,
                "pass_rate": pass_rate, "count": count,
                "gates": {"identity": False, "quality": False, "source": False},
                "reason": g1_reason,
            }

        rl = rules_list or []
        g2_ok, g2_reason = self._check_gate_2_quality(rl)
        if not g2_ok:
            return {
                "suggested": False,
                "pass_rate": pass_rate, "count": count,
                "gates": {"identity": True, "quality": False, "source": False},
                "reason": g2_reason,
            }

        g3_ok, g3_reason = self._check_gate_3_source(rl)
        gates = {"identity": True, "quality": True, "source": g3_ok}
        return {
            "suggested": g3_ok,
            "pass_rate": pass_rate, "count": count,
            "gates": gates,
            "reason": g3_reason,
        }
