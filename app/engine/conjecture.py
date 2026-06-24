"""Conjecture Miner — statistical pattern discovery from validation history.
Uses association rule learning (support/confidence/lift) to discover
potential new rules from accumulated evidence chains.
Three types of conjectures:
1. CO_OCCURRENCE_SUGGESTION — "term A appears, term B always follows" (but no rule captures this)
2. MUTUAL_EXCLUSION_SUGGESTION — "term A and term B never co-occur" across all runs (no rule yet)
3. RULE_TRIGGER_PATTERN — "when rules X,Y,Z all fire, rule W also fires" (rule dependency)
Conjectures are NOT verdicts. They are statistical hints that require human review
before being upgraded to formal rules. Marked as status CONJECTURE, never auto-applied.
"""
import math
import re as _re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class TermStats:
    """Statistics for a single term across all validation runs."""
    term: str
    total_runs: int
    present_count: int
    presence_rate: float
    co_occurrence: dict = field(default_factory=dict)

@dataclass
class Association:
    """An association rule: X → Y."""
    antecedent: str
    consequent: str
    co_occurrence_count: int
    antecedent_count: int
    consequent_count: int
    total_runs: int
    @property
    def support(self) -> float:
        return self.co_occurrence_count / max(self.total_runs, 1)
    @property
    def confidence(self) -> float:
        return self.co_occurrence_count / max(self.antecedent_count, 1)
    @property
    def lift(self) -> float:
        p_y = self.consequent_count / max(self.total_runs, 1)
        if p_y == 0:
            return float("inf") if self.confidence > 0 else 0
        return self.confidence / p_y
    @property
    def is_significant(self) -> bool:
        return (
            self.support >= 0.10
            and self.lift >= 2.0
            and self.co_occurrence_count >= 3
        )

@dataclass
class Conjecture:
    """A suggested rule discovered via statistical mining."""
    conjecture_id: str
    conjecture_type: str
    antecedent: str
    consequent: str
    support: float
    confidence: float
    lift: float
    co_occurrence_count: int
    total_runs: int
    suggested_rule_name: str = ""
    suggested_rule_type: str = ""
    suggested_severity: str = "warning"
    suggested_terms: list = field(default_factory=list)
    suggested_message: str = ""
    suggested_threshold: int = 2
    source_evidence_ids: list = field(default_factory=list)

# ═══════════════════════════════════════════════════════════════════════
# Conjecture Miner
# ═══════════════════════════════════════════════════════════════════════
class ConjectureMiner:
    """Discovers potential new rules from historical validation evidence.
    Now includes outer-region cross analysis for L3_OUTER_POSSIBILITY discoveries.
    """
    MIN_SUPPORT = 0.10
    MIN_CONFIDENCE = 0.85
    MIN_LIFT = 2.0
    MIN_COOCCURRENCE = 3
    MAX_CONJECTURES = 20

    def mine(self, historical_runs):
        """Analyze historical validation runs and return discovered conjectures."""
        if len(historical_runs) < self.MIN_COOCCURRENCE:
            return []
        total = len(historical_runs)
        # Step 1: Extract term sets
        run_sets = []
        for run in historical_runs:
            evidence = run.get("evidence_chain", [])
            terms_failed = set()
            terms_all = set()
            rules_failed = set()
            for ev in evidence:
                matched = ev.get("matched_terms", [])
                for t in matched:
                    t = t.lower().strip()
                    if t:
                        terms_all.add(t)
                        if ev.get("status") == "FAILED":
                            terms_failed.add(t)
                if ev.get("status") == "FAILED":
                    rules_failed.add(ev.get("rule_id", ""))
            run_sets.append({"terms_failed": terms_failed, "terms_all": terms_all, "rules_failed": rules_failed})
        # Step 2: Build term statistics
        term_stats = {}
        for ts in run_sets:
            for term in ts["terms_all"]:
                if term not in term_stats:
                    term_stats[term] = TermStats(term=term, total_runs=total, present_count=0, presence_rate=0.0)
                term_stats[term].present_count += 1
        for t, s in term_stats.items():
            s.presence_rate = s.present_count / total
        # Step 2b: Build co-occurrence matrix
        for ts in run_sets:
            terms = list(ts["terms_all"])
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    a, b = terms[i], terms[j]
                    term_stats[a].co_occurrence[b] = term_stats[a].co_occurrence.get(b, 0) + 1
                    term_stats[b].co_occurrence[a] = term_stats[b].co_occurrence.get(a, 0) + 1
        # Step 3: Find positive associations
        associations = []
        for term_a, stats_a in term_stats.items():
            for term_b, co_count in stats_a.co_occurrence.items():
                if term_a >= term_b:
                    continue
                if term_b not in term_stats:
                    continue
                stats_b = term_stats[term_b]
                asc = Association(antecedent=term_a, consequent=term_b, co_occurrence_count=co_count,
                                  antecedent_count=stats_a.present_count, consequent_count=stats_b.present_count,
                                  total_runs=total)
                if asc.is_significant:
                    associations.append(asc)
        # Step 4: Find negative associations
        negative_associations = []
        for term_a, stats_a in term_stats.items():
            if stats_a.presence_rate < self.MIN_SUPPORT:
                continue
            for term_b, stats_b in term_stats.items():
                if term_a >= term_b:
                    continue
                if stats_b.presence_rate < self.MIN_SUPPORT:
                    continue
                co_count = stats_a.co_occurrence.get(term_b, 0)
                if co_count == 0:
                    negative_associations.append(Association(
                        antecedent=term_a, consequent=term_b, co_occurrence_count=0,
                        antecedent_count=stats_a.present_count, consequent_count=stats_b.present_count,
                        total_runs=total))
        # Step 5: Build Conjecture objects
        conjectures = []
        for asc in sorted(associations, key=lambda a: -a.lift):
            conj = Conjecture(
                conjecture_id=f"CONJ_{uuid.uuid4().hex[:8]}",
                conjecture_type="co_occurrence_suggestion",
                antecedent=asc.antecedent, consequent=asc.consequent,
                support=round(asc.support, 4), confidence=round(asc.confidence, 4), lift=round(asc.lift, 2),
                co_occurrence_count=asc.co_occurrence_count, total_runs=asc.total_runs,
                suggested_rule_name=f"自动发现: {asc.antecedent} ⇒ {asc.consequent}",
                suggested_rule_type="co_occurrence", suggested_terms=[asc.antecedent, asc.consequent],
                suggested_message=(
                    f"统计发现：在 {asc.total_runs} 次校验中，当 '{asc.antecedent}' 出现时，"
                    f"'{asc.consequent}' 在 {asc.confidence*100:.0f}% 的情况下同时出现 "
                    f"（独立出现率 {asc.consequent_count/asc.total_runs*100:.0f}%，lift={asc.lift:.1f}）。"
                    f"建议确认是否应建立共存规则。"),
            )
            conjectures.append(conj)
        for asc in negative_associations:
            if len(conjectures) >= self.MAX_CONJECTURES:
                break
            conj = Conjecture(
                conjecture_id=f"CONJ_{uuid.uuid4().hex[:8]}",
                conjecture_type="mutual_exclusion_suggestion",
                antecedent=asc.antecedent, consequent=asc.consequent,
                support=0.0, confidence=0.0, lift=float("inf"),
                co_occurrence_count=0, total_runs=asc.total_runs,
                suggested_rule_name=f"自动发现: {asc.antecedent} ⟂ {asc.consequent}",
                suggested_rule_type="mutual_exclusion",
                suggested_terms=[asc.antecedent, asc.consequent], suggested_threshold=2,
                suggested_message=(
                    f"统计发现：'{asc.antecedent}'（出现 {asc.antecedent_count} 次）"
                    f"和 '{asc.consequent}'（出现 {asc.consequent_count} 次）"
                    f"在 {asc.total_runs} 次校验中从未同时出现。"
                    f"建议确认是否应建立互斥规则。"),
            )
            conjectures.append(conj)
        return conjectures[:self.MAX_CONJECTURES]

    # ── Outer-region cross analysis ──
    def mine_outer_region(self, text, evidence_chain, existing_rule_terms=None):
        """Analyze text regions NOT covered by any triggered rule.

        Returns:
            (regressed_conjectures, outer_discoveries)
            - regressed_conjectures: relate to known rules → should be considered for upgrade
            - outer_discoveries: no rule overlap → L3_OUTER_POSSIBILITY, kept for review
        """
        existing_terms = existing_rule_terms or set()
        covered_terms = set()
        covered_fragments = []
        for ev in evidence_chain:
            for t in ev.get("matched_terms", []):
                covered_terms.add(t.lower().strip())
            frag = ev.get("input_fragment", "")
            if frag:
                covered_fragments.append(frag)

        sentences = _re.split(r'[。；\n;]', text)
        uncovered_regions = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 8:
                continue
            covered = False
            for cf in covered_fragments:
                cf_clean = cf.strip("...")
                if cf_clean[:30] in sent or sent[:30] in cf_clean:
                    covered = True
                    break
            if not covered:
                uncovered_regions.append(sent)

        if len(uncovered_regions) < 2:
            return [], []

        term_re = _re.compile(r'[一-鿿]{2,8}|[a-zA-Z_]{3,20}')
        uncovered_term_counter = {}
        for region in uncovered_regions:
            for t in term_re.findall(region):
                t = t.lower().strip()
                if len(t) >= 2:
                    uncovered_term_counter[t] = uncovered_term_counter.get(t, 0) + 1

        regressed_conjectures = []
        outer_discoveries = []

        for term, count in uncovered_term_counter.items():
            if count < 2:
                continue
            overlap = bool(term in covered_terms or term in existing_terms)
            partial = any((term in et or et in term) for et in covered_terms | existing_terms if len(et) > 1)

            if overlap or partial:
                target_terms = [et for et in covered_terms | existing_terms if len(et) > 1 and (term in et or et in term)]
                conj = Conjecture(
                    conjecture_id=f"OUTREG_{uuid.uuid4().hex[:8]}",
                    conjecture_type="outer_region_discovery",
                    antecedent=term, consequent=target_terms[0] if target_terms else "",
                    support=count / max(len(uncovered_regions), 1),
                    confidence=0.5 if partial else 0.8,
                    lift=count / max(len(uncovered_regions) * 0.1, 1),
                    co_occurrence_count=count, total_runs=len(uncovered_regions),
                    suggested_rule_name=f"规则外回归: {term} -> 已有规则",
                    suggested_rule_type="co_occurrence", suggested_terms=[term] + target_terms[:2],
                    suggested_message=(
                        f"规则外发现：术语「{term}」在未覆盖区域出现 {count} 次，"
                        f"与已有规则术语 {'/'.join(target_terms[:3])} 相关。"
                        f"建议扩充已有规则以覆盖此术语。"),
                    source_evidence_ids=uncovered_regions[:3],
                )
                regressed_conjectures.append(conj)
            else:
                outer_discoveries.append({
                    "discovery_id": f"OUTER_{uuid.uuid4().hex[:8]}",
                    "term": term, "occurrences": count,
                    "total_uncovered_regions": len(uncovered_regions),
                    "presence_rate": round(count / len(uncovered_regions), 3),
                    "layer": "L3_OUTER_POSSIBILITY",
                    "description": (
                        f"规则外发现：术语「{term}」在 {len(uncovered_regions)} 个"
                        f"未覆盖文本段中出现 {count} 次，"
                        f"与任何已加载规则均无关联。建议人工审查是否需新建规则。"),
                    "sample_regions": [r[:200] for r in uncovered_regions if term in r.lower()][:3],
                })

        regressed_conjectures.sort(key=lambda c: -c.lift)
        return regressed_conjectures[:self.MAX_CONJECTURES], outer_discoveries[:self.MAX_CONJECTURES * 2]

    def upgrade_to_rule(self, conjecture):
        """Convert a human-confirmed conjecture into a formal Rule dict."""
        rule_type = conjecture.suggested_rule_type
        if rule_type == "co_occurrence":
            condition = {"type": "co_occurrence", "antecedent": conjecture.antecedent, "consequent": conjecture.consequent}
        elif rule_type == "mutual_exclusion":
            condition = {"type": "mutual_exclusion", "terms": conjecture.suggested_terms, "threshold": conjecture.suggested_threshold}
        else:
            condition = {"type": "forbidden_pattern", "pattern": f"(?i)({'|'.join(conjecture.suggested_terms)})"}
        return {
            "id": f"UPGRADED_{conjecture.conjecture_id}",
            "name": conjecture.suggested_rule_name,
            "condition": condition,
            "severity": conjecture.suggested_severity,
            "message": conjecture.suggested_message,
            "category": f"conjecture.upgraded.{rule_type}",
        }
