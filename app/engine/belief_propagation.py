"""Belief Propagation Network — Pearl's sum-product algorithm on a factor graph
of validation rules.  Replaces linear credibility weighting with probabilistic
message passing, allowing high-credibility rules to strengthen their neighbors
and low-credibility rules to be tempered by evidence from related rules.

Zero external dependencies.  Pure Python implementation of the sum-product
algorithm for discrete binary variables (PASS / FAIL).
"""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class FactorNode:
    """A compatibility function (factor) between connected variable nodes.

    The potential table maps joint assignments of the connected rules to
    a compatibility score.  Higher scores mean more compatible assignments.
    """
    factor_id: str
    connected_rules: list[str]          # rule IDs this factor connects
    potential: dict[tuple, float] = field(default_factory=dict)
    edge_type: str = ""                 # "SHARES_ENTITY" | "SAME_CATEGORY" | "LOGICAL"
    weight: float = 1.0                 # factor strength

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "connected_rules": self.connected_rules,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "potential_keys": [
                {f"rule_{self.connected_rules[i]}": "PASS" if v == 0 else "FAIL"
                 for i, v in enumerate(k)}
                for k in list(self.potential.keys())[:8]
            ],
        }


@dataclass
class BeliefState:
    """Marginal belief for a single rule (variable node)."""
    rule_id: str
    prior_credibility: float          # from SourceProfile (0–1)
    posterior_credibility: float      # after belief propagation (0–1)
    belief_pass: float                # P(rule is truly PASSED | evidence)
    belief_fail: float                # P(rule is truly FAILED | evidence)
    observed: Optional[str] = None    # "PASSED" | "FAILED" if evidence-clamped
    neighbors: list[str] = field(default_factory=list)

    @property
    def entropy(self) -> float:
        """Shannon entropy of the belief distribution."""
        p, q = self.belief_pass, self.belief_fail
        if p <= 0 or q <= 0:
            return 0.0
        return -(p * math.log2(p) + q * math.log2(q))

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "prior_credibility": round(self.prior_credibility, 4),
            "posterior_credibility": round(self.posterior_credibility, 4),
            "belief_pass": round(self.belief_pass, 4),
            "belief_fail": round(self.belief_fail, 4),
            "entropy": round(self.entropy, 4),
            "observed": self.observed,
            "neighbors": self.neighbors,
        }


@dataclass
class BeliefReport:
    """Complete report after belief propagation."""
    rule_beliefs: dict[str, BeliefState]
    iteration_count: int
    converged: bool
    max_delta: float                 # max belief change in final iteration
    credibility_adjustments: dict[str, float]  # rule_id -> delta
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_beliefs": {rid: b.to_dict() for rid, b in self.rule_beliefs.items()},
            "iteration_count": self.iteration_count,
            "converged": self.converged,
            "max_delta": self.max_delta,
            "credibility_adjustments": {
                rid: round(v, 4) for rid, v in self.credibility_adjustments.items()
            },
            "summary": self.summary,
        }


# ======================================================================
# BeliefNetwork — sum-product belief propagation
# ======================================================================

class BeliefNetwork:
    """Factor-graph belief propagation for rule credibility.

    Variable nodes represent rules (binary state: PASS/FAIL truth).
    Factor nodes represent compatibility between connected rules.
    Prior beliefs come from source_credibility.  Evidence (observed
    verdicts) clamps the corresponding variable nodes.

    The sum-product algorithm iteratively passes messages:
    - Variable → Factor: product of incoming messages from other factors
    - Factor → Variable: sum over neighbor states, weighted by potential
    """

    # State indices
    PASS = 0
    FAIL = 1
    N_STATES = 2

    def __init__(self, max_iterations: int = 50,
                 convergence_threshold: float = 0.001,
                 damping: float = 0.5):
        """Initialize the belief network.

        Args:
            max_iterations: Maximum message-passing iterations.
            convergence_threshold: Stop when max belief change < this.
            damping: Message damping factor (0=none, 1=full damping).
                     Higher damping prevents oscillation.
        """
        self._max_iter = max_iterations
        self._threshold = convergence_threshold
        self._damping = damping
        self._variables: dict[str, BeliefState] = {}
        self._factors: list[FactorNode] = []
        # Messages: (src_type, src_id, dst_id) → [PASS_prob, FAIL_prob]
        # src_type: "V" (variable) or "F" (factor)
        self._messages: dict[tuple[str, str, str], list[float]] = {}
        # Factor → Variable adjacency
        self._factor_adj: dict[str, list[str]] = defaultdict(list)
        # Variable → Factor adjacency
        self._var_factors: dict[str, list[str]] = defaultdict(list)

    # ── Construction ──────────────────────────────────────────────────

    def build_network(self, rules: list[dict],
                      source_profiles: dict[str, Any] | None = None,
                      graph=None) -> None:
        """Create the factor graph from rules.

        Args:
            rules: List of rule dicts (as in rules.json).
            source_profiles: Optional dict of rule_id → SourceProfile (or any
                             object with a 'credibility' attribute).
            graph: Optional RuleGraph instance for edge discovery.
                   If not provided, edges are built from rule metadata only.
        """
        self._clear()

        # Step 1: create variable nodes
        for r in rules:
            rid = r["id"]
            prior = r.get("source_credibility", 0.5)
            if source_profiles and rid in source_profiles:
                prof = source_profiles[rid]
                if hasattr(prof, 'credibility'):
                    prior = prof.credibility
                elif isinstance(prof, dict) and 'credibility' in prof:
                    prior = prof['credibility']
            prior = max(0.01, min(0.99, prior))
            self._variables[rid] = BeliefState(
                rule_id=rid,
                prior_credibility=prior,
                posterior_credibility=prior,
                belief_pass=prior,
                belief_fail=1.0 - prior,
            )

        # Step 2: discover edges
        edges = self._discover_edges(rules, graph)

        # Step 3: create factor nodes for each pair of related rules
        for i, (a_id, b_id, etype, weight) in enumerate(edges):
            fid = f"F_{i:04d}"
            factor = FactorNode(
                factor_id=fid,
                connected_rules=[a_id, b_id],
                edge_type=etype,
                weight=weight,
            )
            # Initialize potentials based on edge type
            factor.potential = self._init_potentials(a_id, b_id, etype, weight)
            self._factors.append(factor)
            self._factor_adj[fid] = [a_id, b_id]
            self._var_factors[a_id].append(fid)
            self._var_factors[b_id].append(fid)
            self._variables[a_id].neighbors.append(b_id)
            self._variables[b_id].neighbors.append(a_id)

        # Step 4: initialize messages to uniform
        self._init_messages()

    # ── Inference ─────────────────────────────────────────────────────

    def propagate(self, evidence: dict[str, str]) -> BeliefReport:
        """Run sum-product belief propagation.

        Args:
            evidence: dict mapping rule_id → "PASSED" | "FAILED" | "NOT_APPLICABLE".
                      Rules with PASSED/FAILED are clamped.  NOT_APPLICABLE rules
                      are treated as unobserved.

        Returns:
            BeliefReport with posterior beliefs and credibility adjustments.
        """
        # Step 1: clamp observed nodes
        for rid, verdict in evidence.items():
            if rid in self._variables and verdict in ("PASSED", "FAILED"):
                self._variables[rid].observed = verdict
                # Clamp belief to observed state
                if verdict == "PASSED":
                    self._variables[rid].belief_pass = 0.99
                    self._variables[rid].belief_fail = 0.01
                else:
                    self._variables[rid].belief_pass = 0.01
                    self._variables[rid].belief_fail = 0.99

        # Step 2: iterative message passing
        converged = False
        max_delta = float('inf')
        for it in range(self._max_iter):
            old_beliefs = {
                rid: (v.belief_pass, v.belief_fail)
                for rid, v in self._variables.items()
            }

            # Variable → Factor messages
            for rid, var in self._variables.items():
                if var.observed:
                    # Clamped: send deterministic message
                    for fid in self._var_factors.get(rid, []):
                        msg_key = ("V", rid, fid)
                        if var.observed == "PASSED":
                            self._messages[msg_key] = [0.99, 0.01]
                        else:
                            self._messages[msg_key] = [0.01, 0.99]
                else:
                    for fid in self._var_factors.get(rid, []):
                        msg = self._compute_var_to_factor(rid, fid)
                        self._messages[("V", rid, fid)] = msg

            # Factor → Variable messages
            for factor in self._factors:
                for rid in factor.connected_rules:
                    msg = self._compute_factor_to_var(factor.factor_id, rid)
                    self._messages[("F", factor.factor_id, rid)] = msg

            # Update marginals
            max_delta = self._update_marginals(old_beliefs)

            if max_delta < self._threshold:
                converged = True
                break

        # Step 3: build report
        adjustments: dict[str, float] = {}
        for rid, var in self._variables.items():
            posterior = (var.belief_pass * var.prior_credibility +
                         var.belief_fail * (1 - var.prior_credibility))
            posterior = max(0.01, min(0.99, posterior))
            var.posterior_credibility = posterior
            if not var.observed:
                delta = posterior - var.prior_credibility
                adjustments[rid] = delta

        iters_str = f"{self._max_iter} (未收敛)" if not converged else "已收敛"
        lines = [
            f"信念传播完成: {len(self._variables)} 节点, {len(self._factors)} 因子",
            f"迭代: {iters_str}, max_delta={max_delta:.6f}",
            f"收敛: {'是' if converged else '否'}",
            f"可信度调整: {len(adjustments)} 条规则被更新",
        ]

        return BeliefReport(
            rule_beliefs=dict(self._variables),
            iteration_count=self._max_iter if not converged else 0,
            converged=converged,
            max_delta=max_delta,
            credibility_adjustments=adjustments,
            summary="\n".join(lines),
        )

    def marginal_belief(self, rule_id: str) -> Optional[BeliefState]:
        """Get posterior belief for a rule after propagation."""
        return self._variables.get(rule_id)

    def calibrate_credibility(self, rules: list[dict],
                               historical_runs: list[dict[str, str]]) -> dict[str, float]:
        """Use historical validation runs to learn empirical factor potentials.

        For each run, compare observed co-occurrence patterns of verdicts
        between connected rules.  Adjust factor potentials to match empirical
        frequencies.

        Args:
            rules: Rule dicts.
            historical_runs: List of {rule_id: verdict, ...} from past runs.

        Returns:
            Updated source_credibility values for each rule.
        """
        if not historical_runs:
            return {r["id"]: r.get("source_credibility", 0.5) for r in rules}

        # Count co-occurrence patterns
        pair_counts: dict[tuple[str, str], dict[tuple, int]] = defaultdict(
            lambda: defaultdict(int))
        for run in historical_runs:
            for factor in self._factors:
                if len(factor.connected_rules) != 2:
                    continue
                a_id, b_id = factor.connected_rules
                a_v = 0 if run.get(a_id) == "PASSED" else 1
                b_v = 0 if run.get(b_id) == "PASSED" else 1
                pair_counts[(a_id, b_id)][(a_v, b_v)] += 1

        # Update potentials based on empirical frequencies
        for factor in self._factors:
            if len(factor.connected_rules) != 2:
                continue
            a_id, b_id = factor.connected_rules
            counts = pair_counts.get((a_id, b_id), {})
            total = sum(counts.values())
            if total > 0:
                new_pot = {}
                for s_a in range(self.N_STATES):
                    for s_b in range(self.N_STATES):
                        cnt = counts.get((s_a, s_b), 0)
                        # Laplace smoothing
                        new_pot[(s_a, s_b)] = (cnt + 1) / (total + self.N_STATES ** 2)
                factor.potential = new_pot

        # Compute empirical credibility
        rule_pass_rate: dict[str, float] = {}
        for rid in [r["id"] for r in rules]:
            passes = sum(1 for run in historical_runs if run.get(rid) == "PASSED")
            total = len(historical_runs)
            rule_pass_rate[rid] = passes / total if total > 0 else 0.5

        return {
            rid: 0.7 * rules_dict.get(rid, {}).get("source_credibility", 0.5) + 0.3 * rate
            for rid, rate in rule_pass_rate.items()
            for rules_dict in [{r["id"]: r for r in rules}]
        }

    # ── Internal: message computation ─────────────────────────────────

    def _compute_var_to_factor(self, var_id: str, target_fid: str) -> list[float]:
        """Variable → Factor message: product of incoming messages from
        all OTHER factors connected to this variable."""
        msg = [1.0, 1.0]
        has_input = False
        for fid in self._var_factors.get(var_id, []):
            if fid == target_fid:
                continue
            in_msg = self._messages.get(("F", fid, var_id), [0.5, 0.5])
            msg[0] *= in_msg[0]
            msg[1] *= in_msg[1]
            has_input = True
        if not has_input:
            var = self._variables[var_id]
            msg = [var.belief_pass, var.belief_fail]
        # Normalize
        total = msg[0] + msg[1]
        if total > 0:
            msg = [msg[0] / total, msg[1] / total]
        else:
            msg = [0.5, 0.5]
        return msg

    def _compute_factor_to_var(self, factor_id: str, target_var: str) -> list[float]:
        """Factor → Variable message: sum over neighbor states (except target),
        weighted by the factor potential and incoming variable messages."""
        factor = self._get_factor(factor_id)
        if factor is None or len(factor.connected_rules) != 2:
            return [0.5, 0.5]

        other_var = (factor.connected_rules[0]
                     if factor.connected_rules[1] == target_var
                     else factor.connected_rules[1])
        other_msg = self._messages.get(("V", other_var, factor_id), [0.5, 0.5])

        msg = [0.0, 0.0]
        for s_tgt in range(self.N_STATES):
            for s_other in range(self.N_STATES):
                key = ((s_tgt, s_other) if factor.connected_rules[0] == target_var
                       else (s_other, s_tgt))
                pot = factor.potential.get(key, 1.0 / self.N_STATES ** 2)
                msg[s_tgt] += pot * other_msg[s_other]

        # Normalize
        total = sum(msg)
        if total > 0:
            msg = [m / total for m in msg]
        else:
            msg = [0.5, 0.5]

        # Damping
        old_key = ("F", factor_id, target_var)
        if old_key in self._messages and self._damping > 0:
            old = self._messages[old_key]
            msg = [
                self._damping * old[i] + (1 - self._damping) * msg[i]
                for i in range(self.N_STATES)
            ]

        return msg

    def _update_marginals(self, old_beliefs: dict[str, tuple[float, float]]) -> float:
        """Update all variable beliefs from current factor→variable messages.
        Returns the max absolute change across all beliefs."""
        max_delta = 0.0
        for rid, var in self._variables.items():
            if var.observed:
                continue
            belief = [1.0, 1.0]
            has_input = False
            for fid in self._var_factors.get(rid, []):
                in_msg = self._messages.get(("F", fid, rid), [0.5, 0.5])
                belief[0] *= in_msg[0]
                belief[1] *= in_msg[1]
                has_input = True
            if not has_input:
                continue
            total = belief[0] + belief[1]
            if total > 0:
                belief = [belief[0] / total, belief[1] / total]
            else:
                belief = [0.5, 0.5]

            old = old_beliefs.get(rid, (0.5, 0.5))
            delta = max(abs(belief[0] - old[0]), abs(belief[1] - old[1]))
            max_delta = max(max_delta, delta)

            var.belief_pass = belief[0]
            var.belief_fail = belief[1]
        return max_delta

    # ── Internal: initialization ──────────────────────────────────────

    def _init_potentials(self, a_id: str, b_id: str,
                          etype: str, weight: float) -> dict[tuple, float]:
        """Initialize factor potential table based on edge type.

        The potential phi(s_a, s_b) encodes how compatible states s_a and s_b are.
        Higher values = more compatible.
        """
        # Base: uniform
        pot = {}
        for s_a in range(self.N_STATES):
            for s_b in range(self.N_STATES):
                pot[(s_a, s_b)] = 1.0 / self.N_STATES ** 2

        # Bias based on edge type
        if etype == "SHARES_ENTITY":
            # Rules sharing entities tend to have same verdict
            bias = 0.45 * weight
            pot[(self.PASS, self.PASS)] += bias
            pot[(self.FAIL, self.FAIL)] += bias
            pot[(self.PASS, self.FAIL)] -= bias / 2
            pot[(self.FAIL, self.PASS)] -= bias / 2
        elif etype == "SAME_CATEGORY":
            # Same category: moderate same-verdict tendency
            bias = 0.30 * weight
            pot[(self.PASS, self.PASS)] += bias
            pot[(self.FAIL, self.FAIL)] += bias
            pot[(self.PASS, self.FAIL)] -= bias / 3
            pot[(self.FAIL, self.PASS)] -= bias / 3
        elif etype == "LOGICAL_CHAIN":
            # Logical dependency: if predecessor FAILS, successor likely FAILS
            # But successor can PASS while predecessor FAILS
            bias = 0.35 * weight
            pot[(self.FAIL, self.FAIL)] += bias
            pot[(self.PASS, self.PASS)] += bias * 0.5
            pot[(self.FAIL, self.PASS)] -= bias * 0.3
        elif etype == "CLAUSE_TYPE_CHAIN":
            # Same clause type: weak same-verdict tendency
            bias = 0.20 * weight
            pot[(self.PASS, self.PASS)] += bias
            pot[(self.FAIL, self.FAIL)] += bias

        # Normalize
        total = sum(pot.values())
        return {k: v / total for k, v in pot.items()}

    def _discover_edges(self, rules: list[dict],
                         graph=None) -> list[tuple[str, str, str, float]]:
        """Discover edges between rules.

        Uses RuleGraph if provided, otherwise builds edges from metadata.
        Returns list of (rule_id_a, rule_id_b, edge_type, weight).
        """
        if graph is not None:
            # Use existing RuleGraph
            edges = []
            for e in graph.get_edges():
                edges.append((e.source_id, e.target_id, e.edge_type, e.weight))
            return edges

        # Fallback: build from rule metadata
        edges = []
        rule_map = {r["id"]: r for r in rules}
        rule_ids = list(rule_map.keys())

        for i in range(len(rule_ids)):
            for j in range(i + 1, len(rule_ids)):
                a_id, b_id = rule_ids[i], rule_ids[j]
                a, b = rule_map[a_id], rule_map[b_id]

                # Same category
                if a.get("category") and a["category"] == b.get("category"):
                    edges.append((a_id, b_id, "SAME_CATEGORY", 0.5))
                    edges.append((b_id, a_id, "SAME_CATEGORY", 0.5))

                # Same clause_type
                if a.get("clause_type") and a["clause_type"] == b.get("clause_type"):
                    edges.append((a_id, b_id, "CLAUSE_TYPE_CHAIN", 0.4))
                    edges.append((b_id, a_id, "CLAUSE_TYPE_CHAIN", 0.4))

        return edges

    def _init_messages(self) -> None:
        """Initialize all messages to uniform [0.5, 0.5]."""
        for rid in self._variables:
            for fid in self._var_factors.get(rid, []):
                self._messages[("V", rid, fid)] = [0.5, 0.5]
        for factor in self._factors:
            for rid in factor.connected_rules:
                self._messages[("F", factor.factor_id, rid)] = [0.5, 0.5]

    def _get_factor(self, factor_id: str) -> Optional[FactorNode]:
        for f in self._factors:
            if f.factor_id == factor_id:
                return f
        return None

    def _clear(self) -> None:
        self._variables.clear()
        self._factors.clear()
        self._messages.clear()
        self._factor_adj.clear()
        self._var_factors.clear()
