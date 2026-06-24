"""Rule Dependency Graph Engine — builds a directed graph of rule relationships,
computes transitive closure, and performs causal chain analysis.

When multiple rules fail on a contract, this engine determines which failures
are root causes and which are downstream consequences.  Zero external dependencies.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class RuleNode:
    """A node in the rule dependency graph."""
    rule_id: str
    rule_name: str
    condition_type: str
    category: str
    clause_type: str
    entities: set[str] = field(default_factory=set)
    terms: list[str] = field(default_factory=list)
    source_credibility: float = 0.5
    severity: str = "warning"

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "condition_type": self.condition_type,
            "category": self.category,
            "clause_type": self.clause_type,
            "entities": sorted(self.entities),
            "terms": self.terms,
            "source_credibility": self.source_credibility,
            "severity": self.severity,
        }


@dataclass
class RuleEdge:
    """A directed edge between two rules in the dependency graph."""
    source_id: str
    target_id: str
    edge_type: str   # "SHARES_ENTITY" | "SAME_CATEGORY" | "LOGICAL_CHAIN" | "CLAUSE_TYPE_CHAIN"
    weight: float    # 0.0–1.0, strength of the dependency
    evidence: str    # human-readable justification for this edge

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "evidence": self.evidence,
        }


@dataclass
class CausalReport:
    """Causal analysis for a single failed rule."""
    rule_id: str
    rule_name: str
    root_causes: list[str]       # rule IDs that are the root cause
    consequences: list[str]      # rule IDs downstream of this failure
    dependency_chain: list[tuple[str, str, str]]  # (from, to, edge_type)
    graph_mermaid: str           # Mermaid-compatible graph for visualization
    classification: str          # "root_cause" | "consequence" | "isolated"
    summary: str

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "root_causes": self.root_causes,
            "consequences": self.consequences,
            "dependency_chain": [
                {"from": f, "to": t, "edge_type": e}
                for f, t, e in self.dependency_chain
            ],
            "graph_mermaid": self.graph_mermaid,
            "classification": self.classification,
            "summary": self.summary,
        }


# ======================================================================
# RuleGraph — the main algorithm class
# ======================================================================

class RuleGraph:
    """Builds and analyzes a dependency graph of validation rules.

    Rules are connected when they share entities (extracted from context_pattern
    and terms fields), belong to the same category, have the same clause_type,
    or when one rule's entities appear in another rule's condition parameters.

    Usage:
        graph = RuleGraph()
        graph.build_from_rules(rules_list)
        graph.transitive_closure()

        # Analyze a single failed rule
        report = graph.find_causal_chain("cn-001", verdicts)

        # Classify all failed rules
        classification = graph.dependency_resolve(verdicts)
    """

    # ── Regex to extract CJK entities from regex patterns ──
    _RE_CJK_RUN = re.compile(r'[一-鿿]{2,8}')
    _RE_REGEX_META = re.compile(r'[.+*?^${}()|\[\]\\]')

    # ── Logical chain patterns: terms in one rule's name that suggest
    #     dependency on another rule's domain ──
    _LOGICAL_CHAIN_HINTS = {
        "保修": ["质量保修", "保修书", "保修期", "保修期限", "保修金"],
        "付款": ["付款比例", "付款期限", "付款节点", "付款逾期", "工程款"],
        "违约金": ["违约金上限", "违约金基数", "违约金比例", "逾期违约金"],
        "验收": ["竣工验收", "验收程序", "验收标准", "验收报告"],
        "争议": ["争议解决", "管辖法院", "协商前置", "仲裁"],
        "质保": ["质量保修书", "质保期", "质保金", "保修"],
        "担保": ["履约担保", "投标保证金", "担保比例"],
    }

    def __init__(self):
        self._nodes: dict[str, RuleNode] = {}
        self._edges: list[RuleEdge] = []
        # adjacency: node_id -> set of (target_id, edge_type, weight)
        self._adj: dict[str, set[tuple[str, str, float]]] = defaultdict(set)
        # reverse adjacency for backward traversal
        self._rev_adj: dict[str, set[tuple[str, str, float]]] = defaultdict(set)
        # transitive closure: node_id -> set of reachable node_ids
        self._closure: dict[str, set[str]] = {}

    # ── Construction ──────────────────────────────────────────────────

    def build_from_rules(self, rules: list[dict]) -> None:
        """Build the dependency graph from a list of rule dicts (as in rules.json).

        Args:
            rules: List of rule dicts, each with id, name, condition, category,
                   clause_type, source_credibility, severity fields.
        """
        self._clear()

        # Step 1: create nodes
        for r in rules:
            cond = r.get("condition", {})
            entities = self._extract_entities(cond)
            terms = cond.get("terms", []) if isinstance(cond.get("terms"), list) else []
            node = RuleNode(
                rule_id=r["id"],
                rule_name=r.get("name", r["id"]),
                condition_type=cond.get("type", ""),
                category=r.get("category", ""),
                clause_type=r.get("clause_type", ""),
                entities=entities,
                terms=terms,
                source_credibility=r.get("source_credibility", 0.5),
                severity=r.get("severity", "warning"),
            )
            self._nodes[r["id"]] = node

        # Step 2: create edges
        node_ids = list(self._nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a_id, b_id = node_ids[i], node_ids[j]
                a, b = self._nodes[a_id], self._nodes[b_id]
                edges = self._find_edges(a, b)
                for e in edges:
                    self._add_edge(e)

    def transitive_closure(self) -> None:
        """Compute full reachability via BFS from every node.

        After this call, self._closure[node_id] contains all rule IDs
        reachable from node_id through directed paths.
        """
        self._closure.clear()
        for node_id in self._nodes:
            reachable = set()
            queue = deque([node_id])
            while queue:
                current = queue.popleft()
                for target_id, _etype, _w in self._adj.get(current, set()):
                    if target_id not in reachable and target_id != node_id:
                        reachable.add(target_id)
                        queue.append(target_id)
            self._closure[node_id] = reachable

    # ── Analysis ──────────────────────────────────────────────────────

    def find_causal_chain(self, failed_rule_id: str,
                          verdicts: dict[str, str]) -> CausalReport:
        """Given a FAILED rule, trace backward to find root causes and forward
        to find consequences among other FAILED rules.

        Args:
            failed_rule_id: The rule ID that FAILED.
            verdicts: Mapping of rule_id -> "PASSED" | "FAILED" | "NOT_APPLICABLE".

        Returns:
            CausalReport with root causes, consequences, and Mermaid graph.
        """
        node = self._nodes.get(failed_rule_id)
        if node is None:
            return CausalReport(
                rule_id=failed_rule_id, rule_name="",
                root_causes=[], consequences=[], dependency_chain=[],
                graph_mermaid="", classification="unknown",
                summary=f"Rule {failed_rule_id} not found in graph.",
            )

        failed_set = {rid for rid, v in verdicts.items() if v == "FAILED"}
        if failed_rule_id not in failed_set:
            failed_set.add(failed_rule_id)

        # Step 1: find predecessors (rules that point TO this one) among FAILED
        root_causes = []
        dep_chain = []
        visited = set()
        queue = deque([failed_rule_id])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            predecessors = [
                (src, etype, w) for src, etype, w in self._rev_adj.get(current, set())
                if src in failed_set and src != current
            ]
            if not predecessors:
                # No failed predecessor — this is a root cause (or the target itself)
                if current != failed_rule_id or not self._has_failed_predecessor(current, failed_set):
                    if current not in root_causes:
                        root_causes.append(current)
            else:
                for src, etype, w in predecessors:
                    dep_chain.append((src, current, etype))
                    if src not in visited:
                        queue.append(src)

        # Step 2: find consequences — forward BFS from failed_rule_id
        consequences = []
        visited_fwd = {failed_rule_id}
        queue_fwd = deque([failed_rule_id])
        while queue_fwd:
            current = queue_fwd.popleft()
            for target, etype, _w in self._adj.get(current, set()):
                if target in failed_set and target not in visited_fwd:
                    consequences.append(target)
                    visited_fwd.add(target)
                    queue_fwd.append(target)

        # Step 3: classification
        if not root_causes and not consequences:
            classification = "isolated"
        elif failed_rule_id in root_causes:
            classification = "root_cause"
        else:
            classification = "consequence"

        # Step 4: build Mermaid graph
        highlight = {failed_rule_id} | set(root_causes) | set(consequences)
        mermaid = self._mermaid_subgraph(highlight, failed_set)

        # Step 5: summary
        node_name = node.rule_name or failed_rule_id
        parts = []
        if root_causes:
            rc_names = [self._nodes[r].rule_name if r in self._nodes else r for r in root_causes]
            parts.append(f"根因: {', '.join(rc_names)}")
        if consequences:
            cs_names = [self._nodes[c].rule_name if c in self._nodes else c for c in consequences]
            parts.append(f"导致: {', '.join(cs_names)}")
        if classification == "isolated":
            parts.append("独立失败，无因果关联")

        return CausalReport(
            rule_id=failed_rule_id,
            rule_name=node_name,
            root_causes=root_causes,
            consequences=consequences,
            dependency_chain=dep_chain,
            graph_mermaid=mermaid,
            classification=classification,
            summary="；".join(parts) if parts else "分析完成",
        )

    def dependency_resolve(self, verdicts: dict[str, str]) -> dict[str, str]:
        """Classify all FAILED rules as 'root_cause', 'consequence', or 'isolated'.

        Returns:
            Dict mapping rule_id -> classification.
        """
        failed_set = {rid for rid, v in verdicts.items() if v == "FAILED"}
        classification: dict[str, str] = {}

        for rid in failed_set:
            has_failed_pred = self._has_failed_predecessor(rid, failed_set)
            has_failed_succ = self._has_failed_successor(rid, failed_set)

            if not has_failed_pred and not has_failed_succ:
                classification[rid] = "isolated"
            elif not has_failed_pred:
                classification[rid] = "root_cause"
            else:
                classification[rid] = "consequence"

        return classification

    # ── Graph introspection ───────────────────────────────────────────

    def get_node(self, rule_id: str) -> Optional[RuleNode]:
        """Return the RuleNode for a given rule ID."""
        return self._nodes.get(rule_id)

    def get_edges(self) -> list[RuleEdge]:
        """Return all edges in the graph."""
        return list(self._edges)

    def get_neighbors(self, rule_id: str) -> list[tuple[str, str, float]]:
        """Return all outgoing edges from a node."""
        return sorted(self._adj.get(rule_id, set()), key=lambda x: -x[2])

    def get_predecessors(self, rule_id: str) -> list[tuple[str, str, float]]:
        """Return all incoming edges to a node."""
        return sorted(self._rev_adj.get(rule_id, set()), key=lambda x: -x[2])

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def is_reachable(self, source_id: str, target_id: str) -> bool:
        """Check if target_id is reachable from source_id via any path."""
        return target_id in self._closure.get(source_id, set())

    def to_dict(self) -> dict:
        """Serialize the full graph."""
        return {
            "nodes": {rid: n.to_dict() for rid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
        }

    def mermaid_full(self) -> str:
        """Generate a full Mermaid flowchart of the entire graph."""
        lines = ["graph TD"]
        seen_edges = set()
        for e in self._edges:
            key = (e.source_id, e.target_id)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            style = _MERMAID_STYLE.get(e.edge_type, "-->")
            src_label = self._nodes[e.source_id].rule_name if e.source_id in self._nodes else e.source_id
            tgt_label = self._nodes[e.target_id].rule_name if e.target_id in self._nodes else e.target_id
            lines.append(f'    {_safe_id(e.source_id)}["{src_label}"] {style} {_safe_id(e.target_id)}["{tgt_label}"]')
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────

    def _clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._adj.clear()
        self._rev_adj.clear()
        self._closure.clear()

    def _extract_entities(self, condition: dict) -> set[str]:
        """Extract CJK entity terms from condition parameters."""
        entities: set[str] = set()

        # From context_pattern (regex pattern — extract CJK runs)
        pattern = condition.get("context_pattern", "")
        if pattern:
            entities.update(self._RE_CJK_RUN.findall(pattern))

        # From terms list
        terms = condition.get("terms", [])
        if isinstance(terms, list):
            for t in terms:
                t_clean = str(t).strip()
                if t_clean:
                    entities.add(t_clean)

        # From label
        label = condition.get("label", "")
        if label:
            entities.update(self._RE_CJK_RUN.findall(label))

        # Filter: remove single chars, keep 2-8 char CJK runs
        return {e for e in entities if 2 <= len(e) <= 8}

    def _find_edges(self, a: RuleNode, b: RuleNode) -> list[RuleEdge]:
        """Find all edges between two nodes."""
        edges: list[RuleEdge] = []

        # SHARES_ENTITY: overlapping entity sets
        shared = a.entities & b.entities
        if shared:
            jaccard = len(shared) / len(a.entities | b.entities) if a.entities | b.entities else 0
            weight = min(1.0, jaccard * 2.0)  # amplify Jaccard
            evidence = f"共享实体: {', '.join(sorted(shared)[:5])}"
            # Direction: bi-directional
            edges.append(RuleEdge(a.rule_id, b.rule_id, "SHARES_ENTITY", weight, evidence))
            edges.append(RuleEdge(b.rule_id, a.rule_id, "SHARES_ENTITY", weight, evidence))

        # SAME_CATEGORY: identical category field
        if a.category and b.category and a.category == b.category:
            edges.append(RuleEdge(a.rule_id, b.rule_id, "SAME_CATEGORY", 0.5, f"同属 {a.category}"))
            edges.append(RuleEdge(b.rule_id, a.rule_id, "SAME_CATEGORY", 0.5, f"同属 {b.category}"))

        # CLAUSE_TYPE_CHAIN: same clause_type
        if a.clause_type and b.clause_type and a.clause_type == b.clause_type:
            edges.append(RuleEdge(a.rule_id, b.rule_id, "CLAUSE_TYPE_CHAIN", 0.4, f"同条款类型 {a.clause_type}"))
            edges.append(RuleEdge(b.rule_id, a.rule_id, "CLAUSE_TYPE_CHAIN", 0.4, f"同条款类型 {b.clause_type}"))

        # LOGICAL_CHAIN: one rule's entities appear in another's name/terms
        for cat, hints in self._LOGICAL_CHAIN_HINTS.items():
            a_has = any(h in a.rule_name or h in str(a.terms) for h in hints)
            b_has = any(h in b.rule_name or h in str(b.terms) for h in hints)
            if a_has and b_has:
                # Both relate to the same logical domain
                edges.append(RuleEdge(a.rule_id, b.rule_id, "LOGICAL_CHAIN", 0.6, f"逻辑域: {cat}"))
                edges.append(RuleEdge(b.rule_id, a.rule_id, "LOGICAL_CHAIN", 0.6, f"逻辑域: {cat}"))

        return edges

    def _add_edge(self, edge: RuleEdge) -> None:
        self._edges.append(edge)
        self._adj[edge.source_id].add((edge.target_id, edge.edge_type, edge.weight))
        self._rev_adj[edge.target_id].add((edge.source_id, edge.edge_type, edge.weight))

    def _has_failed_predecessor(self, rule_id: str, failed_set: set[str]) -> bool:
        """Check if any FAILED rule points TO this rule."""
        for src, _etype, _w in self._rev_adj.get(rule_id, set()):
            if src in failed_set and src != rule_id:
                return True
        return False

    def _has_failed_successor(self, rule_id: str, failed_set: set[str]) -> bool:
        """Check if this rule points TO any FAILED rule."""
        for tgt, _etype, _w in self._adj.get(rule_id, set()):
            if tgt in failed_set and tgt != rule_id:
                return True
        return False

    def _mermaid_subgraph(self, highlight_ids: set[str],
                          failed_ids: set[str]) -> str:
        """Generate Mermaid subgraph highlighting relevant nodes."""
        lines = ["graph TD"]
        seen = set()

        # Collect all edges where at least one endpoint is in highlight_ids
        relevant_edges = []
        for e in self._edges:
            if e.source_id in highlight_ids or e.target_id in highlight_ids:
                relevant_edges.append(e)

        for e in relevant_edges:
            key = (e.source_id, e.target_id)
            if key in seen:
                continue
            seen.add(key)
            style = _MERMAID_STYLE.get(e.edge_type, "-->")
            src_label = self._nodes[e.source_id].rule_name if e.source_id in self._nodes else e.source_id
            tgt_label = self._nodes[e.target_id].rule_name if e.target_id in self._nodes else e.target_id

            # Style nodes based on status
            src_extra = ":::failed" if e.source_id in failed_ids else ""
            tgt_extra = ":::failed" if e.target_id in failed_ids else ""

            lines.append(
                f'    {_safe_id(e.source_id)}["{src_label}"]{src_extra} '
                f'{style} '
                f'{_safe_id(e.target_id)}["{tgt_label}"]{tgt_extra}'
            )

        # Add class definitions
        lines.append("    classDef failed fill:#ffcccc,stroke:#ff0000")
        lines.append("    classDef rootCause fill:#ff9999,stroke:#cc0000,stroke-width:3px")

        return "\n".join(lines)


# ======================================================================
# Helpers
# ======================================================================

_MERMAID_STYLE = {
    "SHARES_ENTITY": "-- entity -->",
    "SAME_CATEGORY": "-- category -->",
    "LOGICAL_CHAIN": "==>",
    "CLAUSE_TYPE_CHAIN": "-.->",
}

_MARKDOWN_SAFE_RE = re.compile(r'[^a-zA-Z0-9_-]')


def _safe_id(s: str) -> str:
    """Convert a rule ID to a Mermaid-safe node identifier."""
    return _MARKDOWN_SAFE_RE.sub('_', s)


# ======================================================================
# Convenience function
# ======================================================================

def build_graph_from_packages(packages: dict[str, Any]) -> RuleGraph:
    """Build a RuleGraph from compiled packages (as stored in engine._packages).

    Args:
        packages: Dict of package_id -> CompiledPackage or dict with 'rules' key.

    Returns:
        A RuleGraph built from all rules in all packages.
    """
    from dataclasses import asdict, is_dataclass

    graph = RuleGraph()
    all_rules: list[dict] = []
    for pkg_id, pkg in packages.items():
        # Handle both CompiledPackage objects and plain dicts
        if hasattr(pkg, 'rules'):
            rules = pkg.rules
        elif isinstance(pkg, dict) and 'rules' in pkg:
            rules = pkg['rules']
        else:
            continue
        for rule in rules:
            # Convert CompiledRule / dataclass to dict
            if isinstance(rule, dict):
                rd = rule
            elif is_dataclass(rule) and not isinstance(rule, type):
                rd = asdict(rule)
            elif hasattr(rule, '__dict__'):
                rd = {k: v for k, v in rule.__dict__.items() if not k.startswith('_')}
            else:
                continue
            # Normalize: ensure 'condition' key exists
            if 'condition' not in rd and 'condition_type' in rd:
                cp = rd.get('condition_params', {})
                rd = {
                    **rd,
                    'condition': {
                        'type': rd.get('condition_type', ''),
                        'terms': cp.get('terms', []),
                        'context_pattern': cp.get('context_pattern', ''),
                        'label': cp.get('label', ''),
                    }
                }
            all_rules.append(rd)
    graph.build_from_rules(all_rules)
    graph.transitive_closure()
    return graph
