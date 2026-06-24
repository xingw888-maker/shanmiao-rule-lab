"""Test Algorithm 1: Rule Dependency Graph Engine."""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.rule_graph import RuleGraph, RuleNode, RuleEdge, CausalReport, build_graph_from_packages


def load_rules() -> list[dict]:
    """Load construction rules from rules.json."""
    rules_path = Path(__file__).resolve().parent.parent / "domains" / "validated" / "construction" / "rules.json"
    data = json.loads(rules_path.read_text("utf-8"))
    return data.get("rules", [])


# ── Test 1: graph build ──

def test_graph_build():
    """Build graph from construction rules.json. Verify node/edge counts."""
    rules = load_rules()
    assert len(rules) >= 24, f"Expected 24+ rules, got {len(rules)}"

    graph = RuleGraph()
    graph.build_from_rules(rules)

    assert graph.node_count() == len(rules), \
        f"Node count mismatch: {graph.node_count()} vs {len(rules)}"
    assert graph.edge_count() > 0, \
        "Expected at least some edges between rules"

    # Verify all nodes are accessible
    for r in rules:
        node = graph.get_node(r["id"])
        assert node is not None, f"Node {r['id']} not found"
        assert node.rule_name, f"Node {r['id']} has empty name"

    print(f"  ✓ graph_build: {graph.node_count()} nodes, {graph.edge_count()} edges")
    return True


# ── Test 2: transitive closure ──

def test_transitive_closure():
    """Verify transitive closure: all nodes reachable from themselves via path."""
    rules = load_rules()
    graph = RuleGraph()
    graph.build_from_rules(rules)
    graph.transitive_closure()

    # Verify closure is non-empty for rules with edges
    reachable_count = 0
    for r in rules:
        rid = r["id"]
        closure = graph._closure.get(rid, set())
        if closure:
            reachable_count += 1

    # At least some rules should have reachable neighbors
    assert reachable_count > 0, "Expected transitive closure to find reachable nodes"

    print(f"  ✓ transitive_closure: {reachable_count}/{len(rules)} nodes have reachable neighbors")
    return True


# ── Test 3: causal chain ──

def test_causal_chain():
    """Simulate multiple FAILED rules and verify causal chain analysis."""
    rules = load_rules()
    graph = RuleGraph()
    graph.build_from_rules(rules)
    graph.transitive_closure()

    # Simulate: cn-001 (waterproof warranty) and cn-014 (warranty certificate) both FAILED
    # These share category "保修" or clause type
    verdicts = {r["id"]: "PASSED" for r in rules}  # all pass by default
    verdicts["cn-001"] = "FAILED"  # waterproof warranty too short
    verdicts["cn-014"] = "FAILED"  # missing warranty certificate

    # Analyze cn-001
    report = graph.find_causal_chain("cn-001", verdicts)
    assert report is not None, "Expected causal report for cn-001"
    assert report.rule_id == "cn-001"
    assert report.classification in ("root_cause", "consequence", "isolated"), \
        f"Unexpected classification: {report.classification}"

    # The report should have a mermaid graph
    assert "graph" in report.graph_mermaid.lower() or "TD" in report.graph_mermaid, \
        "Expected Mermaid graph in report"

    print(f"  ✓ causal_chain: cn-001 classified as '{report.classification}', "
          f"root_causes={report.root_causes}, consequences={report.consequences}")
    return True


# ── Test 4: dependency resolve ──

def test_dependency_resolve():
    """Classify all FAILED rules and verify no classification errors."""
    rules = load_rules()
    graph = RuleGraph()
    graph.build_from_rules(rules)
    graph.transitive_closure()

    # Simulate a mix of FAILED rules
    verdicts = {r["id"]: "PASSED" for r in rules}
    verdicts["cn-001"] = "FAILED"
    verdicts["cn-022"] = "FAILED"  # dispute resolution — should be isolated
    verdicts["cn-027"] = "FAILED"  # payment deadline

    classification = graph.dependency_resolve(verdicts)
    failed_ids = {rid for rid, v in verdicts.items() if v == "FAILED"}

    for rid in failed_ids:
        assert rid in classification, f"FAILED rule {rid} not classified"
        assert classification[rid] in ("root_cause", "consequence", "isolated"), \
            f"Invalid classification for {rid}: {classification[rid]}"

    print(f"  ✓ dependency_resolve: {classification}")
    return True


# ── Test 5: Mermaid output ──

def test_mermaid_output():
    """Verify Mermaid graph generation."""
    rules = load_rules()
    graph = RuleGraph()
    graph.build_from_rules(rules)

    # Full graph mermaid
    mermaid = graph.mermaid_full()
    assert mermaid.startswith("graph TD"), f"Expected 'graph TD', got: {mermaid[:50]}"
    assert len(mermaid.split("\n")) >= 3, "Expected multi-line Mermaid output"

    # Subgraph mermaid
    verdicts = {r["id"]: "PASSED" for r in rules}
    verdicts["cn-001"] = "FAILED"
    report = graph.find_causal_chain("cn-001", verdicts)
    assert "graph TD" in report.graph_mermaid, "Subgraph mermaid should start with 'graph TD'"

    print(f"  ✓ mermaid: full graph {len(mermaid.split(chr(10)))} lines, subgraph valid")
    return True


# ── Test 6: build_graph_from_packages convenience ──

def test_build_from_packages():
    """Verify the convenience function works with dict-based packages."""
    # Create mock packages
    packages = {
        "test-pkg": {
            "rules": [
                {
                    "id": "test-001",
                    "name": "Test Rule 1",
                    "condition": {"type": "numeric_comparison", "context_pattern": "保修|防水"},
                    "category": "保修",
                    "clause_type": "保修",
                },
                {
                    "id": "test-002",
                    "name": "Test Rule 2",
                    "condition": {"type": "numeric_comparison", "context_pattern": "保修|主体结构"},
                    "category": "保修",
                    "clause_type": "保修",
                },
            ]
        }
    }

    graph = build_graph_from_packages(packages)
    assert graph.node_count() == 2, f"Expected 2 nodes, got {graph.node_count()}"
    assert graph.edge_count() > 0, "Expected edges between test rules (shared entity '保修')"

    print(f"  ✓ build_from_packages: {graph.node_count()} nodes, {graph.edge_count()} edges")
    return True


# ── Run all ──

if __name__ == "__main__":
    passed = 0
    total = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            total += 1
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)
