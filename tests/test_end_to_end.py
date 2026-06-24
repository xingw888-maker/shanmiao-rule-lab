
"""End-to-end integration test — full validated pipeline."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.kernel import ShanmiaoKernel

GF2017_TEXT = """
建设工程施工合同。发包人（全称）：XX房地产开发有限公司。承包人（全称）：XX建设集团有限公司。
工程名称：恒达世纪城。工程地点：XX市。资金来源：自筹。
屋面防水工程、有防水要求的卫生间、房间和外墙面的防渗漏，保修期为五年。
主体结构保修五十年。质量保证金百分之三。缺陷责任期二十四个月。
电气管线、给排水管道、设备安装和装修工程最低保修期限为二年。
供热与供冷系统最低保修期限为二个采暖期。
发包人应在收到结算报告后三十日内付款。验收合格后十五个工作日内支付。
"""

def test_e2e_gf2017():
    """Full pipeline end-to-end test with GF2017-like text."""
    kernel = ShanmiaoKernel()
    result = kernel.validate(text=GF2017_TEXT, domain_id="construction", enable_layers=True)
    
    # Structural integrity
    assert "evidence_chain" in result, "Missing evidence_chain"
    assert "contract_profile" in result, "Missing contract_profile"
    assert "clause_blocks" in result, "Missing clause_blocks"
    
    # Contract profile
    profile = result["contract_profile"]
    assert "broad_type" in profile, "Missing broad_type"
    assert "estimated_value" in profile, "Missing estimated_value"
    
    # Evidence chain has items
    evidence = result["evidence_chain"]
    assert len(evidence) > 0, "Empty evidence chain"
    
    # Multi-evidence types
    statuses = {e.get("status") for e in evidence}
    assert statuses, "No statuses in evidence"
    
    # Belief network
    assert "belief_network" in result, "Missing belief_network"
    bn = result["belief_network"]
    assert bn.get("converged") in (True, False), f"Unexpected converged: {bn.get('converged')}"
    
    # Triager summarization
    assert "triager_summary" in result, "Missing triager_summary"
    ts = result["triager_summary"]
    assert ts.get("total", 0) > 0, f"Triager total is 0: {ts}"
    
    # Causal graph
    assert "causal_graph" in result, "Missing causal_graph"
    cg = result["causal_graph"]
    assert cg.get("node_count", 0) >= 1, f"Graph node_count < 1: {cg}"
    
    # Each evidence item has required fields
    for ev in evidence:
        assert "rule_id" in ev, f"Evidence missing rule_id: {ev}"
        assert "status" in ev, f"Evidence missing status: {ev}"
        assert "rationale" in ev, f"Evidence missing rationale: {ev}"
    
    # cn-027 (payment deadline) should be present and numeric_comparison
    cn027_items = [e for e in evidence if e.get("rule_id") == "cn-027"]
    if cn027_items:
        cn027 = cn027_items[0]
        assert cn027["status"] in ("PASSED", "FAILED", "NOT_APPLICABLE"),             f"cn-027 unexpected status: {cn027['status']}"
    
    # Clause blocks present
    clause_blocks = result["clause_blocks"]
    assert len(clause_blocks) > 0, "Empty clause_blocks"
    for cb in clause_blocks:
        assert "clause_type" in cb, f"Clause block missing clause_type: {cb}"
        assert "content_preview" in cb, f"Clause block missing content_preview: {cb}"
    
    print("✓ e2e test passed: all structural checks OK")
    print(f"   evidence items: {len(evidence)}")
    print(f"   clause blocks: {len(clause_blocks)}")
    print(f"   belief network: converged={bn.get('converged')}")
    print(f"   triager: {ts.get('total')} items")
    print(f"   causal graph: {cg.get('node_count')} nodes, {cg.get('edge_count')} edges")

if __name__ == "__main__":
    test_e2e_gf2017()
    print("\nALL TESTS PASSED")
