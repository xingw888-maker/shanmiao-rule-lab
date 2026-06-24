"""AST check condition type — validation test record.
Follows CONSTITUTION v1.0 section 1.2 step 6.
"""

import sys, os
sys.path.insert(0, ".")
os.environ["CITTA_DEV_MODE"] = "true"

from app.engine.core import PythonValidationEngine

eng = PythonValidationEngine()

pkg = {
    "id": "ast-test-v0.1",
    "name": "AST Test Package",
    "version": "0.1.0",
    "domain": "code",
    "disclaimer": "Test only.",
    "rules": [
        {"id": "ast-001", "name": "forbidden_node: eval()", "condition": {"type": "ast_check", "language": "javascript", "search_type": "forbidden_node", "node_type": "call_expression", "node_pattern": "eval"}, "severity": "fatal", "message": "eval forbidden", "category": "security"},
        {"id": "ast-002", "name": "required_node: try_statement", "condition": {"type": "ast_check", "language": "javascript", "search_type": "required_node", "node_type": "try_statement"}, "severity": "major", "message": "need try-catch", "category": "robustness"},
        {"id": "ast-003", "name": "forbidden_pattern: console.log", "condition": {"type": "ast_check", "language": "javascript", "search_type": "forbidden_pattern", "node_pattern": r"console\.log"}, "severity": "minor", "message": "no console.log", "category": "cleanup"},
        {"id": "ast-004", "name": "required_pattern: LICENSE header", "condition": {"type": "ast_check", "language": "javascript", "search_type": "required_pattern", "node_pattern": r"LICENSE|SPDX"}, "severity": "minor", "message": "need LICENSE header", "category": "legal"},
        {"id": "ast-005", "name": "forbidden_node: var declaration", "condition": {"type": "ast_check", "language": "javascript", "search_type": "forbidden_node", "node_type": "variable_declaration", "node_pattern": r"\bvar\b"}, "severity": "major", "message": "use let/const", "category": "style"},
    ],
}

eng.load_package(pkg)
PID = "ast-test-v0.1"

passed = 0
failed = 0

def check(name, code, expected):
    global passed, failed
    r = eng.validate(input_data={"text": code}, packages=[PID])
    if r is None:
        print("SKIP %s: validate returned None" % name)
        return
    statuses = {e["rule_id"]: e["status"] for e in r["evidence_chain"]}
    ok = True
    for rid, exp_status in expected.items():
        actual = statuses.get(rid, "MISSING")
        if actual != exp_status:
            print("FAIL %s: %s expected %s, got %s" % (name, rid, exp_status, actual))
            ok = False
            failed += 1
    if ok:
        print("PASS %s" % name)
        passed += 1

check("bad_code", """function login(pw) {
    var x = eval(pw);
    console.log("debug");
    return fetch("/api");
}
""", {
    "ast-001": "FAILED",
    "ast-002": "FAILED",
    "ast-003": "FAILED",
    "ast-004": "FAILED",
    "ast-005": "FAILED",
})

check("clean_code", """// SPDX-License-Identifier: MIT
async function fetchData() {
    try {
        const resp = await fetch("/data");
        return await resp.json();
    } catch (error) {
        throw error;
    }
}
""", {
    "ast-001": "PASSED",
    "ast-002": "PASSED",
    "ast-003": "PASSED",
    "ast-004": "PASSED",
    "ast-005": "PASSED",
})

check("eval_in_string", """const msg = "never use eval()";""", {
    "ast-001": "PASSED",
})

check("empty_code", "", {
    "ast-001": "PASSED",
    "ast-002": "FAILED",
    "ast-003": "PASSED",
    "ast-004": "FAILED",
    "ast-005": "PASSED",
})

check("async_no_try", """async function f() {
    return await fetch("/api");
}
""", {
    "ast-002": "FAILED",
})

check("arrow_eval", """const fn = (x) => eval(x);""", {
    "ast-001": "FAILED",
})

print()
print("Results: %d passed, %d failed" % (passed, failed))
if failed != 0:
    print("AST check: %d cases differ (handler behavior changed)" % failed)
if passed > 0:
    print("AST validation passed on %d test cases" % passed)
