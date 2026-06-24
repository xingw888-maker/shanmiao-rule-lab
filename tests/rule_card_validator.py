"""
RuleCard verification pipeline -- WO-15
tests/rule_card_validator.py

Three pipeline functions:
  1. validate_rule_card(card: dict) -> dict     Schema check (manual, no jsonschema dep)
  2. run_verification(card: dict, engine) -> dict  Run 4 verification samples
  3. check_card(card: dict, engine) -> dict     Combined gate (schema + verify -> ready)
"""

import json
import os
import re
import sys
import logging
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rule_card_validator")

# ── Constants (mirrored from rule_card_schema.json) ──
TOP_REQUIRED = [
    "id", "name", "version", "domain", "layer",
    "source", "logic", "scope", "verification", "boundaries", "evidence",
]
SOURCE_REQUIRED = [
    "legal_ref", "authority", "credibility", "extraction",
    "date_effective", "last_reviewed",
]
LOGIC_REQUIRED = ["condition_type", "rule_kind", "scope_anchor"]
SCOPE_REQUIRED = ["applies_to", "not_applies_to", "trigger_condition"]
VERIFICATION_REQUIRED = [
    "should_pass", "should_fail", "should_not_apply", "should_not_false_positive",
]
BOUNDARIES_REQUIRED = ["checks", "does_not_check", "known_limitations"]
EVIDENCE_REQUIRED = ["on_pass", "on_fail", "on_na"]

VALID_LAYERS = {"L0_VALIDATED", "L1_CONJECTURE", "L2_SOURCE_UNCERTAIN"}
VALID_EXTRACTIONS = {"manual", "keyword_scan", "llm_extract"}
VALID_CONDITION_TYPES = {
    "numeric_comparison", "required_pattern", "forbidden_pattern",
    "mutual_exclusion", "co_occurrence", "definition_contains",
    "text_contains", "clause_structure", "cross_reference", "positional",
}
VALID_RULE_KINDS = {"mandatory", "completeness", "consistency", "advisory"}
VALID_SCOPE_ANCHORS = {"clause", "global", "cross-clause"}

ID_PATTERN = re.compile(r'^[a-z]+-[A-Z0-9]+$')
SEMVER_PATTERN = re.compile(r'^\d+\.\d+\.\d+$')


def validate_rule_card(card: dict) -> dict:
    """Validate RuleCard fields, types and enum values. Returns {valid, errors}."""
    errors: list[str] = []
    if not isinstance(card, dict):
        return {"valid": False, "errors": ["card must be a JSON object"]}

    for field in TOP_REQUIRED:
        if field not in card:
            errors.append(f"Missing top-level field: '{field}'")

    if "id" in card:
        rid = card["id"]
        if not isinstance(rid, str):
            errors.append(f"'id' must be a string, got {type(rid).__name__}")
        elif not ID_PATTERN.match(rid):
            errors.append(f"'id' '{rid}' does not match pattern <domain>-<CODE> (e.g. cn-001)")

    if "name" in card:
        nm = card["name"]
        if not isinstance(nm, str) or len(nm.strip()) == 0:
            errors.append("'name' must be a non-empty string")
        elif len(nm) > 120:
            errors.append(f"'name' exceeds 120 chars ({len(nm)})")

    if "version" in card:
        v = card["version"]
        if not isinstance(v, str) or not SEMVER_PATTERN.match(v):
            errors.append(f"'version' must be semver (e.g. '1.0.0'), got '{v}'")

    if "domain" in card:
        d = card["domain"]
        if not isinstance(d, str) or len(d.strip()) == 0:
            errors.append("'domain' must be a non-empty string")

    if "layer" in card:
        if card["layer"] not in VALID_LAYERS:
            errors.append(f"'layer' must be one of {VALID_LAYERS}, got '{card['layer']}'")

    if "source" in card:
        src = card["source"]
        if not isinstance(src, dict):
            errors.append("'source' must be an object")
        else:
            for f in SOURCE_REQUIRED:
                if f not in src:
                    errors.append(f"Missing source.{f}")
            if "credibility" in src:
                c = src["credibility"]
                if not isinstance(c, (int, float)):
                    errors.append(f"source.credibility must be a number, got {type(c).__name__}")
                elif not (0 <= c <= 1):
                    errors.append(f"source.credibility must be 0-1, got {c}")
            if "extraction" in src and src["extraction"] not in VALID_EXTRACTIONS:
                errors.append(f"source.extraction must be one of {VALID_EXTRACTIONS}, got '{src['extraction']}'")

    if "logic" in card:
        log = card["logic"]
        if not isinstance(log, dict):
            errors.append("'logic' must be an object")
        else:
            for f in LOGIC_REQUIRED:
                if f not in log:
                    errors.append(f"Missing logic.{f}")
            if "condition_type" in log:
                ct = log["condition_type"]
                if ct not in VALID_CONDITION_TYPES:
                    errors.append(f"logic.condition_type must be one of {VALID_CONDITION_TYPES}, got '{ct}'")
            if "rule_kind" in log and log["rule_kind"] not in VALID_RULE_KINDS:
                errors.append(f"logic.rule_kind must be one of {VALID_RULE_KINDS}, got '{log['rule_kind']}'")
            if "scope_anchor" in log and log["scope_anchor"] not in VALID_SCOPE_ANCHORS:
                errors.append(f"logic.scope_anchor must be one of {VALID_SCOPE_ANCHORS}, got '{log['scope_anchor']}'")

    if "scope" in card:
        sc = card["scope"]
        if not isinstance(sc, dict):
            errors.append("'scope' must be an object")
        else:
            for f in SCOPE_REQUIRED:
                if f not in sc:
                    errors.append(f"Missing scope.{f}")
            if "applies_to" in sc:
                at = sc["applies_to"]
                if not isinstance(at, list) or len(at) == 0:
                    errors.append("scope.applies_to must be a non-empty array")
            if "trigger_condition" in sc:
                tc = sc["trigger_condition"]
                if not isinstance(tc, str) or len(tc.strip()) == 0:
                    errors.append("scope.trigger_condition must be a non-empty string")

    if "verification" in card:
        ver = card["verification"]
        if not isinstance(ver, dict):
            errors.append("'verification' must be an object")
        else:
            for f in VERIFICATION_REQUIRED:
                if f not in ver:
                    errors.append(f"Missing verification.{f}")
            _validate_ver_sample(ver.get("should_pass"), "verification.should_pass", "PASSED", errors)
            _validate_ver_sample(ver.get("should_fail"), "verification.should_fail", "FAILED", errors)
            _validate_ver_sample(ver.get("should_not_apply"), "verification.should_not_apply", {"NOT_APPLICABLE", "PASSED"}, errors)
            _validate_ver_sample(ver.get("should_not_false_positive"), "verification.should_not_false_positive", {"NOT_APPLICABLE", "PASSED"}, errors)

    if "boundaries" in card:
        bnd = card["boundaries"]
        if not isinstance(bnd, dict):
            errors.append("'boundaries' must be an object")
        else:
            for f in BOUNDARIES_REQUIRED:
                if f not in bnd:
                    errors.append(f"Missing boundaries.{f}")
                elif not isinstance(bnd.get(f, ""), str) or len(str(bnd.get(f, "")).strip()) == 0:
                    errors.append(f"boundaries.{f} must be a non-empty string")

    if "evidence" in card:
        ev = card["evidence"]
        if not isinstance(ev, dict):
            errors.append("'evidence' must be an object")
        else:
            for f in EVIDENCE_REQUIRED:
                if f not in ev:
                    errors.append(f"Missing evidence.{f}")
                elif not isinstance(ev.get(f, ""), str) or len(str(ev.get(f, "")).strip()) == 0:
                    errors.append(f"evidence.{f} must be a non-empty string")

    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True}


def _validate_ver_sample(sample, label, expected, errors):
    if sample is None:
        errors.append(f"{label}: missing (required)")
        return
    if not isinstance(sample, dict):
        errors.append(f"{label}: must be an object")
        return
    if "text" not in sample:
        errors.append(f"{label}: missing 'text'")
    elif not isinstance(sample["text"], str) or len(sample["text"].strip()) == 0:
        errors.append(f"{label}.text: must be a non-empty string")
    if "expected" not in sample:
        errors.append(f"{label}: missing 'expected'")
    else:
        exp = sample["expected"]
        allowed = expected if isinstance(expected, set) else {expected}
        if exp not in allowed:
            errors.append(f"{label}.expected: must be one of {allowed}, got '{exp}'")


# ── RuleCard → engine rule bridge ──
_SEVERITY_MAP = {
    "mandatory": "fatal", "completeness": "fatal",
    "consistency": "major", "advisory": "warning",
}


def _rule_kind_to_severity(kind):
    return _SEVERITY_MAP.get(kind, "error")


def _card_to_engine_rule(card):
    logic = card["logic"]
    source = card["source"]
    cond_type = logic["condition_type"]
    params = dict(logic.get("params", {}))
    params["legal_ref"] = source.get("legal_ref", "")
    if cond_type == "numeric_comparison" and "label" not in params:
        params["label"] = card.get("name", card.get("id", ""))
    return {
        "id": card["id"],
        "name": card["name"],
        "condition": {"type": cond_type, **params},
        "severity": _rule_kind_to_severity(logic.get("rule_kind", "mandatory")),
        "message": logic.get("formal_predicate", ""),
        "category": card.get("domain", ""),
        "source": source.get("authority", ""),
        "source_credibility": source.get("credibility", 0.5),
        "extraction_method": source.get("extraction", ""),
        "version": card.get("version", "0.0.0"),
    }


def _card_to_temp_package(card):
    rule = _card_to_engine_rule(card)
    return {
        "id": f"_verify_{card['id']}",
        "name": f"_verify_{card['id']}",
        "version": "1.0.0",
        "domain": card.get("domain", ""),
        "rules": [rule],
    }


def run_verification(card, engine):
    """Run four verification samples against engine, compare actual vs expected."""
    ver = card.get("verification", {})
    pkg = _card_to_temp_package(card)
    pkg_id = pkg["id"]
    try:
        engine.unload_package(pkg_id)
    except Exception:
        pass
    engine.load_package(pkg, domain_id=card.get("domain", ""))

    passed = []
    unexpected = []
    samples = [
        ("should_pass", {"PASSED"}),
        ("should_fail", {"FAILED"}),
        ("should_not_apply", {"NOT_APPLICABLE", "PASSED"}),
        ("should_not_false_positive", {"NOT_APPLICABLE", "PASSED"}),
    ]
    try:
        for name, allowed in samples:
            sample = ver.get(name)
            if not sample:
                unexpected.append(f"{name}: sample missing from verification")
                continue
            text = sample["text"]
            expected = sample["expected"]
            result = engine.validate({"text": text}, [pkg_id])
            evidence = result.get("evidence_chain", [])
            if not evidence:
                unexpected.append(f"{name}: no evidence produced (expected {expected})")
                continue
            actual = evidence[0].get("status", "")
            if actual in allowed:
                passed.append(name)
            else:
                unexpected.append(f"{name}: expected {expected}, got {actual}")
    finally:
        try:
            engine.unload_package(pkg_id)
        except Exception:
            pass
    return {"passed": passed, "unexpected": unexpected}


def check_card(card, engine):
    """Schema + verification gate. Returns ready_for_review or blocked.

    Checks:
      - Schema validity (validate_rule_card)
      - All four verification samples are present and non-empty
      - Engine verification results match expected
    """
    schema_ok = validate_rule_card(card)
    if not schema_ok["valid"]:
        return {"status": "blocked", "schema_errors": schema_ok.get("errors", []), "verification_failures": []}

    # ── Hard gate: all four sample types must be present and non-empty ──
    ver = card.get("verification", {})
    FOUR_SAMPLES = ["should_pass", "should_fail", "should_not_apply", "should_not_false_positive"]
    missing = []
    for key in FOUR_SAMPLES:
        sample = ver.get(key)
        if not isinstance(sample, dict):
            missing.append(key)
        elif "text" not in sample or not isinstance(sample.get("text"), str) or len(sample["text"].strip()) == 0:
            missing.append(key)
    if missing:
        return {
            "status": "blocked",
            "schema_errors": [],
            "verification_failures": [],
            "missing_samples": missing,
            "message": f"Missing or empty verification samples: {', '.join(missing)}. "
                       f"All four sample types (should_pass, should_fail, should_not_apply, should_not_false_positive) "
                       f"are required for ready_for_review.",
        }

    ver_result = run_verification(card, engine)
    if ver_result.get("unexpected"):
        return {"status": "blocked", "schema_errors": [], "verification_failures": ver_result["unexpected"]}
    return {"status": "ready_for_review"}


# ── CLI self-test ──
def _run_tests():
    from app.engine.core import PythonValidationEngine
    CARDS_DIR = os.path.join(_PROJECT_ROOT, "domains", "construction", "cards")
    cn001_path = os.path.join(CARDS_DIR, "cn-001.json")
    r = {}
    if not os.path.exists(cn001_path):
        r["cn001_load"] = f"ERROR: {cn001_path} not found"
        return r
    with open(cn001_path, "r", encoding="utf-8") as f:
        cn001 = json.load(f)
    engine = PythonValidationEngine()

    r["cn001_check_card"] = check_card(cn001, engine)

    missing_field = dict(cn001)
    del missing_field["name"]
    r["missing_field"] = validate_rule_card(missing_field)

    missing_sub = json.loads(json.dumps(cn001))
    del missing_sub["source"]["legal_ref"]
    r["missing_sub_field"] = validate_rule_card(missing_sub)

    missing_ver = json.loads(json.dumps(cn001))
    del missing_ver["verification"]["should_fail"]
    r["missing_ver_sample"] = validate_rule_card(missing_ver)

    bad_expected = json.loads(json.dumps(cn001))
    bad_expected["verification"]["should_fail"]["expected"] = "PASSED"
    r["bad_expected_verification"] = run_verification(bad_expected, engine)

    bad_layer = json.loads(json.dumps(cn001))
    bad_layer["layer"] = "L99_INVALID"
    r["bad_layer_enum"] = validate_rule_card(bad_layer)

    bad_extraction = json.loads(json.dumps(cn001))
    bad_extraction["source"]["extraction"] = "ai_generated"
    r["bad_extraction_enum"] = validate_rule_card(bad_extraction)

    bad_id = json.loads(json.dumps(cn001))
    bad_id["id"] = "CN-001"
    r["bad_id_format"] = validate_rule_card(bad_id)

    return r


def _print_results(r):
    print("\n" + "=" * 72)
    print("  RuleCard verification pipeline -- WO-15 self-test")
    print("=" * 72)
    tests = [
        ("cn-001 full check_card", "cn001_check_card"),
        ("Missing top field (name)", "missing_field"),
        ("Missing sub field (source.legal_ref)", "missing_sub_field"),
        ("Missing ver sample (should_fail)", "missing_ver_sample"),
        ("Bad enum (layer)", "bad_layer_enum"),
        ("Bad enum (extraction)", "bad_extraction_enum"),
        ("Bad id format (CN-001)", "bad_id_format"),
    ]
    all_ok = True
    for label, key in tests:
        val = r.get(key)
        if val is None:
            print(f"\n  [{label}]  SKIP")
            continue
        if key == "cn001_check_card":
            status = val.get("status", "???")
            ok = (status == "ready_for_review")
            tag = "PASS" if ok else "FAIL"
            print(f"\n  [{label}]  {tag}  status={status}")
            if not ok:
                print(f"    schema_errors:      {val.get('schema_errors', [])}")
                print(f"    verification_failures: {val.get('verification_failures', [])}")
                all_ok = False
        elif key == "bad_expected_verification":
            passed = val.get("passed", [])
            unexpected = val.get("unexpected", [])
            has_mismatch = any("should_fail" in u for u in unexpected)
            tag = "PASS" if has_mismatch else "FAIL"
            print(f"\n  [should_fail->PASSED mislabel detect]  {tag}")
            print(f"    passed:     {passed}")
            print(f"    unexpected: {unexpected}")
            if not has_mismatch:
                all_ok = False
        else:
            valid = val.get("valid", False)
            errors = val.get("errors", [])
            tag = "PASS" if not valid else "FAIL"
            print(f"\n  [{label}]  {tag}")
            if valid:
                print(f"    valid=True (should be invalid -- missed)")
                all_ok = False
            else:
                print(f"    valid=False  errors={len(errors)}")
                for e in errors[:3]:
                    print(f"      - {e}")
                if len(errors) > 3:
                    print(f"      ... and {len(errors) - 3} more")
    print("\n" + "-" * 72)
    msg = "ALL PASSED" if all_ok else "HAS FAILURES (see above)"
    print(f"  Result: {msg}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    res = _run_tests()
    _print_results(res)
