#!/usr/bin/env python3
"""T5.2 — Dual-path comparison: regex vs LLM extraction on gold dataset.

Runs each gold sample through two paths and compares results:
  Path A (regex): kernel.validate(text) — existing regex-based handler
  Path B (extracted): kernel.validate(text, structured_extractions=...) — LLM-injected

Outputs per-sample comparison table and structured report.

Usage:
    python tests/road2_compare.py              # live LLM mode (needs DEEPSEEK_API_KEY)
    python tests/road2_compare.py --mock       # mock mode (regex-only, no API)
    python tests/road2_compare.py --max=5      # limit to 5 samples
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# ── Mock pydantic ──
class _MockObj: ...
_fake_ps = type(sys)("pydantic_settings"); _fake_ps.BaseSettings = _MockObj
_fake_pd = type(sys)("pydantic"); _fake_pd.BaseModel = object
if "pydantic_settings" not in sys.modules:
    sys.modules["pydantic_settings"] = _fake_ps
    sys.modules["pydantic"] = _fake_pd
    import app.config
    app.config.settings = _MockObj()
    for _a in ["RUST_ENABLED","LLM_API_URL","LLM_API_KEY","LLM_MODEL",
               "RATE_LIMIT_PER_MINUTE","DEFAULT_TIMEOUT_MS","MAX_INPUT_CHARS","DATABASE_URL"]:
        setattr(app.config.settings, _a, False if _a=="RUST_ENABLED" else "")

from app.kernel import ShanmiaoKernel
from tests.road2_llm_extractor import NumericFieldExtractor

SAMPLES_PATH = HERE / "road2_gold_samples.json"
REPORT_PATH  = HERE / "road2_compare_report.json"
KERNEL_DOMAIN = "validated/construction"


def load_samples() -> list[dict]:
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_evidence(result: dict, rule_id: str) -> dict | None:
    for ev in result.get("evidence_chain", []):
        if ev.get("rule_id") == rule_id:
            return ev
    return None


def _clear_matcher(kernel: ShanmiaoKernel) -> None:
    """Clear structured_inputs to prevent state leak between runs."""
    try:
        if hasattr(kernel, "_engine") and hasattr(kernel._engine, "_matcher"):
            kernel._engine._matcher._structured_inputs = {}
    except Exception:
        pass


def classify_result(expected: str, actual: str) -> str | None:
    if actual == expected:
        return None
    if expected == "NOT_APPLICABLE" and actual != "NOT_APPLICABLE":
        return "false_positive"
    if expected in ("PASSED", "FAILED") and actual == "MISSING":
        return "false_negative"
    if expected == "PASSED" and actual == "FAILED":
        return "false_negative"
    if expected == "FAILED" and actual == "PASSED":
        return "false_positive"
    return "mismatch"


def normalize(actual: str, expected: str) -> tuple[str, bool]:
    if actual == expected:
        return actual, True
    if actual == "MISSING" and expected == "NOT_APPLICABLE":
        return "NOT_APPLICABLE", True
    return actual, False


def run_regex_path(kernel: ShanmiaoKernel, sample: dict) -> dict:
    _clear_matcher(kernel)
    try:
        result = kernel.validate(
            text=sample["text"], domain_id=KERNEL_DOMAIN,
            enable_layers=False, timeout_ms=30000)
    except Exception as exc:
        return {"status": "ERROR", "rationale": f"{type(exc).__name__}: {exc}", "extracted_value": None}
    ev = _find_evidence(result, sample["rule_id"])
    if ev is None:
        return {"status": "MISSING", "rationale": "", "extracted_value": None}
    return {"status": ev.get("status", "?"), "rationale": (ev.get("rationale","") or "")[:200],
            "extracted_value": ev.get("extracted_value")}


def run_extracted_path(kernel: ShanmiaoKernel, sample: dict,
                       extractor, cache: dict, use_mock: bool = False) -> dict:
    rule_id = sample["rule_id"]
    structured: dict[str, dict] = {}

    if not use_mock and extractor is not None:
        key = sample["sample_id"]
        if key in cache:
            structured = cache[key]
        else:
            structured = extractor.extract_for_sample(sample)
            cache[key] = structured

    structured_list = []
    for rid, ext in structured.items():
        structured_list.append({
            "field": ext.get("field_label", ""),
            "value": ext.get("value"),
            "unit": ext.get("unit", ""),
            "operator_hint": ext.get("operator_hint", ""),
            "source_text": ext.get("source_text", ""),
            "confidence": ext.get("confidence", 0.5),
        })

    _clear_matcher(kernel)
    try:
        result = kernel.validate(
            text=sample["text"], domain_id=KERNEL_DOMAIN,
            enable_layers=False, timeout_ms=30000,
            structured_extractions=structured_list if structured_list else None)
    except Exception as exc:
        return {"status": "ERROR", "rationale": f"{type(exc).__name__}: {exc}",
                "extracted_value": None, "extract_info": {"n_structured": len(structured_list)}}

    ev = _find_evidence(result, rule_id)
    if ev is None:
        return {"status": "MISSING", "rationale": "", "extracted_value": None,
                "extract_info": {"n_structured": len(structured_list)}}

    return {
        "status": ev.get("status", "?"),
        "rationale": (ev.get("rationale","") or "")[:200],
        "extracted_value": ev.get("extracted_value"),
        "extract_info": {
            "n_structured": len(structured_list),
            "extraction_for_rule": structured.get(rule_id),
        },
    }


def run_comparison(use_mock: bool = False, max_samples: int | None = None) -> dict:
    samples = load_samples()
    if max_samples:
        samples = samples[:max_samples]
    total = len(samples)

    kernel = ShanmiaoKernel()
    kernel.load_domain(KERNEL_DOMAIN)

    extractor = None
    cache: dict[str, dict] = {}
    if not use_mock:
        key = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("LLM_API_KEY", ""))
        if not key:
            print("WARNING: No API key. Falling back to --mock.")
            use_mock = True
        else:
            extractor = NumericFieldExtractor(api_key=key, delay_sec=1.2)
            print(f"LLM Extractor ready (DeepSeek)")

    mode = "MOCK" if use_mock else "LLM (DeepSeek)"
    print(f"T5.2 Dual-path comparison — {mode}")
    print(f"Samples: {SAMPLES_PATH}\n")

    results = []
    regex_ok = ext_ok = 0

    for i, s in enumerate(samples):
        expected = s["expected_status"]
        sid = s["sample_id"]

        ra = run_regex_path(kernel, s)
        ra_norm, ra_correct = normalize(ra["status"], expected)
        ra_err = classify_result(expected, ra_norm)
        if ra_correct: regex_ok += 1

        rb = run_extracted_path(kernel, s, extractor, cache, use_mock)
        rb_norm, rb_correct = normalize(rb["status"], expected)
        rb_err = classify_result(expected, rb_norm)
        if rb_correct: ext_ok += 1

        if not ra_correct and rb_correct:
            delta = "FIXED"
        elif ra_correct and not rb_correct:
            delta = "REGRESSION"
        elif not ra_correct and not rb_correct:
            delta = "BOTH_WRONG"
        else:
            delta = "SAME"

        ei = rb.get("extract_info") or {}
        eri = ei.get("extraction_for_rule") or {}

        rec = {
            "sample_id": sid, "rule_id": s["rule_id"],
            "sample_type": s.get("sample_type",""),
            "expected": expected,
            "regex_status": ra_norm, "regex_correct": ra_correct,
            "regex_error_type": ra_err,
            "regex_rationale": ra.get("rationale","")[:200],
            "extracted_status": rb_norm, "extracted_correct": rb_correct,
            "extracted_error_type": rb_err,
            "extracted_rationale": rb.get("rationale","")[:200],
            "extract_value": eri.get("value"),
            "extract_unit": eri.get("unit"),
            "extract_confidence": eri.get("confidence"),
            "extract_source_text": str(eri.get("source_text",""))[:100],
            "n_structured_extractions": ei.get("n_structured", 0),
            "delta": delta, "text_preview": s["text"][:100],
        }
        results.append(rec)

        ma = "v" if ra_correct else "x"
        mb = "v" if rb_correct else "x"
        dm = {"FIXED":"+", "REGRESSION":"-", "BOTH_WRONG":"=", "SAME":" "}.get(delta,"?")
        print(f"  [{i+1:2d}/{total}] {sid:<20s} regex={ma}({ra_norm:<15s}) "
              f"extracted={mb}({rb_norm:<15s}) {dm} {delta}")

    reg_acc = round(regex_ok/total,4) if total else 0
    ext_acc = round(ext_ok/total,4) if total else 0

    per_rule = {}
    for r in results:
        rid = r["rule_id"]
        if rid not in per_rule:
            per_rule[rid] = {"rule_id":rid,"regex_correct":0,"extracted_correct":0,"total":0}
        per_rule[rid]["total"] += 1
        if r["regex_correct"]: per_rule[rid]["regex_correct"] += 1
        if r["extracted_correct"]: per_rule[rid]["extracted_correct"] += 1

    fixed   = [r for r in results if r["delta"] == "FIXED"]
    regress = [r for r in results if r["delta"] == "REGRESSION"]
    remain  = [r for r in results if r["delta"] == "BOTH_WRONG"]

    report = {
        "test_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": "T5.2 Dual-path comparison: regex vs LLM extraction",
        "mode": "mock" if use_mock else "llm",
        "summary": {
            "total": total, "regex_correct": regex_ok, "regex_accuracy": reg_acc,
            "extracted_correct": ext_ok, "extracted_accuracy": ext_acc,
            "accuracy_delta": round(ext_acc - reg_acc, 4),
            "errors_fixed": len(fixed), "regressions": len(regress),
            "errors_remaining": len(remain),
        },
        "per_rule_comparison": [
            {"rule_id": pr["rule_id"], "regex_correct": pr["regex_correct"],
             "extracted_correct": pr["extracted_correct"], "total": pr["total"],
             "delta": pr["extracted_correct"] - pr["regex_correct"]}
            for pr in sorted(per_rule.values(), key=lambda x: x["rule_id"])
        ],
        "errors_fixed": fixed, "regressions": regress, "errors_remaining": remain,
        "all_results": results,
    }
    return report


def print_summary(report: dict):
    s = report["summary"]
    print(f"\n{'='*64}")
    print(f"  T5.2 Dual-path — {report['mode']}")
    print(f"  Date: {report['test_date']}")
    print(f"{'='*64}\n")
    print(f"  Regex accuracy:      {s['regex_accuracy']:.2%} ({s['regex_correct']}/{s['total']})")
    print(f"  Extracted accuracy:  {s['extracted_accuracy']:.2%} ({s['extracted_correct']}/{s['total']})")
    print(f"  Delta:               {s['accuracy_delta']:+.2%}")
    print(f"  Fixed: {s['errors_fixed']}  Regressions: {s['regressions']}  Remaining: {s['errors_remaining']}\n")
    print("  Per-rule:")
    for pr in report["per_rule_comparison"]:
        d = f"+{pr['delta']}" if pr["delta"]>0 else str(pr["delta"])
        print(f"    {pr['rule_id']:<6s} regex={pr['regex_correct']}/{pr['total']}  "
              f"extracted={pr['extracted_correct']}/{pr['total']}  delta={d}")

    for label, items in [("Fixed", report["errors_fixed"]),
                          ("Regressions", report["regressions"]),
                          ("Still wrong", report["errors_remaining"])]:
        if not items: continue
        print(f"\n  --- {label} ---")
        for e in items:
            print(f"    {e['sample_id']:<20s} exp={e['expected']:<15s} "
                  f"regex={e['regex_status']:<15s} extracted={e['extracted_status']}")
            print(f"      text: {e['text_preview']}")
            if e.get("extract_value") is not None:
                print(f"      value={e['extract_value']} unit={e['extract_unit']} "
                      f"conf={e['extract_confidence']}")
    print(f"\n  Report: {REPORT_PATH}")


def save_report(report: dict):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    use_mock = "--mock" in sys.argv
    max_samples = None
    for a in sys.argv:
        if a.startswith("--max="):
            max_samples = int(a.split("=")[1])
    report = run_comparison(use_mock=use_mock, max_samples=max_samples)
    save_report(report)
    print_summary(report)
