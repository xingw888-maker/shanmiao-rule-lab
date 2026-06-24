#!/usr/bin/env python3
"""路二阶段一 — 黄金数据集评测脚本

加载 road2_gold_samples.json，对每个样本调用 kernel.validate()，
对比期望判定与实际判定，输出假阳性率基线报告。

只度量当前正则路径，不涉及 LLM。

用法:
    python tests/road2_eval.py

输出:
    - stdout: 人类可读的评测摘要
    - tests/road2_eval_report.json: 结构化评测报告
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.kernel import ShanmiaoKernel

# ── 路径 ──
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "road2_gold_samples.json")
REPORT_PATH  = os.path.join(os.path.dirname(__file__), "road2_eval_report.json")


def load_samples() -> list[dict]:
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_one(kernel: ShanmiaoKernel, sample: dict) -> dict:
    """对单个样本运行 kernel.validate() 并提取对应 rule_id 的 evidence 条目。"""
    start = time.time()
    try:
        result = kernel.validate(
            text=sample["text"],
            domain_id="validated/construction",
            enable_layers=False,
            timeout_ms=30000,
            validation_mode="clause",
        )
    except Exception as exc:
        return {
            "sample_id": sample["sample_id"],
            "rule_id": sample["rule_id"],
            "expected": sample["expected_status"],
            "actual": "ERROR",
            "error_type": "exception",
            "error_detail": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.time() - start) * 1000),
            "evidence_chain_entry": None,
        }

    latency_ms = round((time.time() - start) * 1000)
    evidence_chain = result.get("evidence_chain", [])

    # 找对应 rule_id 的 evidence 条目
    match_ev = None
    for ev in evidence_chain:
        if ev.get("rule_id") == sample["rule_id"]:
            match_ev = ev
            break

    if match_ev is None:
        actual = "MISSING"
    else:
        actual = match_ev.get("status", "?")

    # 判定错误类型
    error_type = None
    expected = sample["expected_status"]
    if actual == expected:
        error_type = None
    elif actual == "MISSING":
        # 规则未触发 — 看期望是什么
        if expected == "NOT_APPLICABLE":
            error_type = None  # 没触发 = NOT_APPLICABLE 视为正确
            actual = "NOT_APPLICABLE"
        elif expected == "PASSED":
            error_type = "false_negative"
        else:  # expected FAILED
            error_type = "false_negative"
    elif expected == "NOT_APPLICABLE" and actual != "NOT_APPLICABLE":
        error_type = "false_positive"
    elif expected == "PASSED" and actual == "FAILED":
        error_type = "false_negative"
    elif expected == "FAILED" and actual == "PASSED":
        error_type = "false_positive"
    else:
        error_type = "mismatch"

    ev_summary = None
    if match_ev:
        ev_summary = {
            "status": match_ev.get("status"),
            "rationale": (match_ev.get("rationale", "") or "")[:200],
            "extracted_value": match_ev.get("extracted_value"),
            "legal_ref": match_ev.get("legal_ref", ""),
        }

    return {
        "sample_id": sample["sample_id"],
        "rule_id": sample["rule_id"],
        "sample_type": sample.get("sample_type", ""),
        "expected": expected,
        "actual": actual,
        "correct": actual == expected,
        "error_type": error_type,
        "text_preview": sample["text"][:100],
        "latency_ms": latency_ms,
        "evidence_chain_entry": ev_summary,
    }


def run_eval() -> dict:
    samples = load_samples()
    kernel = ShanmiaoKernel()

    # 预加载 construction 域
    kernel.load_domain("validated/construction")

    results = []
    for i, s in enumerate(samples):
        r = validate_one(kernel, s)
        results.append(r)
        marker = "✓" if r["correct"] else "✗"
        err  = f" ({r['error_type']})" if r["error_type"] else ""
        print(f"  [{i+1:2d}/{len(samples)}] {marker} {s['sample_id']:<20s} "
              f"expected={s['expected_status']:<15s} actual={r['actual']:<15s}{err}")

    # ── 汇总 ──
    total       = len(results)
    correct     = sum(1 for r in results if r["correct"])
    fp          = sum(1 for r in results if r["error_type"] == "false_positive")
    fn          = sum(1 for r in results if r["error_type"] == "false_negative")
    errors      = sum(1 for r in results if r["error_type"] == "exception")
    accuracy    = round(correct / total, 4) if total else 0.0
    fp_rate     = round(fp / total, 4) if total else 0.0

    # 分规则统计
    per_rule: dict[str, dict] = {}
    for r in results:
        rid = r["rule_id"]
        if rid not in per_rule:
            per_rule[rid] = {"rule_id": rid, "total": 0, "correct": 0,
                             "fp": 0, "fn": 0, "errors": 0}
        per_rule[rid]["total"] += 1
        if r["correct"]:
            per_rule[rid]["correct"] += 1
        if r["error_type"] == "false_positive":
            per_rule[rid]["fp"] += 1
        if r["error_type"] == "false_negative":
            per_rule[rid]["fn"] += 1
        if r["error_type"] == "exception":
            per_rule[rid]["errors"] += 1

    # 分样本类型统计
    by_type: dict[str, dict] = {}
    for r in results:
        st = r.get("sample_type", "?")
        if st not in by_type:
            by_type[st] = {"total": 0, "correct": 0, "fp": 0, "fn": 0}
        by_type[st]["total"] += 1
        if r["correct"]:
            by_type[st]["correct"] += 1
        if r["error_type"] == "false_positive":
            by_type[st]["fp"] += 1
        if r["error_type"] == "false_negative":
            by_type[st]["fn"] += 1

    error_list = [r for r in results if r["error_type"]]

    report = {
        "test_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": "路二阶段一 黄金数据集评测 — 正则路径假阳性基线",
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "false_positives": fp,
            "false_negatives": fn,
            "fp_rate": fp_rate,
            "exceptions": errors,
            "per_sample_type": {
                st: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "fp": v["fp"],
                    "fn": v["fn"],
                    "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
                }
                for st, v in sorted(by_type.items())
            },
        },
        "per_rule": sorted(per_rule.values(), key=lambda x: x["rule_id"]),
        "errors": error_list,
    }

    return report


def print_summary(report: dict):
    s = report["summary"]
    print()
    print("=" * 72)
    print("  路二阶段一 — 黄金数据集评测基线")
    print(f"  日期: {report['test_date']}")
    print("=" * 72)
    print()
    print(f"  样本总数:          {s['total']}")
    print(f"  判定正确:          {s['correct']}")
    print(f"  准确率:             {s['accuracy']:.2%}")
    print(f"  假阳性 (FP):       {s['false_positives']}")
    print(f"  假阴性 (FN):       {s['false_negatives']}")
    print(f"  假阳性率:          {s['fp_rate']:.2%}")
    print(f"  异常:              {s['exceptions']}")
    print()
    print("  --- 按样本类型 ---")
    for st, v in s["per_sample_type"].items():
        print(f"  {st:<20s}  total={v['total']:<2d}  correct={v['correct']:<2d}  "
              f"acc={v['accuracy']:.2%}  fp={v['fp']}  fn={v['fn']}")
    print()
    print("  --- 按规则 ---")
    for pr in report["per_rule"]:
        print(f"  {pr['rule_id']:<6s}  total={pr['total']:<2d}  correct={pr['correct']:<2d}"
              f"  fp={pr['fp']}  fn={pr['fn']}  err={pr['errors']}")
    print()
    if report["errors"]:
        print("  --- 错误详情 ---")
        for e in report["errors"]:
            print(f"  ✗ {e['sample_id']}  rule={e['rule_id']}  "
                  f"expected={e['expected']}  actual={e['actual']}  "
                  f"type={e['error_type']}")
            print(f"    text: {e['text_preview']}")
            ev = e.get("evidence_chain_entry")
            if ev:
                print(f"    rationale: {ev.get('rationale', '')[:120]}")
            print()
    print(f"  报告: {REPORT_PATH}")


def save_report(report: dict):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("路二阶段一 黄金数据集评测")
    print(f"样本文件: {SAMPLES_PATH}")
    print()
    report = run_eval()
    save_report(report)
    print_summary(report)
