"""Verification test for StructuredRuleExtractor.

Tests two domains:
1. 国务院令第279号《建设工程质量管理条例》第40条 — warranty periods
2. 《劳动合同法》第19条 — probation period limits

Cross-domain verification ensures templates are syntactic, not domain-specific.
"""

import sys, os, json
from typing import Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.rule_extractor import StructuredRuleExtractor, RuleCandidate

EXTRACTOR = StructuredRuleExtractor()

# ---------------------------------------------------------------------------
# Test data: 279 Decree (construction warranty)
# ---------------------------------------------------------------------------

SENTENCES_279 = {
    "主体结构": (
        "基础设施工程、房屋建筑的地基基础工程和主体结构工程，"
        "为设计文件规定的该工程的合理使用年限"
    ),
    "屋面防水5年": (
        "屋面防水工程、有防水要求的卫生间、房间和外墙面的防渗漏，为5年"
    ),
    "地下室防水5年": (
        "有防水要求的地下室防渗漏，为5年"
    ),
    "供热供冷2年采暖期": (
        "供热与供冷系统，为2个采暖期、供冷期"
    ),
    "电气给排水2年": (
        "电气管线、给排水管道、设备安装和装修工程，为2年"
    ),
}

EXPECTED_279 = [
    ("屋面防水", 5, 5, "年"),
    ("地下室", 5, 5, "年"),
    ("供热", 2, 2, "采暖期"),
    ("供冷", 2, 2, "年"),
    ("电气", 2, 2, "年"),
    ("给排水", 2, 2, "年"),
    ("装修", 2, 2, "年"),
    ("主体结构", None, None, None),
]

# ---------------------------------------------------------------------------
# Test data: Labor Law (cross-domain)
# ---------------------------------------------------------------------------

SENTENCES_LABOR = {
    "试用期1月": (
        "劳动合同期限三个月以上不满一年的，试用期不得超过一个月"
    ),
    "试用期2月": (
        "劳动合同期限一年以上不满三年的，试用期不得超过二个月"
    ),
    "试用期6月": (
        "三年以上固定期限和无固定期限劳动合同，试用期不得超过六个月"
    ),
}

EXPECTED_LABOR = [
    ("试用期", 1, 1, "月"),
    ("试用期", 2, 2, "月"),
    ("试用期", 6, 6, "月"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subject_contains(subject: str, keyword: str) -> bool:
    return keyword in subject

def _value_matches(val, lo, hi) -> bool:
    if val is None:
        return lo is None and hi is None
    if lo is None:
        return True
    return lo <= val <= hi

def _unit_matches(unit, expected_unit) -> bool:
    if expected_unit is None:
        return unit is None
    if unit is None:
        return expected_unit is None
    u = unit.replace("个月", "月")
    e = expected_unit.replace("个月", "月")
    return u == e

# ---------------------------------------------------------------------------
# Verification 1: 279 Decree
# ---------------------------------------------------------------------------

def verify_279_decree():
    total_sentences = len(SENTENCES_279)
    total_numeric_sentences = 4

    candidates = []
    for name, text in SENTENCES_279.items():
        extracted = EXTRACTOR.extract(text)
        candidates.extend(extracted)
        print(f"\n  提取自「{name}」:")
        for c in extracted:
            print(f"    [{c.condition_type}] subject='{c.subject}' "
                  f"op={c.operator} val={c.expected_value} "
                  f"unit={c.unit} conf={c.confidence:.3f}")

    dedup = {}
    for c in candidates:
        dedup[(c.subject, c.condition_type)] = c
    unique = list(dedup.values())

    matched_expected = 0
    total_expected = len(EXPECTED_279)
    match_details = []

    for keyword, lo, hi, unit in EXPECTED_279:
        found = False
        for c in unique:
            if _subject_contains(c.subject, keyword):
                if _value_matches(c.expected_value, lo, hi) and _unit_matches(c.unit, unit):
                    found = True
                    match_details.append((keyword, True, c.subject, c.expected_value, c.unit))
                    break
        if not found:
            if keyword == "主体结构":
                for c in unique:
                    if _subject_contains(c.subject, keyword):
                        found = True
                        match_details.append((keyword, True, c.subject, c.expected_value, c.unit, "non-numeric"))
                        break
            if not found:
                match_details.append((keyword, False, "", None, ""))
        if found:
            matched_expected += 1

    numeric_hits = 0
    for c in unique:
        if c.condition_type == "numeric_comparison" and c.expected_value is not None:
            for keyword, lo, hi, unit in EXPECTED_279[:6]:
                if _subject_contains(c.subject, keyword) and _value_matches(c.expected_value, lo, hi):
                    numeric_hits += 1
                    break

    matched_subjects = set()
    for c in unique:
        for keyword, lo, hi, unit in EXPECTED_279:
            if _subject_contains(c.subject, keyword):
                matched_subjects.add(keyword)

    print("\n  ── 对比预期结果 ──")
    for kw, ok, *rest in match_details:
        status = "OK" if ok else "MISS"
        print(f"  {status}: {kw}")

    return {
        "total_candidates": len(candidates),
        "unique_candidates": len(unique),
        "matched_expected": matched_expected,
        "total_expected": total_expected,
        "numeric_hits": numeric_hits,
        "numeric_target": total_numeric_sentences,
        "matched_subjects": matched_subjects,
        "all_unique": unique,
    }

# ---------------------------------------------------------------------------
# Verification 2: Labor Law (cross-domain)
# ---------------------------------------------------------------------------

def verify_labor_law():
    candidates = []
    for name, text in SENTENCES_LABOR.items():
        extracted = EXTRACTOR.extract(text)
        candidates.extend(extracted)
        print(f"\n  提取自「{name}」:")
        for c in extracted:
            print(f"    [{c.condition_type}] subject='{c.subject}' "
                  f"op={c.operator} val={c.expected_value} "
                  f"unit={c.unit} conf={c.confidence:.3f}")

    dedup = {}
    for c in candidates:
        dedup[(c.subject, c.condition_type)] = c
    unique = list(dedup.values())

    matched_expected = 0
    total_expected = len(EXPECTED_LABOR)
    match_details = []

    for keyword, lo, hi, unit in EXPECTED_LABOR:
        found = False
        for c in unique:
            subject_ok = _subject_contains(c.subject, keyword) or keyword in c.source_text
            if subject_ok and c.condition_type == "numeric_comparison":
                if _value_matches(c.expected_value, lo, hi) and _unit_matches(c.unit, unit):
                    found = True
                    match_details.append((keyword, lo, True, c.subject))
                    break
        if not found:
            match_details.append((keyword, lo, False, ""))
        if found:
            matched_expected += 1

    print("\n  ── 跨域验证结果 ──")
    for kw, lo, ok, subj in match_details:
        status = "OK" if ok else "MISS"
        print(f"  {status}: {kw} >= {lo} (subject='{subj}')")

    return {
        "total_candidates": len(candidates),
        "unique_candidates": len(unique),
        "matched_expected": matched_expected,
        "total_expected": total_expected,
        "all_unique": unique,
    }

# ---------------------------------------------------------------------------
# Overlap with cn-001~cn-005
# ---------------------------------------------------------------------------

EXISTING_RULES_CN = {
    "cn-001": {"name": "屋面防水工程保修期限", "expected": 5, "unit": "年"},
    "cn-002": {"name": "地下室防水工程保修期限", "expected": 5, "unit": "年"},
    "cn-003": {"name": "主体结构保修期限", "expected": 50, "unit": "年"},
    "cn-004": {"name": "电气管线保修期限", "expected": 2, "unit": "年"},
    "cn-005": {"name": "给排水管道保修期限", "expected": 2, "unit": "年"},
}

CN_RULE_KEYWORDS = {
    "cn-001": ["屋面防水"],
    "cn-002": ["地下室", "地下室防水"],
    "cn-003": ["主体结构"],
    "cn-004": ["电气", "电气管线"],
    "cn-005": ["给排水", "给排水管道"],
}

def compute_overlap(extracted):
    overlap = {}
    for rid, keywords in CN_RULE_KEYWORDS.items():
        matched = False
        for c in extracted:
            for kw in keywords:
                if _subject_contains(c.subject, kw):
                    matched = True
                    overlap[rid] = {
                        "rule_name": EXISTING_RULES_CN[rid]["name"],
                        "matched_subject": c.subject,
                        "matched_value": c.expected_value,
                        "matched_unit": c.unit,
                        "confidence": c.confidence,
                    }
                    break
            if matched:
                break
        if not matched:
            overlap[rid] = {
                "rule_name": EXISTING_RULES_CN[rid]["name"],
                "matched_subject": None,
                "matched_value": None,
                "matched_unit": None,
                "confidence": 0.0,
            }
    return overlap

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  验证 1: 国务院令第279号 第40条 保修期限提取")
    print("=" * 72)
    result_279 = verify_279_decree()
    overlap = compute_overlap(result_279["all_unique"])

    print("\n  与 cn-001~cn-005 重叠:")
    for rid in sorted(overlap.keys()):
        info = overlap[rid]
        status = "OK" if info["matched_subject"] else "MISS"
        print(f"    {rid} ({info['rule_name']}): {status}")
        if info["matched_subject"]:
            print(f"      -> subject='{info['matched_subject']}', "
                  f"value={info['matched_value']}, unit={info['matched_unit']}")

    print("\n" + "=" * 72)
    print("  验证 2: 劳动合同法 第19条 试用期上限 (跨域)")
    print("=" * 72)
    result_labor = verify_labor_law()

    print("\n" + "=" * 72)
    print("  验证摘要报告")
    print("=" * 72)

    print("\n  [1] 279号令提取:")
    print(f"      总候选数:       {result_279['total_candidates']}")
    print(f"      去重候选数:     {result_279['unique_candidates']}")
    print(f"      预期匹配:       {result_279['matched_expected']}/{result_279['total_expected']}")
    print(f"      数值条款命中:   {result_279['numeric_hits']}/{result_279['numeric_target']}")
    print(f"      匹配的预期 subject: {result_279['matched_subjects']}")

    print("\n  [2] cn-001~cn-005 重叠:")
    matched_overlap = sum(1 for v in overlap.values() if v["matched_subject"] is not None)
    print(f"      重叠数:         {matched_overlap}/5")
    print(f"      重叠率:         {matched_overlap / 5 * 100:.0f}%")

    print("\n  [3] 跨域验证 (劳动合同法):")
    print(f"      总候选数:       {result_labor['total_candidates']}")
    print(f"      去重候选数:     {result_labor['unique_candidates']}")
    print(f"      预期匹配:       {result_labor['matched_expected']}/{result_labor['total_expected']}")

    print("\n  [4] 结论:")

    numeric_ok = result_279['numeric_hits'] >= 3
    overlap_ok = matched_overlap >= 2
    labor_ok = result_labor['matched_expected'] >= 2
    all_pass = numeric_ok and overlap_ok and labor_ok

    if all_pass:
        print("      OK: 所有验证通过 — 模板提取可以替代LLM进行首轮候选规则生成")
    else:
        print("      WARN: 部分验证未通过 — 详情见上")

    print(f"      数值条款覆盖:    {'PASS' if numeric_ok else 'FAIL'} ({result_279['numeric_hits']}/{result_279['numeric_target']})")
    print(f"      与现有规则重叠:  {'PASS' if overlap_ok else 'FAIL'} ({matched_overlap}/5)")
    print(f"      跨域验证:        {'PASS' if labor_ok else 'FAIL'} ({result_labor['matched_expected']}/{result_labor['total_expected']})")

    print()

    if all_pass:
        print("ALL VERIFICATION TESTS PASSED")
        return 0
    else:
        print("SOME VERIFICATION TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
