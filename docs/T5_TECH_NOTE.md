# T5 Technical Note: Rule Reachability and Structured Numeric Extraction

This note summarizes the T5 benchmark work in Shanmiao Rule Lab.

Shanmiao Rule Lab is a research sandbox for deterministic domain-rule validation. It is not a legal service and does not provide legal advice. The goal of this benchmark is narrow: test whether structured numeric extraction can improve rule evaluation while keeping final verdicts deterministic and auditable.

## Problem

The Road2 benchmark started with 46 gold samples for 15 construction-domain numeric rules.

The first baseline was misleading:

- Accuracy: 58.7%.
- False positives: 1.
- Many false negatives were caused by missing rule reachability, not extraction quality.

Seven numeric rules existed in `rules.json` but were not loaded through the active construction rule packages. In other words, part of the rule set was not on the field.

## T5.1: Rule Reachability

T5.1 fixed the rule-dispatch issue so that all 15 numeric or sum-numeric construction rules are visible to `validate()`.

After the fix:

- All 15 numeric rules are reachable.
- Regex-only baseline accuracy: 76.09%.
- The remaining errors became real extraction or rule-design problems rather than package-loading failures.

This step matters because model comparison is only meaningful after the deterministic rule path is complete.

## T5.2: Structured Numeric Extraction

T5.2 added a dual-path benchmark:

- Path A: regex-only validation.
- Path B: structured numeric extraction followed by deterministic rule handlers.

The extraction path is intentionally limited. It extracts fields such as value, unit, source text, and confidence. It does not make compliance judgments. The existing handlers still produce `PASSED`, `FAILED`, or `NOT_APPLICABLE`.

Result on the 46-sample Road2 set:

| Metric | Regex only | Structured extraction |
| --- | ---: | ---: |
| Accuracy | 76.09% | 89.13% |
| Correct samples | 35 / 46 | 41 / 46 |
| Delta | - | +13.04% |
| Fixed errors | - | 7 |
| Regressions | - | 1 |
| Remaining errors | - | 4 |

The 7 fixed cases were mainly Chinese numeric and fraction expressions that regex did not handle well:

- `三年`
- `二十四个月`
- `百分之三`
- `万分之一`
- `千分之一`
- `日千分之一`

## Known Regression

One regression remained:

- `cn-002-FP-01`: the text mentioned underground waterproofing details but did not state a warranty period. The extraction path inferred a 5-year warranty from context, producing a false positive.

This is an extraction hallucination, not a deterministic rule-handler bug. Possible mitigations include a higher confidence threshold, stricter prompting, and a requirement that the source span include an explicit duration.

## T5.3: Clause Mode and Regression Removal

T5.3 added an explicit `validation_mode="clause"` path for short rule-level samples. This mode is opt-in: the default `validation_mode="document"` behavior is unchanged. In clause mode, callers must pass an explicit `domain_id`; the kernel then trusts that domain and skips broad document-level rejection for short clause fragments.

T5.3 also tightened structured extraction validation so hallucinated source spans are rejected before handler injection.

Result on the same 46-sample Road2 set:

| Metric | Regex only | Structured extraction |
| --- | ---: | ---: |
| Accuracy | 78.26% | 93.48% |
| Correct samples | 36 / 46 | 43 / 46 |
| Delta | - | +15.22% |
| Fixed errors | - | 7 |
| Regressions | - | 0 |
| Remaining errors | - | 3 |

The previous regression was removed. `cn-020-POS-01` also stopped failing as a short-text domain-classification case because the benchmark now uses explicit clause-fragment mode.

## T5.4: Clause-Type Gating

T5.4 made clause-type routing explicit. The splitter now infers stable clause types such as `保修`, `付款`, `验收`, and `担保`; numeric rules can then avoid searching unrelated clause blocks.

Main implementation changes:

- `clause_splitter.py`: fixed clause-type keyword inference.
- `core.py`: clause-type mismatch now returns `NOT_APPLICABLE` instead of falling back to full-text search.
- `kernel.py`: clause splitting uses the stable splitter path.
- `construction/rules.json`: `cn-020` and `cn-021` moved from `付款` to `担保`.

Result on the 46-sample Road2 set:

| Metric | Regex only | Structured extraction |
| --- | ---: | ---: |
| Accuracy | 80.43% | 93.48% |
| Correct samples | 37 / 46 | 43 / 46 |
| Delta | - | +13.05% |
| Fixed errors | - | 7 |
| Regressions | - | 1 |
| Remaining errors | - | 2 |

The remaining regression is `cn-008-FP-01`: the text mentions completion acceptance but the 28-day number belongs to settlement-material submission, not acceptance organization. The clause gate helps the regex path, but the structured extraction path still treats it as an acceptance deadline. This is a semantic disambiguation limit.

## Remaining Errors

Two stable errors remain after T5.4:

- `cn-003-POS-02`: "reasonable service life" is a legal concept rather than a numeric value.
- `cn-010-POS-01`: payment-ratio sum logic requires multi-value extraction and semantic grouping.

These are protocol, rule-design, or semantic-disambiguation problems. They should not be counted as simple numeric extraction failures.

## Interpretation

T5.1 and T5.2 support a limited but useful architecture:

1. Use deterministic rules for final verdicts.
2. Use structured extraction to provide facts to those rules.
3. Preserve evidence: source text, value, unit, confidence, and rule rationale.
4. Keep candidate and extraction outputs reviewable rather than treating them as authoritative.

The result is not "AI legal judgment." It is a measurable improvement to numeric fact extraction inside an auditable rule engine.

## Next Work

The most useful next step is a small T5.3 pass:

- Prevent the `cn-002` false positive by requiring explicit duration evidence.
- Tighten `cn-008` context handling for completion acceptance.
- Fix the `cn-020` rule-triggering gap.
- Improve `cn-010` multi-value payment-ratio handling.

After that, the same benchmark pattern can be expanded to other domains such as purchase and NDA rules.
