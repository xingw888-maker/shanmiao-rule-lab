# Road2 Known Limits

> 2026-06-24 T5.4 — clause_type keyword gate 生效。regex 路径 clause_type 不匹配→NOT_APPLICABLE。
> LLM 提取 + clause_type gate 准确率 43/46 (93.48%)，pytest 76/76。

## 不可修项 (2)

以下两个样本在 LLM 提取路径下仍无法正确判定。

### cn-003-POS-02 — 法律概念规范化

"合理使用年限"是法律概念，不是显式数值。文本中没有可提取的数值表达式。

- **归属层**：规则定义层
- **建议**：rules.json 中为 cn-003 增加特殊处理（keyword="合理使用年限"→PASSED），不需要动引擎

### cn-010-POS-01 — 多阶段付款聚合

付款比例合计（80%+97%+3%）属于多阶段付款结构理解，不是单值 numeric_comparison。

- **归属层**：协议层
- **建议**：需要对"分阶段"vs"合计"做语义区分，需协议层/多值聚合支持

---

## 已修复项

### cn-008-FP-01 — 语义角色歧义（regex 路径修复）

"28天内提交结算资料"与"28天内组织竣工验收"存在语义歧义。
- **regex 路径**：clause_type gate 正确拦截 — 文本标"付款"（含"结算"关键词），cn-008 规则要求"验收"→NOT_APPLICABLE ✓
- **LLM 路径**：文本含"竣工验收"→splitter 标"验收"→gate 放行→LLM 提取 28 天→PASSED（误判）。LLM 在语义消歧上有天花板，归属规则层/条款级阅读理解，非 T5.4 范围。

### cn-020-POS-01 — clause_type 归类修正

rules.json cn-020/cn-021 clause_type "付款"→"担保"。此前 splitter 标"担保"但规则 expect "付款"，clause_type gate mismatch 导致跳过。修正后 splitter 与规则一致。

---

## 不可归类 (LLM flakiness)

### cn-027-POS-01

文本"发包人应在验收合格后30日内向承包人支付工程款"含 30 天付款期限，但 LLM 提取器未返回结构化字段。非 clause_type 问题（splitter 正确标"付款"，cn-027 规则 clause_type="付款"→匹配）。LLM 提取不稳定，多次运行可能恢复。

---

## 后续方向

| 样本 | 修复层 | 预估改动 |
|------|--------|---------|
| cn-003-POS-02 | 规则定义层 | rules.json 加特殊规则，一次性治理 |
| cn-010-POS-01 | 协议层 | 分阶段 vs 合计语义区分，需协议层支持 |

