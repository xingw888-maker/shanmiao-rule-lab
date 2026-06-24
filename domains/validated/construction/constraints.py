from app.engine.solver import LegalConstraint, ViolationSeverity


def build_constraints():
    return [
        LegalConstraint(field="屋面防水工程保修期限", operator=">=", threshold=5.0, unit="年", severity=ViolationSeverity.FATAL, legal_ref="国务院令第279号第40条(二)"),
        LegalConstraint(field="地下室防水工程保修期限", operator=">=", threshold=5.0, unit="年", severity=ViolationSeverity.FATAL, legal_ref="国务院令第279号第40条(二)"),
        LegalConstraint(field="主体结构保修期限", operator=">=", threshold=50.0, unit="年", severity=ViolationSeverity.FATAL, legal_ref="国务院令第279号第40条(一)"),
        LegalConstraint(field="电气管线保修期限", operator=">=", threshold=2.0, unit="年", severity=ViolationSeverity.FATAL, legal_ref="国务院令第279号第40条(四)"),
        LegalConstraint(field="给排水管道保修期限", operator=">=", threshold=2.0, unit="年", severity=ViolationSeverity.FATAL, legal_ref="国务院令第279号第40条(四)"),
        LegalConstraint(field="质量保证金比例上限", operator="<=", threshold=3.0, unit="%", severity=ViolationSeverity.MAJOR, legal_ref="建质[2017]138号第7条"),
        LegalConstraint(field="付款比例合计一致性", operator="<=", threshold=110.0, unit="%", severity=ViolationSeverity.MINOR, legal_ref="合同内部一致性校验"),  # 110% aligns with rules.json cn-010 (100% base + VAT tolerance)
        LegalConstraint(field="缺陷责任期上限", operator="<=", threshold=24.0, unit="月", severity=ViolationSeverity.MAJOR, legal_ref="建质[2017]138号第2条"),
        LegalConstraint(field="竣工验收组织期限", operator="<=", threshold=28.0, unit="天", severity=ViolationSeverity.MAJOR, legal_ref="GF-2017-0201第13.2.2条"),
        LegalConstraint(field="逾期违约金日罚率上限", operator="<=", threshold=0.05, unit="%", severity=ViolationSeverity.MINOR, legal_ref="民法典第585条"),
    ]
