#!/usr/bin/env python3
"""T5.2 -- DeepSeek LLM structured field extractor for numeric_comparison rules.

Extracts numeric values from contract clause text for 15 construction-domain
numeric_comparison fields. Returns structured JSON -- does NOT make compliance
judgments (that stays in the handler).

Usage:
    from tests.road2_llm_extractor import NumericFieldExtractor
    extractor = NumericFieldExtractor()
    result = extractor.extract("foo")

Config from env vars: DEEPSEEK_API_KEY, DEEPSEEK_API_URL
"""

from __future__ import annotations

import json, os, sys, time, urllib.request
from pathlib import Path
from typing import Any

class _MockObj: ...
_fake_ps = type(sys)("pydantic_settings"); _fake_ps.BaseSettings = _MockObj
_fake_pd = type(sys)("pydantic"); _fake_pd.BaseModel = object
sys.modules["pydantic_settings"] = _fake_ps
sys.modules["pydantic"] = _fake_pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import app.config
app.config.settings = _MockObj()
for _a in ["RUST_ENABLED","LLM_API_URL","LLM_API_KEY","LLM_MODEL",
           "RATE_LIMIT_PER_MINUTE","DEFAULT_TIMEOUT_MS","MAX_INPUT_CHARS","DATABASE_URL"]:
    setattr(app.config.settings, _a, False if _a=="RUST_ENABLED" else "")

DEEPSEEK_URL = os.environ.get("DEEPSEEK_API_URL",
    os.environ.get("LLM_API_URL","https://api.deepseek.com/v1/chat/completions"))
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("LLM_API_KEY",""))
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_MODEL","deepseek-chat"))
CONFIDENCE_FALLBACK_THRESHOLD = 0.8
REQUEST_DELAY_SEC = 0.0

NUMERIC_FIELDS = [
    {"label":"屋面防水保修期","unit":"年","rule_id":"cn-001",
     "guidance":"屋面防水工程保修年限。"},
    {"label":"地下室防水保修期","unit":"年","rule_id":"cn-002",
     "guidance":"地下室防水防渗漏保修年限。"},
    {"label":"主体结构保修期","unit":"年","rule_id":"cn-003",
     "guidance":"主体结构或地基基础保修年限。合理使用年限不提取。"},
    {"label":"电气管线保修期","unit":"年","rule_id":"cn-004",
     "guidance":"电气管线或设备安装工程保修年限。"},
    {"label":"给排水管道保修期","unit":"年","rule_id":"cn-005",
     "guidance":"给排水管道保修年限。"},
    {"label":"质量保证金比例","unit":"%","rule_id":"cn-006",
     "guidance":"质量保证金或质保金占工程价款百分比。百分之三=3。"},
    {"label":"缺陷责任期","unit":"月","rule_id":"cn-007",
     "guidance":"缺陷责任期月数。年换算为月:2年=24月。"},
    {"label":"竣工验收组织期限","unit":"天","rule_id":"cn-008",
     "guidance":"发包人组织竣工验收的天数。28天提交结算资料不是验收期限。"},
    {"label":"逾期违约金日罚率","unit":"%","rule_id":"cn-009",
     "guidance":"逾期违约金日罚率。万分之一=0.01,万分之五=0.05,千分之一=0.1。"},
    {"label":"付款比例","unit":"%","rule_id":"cn-010",
     "guidance":"各阶段付款比例:预付款、进度款、结算款、质保金。分别提取。"},
    {"label":"付款逾期利息","unit":"%","rule_id":"cn-019",
     "guidance":"付款逾期利息日率。千分之一=0.1。"},
    {"label":"履约担保比例","unit":"%","rule_id":"cn-020",
     "guidance":"履约担保或履约保函占合同价百分比。"},
    {"label":"投标保证金比例","unit":"%","rule_id":"cn-021",
     "guidance":"投标保证金占估算价百分比。"},
    {"label":"单次付款比例","unit":"%","rule_id":"cn-026",
     "guidance":"单次付款至百分比。80%、97%等。"},
    {"label":"付款期限","unit":"日","rule_id":"cn-027",
     "guidance":"付款天数。30日内支付=30。付款后15工作日发货不是付款期限。"},
]
FIELD_LABELS_BY_RULE = {f["rule_id"]: f["label"] for f in NUMERIC_FIELDS}


def _build_extraction_prompt(text: str) -> str:
    fds = "\n".join(f'  "{f["label"]}": {f["guidance"]}' for f in NUMERIC_FIELDS)
    return f"""你是合同条款结构化数值提取器。从文本中提取数值字段。

字段定义:
{fds}

规则:
1. 中文数字转换: 五年=5, 二十四个月=24, 百分之三=3
2. 中文分数转换: 万分之一=0.01, 万分之五=0.05, 千分之一=0.1
3. 单位保持原样, 缺陷责任期年换月(2年=24月)
4. 严禁自行推理默认值! 文本没有明确数值就不输出。防水材料进场复试不含保修年限不输出。不要猜测法定值。
5. confidence: 0.9+文本明确提到, 0.8-0.9弱推断, <0.8省略
6. 只输出JSON

文本: {text}

示例: {{"屋面防水保修期":{{"value":5,"unit":"年","source_text":"保修期为五年","confidence":0.95}}}}"""


def _call_deepseek(prompt: str) -> dict:
    if not DEEPSEEK_KEY:
        return {"error": "no key"}
    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role":"system","content":"你是精确的合同条款数值提取器。只输出JSON。禁止推理默认值。"},
            {"role":"user","content":prompt},
        ],
        "max_tokens": 800, "temperature": 0,
        "response_format": {"type":"json_object"},
    }).encode()
    req = urllib.request.Request(DEEPSEEK_URL, data=payload,
        headers={"Authorization":f"Bearer {DEEPSEEK_KEY}","Content-Type":"application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=45)
        return json.loads(json.loads(resp.read().decode())["choices"][0]["message"]["content"])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def parse_extractions(raw: dict) -> list[dict]:
    results = []
    for label, data in raw.items():
        if not isinstance(data, dict): continue
        if data.get("value") is None: continue
        try: value = float(data["value"])
        except (TypeError, ValueError): continue
        conf = float(data.get("confidence",0.5))
        if conf < CONFIDENCE_FALLBACK_THRESHOLD: continue
        results.append({"field_label":label,"value":value,"unit":data.get("unit",""),
                        "source_text":data.get("source_text",""),"confidence":conf})
    return results


class NumericFieldExtractor:
    def __init__(self, api_key: str|None=None, delay_sec: float|None=None):
        self.api_key = api_key or DEEPSEEK_KEY
        self.delay_sec = delay_sec if delay_sec is not None else REQUEST_DELAY_SEC
        self._last_call_at = 0.0

    def extract(self, text: str) -> list[dict]:
        if not self.api_key: return []
        elapsed = time.time() - self._last_call_at
        if elapsed < self.delay_sec: time.sleep(self.delay_sec - elapsed)
        raw = _call_deepseek(_build_extraction_prompt(text))
        self._last_call_at = time.time()
        if "error" in raw: print(f"  [LLM] {raw['error']}"); return []
        return parse_extractions(raw)

    def extract_for_sample(self, sample: dict) -> dict[str,dict]:
        extras = self.extract(sample["text"])
        result = {}
        for ext in extras:
            label = ext["field_label"]
            rid = FIELD_LABELS_BY_RULE.get(label)
            if rid is None:
                for r_id, r_lb in FIELD_LABELS_BY_RULE.items():
                    if r_lb in label or label in r_lb: rid = r_id; break
            if rid: result[rid] = ext
        return result
