# Shanmiao Rule Lab

A small deterministic domain-rule validation research sandbox.

This repository is for education, research, and rule-engine experiments. It is not a legal service and does not provide legal advice. Candidate rule packages are experimental and unreviewed.

## Current Scope

- Multi-domain rule pipeline for local experiments.
- 4 reviewed demo domains under `domains/validated/`.
- 5 candidate domains under `domains/candidate/`.
- Road2 numeric benchmark for tracking numeric-expression extraction limits.
- Deterministic evidence-chain output.

Known limitation: this public repository is primarily a rule-validation sandbox, not a finished real-document extraction product. Chinese numeric expressions such as `万分之一` and `二十四` still need stronger extraction. The current benchmark keeps this visible instead of hiding it.

## Install

```bash
git clone https://github.com/xingw888-maker/shanmiao-rule-lab.git
cd shanmiao-rule-lab
pip install -r requirements.txt
```

## Quick Check

```bash
python -m pytest tests/unit/test_handlers.py -q
python tests/road2_eval.py
```

`road2_eval.py` is a diagnostic benchmark. It is expected to expose current misses in real-text numeric extraction; it is not a production-readiness score.

The larger upstream working tree has more local tests. This clean repository intentionally excludes private corpora, server backups, work logs, and real contract text.

## Optional Local API

```bash
python dev_server.py
curl -X POST http://127.0.0.1:8000/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"input":{"text":"工程质量标准不得低于国家强制性标准。"},"domain":"validated/construction"}'
```

## Repository Layout

```text
app/                  engine and pipeline code
domains/validated/    reviewed demo rule packages
domains/candidate/    experimental unreviewed packages
tests/                public smoke tests and Road2 benchmark
docs/                 selected project notes
API_SPEC.md           local API shape
```

## Domain Status

Validated demo domains:

- `validated/construction`
- `validated/foundation`
- `validated/nda`
- `validated/purchase`

Candidate domains:

- `candidate/civil_code`
- `candidate/civil_procedure`
- `candidate/immigration_law`
- `candidate/labor_law`
- `candidate/nationality_law`

Candidate domains are included to exercise the multi-domain loader. They are not reviewed rule products.

## What This Is Not

This project is not:

- a production legal system,
- a contract judgement system,
- a finished real-world document extraction system,
- a substitute for a lawyer or domain expert,
- a hosted service release,
- a collection of authoritative legal rules.

Outputs such as `PASSED`, `FAILED`, and `NOT_APPLICABLE` are rule-engine results for the loaded rule package. They are not professional conclusions.

## T5.1 Note

T5.1 fixed rule reachability in the multi-domain pipeline and added a Road2 numeric benchmark. The remaining benchmark errors are mainly numeric-expression extraction limitations, not rule-dispatch failures.

## License

Apache License 2.0. See `LICENSE`.
