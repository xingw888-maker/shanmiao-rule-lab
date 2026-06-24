# API Spec

This is a small local API for rule-engine experiments. It is not a hosted production API and should not be presented as a finished real-document extraction service.

## GET /health

Returns basic service status.

```json
{"status":"ok"}
```

## GET /v1/domains

Lists domain packages discovered under `domains/`.

## POST /v1/validate

Validates text against one loaded domain package.

Request:

```json
{
  "input": {"text": "工程质量标准不得低于国家强制性标准。"},
  "domain": "validated/construction"
}
```

Response shape:

```json
{
  "status": "VALIDATED",
  "evidence_chain": [
    {
      "rule_id": "cn-001",
      "status": "PASSED",
      "matched_terms": [],
      "rationale": "..."
    }
  ],
  "disclaimer": "Research sandbox output. Not professional advice."
}
```

`PASSED`, `FAILED`, and `NOT_APPLICABLE` are rule-engine statuses for the loaded rule package. They are not legal or professional conclusions.
