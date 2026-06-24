from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.kernel import ShanmiaoKernel

ROOT = Path(__file__).resolve().parent
kernel = ShanmiaoKernel(domain_path=str(ROOT / "domains"))
app = FastAPI(title="Shanmiao Rule Lab", version="0.1.0")


class ValidateRequest(BaseModel):
    input: dict[str, Any]
    domain: str = "validated/construction"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/domains")
def domains() -> dict[str, Any]:
    return {"domains": kernel.list_domains()}


@app.post("/v1/validate")
def validate(req: ValidateRequest) -> dict[str, Any]:
    text = str(req.input.get("text", ""))
    kernel.load_domain(req.domain)
    result = kernel.validate(text=text, domain_id=req.domain, enable_layers=False)
    result["disclaimer"] = "Research sandbox output. Not professional advice."
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
