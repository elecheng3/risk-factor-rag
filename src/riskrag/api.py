"""REST API (FastAPI). Start with:  python -m riskrag serve   ->  http://127.0.0.1:8000/docs"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analysis import compare, risk_changes
from .config import settings
from .extract import extract_ticker, load_extraction
from .llm import PROVIDERS
from .qa import ask
from .store import ingest_ticker, ingested_tickers
from .taxonomy import PROFILES

app = FastAPI(title="Risk-factor RAG", version="0.1.0",
              description="Extract, score and compare risk factors from SEC 10-K filings.")


class IngestRequest(BaseModel):
    tickers: list[str] = Field(examples=[["TXN", "MU"]])
    years: int = 2


class ExtractRequest(BaseModel):
    tickers: list[str]
    provider: str | None = None
    limit: int | None = None


class AskRequest(BaseModel):
    question: str = Field(examples=["How do these companies describe export-control exposure?"])
    tickers: list[str]
    provider: str | None = None


def _provider(p: str | None) -> str:
    p = p or settings().provider
    if p not in PROVIDERS:
        raise HTTPException(400, f"provider must be one of {PROVIDERS}")
    return p


def _run(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@app.get("/health")
def health():
    cfg = settings()
    return {"status": "ok", "default_provider": cfg.provider, "providers_with_keys": cfg.available_providers()}


@app.get("/companies")
def companies():
    return {"tickers": ingested_tickers(settings())}


@app.post("/ingest")
def ingest(req: IngestRequest):
    cfg = settings()
    out = {}
    for t in req.tickers:
        out[t] = [{"report_date": f.report_date, "accession": f.accession, "risk_items": n}
                  for f, n in _run(ingest_ticker, cfg, t, req.years)]
    return out


@app.post("/extract")
def extract(req: ExtractRequest):
    cfg, p = settings(), _provider(req.provider)
    return {t: {k: v for k, v in _run(extract_ticker, cfg, t, p, req.limit).items() if k != "risks"} for t in req.tickers}


@app.get("/risks/{ticker}")
def risks(ticker: str, provider: str | None = None):
    return _run(load_extraction, settings(), ticker, _provider(provider))


@app.get("/compare")
def compare_endpoint(tickers: str, provider: str | None = None, profile: str = "semiconductor"):
    if profile not in PROFILES:
        raise HTTPException(400, f"profile must be one of {list(PROFILES)}")
    r = _run(compare, settings(), [t.strip() for t in tickers.split(",")], _provider(provider), profile)
    return {"items": r["items"].to_dict(), "mean_severity": r["mean_severity"].fillna(0).to_dict(),
            "weighted_index": r["weighted_index"].to_dict()}


@app.get("/changes/{ticker}")
def changes(ticker: str):
    return _run(risk_changes, settings(), ticker).to_dict(orient="records")


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return _run(ask, settings(), req.question, req.tickers, _provider(req.provider))
