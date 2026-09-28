"""Ingest filings from EDGAR (or local HTML files) into per-filing JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .edgar import EdgarClient, Filing
from .sections import RiskItem, extract_risk_items


def _filing_path(cfg: Settings, ticker: str, accession: str) -> Path:
    return cfg.filings_dir / ticker.upper() / f"{accession}.json"


def save_filing(cfg: Settings, meta: dict, items: list[RiskItem]) -> Path:
    path = _filing_path(cfg, meta["ticker"], meta["accession"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"filing": meta, "items": [i.to_dict() for i in items]}, indent=1))
    return path


def ingest_html(cfg: Settings, html: str, meta: dict) -> tuple[Path, int]:
    """Parse one 10-K document; `meta` needs ticker, accession, report_date."""
    prefix = f"{meta['ticker'].upper()}-{meta['report_date'][:4]}"
    items = extract_risk_items(html, prefix)
    return save_filing(cfg, meta, items), len(items)


def ingest_ticker(cfg: Settings, ticker: str, years: int = 2) -> list[tuple[Filing, int]]:
    client = EdgarClient(cfg.sec_user_agent, cfg.raw_dir)
    done = []
    for f in client.ten_k_filings(ticker, limit=years):
        _, n = ingest_html(cfg, client.document(f), f.to_dict())
        done.append((f, n))
    return done


def load_filings(cfg: Settings, ticker: str) -> list[dict]:
    """All ingested filings for a ticker, newest first."""
    d = cfg.filings_dir / ticker.upper()
    if not d.exists():
        return []
    rows = [json.loads(p.read_text()) for p in d.glob("*.json")]
    return sorted(rows, key=lambda r: r["filing"]["report_date"], reverse=True)


def latest(cfg: Settings, ticker: str) -> dict:
    rows = load_filings(cfg, ticker)
    if not rows:
        raise LookupError(f"{ticker} has not been ingested; run: python -m riskrag ingest {ticker}")
    return rows[0]


def ingested_tickers(cfg: Settings) -> list[str]:
    if not cfg.filings_dir.exists():
        return []
    return sorted(p.name for p in cfg.filings_dir.iterdir() if p.is_dir() and any(p.glob("*.json")))
