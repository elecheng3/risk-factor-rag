"""SEC EDGAR client: ticker -> CIK -> latest 10-K filings -> primary document.

SEC requires a descriptive User-Agent ("Name email@example.com") and at most
10 requests per second: https://www.sec.gov/os/accessing-edgar-data
Downloaded documents are cached under data/raw so each filing is fetched once.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


@dataclass
class Filing:
    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document)

    def to_dict(self) -> dict:
        return asdict(self) | {"url": self.url}


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path, min_interval: float = 0.15):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with your name and email, e.g. "
                "SEC_USER_AGENT='Jane Doe jane@example.com' in your .env"
            )
        self.http = httpx.Client(headers={"User-Agent": user_agent}, timeout=30, follow_redirects=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0

    def _get(self, url: str) -> httpx.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self.http.get(url)
        r.raise_for_status()
        return r

    def _cached_json(self, url: str, name: str) -> dict:
        path = self.cache_dir / name
        if path.exists():
            return json.loads(path.read_text())
        data = self._get(url).json()
        path.write_text(json.dumps(data))
        return data

    def lookup(self, ticker: str) -> tuple[int, str]:
        data = self._cached_json(TICKERS_URL, "company_tickers.json")
        for row in data.values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"]), row["title"]
        raise KeyError(f"Ticker {ticker} not found in SEC company list")

    def ten_k_filings(self, ticker: str, limit: int = 2) -> list[Filing]:
        """Most recent 10-K filings (amendments excluded), newest first."""
        cik, company = self.lookup(ticker)
        sub = self._cached_json(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik}.json")
        recent = sub["filings"]["recent"]
        out = []
        for i, form in enumerate(recent["form"]):
            if form != "10-K":
                continue
            out.append(
                Filing(
                    ticker=ticker.upper(),
                    company=company,
                    cik=cik,
                    form=form,
                    accession=recent["accessionNumber"][i],
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    primary_document=recent["primaryDocument"][i],
                )
            )
            if len(out) == limit:
                break
        if not out:
            raise LookupError(f"No 10-K filings found for {ticker} (foreign issuers file 20-F instead)")
        return out

    def document(self, filing: Filing) -> str:
        path = self.cache_dir / f"{filing.ticker}_{filing.accession}.html"
        if path.exists():
            return path.read_text(errors="ignore")
        html = self._get(filing.url).text
        path.write_text(html)
        return html
