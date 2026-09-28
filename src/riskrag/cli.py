"""Command line: python -m riskrag <command> ...

  demo                              offline run on two fictional companies (no keys needed)
  ingest TXN MU --years 2           download and parse the latest 10-Ks from SEC EDGAR
  extract TXN MU --provider openai  classify risk items with an LLM
  compare TXN MU --provider openai --profile semiconductor
  items TXN                         list the risk headings parsed from the latest 10-K
  changes TXN                       risks added / removed versus the prior 10-K
  ask "question" --tickers TXN MU --provider anthropic
  providers TXN --providers openai anthropic gemini --limit 24
  serve                             start the REST API (FastAPI) on :8000
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .config import settings

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)


def _provider(args, cfg) -> str:
    return args.provider or cfg.provider


def main(argv: list[str] | None = None) -> None:
    cfg = settings()
    ap = argparse.ArgumentParser(prog="riskrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo")
    p = sub.add_parser("ingest"); p.add_argument("tickers", nargs="+"); p.add_argument("--years", type=int, default=2)
    p = sub.add_parser("extract"); p.add_argument("tickers", nargs="+"); p.add_argument("--provider"); p.add_argument("--limit", type=int)
    p = sub.add_parser("compare"); p.add_argument("tickers", nargs="+"); p.add_argument("--provider"); p.add_argument("--profile", default="semiconductor")
    p = sub.add_parser("items"); p.add_argument("ticker")
    p = sub.add_parser("changes"); p.add_argument("ticker")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--tickers", nargs="+", required=True); p.add_argument("--provider")
    p = sub.add_parser("providers"); p.add_argument("tickers", nargs="+"); p.add_argument("--providers", nargs="+")
    p.add_argument("--limit", type=int); p.add_argument("--labels"); p.add_argument("--out", default="results")
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        from .demo import load_demo
        from .extract import extract_ticker
        from .analysis import compare, risk_changes
        from .qa import ask

        tickers = load_demo(cfg)
        provider = cfg.provider
        print(f"Ingested fictional companies {tickers}; provider = {provider}\n")
        for t in tickers:
            extract_ticker(cfg, t, provider)
        r = compare(cfg, tickers, provider)
        print("Risk items per category:\n", r["items"], "\n\nWeighted risk index (semiconductor profile):\n", r["weighted_index"])
        print("\nACME year-over-year changes:\n", risk_changes(cfg, "ACME")[["change", "heading"]].to_string(index=False))
        a = ask(cfg, "Which company is more exposed to export controls and China?", tickers, provider)
        print("\nQ&A:", a["answer"], "\nsources:", [s["id"] for s in a["sources"]])
        return

    if args.cmd == "ingest":
        from .store import ingest_ticker
        for t in args.tickers:
            for f, n in ingest_ticker(cfg, t, args.years):
                print(f"{t}: 10-K for {f.report_date} ({f.accession}) -> {n} risk items")
    elif args.cmd == "extract":
        from .extract import extract_ticker
        for t in args.tickers:
            out = extract_ticker(cfg, t, _provider(args, cfg), args.limit)
            u = out["usage"]
            print(f"{t}: {len(out['risks'])} risks via {out['provider']}/{out['model']} | "
                  f"{u['calls']} calls, {u['input_tokens']}+{u['output_tokens']} tokens, {u['latency_s']:.1f}s")
    elif args.cmd == "compare":
        from .analysis import compare
        r = compare(cfg, args.tickers, _provider(args, cfg), args.profile)
        print(r["items"], "\n\nMean severity:\n", r["mean_severity"], "\n\n", r["weighted_index"])
    elif args.cmd == "items":
        from .store import latest
        f = latest(cfg, args.ticker)
        print(f"{f['filing']['company']} 10-K for {f['filing']['report_date']}: {len(f['items'])} risk items\n")
        group = None
        for it in f["items"]:
            if it["group"] != group:
                group = it["group"]
                print(f"[{group or 'no group header'}]")
            print(f"  {it['id']}  ({len(it['text']):>5} chars)  {it['heading'][:110]}")
    elif args.cmd == "changes":
        from .analysis import risk_changes
        print(risk_changes(cfg, args.ticker).to_string(index=False))
    elif args.cmd == "ask":
        from .qa import ask
        print(json.dumps(ask(cfg, args.question, args.tickers, _provider(args, cfg)), indent=2))
    elif args.cmd == "providers":
        from .evaluate import compare_providers, save_comparison
        provs = args.providers or cfg.available_providers()
        if len(provs) < 2:
            sys.exit("Need at least two providers with API keys (or pass --providers ... mock)")
        res = compare_providers(cfg, args.tickers, provs, args.limit, args.labels)
        print(save_comparison(res, __import__("pathlib").Path(args.out)).read_text())
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("riskrag.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
