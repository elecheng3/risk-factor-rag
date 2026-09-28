# Risk-factor RAG

Pull the **Risk Factors** section (Item 1A) out of companies' SEC 10-K filings, classify every
risk with an LLM, score companies against an industry-weighted index, flag risks that were added
or dropped since last year, and ask questions across companies with cited answers.

The LLM layer is **switchable between OpenAI, Anthropic and Gemini** with one setting, and the
same extraction can be run through all three to measure how much they agree.

## What it does

```
SEC EDGAR API ──► 10-K HTML ──► Item 1A parser ──► risk items (heading + text)
                                                        │
                    ┌───────────────────────────────────┼─────────────────────────┐
                    ▼                                   ▼                         ▼
          LLM classification                  TF-IDF retrieval            year-over-year diff
   (category, severity 1-5, emerging)          + cited answers           (added / removed risks)
                    │                                   │
                    ▼                                   ▼
     industry-weighted risk index        REST API (FastAPI) · Streamlit dashboard · CLI
```

- **Ingestion:** looks up a ticker's CIK, finds its latest 10-K filings through the SEC EDGAR
  submissions API, downloads each primary document (cached, rate-limited as SEC requires).
- **Parsing:** finds the real Item 1A (not the table-of-contents entry), then splits it into
  individual risks using the bold/italic headings filings use, keeping group headers such as
  "Risks Related to Our Operations". Falls back to fixed-size chunks when a filing has no headings.
- **Extraction:** batches risk items to the chosen LLM, which returns JSON validated with
  pydantic: one of 13 categories, severity on a written 1-5 rubric, an "emerging" flag and a short
  summary. Invalid JSON gets one repair attempt; items the model skips are flagged, never dropped.
- **Scoring:** per company and category, item counts and mean severity, plus a 0-100 index weighted
  by industry profile (semiconductor, fintech, manufacturing, neutral; see `taxonomy.py`).
- **Change detection:** compares risk headings with the prior year's 10-K by TF-IDF similarity to
  list risks that were added or removed.
- **Q&A:** retrieves passages balanced across the selected companies, asks the LLM to answer only
  from them with `[ID]` citations, and reports any citation that was not in the retrieved set.
- **Operations:** every LLM response is cached on disk (re-runs cost nothing), and tokens and
  latency per call are logged to `data/usage.jsonl`.

## Switching providers

| Provider | Default model (cheapest general model, Sep 2026) | Key variable |
| --- | --- | --- |
| OpenAI | `gpt-6-luna` | `OPENAI_API_KEY` |
| Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` |
| mock | keyword rules, offline | none |

Set `LLM_PROVIDER` in `.env`, or pass `--provider` to any command. Model names can be changed with
`OPENAI_MODEL`, `ANTHROPIC_MODEL` and `GEMINI_MODEL`.

## Quick start

```bash
pip install -e .                # installs the package and the `riskrag` command
riskrag demo                    # runs everything offline on two fictional companies
```

The demo needs no API key or network: `riskrag/demo.py` generates 10-K-shaped documents for two
**fictional** companies, and the `mock` provider is a keyword classifier standing in for an LLM.

### Real filings

```bash
cp .env.example .env            # add SEC_USER_AGENT and at least one API key

riskrag ingest TXN MU INTC ADI --years 2
riskrag extract TXN MU INTC ADI --provider openai
riskrag compare TXN MU INTC ADI --provider openai --profile semiconductor
riskrag items TXN                # check how Item 1A was split
riskrag changes MU
riskrag ask "How does each company describe export-control exposure?" --tickers TXN MU INTC ADI --provider anthropic

# same risk items through every provider you have keys for
riskrag providers TXN MU --providers openai anthropic gemini

streamlit run dashboard.py      # dashboard
riskrag serve         # REST API, docs at http://127.0.0.1:8000/docs
```

Only US filers work: foreign issuers such as TSMC file a 20-F, which has a different structure.

## Comparing providers

`riskrag providers ...` sends the identical risk items through each provider and writes
`results/provider_comparison.md` with:

- tokens, latency per item and items the model skipped, per provider
- category agreement, Cohen's kappa and mean severity difference, per provider pair
- accuracy against your own labels, if you pass `--labels eval/labels.csv`
  (columns `id,category,severity`; a template is in `eval/labels_template.csv`)

Agreement between models is not accuracy, so the labelled check is what shows whether a cheaper
model is good enough.

## REST API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | status and which providers have keys |
| GET | `/companies` | ingested tickers |
| POST | `/ingest` | `{"tickers": ["TXN"], "years": 2}` |
| POST | `/extract` | `{"tickers": ["TXN"], "provider": "gemini"}` |
| GET | `/risks/{ticker}` | classified risks for the latest 10-K |
| GET | `/compare?tickers=TXN,MU&profile=semiconductor` | category matrices and weighted index |
| GET | `/changes/{ticker}` | risks added or removed versus the prior 10-K |
| POST | `/ask` | `{"question": "...", "tickers": ["TXN", "MU"]}` |

## Design choices

- **TF-IDF retrieval instead of an embedding API.** It behaves identically whichever LLM is chosen
  (Anthropic has no embeddings endpoint), costs nothing, and risk-factor text is keyword-dense.
  An embedding retriever would be the next upgrade for paraphrased questions.
- **Batching.** Eight risk items per call keeps prompts small while cutting per-call overhead.
- **The weighted index is a modelling choice.** Weights live in `taxonomy.py` and are meant to be
  argued with; the per-category tables are shown next to the index for that reason.

## Tests

```bash
python -m pytest -q
```

14 offline tests cover the parser (table-of-contents skipping, group headers, fallback chunking),
the full pipeline on the demo filings, all three provider backends with their SDK clients faked
(request shape, usage parsing, response caching), the JSON repair path, the provider comparison
metrics and the REST API.

## Project structure

```
src/riskrag/edgar.py      SEC EDGAR client (ticker -> CIK -> 10-K documents)
src/riskrag/sections.py   Item 1A extraction and risk splitting
src/riskrag/llm.py        OpenAI / Anthropic / Gemini / mock behind one interface, cache, usage log
src/riskrag/extract.py    LLM classification with pydantic validation
src/riskrag/analysis.py   category tables, weighted index, year-over-year changes
src/riskrag/qa.py         retrieval and cited question answering
src/riskrag/evaluate.py   cross-provider comparison
src/riskrag/api.py        FastAPI app
src/riskrag/cli.py        command line
src/riskrag/demo.py       synthetic filings for two fictional companies
dashboard.py              Streamlit dashboard
```

## Background

Inspired by risk-research work during a consulting internship, where comparing risk disclosures
across companies was a manual, weeks-long task. This is an independent rebuild on public SEC data;
it contains no code, data or material from that engagement.

## Author

Yu Hua (Eleanor) Cheng, M.S. Technology Management, University of Illinois Urbana-Champaign.
