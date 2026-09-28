"""Streamlit dashboard.  Run:  streamlit run dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from riskrag.analysis import compare, risk_changes  # noqa: E402
from riskrag.config import settings  # noqa: E402
from riskrag.extract import load_extraction  # noqa: E402
from riskrag.qa import ask  # noqa: E402
from riskrag.store import ingested_tickers, load_filings  # noqa: E402
from riskrag.taxonomy import PROFILES  # noqa: E402

st.set_page_config(page_title="Risk-factor RAG", layout="wide")
cfg = settings()

st.title("10-K Risk Factor Explorer")
tickers_all = ingested_tickers(cfg)
if not tickers_all:
    st.info("No filings yet. Run `riskrag demo` (offline) or `riskrag ingest TXN MU --years 2`.")
    st.stop()

tickers = st.multiselect("Companies", tickers_all, default=tickers_all[:4])
if not tickers:
    st.stop()

extracted = sorted(p.name for p in cfg.extractions_dir.iterdir()) if cfg.extractions_dir.exists() else []
tab_items, tab_compare, tab_changes, tab_ask = st.tabs(
    ["Risk factors", "Compare companies", "Year-over-year changes", "Ask the filings"])

# ---- 1. parsed risk factors (no LLM needed) ---------------------------------------
with tab_items:
    t = st.selectbox("Company", tickers, key="items_ticker")
    filings = load_filings(cfg, t)
    years = {f"{f['filing']['report_date']}  ({len(f['items'])} risks)": f for f in filings}
    f = years[st.selectbox("10-K", list(years))]
    meta = f["filing"]
    st.caption(f"{meta['company']} · filed {meta['filing_date']} · [source document]({meta['url']})"
               if meta["url"].startswith("http") else f"{meta['company']} · {meta['url']}")

    # classification from any provider that has processed this filing
    labels = {}
    for p in extracted:
        path = cfg.extractions_dir / p / f"{t}_{meta['accession']}.json"
        if path.exists():
            labels = {r["id"]: r for r in pd.read_json(path, typ="series")["risks"]}
            st.caption(f"Categories from **{p}** extraction")
            break

    items = pd.DataFrame(f["items"])
    items["chars"] = items["text"].str.len()
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk items", len(items))
    c2.metric("Group headers", items["group"].replace("", pd.NA).nunique())
    c3.metric("Median length (chars)", int(items["chars"].median()))

    fig = px.bar(items, x="id", y="chars", hover_data={"heading": True, "id": False},
                 labels={"chars": "Characters", "id": ""}, height=260)
    fig.update_traces(marker_color="#2a78d6")
    fig.update_layout(title="Length of each risk item (very long bars may be several risks merged)",
                      xaxis_showticklabels=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    query = st.text_input("Filter by keyword", "")
    shown = items[items["heading"].str.contains(query, case=False) | items["text"].str.contains(query, case=False)] \
        if query else items
    for group, rows in shown.groupby("group", sort=False):
        st.markdown(f"#### {group or 'Risk factors'}")
        for _, r in rows.iterrows():
            lab = labels.get(r["id"])
            tag = f"  ·  `{lab['category']}` severity {lab['severity']}" if lab else ""
            with st.expander(f"{r['heading']}{tag}"):
                if lab:
                    st.info(lab["summary"])
                st.write(r["text"])
                st.caption(f"{r['id']} · {r['chars']:,} characters")

# ---- 2. comparison (needs an LLM extraction) ---------------------------------------
with tab_compare:
    if not extracted:
        st.info("Run `riskrag extract " + " ".join(tickers) + "` to classify the risks, then refresh.")
    else:
        c1, c2 = st.columns(2)
        provider = c1.selectbox("Extraction provider", extracted)
        profile = c2.selectbox("Industry weighting", list(PROFILES), index=list(PROFILES).index("semiconductor"))
        try:
            r = compare(cfg, tickers, provider, profile)
            left, right = st.columns([2, 1])
            fig = px.imshow(r["mean_severity"].fillna(0), text_auto=".1f", color_continuous_scale="Blues",
                            zmin=0, zmax=5, aspect="auto", labels={"color": "Mean severity (1-5)"})
            fig.update_layout(title="Mean severity by risk category (0 = not disclosed)", height=520)
            left.plotly_chart(fig, width="stretch")
            fig2 = px.bar(r["weighted_index"].sort_values(), orientation="h",
                          labels={"value": "Weighted risk index (0-100)", "index": ""})
            fig2.update_traces(marker_color="#2a78d6")
            fig2.update_layout(showlegend=False, title=f"Weighted risk index ({profile} profile)", height=520)
            right.plotly_chart(fig2, width="stretch")
            st.dataframe(r["items"], width="stretch")
        except LookupError as e:
            st.warning(str(e))

# ---- 3. year-over-year changes -------------------------------------------------------
with tab_changes:
    for t in tickers:
        try:
            ch = risk_changes(cfg, t)
            st.markdown(f"**{t}**: {int((ch.change == 'added').sum())} added, "
                        f"{int((ch.change == 'removed').sum())} removed versus the prior 10-K")
            if len(ch):
                st.dataframe(ch, hide_index=True, width="stretch")
        except LookupError as e:
            st.caption(f"{t}: {e}")

# ---- 4. Q&A --------------------------------------------------------------------------
with tab_ask:
    q = st.text_input("Question", "How do these companies describe their exposure to export controls?")
    options = [cfg.provider] + [p for p in cfg.available_providers() if p != cfg.provider]
    qa_provider = st.selectbox("Answer with", options)
    if st.button("Ask") and q:
        try:
            with st.spinner("Retrieving passages and asking the model..."):
                a = ask(cfg, q, tickers, qa_provider)
            st.write(a["answer"])
            if a["invalid_citations"]:
                st.warning(f"Model cited passages that were not retrieved: {a['invalid_citations']}")
            st.dataframe(a["sources"], hide_index=True, width="stretch")
            st.caption(f"{a['provider']}/{a['model']} · {a['tokens']['input']}+{a['tokens']['output']} tokens")
        except (RuntimeError, LookupError) as e:
            st.error(str(e))
