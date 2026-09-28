"""Streamlit dashboard.  Run:  streamlit run dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from riskrag.analysis import compare, risk_changes  # noqa: E402
from riskrag.config import settings  # noqa: E402
from riskrag.qa import ask  # noqa: E402
from riskrag.store import ingested_tickers  # noqa: E402
from riskrag.taxonomy import PROFILES  # noqa: E402

st.set_page_config(page_title="Risk-factor RAG", layout="wide")
cfg = settings()

st.title("10-K Risk Factor Comparison")
tickers_all = ingested_tickers(cfg)
if not tickers_all:
    st.info("No filings yet. Run `python -m riskrag demo` (offline) or `python -m riskrag ingest TXN MU`.")
    st.stop()

extracted = [p.name for p in cfg.extractions_dir.iterdir()] if cfg.extractions_dir.exists() else []
c1, c2, c3 = st.columns([3, 1, 1])
tickers = c1.multiselect("Companies", tickers_all, default=tickers_all[:4])
provider = c2.selectbox("Extraction provider", extracted or [cfg.provider])
profile = c3.selectbox("Industry weighting", list(PROFILES), index=list(PROFILES).index("semiconductor"))
if not tickers:
    st.stop()

try:
    r = compare(cfg, tickers, provider, profile)
except LookupError as e:
    st.warning(str(e))
    st.stop()

left, right = st.columns([2, 1])
sev = r["mean_severity"].fillna(0)
fig = px.imshow(sev, text_auto=".1f", color_continuous_scale="Blues", zmin=0, zmax=5, aspect="auto",
                labels={"color": "Mean severity (1-5)"})
fig.update_layout(title="Mean severity by risk category (0 = not disclosed)", height=520)
left.plotly_chart(fig, width="stretch")

fig2 = px.bar(r["weighted_index"].sort_values(), orientation="h", labels={"value": "Weighted risk index (0-100)", "index": ""})
fig2.update_traces(marker_color="#2a78d6")
fig2.update_layout(showlegend=False, title=f"Weighted risk index ({profile} profile)", height=520)
right.plotly_chart(fig2, width="stretch")

st.subheader("What changed versus the prior 10-K")
for t in tickers:
    try:
        ch = risk_changes(cfg, t)
        st.markdown(f"**{t}** — {int((ch.change == 'added').sum())} added, {int((ch.change == 'removed').sum())} removed")
        if len(ch):
            st.dataframe(ch, hide_index=True, width="stretch")
    except LookupError as e:
        st.caption(f"{t}: {e}")

st.subheader("Ask the filings")
q = st.text_input("Question", "How do these companies describe their exposure to export controls?")
qa_provider = st.selectbox("Answer with", [cfg.provider] + [p for p in cfg.available_providers() if p != cfg.provider])
if st.button("Ask") and q:
    with st.spinner("Retrieving passages and asking the model..."):
        a = ask(cfg, q, tickers, qa_provider)
    st.write(a["answer"])
    if a["invalid_citations"]:
        st.warning(f"Model cited passages that were not retrieved: {a['invalid_citations']}")
    st.dataframe(a["sources"], hide_index=True, width="stretch")
    st.caption(f"{a['provider']}/{a['model']} · {a['tokens']['input']}+{a['tokens']['output']} tokens")
