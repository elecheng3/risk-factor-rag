"""Offline demo data: synthetic 10-K documents for two FICTIONAL companies.

Acme Semiconductor (ACME) and Borealis Devices (BORL) do not exist. Their
"filings" are generated here in the same HTML shape real 10-Ks use (table of
contents, Item 1A with bold group headers and bold risk headings, Item 1B),
so the parser, extraction, scoring and Q&A can run without network or keys.
"""

from __future__ import annotations

from .config import Settings
from .store import ingest_html

COMMON = [
    ("Risks Related to Our Industry", [
        ("Demand for our products is cyclical, and an inventory correction at our customers could materially reduce our revenue.",
         "The semiconductor industry is highly cyclical. Customers built inventory during periods of supply shortage and are now reducing it, which lowered our orders in the second half of the year. A prolonged economic slowdown, higher interest rates or weaker consumer demand could cause further significant declines in revenue and margins."),
        ("We face intense competition and pricing pressure from larger competitors.",
         "Several competitors have greater financial and technical resources. Aggressive pricing, new product introductions and rapid technology changes could reduce our market share. If our research and development investments do not produce competitive products on time, our results could be adversely affected."),
    ]),
    ("Risks Related to Our Operations", [
        ("We depend on a limited number of foundries and subcontractors, some of them single sources, for wafer supply and assembly.",
         "We rely on third-party foundries for a substantial portion of our wafers and on subcontractors in Asia for assembly and test. A single source supplies certain substrates. Capacity constraints, quality problems or the loss of any of these suppliers could delay shipments and materially harm our business."),
        ("Disruptions at our manufacturing sites, including natural disasters, could interrupt production.",
         "Our fabs are located in regions exposed to earthquakes, storms and water shortages. Equipment failures, yield problems or power outages have in the past reduced output, and a significant disruption could impair our ability to meet customer demand."),
        ("Cybersecurity incidents could disrupt our operations and expose confidential information.",
         "We are regularly targeted by attempts to breach our information technology systems, including ransomware and phishing. A successful attack could disrupt manufacturing, compromise intellectual property and customer data, and result in significant remediation costs and liability."),
        ("We may be unable to attract and retain key engineering personnel.",
         "Competition for experienced engineers is intense. The loss of key employees or failure to hire qualified talent could delay product development and affect execution."),
    ]),
    ("Legal, Regulatory and Financial Risks", [
        ("Changes in tax laws and ongoing tax audits could adversely affect our results.",
         "We are subject to taxation in multiple jurisdictions. New legislation, including global minimum tax rules, and the outcome of current audits could increase our effective tax rate."),
        ("We could be subject to intellectual property litigation.",
         "Third parties have asserted, and may assert, patent infringement claims against us. Litigation is costly and could require us to pay damages or change product designs."),
        ("Environmental and climate regulations could increase our costs.",
         "Our manufacturing uses chemicals and gases regulated under environmental laws. New climate disclosure rules and emission limits could require capital investment and increase operating costs."),
        ("Our indebtedness could limit our flexibility.",
         "We have outstanding debt that requires interest payments and includes covenants. Higher interest rates or a decline in cash flow could constrain capital allocation and investment."),
    ]),
]

SPECIFIC = {
    "ACME": {
        "base": [
            ("Risks Related to Our Customers", [
                ("A small number of customers account for a significant portion of our revenue.",
                 "Our three largest customers accounted for 46% of revenue. The loss of, or a significant reduction in orders from, any of them would materially and adversely affect our results."),
            ]),
        ],
        "new": [
            ("Risks Related to Our Industry", [
                ("New export control rules restricting sales of advanced chips to China could significantly reduce our revenue.",
                 "Recent regulations issued this year restrict shipments of certain products to customers in China without a license. China represented 24% of our revenue. Licenses may be delayed or denied, and customers may shift to domestic suppliers, which would materially and adversely affect our business."),
                ("Emerging regulation of artificial intelligence could affect demand for our AI accelerator products.",
                 "Governments are proposing new rules on artificial intelligence systems and on the data centers that train them. Such regulation, or a slowdown in generative AI investment, could reduce demand for the accelerator products that drove most of our recent growth."),
            ]),
        ],
    },
    "BORL": {
        "base": [
            ("Risks Related to Our Customers", [
                ("We sell a significant portion of our products through distributors.",
                 "Distributors accounted for 60% of revenue. If distributors reduce inventory, change terms, or fail to pay, our revenue and cash flow could be affected."),
            ]),
        ],
        "new": [
            ("Risks Related to Our Operations", [
                ("Our expansion of a new 300mm fab in Texas may not be completed on schedule or within budget.",
                 "We are building a new manufacturing facility that requires substantial capital. Construction delays, cost overruns, or lower than expected yields during the ramp could materially reduce our margins and return on investment."),
            ]),
        ],
    },
}

COMPANIES = {"ACME": "Acme Semiconductor Corp. (fictional)", "BORL": "Borealis Devices Inc. (fictional)"}


def _html(company: str, groups: list) -> str:
    risks = []
    for header, items in groups:
        risks.append(f'<p><span style="font-weight:700">{header}</span></p>')
        for heading, body in items:
            risks.append(f'<p><span style="font-weight:700;font-style:italic">{heading}</span></p>')
            risks.append(f'<p><span style="font-weight:400">{body}</span></p>')
    return f"""<html><body>
<div><p>{company} Annual Report on Form 10-K</p>
<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A. Risk Factors</td><td>12</td></tr><tr><td>Item 1B. Unresolved Staff Comments</td><td>30</td></tr></table>
<p><b>Item 1. Business</b></p><p>We design and manufacture analog and embedded semiconductors.</p>
<p><b>Item 1A. Risk Factors</b></p>
<p>Investing in our securities involves risk. You should carefully consider the risks described below.</p>
{''.join(risks)}
<p><b>Item 1B. Unresolved Staff Comments</b></p><p>None.</p>
<p><b>Item 2. Properties</b></p><p>Our headquarters are leased.</p></div></body></html>"""


def filings() -> list[tuple[dict, str]]:
    """Two fiscal years per company; the newer year adds the "new" risks and
    drops the IP-litigation risk, so year-over-year detection has something to find."""
    out = []
    for t, name in COMPANIES.items():
        for year, acc_n, extra in ((2024, 1, []), (2025, 2, SPECIFIC[t]["new"])):
            groups = [(h, list(items)) for h, items in COMMON] + SPECIFIC[t]["base"]
            if year == 2025:
                groups = [(h, [i for i in items if "intellectual property" not in i[0]]) for h, items in groups]
                for h, items in extra:
                    for g in groups:
                        if g[0] == h:
                            g[1][0:0] = items
            meta = {"ticker": t, "company": name, "cik": 0, "form": "10-K", "accession": f"DEMO-{t}-{year}-{acc_n}",
                    "filing_date": f"{year + 1}-02-15", "report_date": f"{year}-12-31", "primary_document": "synthetic",
                    "url": "synthetic (riskrag.demo)"}
            out.append((meta, _html(name, groups)))
    return out


def load_demo(cfg: Settings) -> list[str]:
    for meta, html in filings():
        ingest_html(cfg, html, meta)
    return list(COMPANIES)
