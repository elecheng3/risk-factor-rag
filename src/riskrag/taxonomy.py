"""Risk taxonomy and industry weighting profiles.

The weights express how much a category matters for a given industry when
computing the weighted risk index (1.0 = neutral). They are a documented
modelling choice, meant to be edited, not ground truth.
"""

CATEGORIES = {
    "supply_chain": "Supplier dependence, single sourcing, materials, logistics, capacity at foundries or subcontractors",
    "geopolitical_trade": "Export controls, tariffs, sanctions, cross-border tensions, operations in specific countries",
    "cybersecurity": "Breaches, ransomware, IT system failures, data protection",
    "regulatory_legal": "Laws, regulation, litigation, IP disputes, tax, compliance",
    "customer_concentration": "Dependence on a few customers, distributors or end markets",
    "technology_competition": "Competitors, pricing pressure, product transitions, R&D execution",
    "macroeconomic_demand": "Cyclical demand, inventory corrections, interest rates, FX, recession",
    "operational_manufacturing": "Fab or plant disruptions, yields, capacity expansion, natural disasters",
    "human_capital": "Hiring, retention, key personnel, labor",
    "financial_liquidity": "Debt, capital allocation, impairments, credit, cash needs",
    "esg_climate": "Climate, environmental regulation, sustainability commitments",
    "ai_related": "Risks specific to AI adoption, AI regulation, or AI demand",
    "other": "Anything that fits none of the above",
}

SEVERITY_RUBRIC = """Severity 1-5, judged from the company's own wording:
1 = generic boilerplate that almost every company discloses
2 = relevant but described as limited or well mitigated
3 = material, with specific exposure described
4 = significant, with quantified exposure or a concrete recent event
5 = critical, described as already affecting results or threatening operations"""

PROFILES = {
    "neutral": {c: 1.0 for c in CATEGORIES},
    "semiconductor": {
        **{c: 1.0 for c in CATEGORIES},
        "supply_chain": 1.5,
        "geopolitical_trade": 1.5,
        "technology_competition": 1.3,
        "operational_manufacturing": 1.3,
        "customer_concentration": 1.2,
        "macroeconomic_demand": 1.2,
        "other": 0.5,
    },
    "fintech": {
        **{c: 1.0 for c in CATEGORIES},
        "cybersecurity": 1.6,
        "regulatory_legal": 1.5,
        "financial_liquidity": 1.3,
        "supply_chain": 0.6,
        "operational_manufacturing": 0.6,
        "other": 0.5,
    },
    "manufacturing": {
        **{c: 1.0 for c in CATEGORIES},
        "supply_chain": 1.5,
        "operational_manufacturing": 1.4,
        "esg_climate": 1.2,
        "human_capital": 1.2,
        "other": 0.5,
    },
}
