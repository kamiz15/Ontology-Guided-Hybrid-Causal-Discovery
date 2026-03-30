# 04_forbidden_edges.py
# ============================================================
# Step 4 — Ontology-derived causal constraints.
# Defines forbidden and required edges based on ESG domain
# knowledge and the ESGOnt ontology relationships.
#
# These constraints are injected into the PC algorithm via
# causal-learn's BackgroundKnowledge API in later scripts.
#
# Usage (standalone check):
#   python 04_forbidden_edges.py
#
# Usage (imported by pipeline scripts):
#   from forbidden_edges import FORBIDDEN_EDGES, REQUIRED_EDGES
#   from forbidden_edges import build_background_knowledge
# ============================================================

from __future__ import annotations


# ── Forbidden edges ───────────────────────────────────────────
# Each tuple: (source, target) means "forbid a causal arrow
# FROM source TO target".
# Rationale is in the comment on each line.

FORBIDDEN_EDGES: list[tuple[str, str]] = [

    # --- Composite scores cannot cause their own raw inputs ---
    ("carbon_intensity",            "co2_ch4_n2o_scope_1_3"),
    ("governance_compliance_score", "board_diversity"),
    ("governance_compliance_score", "esg_oversight_policy"),
    ("governance_compliance_score", "auditor_independence_score"),
    ("resilience_score",            "climate_risk_assessment_done"),
    ("reporting_quality_score",     "iso_14001_exists"),
    ("health_impact_score",         "training_hours"),
    ("health_impact_score",         "healthcare_access_employees"),
    ("resource_efficiency_index",   "renewable_energy_share"),
    ("resource_efficiency_index",   "total_energy_consumption"),

    # --- Outcomes cannot precede their drivers (temporal logic) ---
    ("injury_frequency_rate",       "training_hours"),
    ("turnover_rate",               "training_hours"),
    ("carbon_intensity",            "renewable_energy_share"),
    ("environmental_fines",         "iso_14001_exists"),
    ("ethical_breaches",            "esg_oversight_policy"),
    ("corruption_cases",            "esg_oversight_policy"),

    # --- Financial ratios cannot cause their own numerator/denominator ---
    ("roa_eat",                     "total_asset"),
    ("roa_eat",                     "earnings_after_tax"),
    ("roe_eat",                     "total_equity"),
    ("roe_eat",                     "earnings_after_tax"),
    ("debt_to_equity_ratio",        "total_debt"),
    ("debt_to_equity_ratio",        "total_equity"),
    ("pe_ratio",                    "market_price_share"),
    ("pe_ratio",                    "eps"),
    ("pbv",                         "market_value_equity"),
    ("gross_profit_margin",         "gross_sales"),
    ("net_profit_margin",           "net_sales"),

    # --- Cross-domain impossibilities (E ↛ G without operations) ---
    ("co2_ch4_n2o_scope_1_3",      "board_diversity"),
    ("co2_ch4_n2o_scope_1_3",      "ceo_chair_split"),
    ("water_withdrawal",            "ceo_chair_split"),
    ("hazardous_waste_generated",   "board_diversity"),

    # --- Lag variables cannot be caused by current values ---
    # (lag = prior period snapshot — causality runs forward in time)
    ("net_sales",                   "lag_net_sales"),
    ("total_asset",                 "lag_total_asset"),
    ("market_price_share",          "lag_market_price"),
]


# ── Required edges ────────────────────────────────────────────
# Each tuple: (source, target) means "require a causal arrow
# FROM source TO target".
# Use sparingly — only for relationships with very strong
# theoretical and empirical support.

REQUIRED_EDGES: list[tuple[str, str]] = [
    ("emission_reduction_policy",   "co2_ch4_n2o_scope_1_3"),
    ("renewable_energy_share",      "total_energy_consumption"),
    ("training_hours",              "injury_frequency_rate"),
    ("board_diversity",             "governance_compliance_score"),
    ("esg_oversight_policy",        "governance_compliance_score"),
]


# ── Build BackgroundKnowledge object for causal-learn ─────────
def build_background_knowledge(available_columns: list[str]):
    """
    Returns a causal-learn BackgroundKnowledge object with all
    forbidden and required edges that apply to the given columns.

    Parameters
    ----------
    available_columns : list of column names in the dataset

    Returns
    -------
    bk : BackgroundKnowledge
    applied_forbidden : int
    applied_required  : int
    """
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    col_set = set(available_columns)
    bk = BackgroundKnowledge()
    applied_forbidden = 0
    applied_required  = 0
    skipped           = []

    for (src, tgt) in FORBIDDEN_EDGES:
        if src in col_set and tgt in col_set:
            bk.add_forbidden_by_node(src, tgt)
            applied_forbidden += 1
        else:
            skipped.append(("forbidden", src, tgt))

    for (src, tgt) in REQUIRED_EDGES:
        if src in col_set and tgt in col_set:
            bk.add_required_by_node(src, tgt)
            applied_required += 1
        else:
            skipped.append(("required", src, tgt))

    print(f"[constraints] Forbidden edges applied : {applied_forbidden}")
    print(f"[constraints] Required edges applied  : {applied_required}")
    if skipped:
        print(f"[constraints] Skipped (cols not in data): {len(skipped)}")
        for kind, s, t in skipped:
            print(f"    {kind}: {s} -> {t}")

    return bk, applied_forbidden, applied_required


if __name__ == "__main__":
    print(f"Forbidden edges defined : {len(FORBIDDEN_EDGES)}")
    print(f"Required edges defined  : {len(REQUIRED_EDGES)}")
    print("\nVerifying build_background_knowledge with dummy column list...")

    dummy_cols = [
        "co2_ch4_n2o_scope_1_3", "emission_reduction_policy",
        "renewable_energy_share", "total_energy_consumption",
        "training_hours", "injury_frequency_rate",
        "board_diversity", "governance_compliance_score",
        "esg_oversight_policy",
    ]
    bk, f_count, r_count = build_background_knowledge(dummy_cols)
    print(f"\nBackgroundKnowledge object built — {f_count} forbidden, {r_count} required.")
