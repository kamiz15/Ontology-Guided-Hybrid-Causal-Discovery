# 03_build_column_mapping.py
# ============================================================
# Step 3 — Build the column → ontology mapping table.
# Generates a CSV template pre-filled for all known ESG and
# financial columns. Unknown columns are flagged as "REVIEW".
#
# Edit the output CSV manually to fill in causal_role and
# notes for any new columns the supervisor's data introduces.
#
# Usage:
#   python 03_build_column_mapping.py
#   python 03_build_column_mapping.py --input data/processed/data_clean.csv
#
# Output:
#   data/processed/column_mapping.csv
# ============================================================

import argparse
import pandas as pd
from config import CLEAN_DATA_PATH, COLUMN_MAPPING_PATH

# ── Known mappings ────────────────────────────────────────────
# Format: column_name → (esg_domain, ontology_class, unit, causal_role, notes)
KNOWN_MAPPINGS = {
    # -- Environmental --
    "co2_ch4_n2o_scope_1_3":         ("Environmental", "ESG_Metric",           "tCO2e",         "candidate effect", "Scope 1-3 GHG emissions"),
    "carbon_intensity":              ("Environmental", "ESG_Metric",           "tCO2e/output",  "candidate effect", "Derived from emissions + output — cannot cause its own inputs"),
    "emission_reduction_policy":     ("Environmental", "Indicator",            "bool",          "candidate cause",  "Policy existence drives emission outcomes"),
    "carbon_neutral_commitment":     ("Environmental", "Indicator",            "bool",          "candidate cause",  "Strategic commitment drives energy/emission decisions"),
    "air_emissions_sox_nox_pm":      ("Environmental", "ESG_Metric",           "µg/m³",         "candidate effect", "Downstream of industrial process decisions"),
    "land_area_affected":            ("Environmental", "Indicator",            "ha",            "candidate cause",  "Physical footprint driver"),
    "biodiversity_protection_actions":("Environmental","Indicator",            "bool",          "candidate cause",  "Policy-level driver"),
    "fsc_pefc_certified_sourcing":   ("Environmental", "Indicator",            "%",             "candidate cause",  "Supply chain sustainability driver"),
    "climate_risk_assessment_done":  ("Environmental", "Indicator",            "bool",          "candidate cause",  "Risk management input"),
    "resilience_score":              ("Environmental", "ESG_Metric",           "0-100",         "candidate effect", "Composite — derived from risk assessment"),
    "environmental_fines":           ("Environmental", "ESG_Metric",           "Currency",      "candidate effect", "Outcome of compliance failures"),
    "iso_14001_exists":              ("Environmental", "Indicator",            "bool",          "candidate cause",  "Certification drives compliance outcomes"),
    "reporting_quality_score":       ("Environmental", "ESG_Metric",           "0-100",         "candidate effect", "Composite reporting quality"),
    "green_product_revenue":         ("Environmental", "ESG_Metric",           "Currency/%",    "candidate effect", "Outcome of green product strategy"),
    "green_buildings_area":          ("Environmental", "Indicator",            "m²/%",          "candidate cause",  "Physical infrastructure driver"),
    "resource_efficiency_index":     ("Environmental", "ESG_Metric",           "0-100",         "candidate effect", "Composite efficiency score"),
    "total_energy_consumption":      ("Environmental", "ESG_Metric",           "kWh/GJ",        "candidate effect", "Downstream of energy mix decisions"),
    "scope_1_ghg_emissions":         ("Environmental", "ESG_Metric",           "tCO2e",         "candidate effect", "Direct emissions outcome"),
    "scope_2_ghg_emissions":         ("Environmental", "ESG_Metric",           "tCO2e",         "candidate effect", "Purchased energy emissions outcome"),
    "scope_3_ghg_emissions":         ("Environmental", "ESG_Metric",           "tCO2e",         "candidate effect", "Value-chain emissions outcome"),
    "renewable_energy_share":        ("Environmental", "Indicator",            "%",             "candidate cause",  "Energy mix driver"),
    "hazardous_waste_generated":     ("Environmental", "ESG_Metric",           "kg/ton",        "candidate effect", "Industrial process output"),
    "recyclable_packaging_share":    ("Environmental", "Indicator",            "%",             "candidate cause",  "Packaging decision driver"),
    "toxic_spills":                  ("Environmental", "ESG_Metric",           "count",         "candidate effect", "Compliance failure outcome"),
    "waste_recycled_share":          ("Environmental", "Indicator",            "%",             "candidate cause",  "Waste management practice"),
    "water_withdrawal":              ("Environmental", "ESG_Metric",           "m³",            "candidate effect", "Resource consumption outcome"),
    "emf_exposure":                  ("Environmental", "Indicator",            "V/m or W/m²",   "candidate cause",  "Operational risk factor"),
    "gmo_products":                  ("Environmental", "Indicator",            "bool",          "candidate cause",  "Product type driver"),
    "ods_emissions":                 ("Environmental", "ESG_Metric",           "kg CFC-11 eq",  "candidate effect", "Ozone-depleting substance output"),

    # -- Social --
    "safety_transparency_trials":    ("Social",        "Indicator",            "bool",          "candidate cause",  "Governance transparency driver"),
    "collective_bargaining_coverage":("Social",        "Indicator",            "%",             "candidate cause",  "Labour relations driver"),
    "fair_wage_gap":                 ("Social",        "ESG_Metric",           "Currency",      "candidate effect", "Pay equity outcome"),
    "community_investment":          ("Social",        "ESG_Metric",           "Currency",      "candidate effect", "CSR spend outcome"),
    "health_impact_score":           ("Social",        "ESG_Metric",           "0-100",         "candidate effect", "Composite health outcome"),
    "csr_contribution":              ("Social",        "ESG_Metric",           "Currency",      "candidate effect", "CSR financial commitment"),
    "customer_satisfaction_score":   ("Social",        "ESG_Metric",           "0-100",         "candidate effect", "Customer outcome metric"),
    "hiv_program_coverage":          ("Social",        "Indicator",            "%",             "candidate cause",  "Health programme driver"),
    "human_rights_violations":       ("Social",        "ESG_Metric",           "count",         "candidate effect", "Compliance failure outcome"),
    "diversity_representation":      ("Social",        "Indicator",            "%",             "candidate cause",  "Workforce composition driver"),
    "diversity_women_representation":("Social",        "Indicator",            "%",             "candidate cause",  "Gender diversity and representation driver"),
    "health_safety":                 ("Social",        "Indicator",            "mixed",         "candidate cause",  "Workplace health and safety programme indicator"),
    "child_labor_compliance":        ("Social",        "Indicator",            "bool",          "candidate cause",  "Supply chain compliance driver"),
    "indigenous_consent_verification":("Social",       "Indicator",            "bool",          "candidate cause",  "Stakeholder engagement driver"),
    "access_to_services":            ("Social",        "ESG_Metric",           "%",             "candidate effect", "Community access outcome"),
    "healthcare_access_employees":   ("Social",        "Indicator",            "%",             "candidate cause",  "Employee benefit driver"),
    "animal_welfare_compliance":     ("Social",        "Indicator",            "bool",          "candidate cause",  "Operational compliance driver"),
    "training_hours":                ("Social",        "Indicator",            "hrs/employee",  "candidate cause",  "Workforce development driver"),
    "turnover_rate":                 ("Social",        "ESG_Metric",           "%",             "candidate effect", "Downstream of training + compensation"),
    "injury_frequency_rate":         ("Social",        "ESG_Metric",           "rate/100",      "candidate effect", "Safety outcome — downstream of training"),
    "contract_compliance":           ("Social",        "Indicator",            "bool",          "candidate cause",  "Legal compliance driver"),
    "union_membership":              ("Social",        "Indicator",            "%",             "candidate cause",  "Labour relations driver"),
    "product_safety_compliance":     ("Social",        "Indicator",            "%",             "candidate cause",  "Product risk driver"),
    "responsible_marketing_compliance":("Social",      "Indicator",            "bool",          "candidate cause",  "Marketing governance driver"),
    "supplier_audits":               ("Social",        "Indicator",            "bool",          "candidate cause",  "Supply chain governance driver"),

    # -- Governance --
    "esg_oversight_policy":          ("Governance",    "Indicator",            "bool",          "candidate cause",  "Board-level ESG governance driver"),
    "board_diversity":               ("Governance",    "PerformanceIndicator", "%",             "candidate cause",  "Board composition driver"),
    "governance_compliance_score":   ("Governance",    "ESG_Metric",           "0-100",         "candidate effect", "Composite — cannot cause its own inputs"),
    "esg_incentive_bonus":           ("Governance",    "Indicator",            "bool",          "candidate cause",  "Executive incentive structure driver"),
    "assurance_score":               ("Governance",    "ESG_Metric",           "0-100",         "candidate effect", "Third-party assurance outcome"),
    "green_financing":               ("Governance",    "ESG_Metric",           "Currency",      "candidate effect", "Green capital allocation outcome"),
    "board_strategy_esg_oversight":  ("Governance",    "Indicator",            "mixed",         "candidate cause",  "Board-level ESG strategy and oversight driver"),
    "ceo_chair_split":               ("Governance",    "Indicator",            "bool",          "candidate cause",  "Board structure driver"),
    "auditor_independence_score":    ("Governance",    "PerformanceIndicator", "0-100",         "candidate cause",  "Audit quality driver"),
    "ethical_breaches":              ("Governance",    "ESG_Metric",           "count",         "candidate effect", "Compliance failure outcome"),
    "anti_competitive_violations":   ("Governance",    "ESG_Metric",           "count",         "candidate effect", "Regulatory failure outcome"),
    "corruption_cases":              ("Governance",    "ESG_Metric",           "count",         "candidate effect", "Integrity failure outcome"),
    "privacy_compliance":            ("Governance",    "Indicator",            "bool",          "candidate cause",  "Data governance driver"),
    "financial_inclusion":           ("Governance",    "ESG_Metric",           "%",             "candidate effect", "Stakeholder inclusion outcome"),
    "global_compact_membership":     ("Governance",    "Indicator",            "bool",          "candidate cause",  "UN commitment driver"),
    "shareholder_rights_score":      ("Governance",    "PerformanceIndicator", "0-100",         "candidate cause",  "Investor rights driver"),
    "site_closure_plan":             ("Governance",    "Indicator",            "bool",          "candidate cause",  "Operational risk management driver"),
    "tax_transparency_reporting":    ("Governance",    "Indicator",            "bool",          "candidate cause",  "Fiscal governance driver"),
    "lobby_spending":                ("Governance",    "Indicator",            "Currency",      "candidate cause",  "Political influence driver"),
    "reporting_quality":             ("Governance",    "ESG_Metric",           "mixed",         "candidate effect", "Disclosure quality outcome"),
    "systemic_risk_level":           ("Governance",    "ESG_Metric",           "Low/Med/High",  "candidate effect", "Composite risk outcome"),
    "military_connection_risk_head": ("Governance",    "Indicator",            "binary",        "candidate cause",  "Board-level governance risk"),

    # -- Financial --
    "total_asset":                   ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Firm size proxy"),
    "total_debt":                    ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Leverage driver"),
    "total_equity":                  ("Financial",     "Indicator",            "Currency",      "candidate effect", "Outcome of financial decisions"),
    "roa_eat":                       ("Financial",     "ESG_Metric",           "ratio",         "candidate effect", "Financial performance outcome"),
    "roe_eat":                       ("Financial",     "ESG_Metric",           "ratio",         "candidate effect", "Equity performance outcome"),
    "tobins_q":                      ("Financial",     "ESG_Metric",           "ratio",         "candidate effect", "Market valuation vs assets"),
    "eps":                           ("Financial",     "Indicator",            "Currency",      "candidate effect", "Earnings per share"),
    "ebit":                          ("Financial",     "Indicator",            "Currency",      "candidate effect", "Operating profit"),
    "earnings_after_tax":            ("Financial",     "Indicator",            "Currency",      "candidate effect", "Net earnings outcome"),
    "debt_to_equity_ratio":          ("Financial",     "Indicator",            "ratio",         "candidate cause",  "Capital structure driver"),
    "debt_to_asset_ratio":           ("Financial",     "Indicator",            "ratio",         "candidate cause",  "Leverage driver"),
    "pe_ratio":                      ("Financial",     "Indicator",            "ratio",         "candidate effect", "Market pricing outcome"),
    "gross_profit_margin":           ("Financial",     "Indicator",            "%",             "candidate effect", "Operational efficiency outcome"),
    "net_profit_margin":             ("Financial",     "Indicator",            "%",             "candidate effect", "Bottom-line efficiency"),
    "roa_eat":                       ("Financial",     "ESG_Metric",           "ratio",         "candidate effect", "Return on assets"),
    "current_ratio":                 ("Financial",     "Indicator",            "ratio",         "candidate cause",  "Liquidity driver"),
    "quick_ratio":                   ("Financial",     "Indicator",            "ratio",         "candidate cause",  "Liquidity driver"),
    "cash_ratio":                    ("Financial",     "Indicator",            "ratio",         "candidate cause",  "Liquidity driver"),
    "gross_sales":                   ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Revenue driver"),
    "net_sales":                     ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Revenue driver"),
    "total_revenue":                 ("Financial",     "Indicator",            "Currency",      "candidate effect", "Reported revenue outcome"),
    "capital_expenditure":           ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Investment driver"),
    "goodwill":                      ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Intangible asset driver"),
    "retained_earnings":             ("Financial",     "Indicator",            "Currency",      "candidate effect", "Accumulated profit outcome"),
    "market_value_equity":           ("Financial",     "Indicator",            "Currency",      "candidate effect", "Market valuation outcome"),
    "market_price_share":            ("Financial",     "Indicator",            "Currency",      "candidate effect", "Market price outcome"),
    "pbv":                           ("Financial",     "ESG_Metric",           "ratio",         "candidate effect", "Price-to-book ratio"),
    "solvency_ratio":                ("Financial",     "Indicator",            "ratio",         "candidate cause",  "Long-term financial stability"),
    "times_interest_earned":         ("Financial",     "Indicator",            "ratio",         "candidate effect", "Debt service capacity"),
    "asset_growth_pct":              ("Financial",     "Indicator",            "%",             "candidate effect", "Growth outcome"),
    "eat_growth":                    ("Financial",     "Indicator",            "%",             "candidate effect", "Earnings growth outcome"),
    "human_capital_cost":            ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Workforce investment driver"),
    "intangible_assets":             ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Innovation/brand asset driver"),
    "operating_leverage":            ("Financial",     "Indicator",            "ratio",         "candidate cause",  "Cost structure driver"),
    "cash_holding":                  ("Financial",     "Indicator",            "Currency",      "candidate cause",  "Liquidity driver"),
    "dividend_payout_ratio":         ("Financial",     "Indicator",            "%",             "candidate effect", "Shareholder return outcome"),
    "dividend_per_share":            ("Financial",     "Indicator",            "Currency",      "candidate effect", "Dividend outcome"),
    "sustainable_finance_green_financing": ("Financial", "ESG_Metric",         "Currency",      "candidate effect", "Green or sustainable financing outcome"),
}


def build_mapping(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    actual_cols = df.columns.tolist()

    rows = []
    unrecognised = []

    for col in actual_cols:
        if col in KNOWN_MAPPINGS:
            domain, ont_class, unit, causal_role, notes = KNOWN_MAPPINGS[col]
        else:
            domain      = "REVIEW"
            ont_class   = "REVIEW"
            unit        = "REVIEW"
            causal_role = "REVIEW"
            notes       = "Not in known mappings — fill in manually"
            unrecognised.append(col)

        rows.append({
            "column_name":   col,
            "esg_domain":    domain,
            "ontology_class":ont_class,
            "unit":          unit,
            "causal_role":   causal_role,
            "notes":         notes,
        })

    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(output_path, index=False)

    print(f"[mapping] Total columns mapped : {len(rows)}")
    print(f"[mapping] Known mappings applied: {len(rows) - len(unrecognised)}")
    print(f"[mapping] Needs manual review   : {len(unrecognised)}")
    if unrecognised:
        print(f"[mapping] Unrecognised columns: {unrecognised}")
    print(f"[mapping] Saved -> {output_path}")

    return mapping_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build column → ontology mapping table.")
    parser.add_argument("--input",  default=CLEAN_DATA_PATH,   help="Path to clean CSV")
    parser.add_argument("--output", default=COLUMN_MAPPING_PATH, help="Path to save mapping CSV")
    args = parser.parse_args()

    build_mapping(args.input, args.output)
