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

    # --- Emission outcomes cannot retroactively create the policies that drive them ---
    ("scope_1_ghg_emissions",        "emission_reduction_policy"),
    ("scope_2_ghg_emissions",        "emission_reduction_policy"),
    ("scope_3_ghg_emissions",        "emission_reduction_policy"),

    # --- Emission levels do not determine energy mix decisions ---
    ("scope_1_ghg_emissions",        "renewable_energy_share"),
    ("scope_2_ghg_emissions",        "renewable_energy_share"),

    # --- Operational emissions cannot drive board-level governance strategy ---
    ("scope_1_ghg_emissions",        "board_strategy_esg_oversight"),
    ("scope_2_ghg_emissions",        "board_strategy_esg_oversight"),
    ("scope_3_ghg_emissions",        "board_strategy_esg_oversight"),

    # --- Emissions cannot cause workforce composition decisions ---
    ("scope_1_ghg_emissions",        "diversity_women_representation"),
    ("scope_2_ghg_emissions",        "diversity_women_representation"),
    ("scope_3_ghg_emissions",        "diversity_women_representation"),

    # --- Reporting quality is an outcome: it cannot alter actual ESG performance ---
    ("reporting_quality",            "scope_1_ghg_emissions"),
    ("reporting_quality",            "scope_2_ghg_emissions"),
    ("reporting_quality",            "scope_3_ghg_emissions"),
    ("reporting_quality",            "health_safety"),
    ("reporting_quality",            "diversity_women_representation"),
    ("reporting_quality",            "renewable_energy_share"),

    # --- Workforce diversity follows board strategy, not the reverse ---
    ("diversity_women_representation", "board_strategy_esg_oversight"),

    # --- Health & safety outcomes follow governance, not the other way around ---
    ("health_safety",                "board_strategy_esg_oversight"),
]


# ── Required edges ────────────────────────────────────────────
# Each tuple: (source, target) means "require a causal arrow
# FROM source TO target".
# Use sparingly — only for relationships with very strong
# theoretical and empirical support.

REQUIRED_EDGES: list[tuple[str, str]] = [
    # Emission reduction policy directly drives scope 1 (operational) emissions
    ("emission_reduction_policy",    "scope_1_ghg_emissions"),
    # Renewable energy adoption directly reduces scope 2 (purchased energy) emissions
    ("renewable_energy_share",       "scope_2_ghg_emissions"),
    # Board ESG strategy is the upstream driver of emission policies
    ("board_strategy_esg_oversight", "emission_reduction_policy"),
    # Board oversight drives the quality and rigour of ESG reporting
    ("board_strategy_esg_oversight", "reporting_quality"),
    # Board strategy drives workforce diversity targets
    ("board_strategy_esg_oversight", "diversity_women_representation"),
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
    from causallearn.graph.GraphNode import GraphNode

    col_set  = set(available_columns)
    node_map = {col: GraphNode(col) for col in available_columns}
    bk = BackgroundKnowledge()
    applied_forbidden = 0
    applied_required  = 0
    skipped           = []

    for (src, tgt) in FORBIDDEN_EDGES:
        if src in col_set and tgt in col_set:
            bk.add_forbidden_by_node(node_map[src], node_map[tgt])
            applied_forbidden += 1
        else:
            skipped.append(("forbidden", src, tgt))

    for (src, tgt) in REQUIRED_EDGES:
        if src in col_set and tgt in col_set:
            bk.add_required_by_node(node_map[src], node_map[tgt])
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
        "scope_1_ghg_emissions", "scope_2_ghg_emissions", "scope_3_ghg_emissions",
        "emission_reduction_policy", "renewable_energy_share",
        "diversity_women_representation", "health_safety",
        "board_strategy_esg_oversight", "reporting_quality",
    ]
    bk, f_count, r_count = build_background_knowledge(dummy_cols)
    print(f"\nBackgroundKnowledge object built - {f_count} forbidden, {r_count} required.")
