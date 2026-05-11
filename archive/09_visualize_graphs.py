# 09_visualize_graphs.py
# ============================================================
# Causal Graph Visualization & Comparison Figures — Upgraded.
#
# Key upgrades over the previous version:
#   1. Directional arrows clearly visible (larger, offset from nodes,
#      curved so bidirectional pairs show as two separate arcs).
#   2. Edge width & opacity scale with |weight|  -  strong edges pop,
#      weak edges fade into the background.
#   3. Optional weight threshold (--min-weight) filters out noise.
#   4. Graphviz "dot" layout for top-down causal flow
#      (falls back gracefully if pygraphviz is not installed).
#   5. Domain-split subgraph panels (E-only, S-only, G-only, F-only,
#      cross-domain, plus a top-K strongest-edges panel).
#
# Usage:
#   python 09_visualize_graphs.py
#   python 09_visualize_graphs.py --subgraphs
#   python 09_visualize_graphs.py --min-weight 0.3 --subgraphs
#   python 09_visualize_graphs.py --top-n 25 --dpi 300 --subgraphs --interactive
# ============================================================

from __future__ import annotations
import argparse
import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEFAULT_DPI = 300
DEFAULT_READY_DATA_PATH = "data/processed/data_ready.csv"

# ── Colors & domain mapping ──────────────────────────────────
DOMAIN_COLORS = {
    "Environmental": "#1D9E75", "Social": "#D85A30",
    "Governance":    "#534AB7", "Financial": "#378ADD",
    "Unknown":       "#888780",
}

VARIABLE_DOMAINS = {
    # Environmental
    "co2_ch4_n2o_scope_1_3": "Environmental", "carbon_intensity": "Environmental",
    "scope_1_ghg_emissions": "Environmental", "scope_2_ghg_emissions": "Environmental",
    "scope_3_ghg_emissions": "Environmental",
    "emission_reduction_policy": "Environmental", "renewable_energy_share": "Environmental",
    "total_energy_consumption": "Environmental", "environmental_fines": "Environmental",
    "iso_14001_exists": "Environmental", "reporting_quality_score": "Environmental",
    "resource_efficiency_index": "Environmental", "resilience_score": "Environmental",
    "climate_risk_assessment_done": "Environmental", "hazardous_waste_generated": "Environmental",
    "water_withdrawal": "Environmental", "carbon_neutral_commitment": "Environmental",
    "air_emissions_sox_nox_pm": "Environmental", "land_area_affected": "Environmental",
    "biodiversity_protection_actions": "Environmental", "fsc_pefc_certified_sourcing": "Environmental",
    "green_product_revenue": "Environmental", "green_buildings_area": "Environmental",
    "recyclable_packaging_share": "Environmental", "toxic_spills": "Environmental",
    "waste_recycled_share": "Environmental", "emf_exposure": "Environmental",
    "gmo_products": "Environmental", "ods_emissions": "Environmental",
    # Social
    "training_hours": "Social", "injury_frequency_rate": "Social",
    "turnover_rate": "Social", "health_impact_score": "Social",
    "healthcare_access_employees": "Social", "diversity_representation": "Social",
    "diversity_women_representation": "Social", "health_safety": "Social",
    "community_investment": "Social", "customer_satisfaction_score": "Social",
    "human_rights_violations": "Social", "fair_wage_gap": "Social",
    "collective_bargaining_coverage": "Social", "csr_contribution": "Social",
    "safety_transparency_trials": "Social", "child_labor_compliance": "Social",
    "access_to_services": "Social", "union_membership": "Social",
    "product_safety_compliance": "Social", "supplier_audits": "Social",
    # Governance
    "board_diversity": "Governance", "governance_compliance_score": "Governance",
    "esg_oversight_policy": "Governance", "esg_incentive_bonus": "Governance",
    "board_strategy_esg_oversight": "Governance", "reporting_quality": "Governance",
    "auditor_independence_score": "Governance", "ethical_breaches": "Governance",
    "corruption_cases": "Governance", "ceo_chair_split": "Governance",
    "assurance_score": "Governance", "green_financing": "Governance",
    "privacy_compliance": "Governance", "anti_competitive_violations": "Governance",
    "shareholder_rights_score": "Governance", "lobby_spending": "Governance",
    # Financial
    "roa_eat": "Financial", "roe_eat": "Financial", "net_profit_margin": "Financial",
    "gross_profit_margin": "Financial", "debt_to_equity_ratio": "Financial",
    "pe_ratio": "Financial", "pbv": "Financial", "total_asset": "Financial",
    "total_equity": "Financial", "total_debt": "Financial",
    "sustainable_finance_green_financing": "Financial", "total_revenue": "Financial",
    "earnings_after_tax": "Financial", "market_price_share": "Financial",
    "eps": "Financial", "net_sales": "Financial", "gross_sales": "Financial",
    "net_cf_operating": "Financial", "lag_net_sales": "Financial",
    "lag_total_asset": "Financial", "lag_market_price": "Financial",
}

SCATTER_PAIRS = [
    (
        "emission_reduction_policy",
        "board_strategy_esg_oversight",
        "Highest flagged correlation",
    ),
    (
        "total_revenue",
        "scope_3_ghg_emissions",
        "Scale and value-chain emissions",
    ),
    (
        "renewable_energy_share",
        "scope_2_ghg_emissions",
        "Energy mix and purchased-energy emissions",
    ),
    (
        "sustainable_finance_green_financing",
        "reporting_quality",
        "Green financing and reporting quality",
    ),
    (
        "diversity_women_representation",
        "board_strategy_esg_oversight",
        "Diversity and ESG oversight",
    ),
    (
        "community_investment",
        "total_revenue",
        "Community investment and firm scale",
    ),
]

LOG_TRANSFORM_VARS = {
    "scope_1_ghg_emissions",
    "scope_2_ghg_emissions",
    "scope_3_ghg_emissions",
    "community_investment",
    "sustainable_finance_green_financing",
    "total_revenue",
}


# ── Layout helper ────────────────────────────────────────────
def _best_layout(G):
    """Try graphviz 'dot' (top-down causal flow), fall back to
    kamada_kawai, then spring. 'dot' makes arrow direction much
    easier to read because edges flow consistently top-to-bottom."""
    try:
        return nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pass
    try:
        return nx.kamada_kawai_layout(G)
    except Exception:
        return nx.spring_layout(G, k=2, iterations=100, seed=42)


# ── I/O ──────────────────────────────────────────────────────
def load_all(graph_dir="outputs/graphs"):
    graphs = {}
    for p in sorted(glob.glob(os.path.join(graph_dir, "*_adjacency.csv"))):
        name = os.path.basename(p).replace("_adjacency.csv", "")
        if "weight" in name or "edge_prob" in name:
            continue
        try:
            df = pd.read_csv(p, index_col=0)
            graphs[name] = df
            print(f"  {name}: {df.shape[0]} vars, {int((df.values != 0).sum())} edges")
        except Exception as e:
            print(f"  Warn: {p}: {e}")
    return graphs


def df_to_nx(adj_df, min_weight=0.0):
    """Convert adjacency DataFrame -> DiGraph.
    Edges with |weight| < min_weight are dropped."""
    G = nx.DiGraph()
    for c in adj_df.columns:
        d = VARIABLE_DOMAINS.get(c, "Unknown")
        G.add_node(c, domain=d, color=DOMAIN_COLORS.get(d, "#888"))
    for i, s in enumerate(adj_df.columns):
        for j, t in enumerate(adj_df.columns):
            w = float(adj_df.iloc[i, j])
            if w != 0 and abs(w) >= min_weight:
                G.add_edge(s, t, weight=w)
    return G


def _save_figure(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def _pretty_name(name):
    return name.replace("_", " ").title()


def _numeric_frame(df, cols):
    out = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    return out.dropna()


def _transform_for_scatter(values, var_name):
    label = _pretty_name(var_name)
    if var_name in LOG_TRANSFORM_VARS and values.dropna().ge(0).all():
        return np.log10(values + 1), f"log10({label} + 1)"
    return values, label


def _jitter_if_discrete(values, rng):
    clean = pd.Series(values).dropna()
    if clean.empty or clean.nunique() > 8:
        return values
    span = clean.max() - clean.min()
    scale = 0.025 * span if span > 0 else 0.025
    return values + rng.normal(0, scale, size=len(values))


# ── Core plotting ────────────────────────────────────────────
def plot_graph(G, title, ax, top_n=40):
    """Plot G on ax with weight-scaled edges and clearly visible arrows.

    - Width and opacity scale with |weight| so strong edges stand out.
    - Arrows use arrowstyle '-|>' (filled triangle), size 18.
    - Curved edges (arc3 rad=0.15) so A->B and B->A render as two
      separate arcs instead of one ambiguous line.
    """
    # Keep only top_n strongest edges if graph is dense
    if top_n > 0 and len(G.edges) > top_n:
        ew_sorted = sorted(
            [(u, v, abs(d.get("weight", 1.0))) for u, v, d in G.edges(data=True)],
            key=lambda x: x[2], reverse=True,
        )[:top_n]
        keep_edges = {(u, v) for u, v, _ in ew_sorted}
        keep_nodes = {n for uv in keep_edges for n in uv}
        G = G.subgraph(keep_nodes).copy()
        G.remove_edges_from([e for e in list(G.edges()) if e not in keep_edges])

    if not G.nodes:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    pos = _best_layout(G)

    # Node styling
    node_colors = [DOMAIN_COLORS.get(G.nodes[n].get("domain", "Unknown"), "#888")
                   for n in G.nodes]
    node_sizes = [max(500, 200 + G.degree(n) * 80) for n in G.nodes]

    # Edge styling: normalize weights to [0, 1] for width/alpha
    edge_data = list(G.edges(data=True))
    if edge_data:
        weights = np.array([abs(d.get("weight", 1.0)) for _, _, d in edge_data])
        max_w = weights.max() if weights.max() > 0 else 1.0
        norm = weights / max_w
        widths = 0.6 + 3.0 * norm
        alphas = 0.25 + 0.65 * norm
    else:
        widths, alphas = [], []

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        alpha=0.92, edgecolors="#333", linewidths=0.8,
    )

    # Draw edges one at a time to give each its own width & alpha.
    # This is the cleanest way to get per-edge styling with arrows.
    for (u, v, _d), w, a in zip(edge_data, widths, alphas):
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            edge_color="#333",
            width=float(w), alpha=float(a),
            arrows=True, arrowstyle="-|>", arrowsize=16,
            connectionstyle="arc3,rad=0.15",
            node_size=node_sizes,
            min_source_margin=12, min_target_margin=14,
        )

    # Labels
    labels = {n: n.replace("_", "\n")[:18] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=7, font_weight="bold")

    ax.set_title(
        f"{title}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        fontsize=10, fontweight="bold",
    )
    ax.axis("off")


# ── Figure savers ────────────────────────────────────────────
def save_correlation_heatmap(data_path, out_dir, dpi):
    if not os.path.exists(data_path):
        print(f"  Data file not found, skipping correlation heatmap: {data_path}")
        return

    df = pd.read_csv(data_path)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        print("  Not enough numeric columns for correlation heatmap")
        return

    corr = numeric.corr()
    names = [_pretty_name(c) for c in corr.columns]
    size = max(8, 0.75 * len(names))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    for i in range(len(names)):
        for j in range(len(names)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Numeric Variable Correlation Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, os.path.join(out_dir, "correlation_heatmap.png"), dpi)


def save_scatter_diagnostics(data_path, out_dir, dpi):
    if not os.path.exists(data_path):
        print(f"  Data file not found, skipping scatter diagnostics: {data_path}")
        return

    df = pd.read_csv(data_path)
    pairs = [(x, y, title) for x, y, title in SCATTER_PAIRS
             if x in df.columns and y in df.columns]
    if not pairs:
        print("  No configured scatter pairs found in data")
        return

    cols = 2
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.2 * rows))
    axes = np.array(axes).flatten()
    rng = np.random.default_rng(42)

    for ax, (x_col, y_col, title) in zip(axes, pairs):
        data = _numeric_frame(df, [x_col, y_col])
        if data.empty:
            ax.text(0.5, 0.5, "No numeric data", ha="center", va="center",
                    transform=ax.transAxes, color="#777")
            ax.axis("off")
            continue

        x_vals, x_label = _transform_for_scatter(data[x_col], x_col)
        y_vals, y_label = _transform_for_scatter(data[y_col], y_col)
        x_plot = _jitter_if_discrete(x_vals.to_numpy(), rng)
        y_plot = _jitter_if_discrete(y_vals.to_numpy(), rng)
        color = DOMAIN_COLORS.get(VARIABLE_DOMAINS.get(y_col, "Unknown"), "#378ADD")

        ax.scatter(x_plot, y_plot, s=34, color=color, alpha=0.72,
                   edgecolors="#333", linewidths=0.35)

        if len(data) > 2 and x_vals.nunique() > 1 and y_vals.nunique() > 1:
            coef = np.polyfit(x_vals, y_vals, deg=1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="#222", linewidth=1.2, alpha=0.75)
            corr = data[x_col].corr(data[y_col])
            subtitle = f"Pearson r = {corr:.2f}, n = {len(data)}"
        else:
            subtitle = f"n = {len(data)}"

        ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.suptitle("Key Scatter Diagnostics", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, os.path.join(out_dir, "scatter_key_relationships.png"), dpi)


def save_individual(graphs, out_dir, top_n, dpi):
    for name, df in graphs.items():
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        plot_graph(df_to_nx(df), name.replace("_", " ").title(), ax, top_n)
        legend = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLORS.items()]
        ax.legend(handles=legend, loc="lower left", fontsize=8)
        p = os.path.join(out_dir, f"network_{name}.png")
        _save_figure(fig, p, dpi)


def save_grid(graphs, out_dir, top_n, dpi):
    n = len(graphs)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    i = 0
    for i, (name, df) in enumerate(graphs.items()):
        plot_graph(df_to_nx(df), name.replace("_", " ").title(), axes[i], top_n)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    legend = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Causal Graph Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = os.path.join(out_dir, "comparison_grid.png")
    _save_figure(fig, p, dpi)


def save_jaccard(graphs, out_dir, dpi):
    names = list(graphs.keys())
    n = len(names)
    if n < 2:
        return
    common = set(graphs[names[0]].columns)
    for nm in names[1:]:
        common &= set(graphs[nm].columns)
    common = sorted(common)
    if len(common) < 3:
        print("  Not enough common vars for Jaccard"); return
    jac = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ai = graphs[names[i]].loc[common, common].values.flatten() != 0
            aj = graphs[names[j]].loc[common, common].values.flatten() != 0
            inter, union = (ai & aj).sum(), (ai | aj).sum()
            jac[i, j] = inter / union if union else 0
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(jac, cmap="YlOrRd", vmin=0, vmax=1)
    dn = [nm.replace("_", " ").title()[:18] for nm in names]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(dn, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(dn, fontsize=8)
    for i in range(n):
        for j in range(n):
            c = "white" if jac[i, j] > 0.5 else "black"
            ax.text(j, i, f"{jac[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=c)
    fig.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_title("Pairwise Edge Overlap", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "jaccard_heatmap.png")
    _save_figure(fig, p, dpi)


def save_edge_bars(graphs, out_dir, dpi):
    names, counts, colors = [], [], []
    for name, df in graphs.items():
        names.append(name.replace("_", "\n").title()[:22])
        counts.append(int((df.values != 0).sum()))
        if "deci" in name:
            colors.append("#534AB7")
        elif "gemma" in name:
            colors.append("#D85A30")
        elif "constrained" in name:
            colors.append("#1D9E75")
        else:
            colors.append("#378ADD")
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars = ax.bar(range(len(names)), counts, color=colors, alpha=0.85,
                  edgecolor="#444", linewidth=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + max(counts) * 0.02, str(c),
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Directed edges")
    ax.set_title("Edge Count Comparison", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "edge_count_comparison.png")
    _save_figure(fig, p, dpi)


def save_constraint_impact(graphs, out_dir, dpi):
    pairs = [("unconstrained_pc", "constrained_pc", "PC"),
             ("deci_unconstrained", "deci_constrained", "DECI")]
    found = [(u, c, l) for u, c, l in pairs if u in graphs and c in graphs]
    if not found:
        return
    fig, axes = plt.subplots(1, len(found), figsize=(6 * len(found), 5))
    if len(found) == 1:
        axes = [axes]
    for ax, (un, cn, label) in zip(axes, found):
        common = sorted(set(graphs[un].columns) & set(graphs[cn].columns))
        u = graphs[un].loc[common, common].values != 0
        c = graphs[cn].loc[common, common].values != 0
        vals = [int((u & c).sum()), int((u & ~c).sum()), int((~u & c).sum())]
        cats = ["Shared", "Removed\nby ontology", "Added\nby ontology"]
        cs = ["#378ADD", "#E24B4A", "#1D9E75"]
        bars = ax.bar(cats, vals, color=cs, alpha=0.85, edgecolor="#444")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(vals) * 0.03, str(v),
                    ha="center", fontsize=11, fontweight="bold")
        ax.set_title(f"{label}: Ontology Impact", fontsize=11, fontweight="bold")
        ax.set_ylabel("Edges")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "constraint_impact.png")
    _save_figure(fig, p, dpi)


def save_interactive(graphs, out_dir):
    try:
        from pyvis.network import Network
    except ImportError:
        print("  pyvis not installed, skipping HTML. pip install pyvis"); return
    for name, df in graphs.items():
        net = Network(height="700px", width="100%", directed=True,
                      bgcolor="#1a1a2e", font_color="#eee")
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)
        for c in df.columns:
            d = VARIABLE_DOMAINS.get(c, "Unknown")
            net.add_node(c, label=c.replace("_", " ")[:18],
                         color=DOMAIN_COLORS.get(d, "#888"),
                         title=f"{c}\n{d}", size=15)
        for i, s in enumerate(df.columns):
            for j, t in enumerate(df.columns):
                if df.iloc[i, j] != 0:
                    net.add_edge(s, t, value=abs(float(df.iloc[i, j])))
        p = os.path.join(out_dir, f"interactive_{name}.html")
        net.save_graph(p)
        print(f"  -> {p}")


# ── NEW: domain-split subgraphs ──────────────────────────────
def _filter_by_edge_domain(G, keep_fn):
    """Return subgraph keeping only edges where keep_fn(src_dom, tgt_dom) is True.
    Isolated nodes are removed."""
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        sd = G.nodes[u].get("domain", "Unknown")
        td = G.nodes[v].get("domain", "Unknown")
        if keep_fn(sd, td):
            H.add_edge(u, v, **d)
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0])
    return H


def save_domain_subgraphs(graphs, out_dir, top_n=30, dpi=DEFAULT_DPI):
    """For each graph, produce a 2x3 panel with domain-filtered views:
    E-only, S-only, G-only, F-only, cross-domain, and top-K strongest."""
    for name, df in graphs.items():
        G_full = df_to_nx(df)
        if len(G_full.edges) == 0:
            continue

        views = [
            ("Environmental only",
             _filter_by_edge_domain(G_full,
                 lambda s, t: s == "Environmental" and t == "Environmental")),
            ("Social only",
             _filter_by_edge_domain(G_full,
                 lambda s, t: s == "Social" and t == "Social")),
            ("Governance only",
             _filter_by_edge_domain(G_full,
                 lambda s, t: s == "Governance" and t == "Governance")),
            ("Financial only",
             _filter_by_edge_domain(G_full,
                 lambda s, t: s == "Financial" and t == "Financial")),
            ("Cross-domain",
             _filter_by_edge_domain(G_full,
                 lambda s, t: s != t and s != "Unknown" and t != "Unknown")),
            (f"Top {top_n} strongest", G_full),  # plot_graph does top_n filter
        ]

        fig, axes = plt.subplots(2, 3, figsize=(21, 14))
        axes = axes.flatten()
        for ax, (label, H) in zip(axes, views):
            # Full graph panel respects top_n; single-domain panels show everything
            this_top_n = top_n if label.startswith("Top") else 0
            plot_graph(H, label, ax, top_n=this_top_n)

        legend = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLORS.items()]
        fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=10,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{name.replace('_', ' ').title()} — Domain Views",
                     fontsize=14, fontweight="bold", y=1.00)
        fig.tight_layout()

        p = os.path.join(out_dir, f"subgraphs_{name}.png")
        _save_figure(fig, p, dpi)


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default="outputs/graphs")
    parser.add_argument("--out-dir",   default="outputs/figures")
    parser.add_argument("--data-path", default=DEFAULT_READY_DATA_PATH,
                        help="Causal-ready numeric CSV used for EDA plots")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help="Output image resolution")
    parser.add_argument("--top-n", type=int, default=40,
                        help="Keep only this many strongest edges per figure (0 = all)")
    parser.add_argument("--min-weight", type=float, default=0.0,
                        help="Drop edges with |weight| below this threshold")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip scatter and data-correlation diagnostic figures")
    parser.add_argument("--subgraphs", action="store_true",
                        help="Produce domain-split subgraph panels")
    parser.add_argument("--interactive", action="store_true",
                        help="Also save interactive HTML (pyvis)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[viz] Loading from {args.graph_dir}/")
    graphs = load_all(args.graph_dir)
    if not graphs:
        print("[error] No graphs found. Run baselines first."); return

    # Apply min-weight threshold at the DataFrame level so every
    # downstream figure (including edge counts) sees the filtered data.
    if args.min_weight > 0:
        print(f"\n[viz] Applying min-weight threshold: |w| >= {args.min_weight}")
        for nm in list(graphs.keys()):
            m = graphs[nm].copy()
            m[m.abs() < args.min_weight] = 0
            graphs[nm] = m
            print(f"  {nm}: {int((m.values != 0).sum())} edges after threshold")

    print(f"\n  1. Individual plots"); save_individual(graphs, args.out_dir, args.top_n, args.dpi)
    print(f"\n  2. Comparison grid");   save_grid(graphs, args.out_dir, args.top_n, args.dpi)
    print(f"\n  3. Jaccard heatmap");   save_jaccard(graphs, args.out_dir, args.dpi)
    print(f"\n  4. Edge counts");       save_edge_bars(graphs, args.out_dir, args.dpi)
    print(f"\n  5. Constraint impact"); save_constraint_impact(graphs, args.out_dir, args.dpi)
    if not args.skip_eda:
        print(f"\n  6. Data correlation heatmap"); save_correlation_heatmap(args.data_path, args.out_dir, args.dpi)
        print(f"\n  7. Scatter diagnostics");      save_scatter_diagnostics(args.data_path, args.out_dir, args.dpi)
    if args.subgraphs:
        print(f"\n  8. Domain subgraphs"); save_domain_subgraphs(graphs, args.out_dir, args.top_n, args.dpi)
    if args.interactive:
        print(f"\n  9. Interactive HTML"); save_interactive(graphs, args.out_dir)

    print(f"\n[done] All figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
