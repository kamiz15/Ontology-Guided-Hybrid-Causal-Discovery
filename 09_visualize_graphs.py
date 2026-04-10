# 09_visualize_graphs.py
# ============================================================
# Causal Graph Visualization & Comparison Figures
# Loads all adjacency CSVs and produces publication-ready figures.
#
# Usage:
#   python 09_visualize_graphs.py
#   python 09_visualize_graphs.py --top-n 30 --interactive
# ============================================================

from __future__ import annotations
import argparse, os, glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DOMAIN_COLORS = {
    "Environmental": "#1D9E75", "Social": "#D85A30",
    "Governance": "#534AB7", "Financial": "#378ADD", "Unknown": "#888780",
}

VARIABLE_DOMAINS = {
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
    "board_diversity": "Governance", "governance_compliance_score": "Governance",
    "esg_oversight_policy": "Governance", "esg_incentive_bonus": "Governance",
    "board_strategy_esg_oversight": "Governance", "reporting_quality": "Governance",
    "auditor_independence_score": "Governance", "ethical_breaches": "Governance",
    "corruption_cases": "Governance", "ceo_chair_split": "Governance",
    "assurance_score": "Governance", "green_financing": "Governance",
    "privacy_compliance": "Governance", "anti_competitive_violations": "Governance",
    "shareholder_rights_score": "Governance", "lobby_spending": "Governance",
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

def load_all(graph_dir="outputs/graphs"):
    graphs = {}
    for p in sorted(glob.glob(os.path.join(graph_dir, "*_adjacency.csv"))):
        name = os.path.basename(p).replace("_adjacency.csv","")
        if "weight" in name or "edge_prob" in name: continue
        try:
            df = pd.read_csv(p, index_col=0)
            graphs[name] = df
            print(f"  {name}: {df.shape[0]} vars, {int((df.values!=0).sum())} edges")
        except Exception as e:
            print(f"  Warn: {p}: {e}")
    return graphs

def df_to_nx(adj_df, name=""):
    G = nx.DiGraph()
    for c in adj_df.columns:
        d = VARIABLE_DOMAINS.get(c, "Unknown")
        G.add_node(c, domain=d, color=DOMAIN_COLORS.get(d,"#888"))
    for i, s in enumerate(adj_df.columns):
        for j, t in enumerate(adj_df.columns):
            if adj_df.iloc[i,j] != 0:
                G.add_edge(s, t, weight=float(adj_df.iloc[i,j]))
    return G

def plot_graph(G, title, ax, top_n=40):
    if top_n > 0 and len(G.edges) > top_n:
        ew = sorted([(u,v,abs(d.get("weight",1))) for u,v,d in G.edges(data=True)], key=lambda x:x[2], reverse=True)
        nodes = set()
        for u,v,_ in ew[:top_n]: nodes.add(u); nodes.add(v)
        G = G.subgraph(nodes).copy()
    if not G.nodes:
        ax.text(0.5,0.5,"No edges",ha="center",va="center",transform=ax.transAxes); ax.set_title(title); return
    nc = [DOMAIN_COLORS.get(G.nodes[n].get("domain","Unknown"),"#888") for n in G.nodes]
    ns = [max(250, 80+G.degree(n)*60) for n in G.nodes]
    try: pos = nx.kamada_kawai_layout(G)
    except: pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#888", arrows=True, arrowsize=8, width=0.7, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns, alpha=0.85, edgecolors="#444", linewidths=0.5)
    labels = {n: n.replace("_","\n")[:18] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=5.5, font_weight="bold")
    ax.set_title(f"{title}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges", fontsize=10, fontweight="bold")
    ax.axis("off")

def save_individual(graphs, out_dir, top_n):
    for name, df in graphs.items():
        fig, ax = plt.subplots(1,1,figsize=(12,9))
        plot_graph(df_to_nx(df), name.replace("_"," ").title(), ax, top_n)
        legend = [mpatches.Patch(color=c, label=d) for d,c in DOMAIN_COLORS.items()]
        ax.legend(handles=legend, loc="lower left", fontsize=8)
        p = os.path.join(out_dir, f"network_{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
        print(f"  -> {p}")

def save_grid(graphs, out_dir, top_n):
    n = len(graphs)
    if n == 0: return
    cols = min(3, n); rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i,(name,df) in enumerate(graphs.items()):
        plot_graph(df_to_nx(df), name.replace("_"," ").title(), axes[i], top_n)
    for j in range(i+1, len(axes)): axes[j].axis("off")
    legend = [mpatches.Patch(color=c, label=d) for d,c in DOMAIN_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Causal Graph Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = os.path.join(out_dir, "comparison_grid.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  -> {p}")

def save_jaccard(graphs, out_dir):
    names = list(graphs.keys())
    n = len(names)
    if n < 2: return
    common = set(graphs[names[0]].columns)
    for nm in names[1:]: common &= set(graphs[nm].columns)
    common = sorted(common)
    if len(common) < 3: print("  Not enough common vars for Jaccard"); return
    jac = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ai = graphs[names[i]].loc[common,common].values.flatten() != 0
            aj = graphs[names[j]].loc[common,common].values.flatten() != 0
            inter, union = (ai&aj).sum(), (ai|aj).sum()
            jac[i,j] = inter/union if union else 0
    fig, ax = plt.subplots(figsize=(max(6,n*1.2), max(5,n)))
    im = ax.imshow(jac, cmap="YlOrRd", vmin=0, vmax=1)
    dn = [n.replace("_"," ").title()[:18] for n in names]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(dn, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(dn, fontsize=8)
    for i in range(n):
        for j in range(n):
            c = "white" if jac[i,j]>0.5 else "black"
            ax.text(j, i, f"{jac[i,j]:.2f}", ha="center", va="center", fontsize=9, color=c)
    fig.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_title("Pairwise Edge Overlap", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "jaccard_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  -> {p}")

def save_edge_bars(graphs, out_dir):
    names, counts, colors = [], [], []
    for name, df in graphs.items():
        names.append(name.replace("_","\n").title()[:22])
        counts.append(int((df.values!=0).sum()))
        if "deci" in name: colors.append("#534AB7")
        elif "gemma" in name: colors.append("#D85A30")
        elif "constrained" in name: colors.append("#1D9E75")
        else: colors.append("#378ADD")
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.5), 5))
    bars = ax.bar(range(len(names)), counts, color=colors, alpha=0.85, edgecolor="#444", linewidth=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(counts)*0.02, str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Directed edges"); ax.set_title("Edge Count Comparison", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "edge_count_comparison.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  -> {p}")

def save_constraint_impact(graphs, out_dir):
    pairs = [("unconstrained_pc","constrained_pc","PC"), ("deci_unconstrained","deci_constrained","DECI")]
    found = [(u,c,l) for u,c,l in pairs if u in graphs and c in graphs]
    if not found: return
    fig, axes = plt.subplots(1, len(found), figsize=(6*len(found), 5))
    if len(found)==1: axes=[axes]
    for ax,(un,cn,label) in zip(axes, found):
        common = sorted(set(graphs[un].columns) & set(graphs[cn].columns))
        u = graphs[un].loc[common,common].values != 0
        c = graphs[cn].loc[common,common].values != 0
        vals = [int((u&c).sum()), int((u&~c).sum()), int((~u&c).sum())]
        cats = ["Shared","Removed\nby ontology","Added\nby ontology"]
        cs = ["#378ADD","#E24B4A","#1D9E75"]
        bars = ax.bar(cats, vals, color=cs, alpha=0.85, edgecolor="#444")
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.03, str(v), ha="center", fontsize=11, fontweight="bold")
        ax.set_title(f"{label}: Ontology Impact", fontsize=11, fontweight="bold")
        ax.set_ylabel("Edges"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "constraint_impact.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  -> {p}")

def save_interactive(graphs, out_dir):
    try:
        from pyvis.network import Network
    except ImportError:
        print("  pyvis not installed, skipping HTML. pip install pyvis"); return
    for name, df in graphs.items():
        net = Network(height="700px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="#eee")
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)
        for c in df.columns:
            d = VARIABLE_DOMAINS.get(c,"Unknown")
            net.add_node(c, label=c.replace("_"," ")[:18], color=DOMAIN_COLORS.get(d,"#888"), title=f"{c}\n{d}", size=15)
        for i,s in enumerate(df.columns):
            for j,t in enumerate(df.columns):
                if df.iloc[i,j]!=0: net.add_edge(s, t, value=abs(float(df.iloc[i,j])))
        p = os.path.join(out_dir, f"interactive_{name}.html")
        net.save_graph(p); print(f"  -> {p}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default="outputs/graphs")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    out_dir = "outputs/figures"; os.makedirs(out_dir, exist_ok=True)
    print(f"[viz] Loading from {args.graph_dir}/")
    graphs = load_all(args.graph_dir)
    if not graphs: print("[error] No graphs found. Run baselines first."); return
    print(f"\n  1. Individual plots"); save_individual(graphs, out_dir, args.top_n)
    print(f"\n  2. Comparison grid"); save_grid(graphs, out_dir, args.top_n)
    print(f"\n  3. Jaccard heatmap"); save_jaccard(graphs, out_dir)
    print(f"\n  4. Edge counts"); save_edge_bars(graphs, out_dir)
    print(f"\n  5. Constraint impact"); save_constraint_impact(graphs, out_dir)
    if args.interactive: print(f"\n  6. Interactive HTML"); save_interactive(graphs, out_dir)
    print(f"\n[done] All figures in {out_dir}/")

if __name__ == "__main__":
    main()
