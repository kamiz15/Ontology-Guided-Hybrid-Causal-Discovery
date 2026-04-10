# 08_gemma_causal_proposals.py
# ============================================================
# Gemma 4 — LLM-based Causal Edge Proposals for ESG Data
#
# Uses Google's Gemma open-source model to reason about ESG
# causal relationships from variable descriptions alone.
# Compares LLM-proposed edges against data-driven graphs.
#
# Three backend options:
#   1. Ollama (local)    — ollama run gemma4:12b
#   2. Google AI Studio  — free API key from aistudio.google.com
#   3. HuggingFace       — needs GPU with >=16GB VRAM
#
# Usage:
#   python 08_gemma_causal_proposals.py --backend ollama
#   python 08_gemma_causal_proposals.py --backend google --api-key KEY
# ============================================================

from __future__ import annotations
import argparse, json, os, time
from datetime import datetime
import networkx as nx
import numpy as np
import pandas as pd
from config import READY_DATA_PATH

# ── ESG variable descriptions for LLM prompting ─────────────
VARIABLE_DESCRIPTIONS = {
    "co2_ch4_n2o_scope_1_3":       "Total GHG emissions Scope 1-3 (tCO2e)",
    "carbon_intensity":             "Emissions per unit output (tCO2e/output)",
    "scope_1_ghg_emissions":        "Direct operational greenhouse gas emissions (tCO2e)",
    "scope_2_ghg_emissions":        "Indirect electricity and energy emissions (tCO2e)",
    "scope_3_ghg_emissions":        "Value-chain and financed greenhouse gas emissions (tCO2e)",
    "emission_reduction_policy":    "Has formal emission reduction policy (bool)",
    "renewable_energy_share":       "% of energy from renewables",
    "total_energy_consumption":     "Total energy consumed (kWh/GJ)",
    "environmental_fines":          "Environmental violation fines (Currency)",
    "iso_14001_exists":             "Has ISO 14001 certification (bool)",
    "reporting_quality_score":      "Environmental reporting quality (0-100)",
    "resource_efficiency_index":    "Resource utilization efficiency (0-100)",
    "resilience_score":             "Climate resilience score (0-100)",
    "climate_risk_assessment_done": "Climate risk assessment conducted (bool)",
    "hazardous_waste_generated":    "Hazardous waste produced (kg/ton)",
    "water_withdrawal":             "Water withdrawn from all sources (m3)",
    "training_hours":               "Training hours per employee",
    "injury_frequency_rate":        "Injury rate per 100 workers",
    "turnover_rate":                "Employee turnover rate (%)",
    "health_impact_score":          "Employee health/wellbeing score (0-100)",
    "healthcare_access_employees":  "% employees with healthcare access",
    "diversity_representation":     "Workforce diversity (%)",
    "diversity_women_representation":"Women representation or diversity ratio in workforce/leadership",
    "health_safety":                "Health and safety quality or programme strength",
    "community_investment":         "Community investment spending (Currency)",
    "customer_satisfaction_score":  "Customer satisfaction index (0-100)",
    "human_rights_violations":      "Count of human rights violations",
    "board_diversity":              "Board diversity percentage (%)",
    "governance_compliance_score":  "Governance compliance score (0-100)",
    "esg_oversight_policy":         "Board-level ESG oversight exists (bool)",
    "board_strategy_esg_oversight": "Board strategy and ESG oversight quality",
    "esg_incentive_bonus":          "Exec compensation tied to ESG (bool)",
    "auditor_independence_score":   "Auditor independence score (0-100)",
    "ethical_breaches":             "Count of ethical breaches",
    "corruption_cases":             "Count of corruption cases",
    "ceo_chair_split":              "CEO/Chair roles separated (bool)",
    "roa_eat":                      "Return on Assets (earnings after tax)",
    "roe_eat":                      "Return on Equity (earnings after tax)",
    "net_profit_margin":            "Net profit margin ratio",
    "debt_to_equity_ratio":         "Debt to equity ratio",
    "total_asset":                  "Total assets (Currency)",
    "total_equity":                 "Total equity (Currency)",
    "sustainable_finance_green_financing": "Green or sustainable financing volume (Currency)",
    "total_revenue":                "Total revenue or income (Currency)",
    "market_price_share":           "Market price per share (Currency)",
    "reporting_quality":            "Quality and assurance level of ESG reporting",
}

def build_discovery_prompt(variables: dict[str, str]) -> str:
    var_list = "\n".join(f"- {k}: {v}" for k, v in variables.items())
    return f"""You are an ESG domain expert and causal reasoning specialist.

Variables:
{var_list}

Propose ALL plausible DIRECT causal edges between these variables.
Only include edges where a clear causal mechanism exists (not mere correlation).

Rules:
- Composite scores CANNOT cause their own inputs
- Outcomes cannot precede their drivers temporally  
- Financial ratios cannot cause their own numerator/denominator
- Environmental variables cannot directly cause governance variables

For each edge respond with EXACTLY this JSON (one per line, no other text):
{{"source": "var_name", "target": "var_name", "confidence": 0.0-1.0, "rationale": "brief mechanism"}}

Propose 25-50 edges. Output ONLY JSON lines."""

# ── Backend query functions ──────────────────────────────────
def query_ollama(prompt: str, model: str = "gemma4:12b") -> str:
    import urllib.request
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096}
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["response"]

def query_google_ai(prompt: str, api_key: str, model: str = "gemma-4-12b-it") -> str:
    import urllib.request
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    return result["candidates"][0]["content"]["parts"][0]["text"]

def query_huggingface(prompt: str, model_id: str = "google/gemma-4-12b-it") -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=4096, temperature=0.2, do_sample=True)
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

# ── Parse + convert ──────────────────────────────────────────
def parse_edges(response: str) -> list[dict]:
    edges = []
    for line in response.strip().split("\n"):
        line = line.strip().rstrip(",")
        if not line.startswith("{"): continue
        try:
            obj = json.loads(line)
            if "source" in obj and "target" in obj:
                edges.append(obj)
        except json.JSONDecodeError:
            continue
    return edges

def edges_to_adjacency(edges, columns, threshold=0.5):
    col_set = set(columns)
    idx = {c: i for i, c in enumerate(columns)}
    adj = np.zeros((len(columns), len(columns)), dtype=int)
    for e in edges:
        s, t = e.get("source",""), e.get("target","")
        if s in col_set and t in col_set and e.get("confidence",0) >= threshold:
            adj[idx[s], idx[t]] = 1
    return adj

def compare_with_graphs(gemma_adj, columns, graph_dir="outputs/graphs"):
    results = []
    idx = {c: i for i, c in enumerate(columns)}
    for f in sorted(os.listdir(graph_dir)):
        if f.endswith("_adjacency.csv") and "gemma" not in f:
            try:
                df = pd.read_csv(os.path.join(graph_dir, f), index_col=0)
                common = [c for c in columns if c in df.columns and c in df.index]
                if len(common) < 5: continue
                n = len(common)
                g, a = np.zeros((n,n),dtype=bool), np.zeros((n,n),dtype=bool)
                for i,ci in enumerate(common):
                    for j,cj in enumerate(common):
                        g[i,j] = gemma_adj[idx[ci], idx[cj]] if ci in idx and cj in idx else False
                        a[i,j] = df.loc[ci,cj] != 0
                gf, af = g.flatten(), a.flatten()
                inter = int((gf & af).sum())
                union = int((gf | af).sum())
                results.append({
                    "algorithm": f.replace("_adjacency.csv",""),
                    "gemma_edges": int(gf.sum()), "algo_edges": int(af.sum()),
                    "shared": inter, "gemma_only": int((gf & ~af).sum()),
                    "algo_only": int((~gf & af).sum()),
                    "jaccard": round(inter/union,4) if union else 0,
                })
            except Exception as e:
                print(f"  Warn: {f}: {e}")
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", choices=["ollama","google","huggingface"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--input", default=READY_DATA_PATH)
    args = parser.parse_args()

    for d in ["outputs/graphs","outputs/metrics","reports"]: os.makedirs(d, exist_ok=True)

    df = pd.read_csv(args.input)
    data_cols = list(df.select_dtypes(include="number").columns)
    active = {k:v for k,v in VARIABLE_DESCRIPTIONS.items() if k in data_cols}
    print(f"[gemma] {len(active)} variables, backend={args.backend}")

    if args.backend == "ollama":
        m = args.model or "gemma4:12b"
        qfn = lambda p: query_ollama(p, m)
    elif args.backend == "google":
        key = args.api_key or os.environ.get("GOOGLE_AI_KEY","")
        if not key: print("[error] Need --api-key or GOOGLE_AI_KEY env"); return
        m = args.model or "gemma-4-12b-it"
        qfn = lambda p: query_google_ai(p, key, m)
    else:
        m = args.model or "google/gemma-4-12b-it"
        qfn = lambda p: query_huggingface(p, m)

    prompt = build_discovery_prompt(active)
    print(f"  Querying Gemma ({len(prompt)} chars)...")
    t0 = time.time()
    response = qfn(prompt)
    elapsed = time.time() - t0
    print(f"  Response: {elapsed:.1f}s, {len(response)} chars")

    with open("reports/gemma_causal_reasoning.txt","w") as f:
        f.write(f"Gemma Causal Reasoning — {datetime.now().isoformat()}\n{'='*60}\nPROMPT:\n{prompt}\n\n{'='*60}\nRESPONSE:\n{response}\n")

    edges = parse_edges(response)
    print(f"  Parsed {len(edges)} edges")
    for e in sorted(edges, key=lambda x: x.get("confidence",0), reverse=True)[:10]:
        print(f"    {e['source']} -> {e['target']}  conf={e.get('confidence','?')}  {e.get('rationale','')[:50]}")

    adj = edges_to_adjacency(edges, data_cols, args.threshold)
    print(f"  Graph: {np.count_nonzero(adj)} edges")

    adj_df = pd.DataFrame(adj, index=data_cols, columns=data_cols)
    adj_df.to_csv("outputs/graphs/gemma_proposed_adjacency.csv")
    G = nx.DiGraph()
    G.add_nodes_from(data_cols)
    for i,s in enumerate(data_cols):
        for j,t in enumerate(data_cols):
            if adj[i,j]: G.add_edge(s,t)
    nx.write_gml(G, "outputs/graphs/gemma_proposed_graph.gml")
    pd.DataFrame(edges).to_csv("outputs/graphs/gemma_proposed_edges.csv", index=False)

    comp = compare_with_graphs(adj, data_cols)
    if not comp.empty:
        comp.to_csv("outputs/metrics/gemma_vs_data_comparison.csv", index=False)
        print(f"\n  Gemma vs Data-Driven:\n{comp.to_string(index=False)}")

    print(f"\n[done] Outputs in outputs/graphs/ and reports/")

if __name__ == "__main__":
    main()
