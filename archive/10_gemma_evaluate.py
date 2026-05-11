# 10_gemma_evaluate.py
# ============================================================
# Gemma-based qualitative evaluation of causal discovery output.
#
# For every directed edge in every adjacency CSV, queries Gemma for:
#   - plausibility of any causal link (1-5)
#   - most plausible relationship between the two variables
#   - mechanism (one-sentence causal pathway)
#   - confounders (brief note)
#   - confidence (high / medium / low)
#
# Produces:
#   outputs/gemma_eval/edge_scores.csv      per-edge scores across all graphs
#   outputs/gemma_eval/model_summary.csv    aggregated per-graph metrics
#   outputs/gemma_eval/raw_responses.jsonl  audit trail of every API call
#   outputs/gemma_eval/cache.json           deduplicated edge-score cache
#   outputs/gemma_eval/comparison.png       bar-chart model comparison
#   outputs/gemma_eval/distribution.png     plausibility distribution per model
#
# This is a SUPPLEMENTARY plausibility check, not ground truth.
# LLM outputs are stochastic and potentially biased by training data.
# Use it alongside literature-backed ground truth, not in place of it.
#
# ── Setup ────────────────────────────────────────────────────
#
# Option A: Ollama (local, free, needs ~4-8 GB RAM)
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama pull gemma3:4b
#   ollama serve
#   python 10_gemma_evaluate.py --backend ollama --model gemma3:4b
#
# Option B: Google AI Studio (cloud, free tier, rate-limited)
#   pip install google-genai
#   # Get API key at https://aistudio.google.com/apikey
#   export GEMINI_API_KEY="your_key_here"
#   python 10_gemma_evaluate.py --backend google --model gemma-4-26b-a4b-it
#
# ── Usage ────────────────────────────────────────────────────
#   python 10_gemma_evaluate.py                              # default: ollama
#   python 10_gemma_evaluate.py --backend google --model gemma-4-26b-a4b-it
#   python 10_gemma_evaluate.py --max-edges 20               # quick smoke test
#   python 10_gemma_evaluate.py --no-cache                   # force re-query all
# ============================================================

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse domain mapping from the visualizer to keep labels consistent
try:
    from importlib import import_module
    _viz = import_module("09_visualize_graphs")
    VARIABLE_DOMAINS = _viz.VARIABLE_DOMAINS
    DOMAIN_COLORS = _viz.DOMAIN_COLORS
except Exception:
    VARIABLE_DOMAINS = {}
    DOMAIN_COLORS = {
        "Environmental": "#1D9E75", "Social": "#D85A30",
        "Governance":    "#534AB7", "Financial": "#378ADD",
        "Unknown":       "#888780",
    }

RELATIONSHIP_LABELS = (
    "A_causes_B",
    "B_causes_A",
    "bidirectional",
    "confounded",
    "none",
)

RELATIONSHIP_NORMALIZATION = {
    "a_causes_b": "A_causes_B",
    "a causes b": "A_causes_B",
    "b_causes_a": "B_causes_A",
    "b causes a": "B_causes_A",
    "bidirectional": "bidirectional",
    "feedback": "bidirectional",
    "confounded": "confounded",
    "common_cause": "confounded",
    "common cause": "confounded",
    "none": "none",
    "no_link": "none",
    "no link": "none",
}


# ── Prompt ───────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an expert in corporate Environmental, Social, and Governance (ESG) research.

You are given two variables from a corporate ESG dataset:
  Variable A: "{cause}"   (domain: {cause_domain})
  Variable B: "{effect}"  (domain: {effect_domain})

Do NOT assume any particular direction. Consider all possibilities:
  - A could directly cause B
  - B could directly cause A
  - Both could be driven by a common third factor (confounding)
  - They could have a bidirectional/feedback relationship
  - There could be no causal relationship at all

Evaluate independently and respond with ONLY a JSON object:
{{
  "most_plausible_relationship": "<A_causes_B | B_causes_A | bidirectional | confounded | none>",
  "plausibility_of_any_link": <integer 1 to 5>,
  "mechanism_if_any": "<one short sentence, or 'none'>",
  "likely_confounders": "<brief mention, or 'none'>",
  "confidence": "<high|medium|low>"
}}

Plausibility of ANY causal link (in either direction):
  1 = no causal link, likely just correlation or independence
  2 = weak, speculative
  3 = plausible but contested
  4 = well-supported by domain reasoning or empirical literature
  5 = strongly established
"""


def build_prompt(cause: str, effect: str) -> str:
    return PROMPT_TEMPLATE.format(
        cause=cause,
        effect=effect,
        cause_domain=VARIABLE_DOMAINS.get(cause, "Unknown"),
        effect_domain=VARIABLE_DOMAINS.get(effect, "Unknown"),
    )


# ── Response parsing ─────────────────────────────────────────
def parse_json_response(raw: str) -> dict | None:
    """Robust JSON extraction: strips code fences, finds {...} block,
    handles trailing commas. Returns None if unparseable."""
    if not raw:
        return None
    s = raw.strip()
    for fence in ("```json", "```JSON", "```"):
        if s.startswith(fence):
            s = s[len(fence):].lstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = s[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)  # strip trailing commas
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def normalize_relationship(value: object) -> str | None:
    if value is None:
        return None
    key = str(value).strip()
    if key in RELATIONSHIP_LABELS:
        return key
    normalized = key.lower().replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    return RELATIONSHIP_NORMALIZATION.get(normalized)


def validate_response(d: dict) -> dict | None:
    """Validate response has expected keys with sensible types. Returns
    cleaned dict or None."""
    if not isinstance(d, dict):
        return None
    try:
        plaus = int(d.get("plausibility_of_any_link", d.get("plausibility", 0)))
        if not (1 <= plaus <= 5):
            return None
    except (ValueError, TypeError):
        return None
    relationship = normalize_relationship(d.get("most_plausible_relationship"))
    if relationship is None:
        return None
    confidence = str(d.get("confidence", "")).lower().strip()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return {
        "plausibility_of_any_link": plaus,
        "most_plausible_relationship": relationship,
        "mechanism_if_any": str(d.get("mechanism_if_any", d.get("mechanism", ""))).strip()[:300],
        "likely_confounders": str(d.get("likely_confounders", d.get("confounders", ""))).strip()[:200],
        "confidence": confidence,
        "parse_error": False,
    }


# ── Clients ──────────────────────────────────────────────────
class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        import requests
        self.requests = requests
        self.model = model
        self.host = host.rstrip("/")
        # Ping to fail fast if server is down
        try:
            r = self.requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Ollama server not reachable at {self.host}. "
                f"Start it with `ollama serve`. Error: {e}"
            )

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "seed": 42, "num_predict": 400},
        }
        for attempt in range(max_retries):
            try:
                r = self.requests.post(
                    f"{self.host}/api/generate", json=payload, timeout=120
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


class GoogleClient:
    def __init__(self, model: str, api_key: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai not installed. "
                "Run: pip install google-genai"
            )
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Google API key missing. Set GEMINI_API_KEY env var or "
                "pass --api-key. Get one at https://aistudio.google.com/apikey"
            )
        self.types = types
        self.client = genai.Client(api_key=key)
        self.model_name = model
        self.config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=400,
            response_mime_type="application/json",
        )

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.config,
                )
                return resp.text or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Google rate limit — wait longer
                time.sleep(5 * (attempt + 1))
        return ""


# ── Cache ────────────────────────────────────────────────────
class EdgeCache:
    """Cache keyed by (cause, effect). Same pair across different
    graphs only gets queried once. Persisted as JSON."""

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.data: dict[str, dict] = {}
        if enabled and os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
                print(f"[cache] Loaded {len(self.data)} cached edges from {path}")
            except Exception:
                self.data = {}

    @staticmethod
    def key(cause: str, effect: str) -> str:
        return f"{cause}__->__{effect}"

    def get(self, cause: str, effect: str) -> dict | None:
        if not self.enabled:
            return None
        return self.data.get(self.key(cause, effect))

    def set(self, cause: str, effect: str, value: dict):
        self.data[self.key(cause, effect)] = value

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


# ── Evaluation loop ──────────────────────────────────────────
def load_graphs(graph_dir: str) -> dict[str, pd.DataFrame]:
    graphs = {}
    for p in sorted(glob.glob(os.path.join(graph_dir, "*_adjacency.csv"))):
        name = os.path.basename(p).replace("_adjacency.csv", "")
        if "weight" in name or "edge_prob" in name:
            continue
        if name == "gemma_proposed":
            print(f"[load] Skipping {name} to avoid self-evaluation")
            continue
        try:
            df = pd.read_csv(p, index_col=0)
            graphs[name] = df
        except Exception as e:
            print(f"  Warn: {p}: {e}")
    return graphs


def iter_edges(adj_df: pd.DataFrame):
    """Yield (cause, effect, weight) for every non-zero entry."""
    cols = list(adj_df.columns)
    M = adj_df.values
    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            w = float(M[i, j])
            if w != 0:
                yield src, tgt, w


def score_edge(
    client, cause: str, effect: str, cache: EdgeCache, raw_log_fh
) -> dict | None:
    """Return a validated score dict, or None on failure."""
    cached = cache.get(cause, effect)
    if cached is not None:
        cached = validate_response(cached)
        if cached is not None:
            return cached

    prompt = build_prompt(cause, effect)
    raw = client.generate(prompt)

    # Audit trail: always log raw response
    raw_log_fh.write(json.dumps({
        "cause": cause, "effect": effect, "raw": raw,
    }) + "\n")
    raw_log_fh.flush()

    parsed = parse_json_response(raw)
    validated = validate_response(parsed) if parsed else None

    if validated is None:
        # Record failure as placeholder; don't crash the whole run
        validated = {
            "plausibility_of_any_link": 0,  # sentinel: query failed
            "most_plausible_relationship": "parse_error",
            "mechanism_if_any": "PARSE_ERROR",
            "likely_confounders": "",
            "confidence": "low",
            "parse_error": True,
        }
    cache.set(cause, effect, validated)
    return validated


def evaluate_all_graphs(
    graphs: dict[str, pd.DataFrame],
    client,
    out_dir: str,
    max_edges: int | None,
    delay: float,
    no_cache: bool,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    cache = EdgeCache(os.path.join(out_dir, "cache.json"), enabled=not no_cache)
    raw_log_path = os.path.join(out_dir, "raw_responses.jsonl")

    rows = []
    with open(raw_log_path, "a") as raw_fh:
        for gname, df in graphs.items():
            edges = list(iter_edges(df))
            if max_edges and len(edges) > max_edges:
                edges = edges[:max_edges]
            print(f"\n[{gname}] scoring {len(edges)} edges")
            for idx, (src, tgt, w) in enumerate(edges, 1):
                try:
                    score = score_edge(client, src, tgt, cache, raw_fh)
                except Exception as e:
                    print(f"  [{idx}/{len(edges)}] {src}->{tgt} FAILED: {e}")
                    continue
                rows.append({
                    "graph": gname,
                    "cause": src,
                    "effect": tgt,
                    "cause_domain": VARIABLE_DOMAINS.get(src, "Unknown"),
                    "effect_domain": VARIABLE_DOMAINS.get(tgt, "Unknown"),
                    "weight": w,
                    **score,
                    "supports_proposed_direction": (
                        score["most_plausible_relationship"] == "A_causes_B"
                    ),
                })
                if idx % 10 == 0:
                    print(f"  [{idx}/{len(edges)}] cached so far: {len(cache.data)}")
                    cache.save()
                if delay > 0:
                    time.sleep(delay)
            cache.save()  # save after each graph

    cache.save()
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "edge_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[save] edge scores -> {out_csv}  ({len(df)} rows)")
    return df


# ── Aggregation ──────────────────────────────────────────────
def aggregate_per_model(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    valid = scores[scores["plausibility_of_any_link"] > 0].copy()

    rows = []
    for g, sub in valid.groupby("graph"):
        n = len(sub)
        if n == 0:
            continue
        plaus = sub["plausibility_of_any_link"]
        w_abs = sub["weight"].abs()
        w_norm = (w_abs / w_abs.max()) if w_abs.max() > 0 else w_abs
        rows.append({
            "graph": g,
            "n_edges_scored": n,
            "mean_plausibility": round(plaus.mean(), 2),
            "median_plausibility": int(plaus.median()),
            "pct_well_supported": round(100 * (plaus >= 4).mean(), 1),
            "pct_implausible": round(100 * (plaus <= 2).mean(), 1),
            "pct_supports_proposed_direction": round(
                100 * (sub["most_plausible_relationship"] == "A_causes_B").mean(), 1
            ),
            "pct_reverse_direction": round(
                100 * (sub["most_plausible_relationship"] == "B_causes_A").mean(), 1
            ),
            "pct_bidirectional": round(
                100 * (sub["most_plausible_relationship"] == "bidirectional").mean(), 1
            ),
            "pct_confounded": round(
                100 * (sub["most_plausible_relationship"] == "confounded").mean(), 1
            ),
            "pct_no_link": round(
                100 * (sub["most_plausible_relationship"] == "none").mean(), 1
            ),
            "weighted_plausibility": round((plaus * w_norm).sum() / w_norm.sum(), 2)
                if w_norm.sum() > 0 else round(plaus.mean(), 2),
        })
    out = pd.DataFrame(rows).sort_values("mean_plausibility", ascending=False)
    return out


# ── Figures ──────────────────────────────────────────────────
def _color_for(graph_name: str) -> str:
    if "deci" in graph_name:    return "#534AB7"
    if "gemma" in graph_name:   return "#D85A30"
    if "constrained" in graph_name: return "#1D9E75"
    if "lingam" in graph_name:  return "#E2A32E"
    if "notears" in graph_name: return "#9B59B6"
    return "#378ADD"


def plot_comparison(summary: pd.DataFrame, out_path: str):
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: mean plausibility
    ax = axes[0]
    names = [n.replace("_", "\n")[:22] for n in summary["graph"]]
    colors = [_color_for(n) for n in summary["graph"]]
    bars = ax.bar(range(len(names)), summary["mean_plausibility"],
                  color=colors, alpha=0.85, edgecolor="#333")
    for b, v in zip(bars, summary["mean_plausibility"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Mean plausibility (1-5)")
    ax.set_ylim(0, 5.5)
    ax.axhline(3, ls="--", color="#aaa", lw=0.8, label="neutral (3)")
    ax.set_title("Mean Edge Plausibility", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: % well-supported vs % implausible
    ax = axes[1]
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w/2, summary["pct_well_supported"], w, color="#1D9E75",
           alpha=0.85, label="well-supported (>=4)")
    ax.bar(x + w/2, summary["pct_implausible"], w, color="#E24B4A",
           alpha=0.85, label="implausible (<=2)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("% of scored edges")
    ax.set_title("Quality Split", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 3: relationship breakdown
    ax = axes[2]
    bottom = np.zeros(len(x))
    relationship_columns = [
        ("pct_supports_proposed_direction", "A causes B", "#1D9E75"),
        ("pct_bidirectional", "bidirectional", "#378ADD"),
        ("pct_reverse_direction", "B causes A", "#E2A32E"),
        ("pct_confounded", "confounded", "#888780"),
        ("pct_no_link", "none", "#E24B4A"),
    ]
    for col, label, color in relationship_columns:
        values = summary[col].to_numpy()
        ax.bar(x, values, bottom=bottom, color=color, alpha=0.85, label=label)
        bottom = bottom + values
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("% of scored edges")
    ax.set_title("Most Plausible Relationship", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Gemma Qualitative Comparison Across Causal Models",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] comparison figure -> {out_path}")


def plot_distribution(scores: pd.DataFrame, out_path: str):
    if scores.empty or "plausibility_of_any_link" not in scores:
        return
    valid = scores[scores["plausibility_of_any_link"] > 0].copy()
    if valid.empty:
        return
    graphs = sorted(valid["graph"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(graphs) * 1.4), 5))
    data = [valid[valid["graph"] == g]["plausibility_of_any_link"].values for g in graphs]
    bp = ax.boxplot(data, labels=[g.replace("_", "\n")[:18] for g in graphs],
                    patch_artist=True, widths=0.6)
    for patch, g in zip(bp["boxes"], graphs):
        patch.set_facecolor(_color_for(g))
        patch.set_alpha(0.7)
    ax.set_ylabel("Plausibility (1-5)")
    ax.set_ylim(0.5, 5.5)
    ax.axhline(3, ls="--", color="#aaa", lw=0.8)
    ax.set_title("Plausibility Distribution per Model", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.xticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] distribution figure -> {out_path}")


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default="outputs/graphs")
    parser.add_argument("--out-dir",   default="outputs/gemma_eval")
    parser.add_argument("--backend", choices=["ollama", "google"], default="ollama")
    parser.add_argument("--model", default=None,
                        help="Model name (default: gemma3:4b for ollama, gemma-4-26b-a4b-it for google)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama host URL")
    parser.add_argument("--api-key", default=None,
                        help="Google API key (else uses GEMINI_API_KEY env var)")
    parser.add_argument("--max-edges", type=int, default=None,
                        help="Cap edges per graph (useful for smoke tests)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Sleep between API calls (respect rate limits)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache and re-query every edge (recommended after prompt changes)")
    args = parser.parse_args()

    # Resolve model name default by backend
    if args.model is None:
        args.model = "gemma3:4b" if args.backend == "ollama" else "gemma-4-26b-a4b-it"

    print(f"[init] backend={args.backend}  model={args.model}")

    # Build client
    if args.backend == "ollama":
        client = OllamaClient(model=args.model, host=args.host)
    else:
        client = GoogleClient(model=args.model, api_key=args.api_key)

    # Load graphs
    graphs = load_graphs(args.graph_dir)
    if not graphs:
        print(f"[error] No adjacency CSVs found in {args.graph_dir}/")
        sys.exit(1)
    total_edges = sum(int((df.values != 0).sum()) for df in graphs.values())
    print(f"[init] {len(graphs)} graphs, {total_edges} total edges to consider")
    if args.max_edges:
        print(f"[init] capping at {args.max_edges} edges per graph")

    # Score
    scores = evaluate_all_graphs(
        graphs, client, args.out_dir,
        max_edges=args.max_edges, delay=args.delay, no_cache=args.no_cache,
    )

    # Aggregate
    summary = aggregate_per_model(scores)
    summary_path = os.path.join(args.out_dir, "model_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n[save] model summary -> {summary_path}")
    if not summary.empty:
        print("\n" + summary.to_string(index=False))

    # Figures
    plot_comparison(summary, os.path.join(args.out_dir, "comparison.png"))
    plot_distribution(scores, os.path.join(args.out_dir, "distribution.png"))

    print(f"\n[done] Everything in {args.out_dir}/")


if __name__ == "__main__":
    main()
