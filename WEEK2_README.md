# Week 2 — Baseline + Hybrid Pipeline (Apr 6–12)

## Quick Start

### 1. Install dependencies
```bash
pip install causal-learn lingam networkx pandas numpy
# Optional for NOTEARS:
pip install gcastle
```

### 2. Ensure your real data is in place
Your cleaned data should be at `data/processed/data_ready.csv` (output of `02_clean.py`).
If you haven't run cleaning yet:
```bash
python 02_clean.py --input data/raw/esg_raw.csv
```

### 3. Run all three baselines
```bash
python 05_run_baselines.py
```
This produces:
| Output | Description |
|---|---|
| `outputs/graphs/unconstrained_pc_adjacency.csv` | Adjacency matrix — PC without constraints |
| `outputs/graphs/unconstrained_pc_graph.gml` | Graph file for NetworkX/Gephi/Cytoscape |
| `outputs/graphs/unconstrained_lingam_adjacency.csv` | Adjacency matrix — DirectLiNGAM |
| `outputs/graphs/unconstrained_lingam_weights.csv` | Weighted adjacency (actual coefficients) |
| `outputs/graphs/unconstrained_lingam_graph.gml` | Graph file |
| `outputs/graphs/constrained_pc_adjacency.csv` | Adjacency matrix — PC with ontology constraints |
| `outputs/graphs/constrained_pc_graph.gml` | Graph file |
| `outputs/metrics/run_log.csv` | Reproducibility log (algo, params, timestamp, edge count) |

### 4. (Optional) Run NOTEARS via gCastle
```bash
python 06_run_notears.py
```

### 5. Tune hyperparameters
```bash
# Stricter significance level → fewer edges
python 05_run_baselines.py --alpha 0.01

# NOTEARS: more sparsity, stricter pruning
python 06_run_notears.py --lambda1 0.2 --w-threshold 0.5
```

## Deliverables Checklist (Week 2)
- [ ] Three causal graphs: unconstrained PC, unconstrained LiNGAM, constrained PC
- [ ] Saved as .gml + adjacency CSV
- [ ] Reproducible via single script
- [ ] Run log with algorithm name, hyperparameters, timestamp
- [ ] (Bonus) NOTEARS comparison baseline

## File Naming for Import
Python can't import files starting with digits. If you keep the `04_`/`05_` prefix convention:
- Rename for import: `cp 04_forbidden_edges.py forbidden_edges.py`
- Or use symlinks: `ln -s 04_forbidden_edges.py forbidden_edges.py`

The scripts assume `forbidden_edges.py` and `config.py` are importable from the working directory.

## Notes on Algorithms

**PC (Peter-Clark)**: Constraint-based. Uses conditional independence tests (Fisher-Z) to discover the skeleton, then orients edges. Works well with moderate dimensions. The constrained version injects your ontology's forbidden/required edges via `BackgroundKnowledge`.

**DirectLiNGAM**: Assumes linear, non-Gaussian data. Discovers a full causal ordering and weighted adjacency matrix. The weights represent causal effect sizes. Threshold of 0.01 applied to remove noise edges.

**NOTEARS**: Score-based continuous optimization. Treats DAG learning as a constrained optimization problem. The `lambda1` parameter controls sparsity (higher = fewer edges). `w_threshold` prunes weak edges post-optimization.
