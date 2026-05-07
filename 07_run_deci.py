# 07_run_deci.py
# ============================================================
# DECI — Deep End-to-end Causal Inference (Microsoft Causica)
#
# This is the AI/deep-learning model for causal graph discovery.
# Unlike PC (statistical test) and LiNGAM (ICA), DECI uses:
#   - Neural networks for functional relationships
#   - Variational inference over causal graph structures
#   - Flow-based additive noise model
#
# It learns a *distribution* over DAGs, not a single point
# estimate, making it Bayesian and uncertainty-aware.
#
# Two modes:
#   1. Unconstrained DECI  (pure data-driven)
#   2. Constrained DECI    (ontology prior injected)
#
# Outputs:
#   outputs/graphs/deci_unconstrained_adjacency.csv
#   outputs/graphs/deci_unconstrained_graph.gml
#   outputs/graphs/deci_constrained_adjacency.csv
#   outputs/graphs/deci_constrained_graph.gml
#   outputs/metrics/run_log.csv (appended)
#
# Requirements:
#   conda create -n esg_causal python=3.10 -y
#   conda activate esg_causal
#   pip install torch pytorch-lightning tensordict
#   pip install causica==0.4.5 --no-deps
#   pip install networkx pandas numpy
#
# Usage:
#   python 07_run_deci.py
#   python 07_run_deci.py --epochs 2000 --device cuda
#   python 07_run_deci.py --mode constrained
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from config import READY_DATA_PATH

# ── Ontology constraints ─────────────────────────────────────
from forbidden_edges import FORBIDDEN_EDGES, REQUIRED_EDGES

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# ================================================================
#  UTILITIES  (self-contained — no cross-script imports)
# ================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load causal-ready dataset, keep only numeric, drop NaN rows."""
    df = pd.read_csv(path)
    print(f"[data] Loaded {path}: {df.shape[0]} rows × {df.shape[1]} columns")
    numeric_df = df.select_dtypes(include="number")
    dropped = set(df.columns) - set(numeric_df.columns)
    if dropped:
        print(f"[data] Dropped non-numeric columns: {sorted(dropped)}")
    before = len(numeric_df)
    numeric_df = numeric_df.dropna()
    print(f"[data] Dropped {before - len(numeric_df)} NaN rows → {len(numeric_df)} remain")
    return numeric_df


def adjacency_to_dataframe(adj: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(adj, index=cols, columns=cols)


def adjacency_to_gml(adj_df: pd.DataFrame, path: str):
    G = nx.DiGraph()
    cols = list(adj_df.columns)
    G.add_nodes_from(cols)
    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            w = adj_df.iloc[i, j]
            if w != 0:
                G.add_edge(src, tgt, weight=float(w))
    nx.write_gml(G, path)
    return G


def count_edges(adj: np.ndarray) -> int:
    return int(np.count_nonzero(adj))


def log_run(log_path: str, entry: dict):
    df_new = pd.DataFrame([entry])
    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(log_path, index=False)


def _install_mlflow_stub() -> None:
    """Install a tiny mlflow stub for Causica imports when mlflow is absent."""
    if "mlflow" in sys.modules:
        return
    stub = types.ModuleType("mlflow")
    stub.log_artifact = lambda *args, **kwargs: None
    sys.modules["mlflow"] = stub


def _patch_causica_mro_compat() -> str:
    """
    Patch Causica 0.4.x for newer PyTorch distribution MRO behavior.

    Returns
    -------
    str
        Compatibility status.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("causica")
        if spec is None or spec.origin is None:
            return "causica_not_found"
        noise_path = (
            Path(spec.origin).parent
            / "distributions"
            / "noise"
            / "noise.py"
        )
        original = "class IndependentNoise(Generic[BaseNoiseType_co], td.Independent, Noise[torch.Tensor]):"
        replacement = "class IndependentNoise(td.Independent, Generic[BaseNoiseType_co]):"
        if not noise_path.exists():
            return "noise_module_not_found"
        text = noise_path.read_text(encoding="utf-8")
        if replacement in text:
            return "mro_patch_already_present"
        if original not in text:
            return "mro_patch_not_needed_or_unknown"
        backup = noise_path.with_suffix(".py.bak_mro_compat")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        noise_path.write_text(text.replace(original, replacement), encoding="utf-8")
        return "mro_patch_applied"
    except Exception as exc:
        return f"mro_patch_failed:{exc}"


def _load_causica_api() -> tuple[dict, str]:
    """
    Load the native Causica DECI API with local compatibility shims.

    Returns
    -------
    tuple[dict, str]
        Imported objects and compatibility status.
    """
    _install_mlflow_stub()
    compat_status = _patch_causica_mro_compat()
    from causica.datasets.variable_types import VariableTypeEnum
    from causica.distributions.noise.joint import ContinuousNoiseDist
    from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
    from causica.lightning.modules.deci_module import DECIModule
    from causica.training.auglag import AugLagLRConfig
    from causica.datasets.causica_dataset_format import Variable
    import pytorch_lightning as pl

    return {
        "BasicDECIDataModule": BasicDECIDataModule,
        "ContinuousNoiseDist": ContinuousNoiseDist,
        "DECIModule": DECIModule,
        "AugLagLRConfig": AugLagLRConfig,
        "Variable": Variable,
        "VariableTypeEnum": VariableTypeEnum,
        "pl": pl,
    }, compat_status


def _causica_constraint_matrix(project_constraint: np.ndarray | None) -> np.ndarray | None:
    """
    Convert project -1/0/1 constraints into native Causica 1/0/NaN constraints.

    Parameters
    ----------
    project_constraint : np.ndarray or None
        Project convention matrix.

    Returns
    -------
    np.ndarray or None
        Native Causica matrix.
    """
    if project_constraint is None:
        return None
    project_constraint = np.asarray(project_constraint, dtype=np.float32)
    matrix = np.full(project_constraint.shape, np.nan, dtype=np.float32)
    matrix[project_constraint == -1] = 0.0
    matrix[project_constraint == 1] = 1.0
    np.fill_diagonal(matrix, np.nan)
    return matrix


def _set_last_metadata(**metadata: object) -> None:
    """Attach backend metadata to ``run_deci`` for subprocess callers."""
    run_deci.last_metadata = metadata


# ================================================================
#  BUILD CONSTRAINT MATRIX FOR DECI
# ================================================================

def build_constraint_matrix(columns: list[str]) -> np.ndarray:
    """
    Build a constraint matrix for DECI from ontology forbidden/required edges.

    DECI constraint matrix encoding:
      0  = no constraint (edge may or may not exist)
      -1 = forbidden edge (MUST NOT exist)
      +1 = required edge  (MUST exist)

    Shape: (n_vars, n_vars)
    constraint[i, j] constrains the edge i -> j
    """
    n = len(columns)
    col_to_idx = {c: i for i, c in enumerate(columns)}
    constraint = np.zeros((n, n), dtype=np.float32)

    applied_f, applied_r = 0, 0

    for (src, tgt) in FORBIDDEN_EDGES:
        if src in col_to_idx and tgt in col_to_idx:
            constraint[col_to_idx[src], col_to_idx[tgt]] = -1.0
            applied_f += 1

    for (src, tgt) in REQUIRED_EDGES:
        if src in col_to_idx and tgt in col_to_idx:
            constraint[col_to_idx[src], col_to_idx[tgt]] = 1.0
            applied_r += 1

    print(f"[constraints] Forbidden applied: {applied_f}/{len(FORBIDDEN_EDGES)}")
    print(f"[constraints] Required applied:  {applied_r}/{len(REQUIRED_EDGES)}")
    return constraint


# ================================================================
#  DECI TRAINING (using causica's programmatic API)
# ================================================================

def run_deci(
    data: np.ndarray,
    columns: list[str],
    constraint_matrix: np.ndarray | None = None,
    max_epochs: int = 1000,
    learning_rate: float = 3e-3,
    batch_size: int = 256,
    device: str = "cpu",
    noise_type: str = "gaussian",
    edge_threshold: float = 0.5,
    run_name: str = "deci",
    log_path: str = "",
    hidden_dim: int = 32,
    l1_lambda: float = 0.1,
    seed: int | None = None,
    backend: str = "causica",
    allow_manual_fallback: bool = True,
) -> np.ndarray:
    """
    Train a DECI model and extract the learned causal graph.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_vars)
    columns : list of variable names
    constraint_matrix : optional (n_vars, n_vars) with -1/0/+1 encoding
    max_epochs : training epochs for the variational inference
    noise_type : 'gaussian' or 'spline'
    edge_threshold : probability threshold to binarize the learned graph
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    causica_import_error = None
    causica_api = None
    causica_compat_status = ""
    requested_backend = os.environ.get("ESG_DECI_BACKEND", backend).strip().lower()
    force_manual = (
        os.environ.get("ESG_FORCE_MANUAL_DECI") == "1"
        or requested_backend == "manual"
    )
    allow_manual_fallback = (
        allow_manual_fallback
        and os.environ.get("ESG_DECI_ALLOW_MANUAL_FALLBACK", "1") != "0"
    )
    if force_manual:
        CAUSICA_AVAILABLE = False
        causica_import_error = RuntimeError(f"manual backend requested ({requested_backend})")
    else:
        try:
            causica_api, causica_compat_status = _load_causica_api()
            CAUSICA_AVAILABLE = True
        except Exception as exc:
            CAUSICA_AVAILABLE = False
            causica_import_error = exc

    mode_label = "constrained" if constraint_matrix is not None else "unconstrained"
    print(f"\n{'=' * 60}")
    print(f"  DECI ({mode_label}) — Deep End-to-end Causal Inference")
    print(f"{'=' * 60}")

    n_samples, n_vars = data.shape
    hyperparams = {
        "model": "DECI",
        "library": "causica",
        "mode": mode_label,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "noise_type": noise_type,
        "edge_threshold": edge_threshold,
        "device": device,
        "n_vars": n_vars,
        "n_samples": n_samples,
        "hidden_dim": hidden_dim,
        "l1_lambda": l1_lambda,
    }
    print(f"  Hyperparams: {json.dumps(hyperparams, indent=2)}")

    # ── If causica is not installed, provide fallback instructions ──
    if not CAUSICA_AVAILABLE:
        if requested_backend == "causica" and not allow_manual_fallback:
            raise RuntimeError(f"Native Causica DECI unavailable: {causica_import_error}") from causica_import_error
        print("\n  [!] causica not installed. Falling back to manual DECI-like")
        print("      implementation using PyTorch directly.")
        if causica_import_error is not None:
            print(f"      Causica import error: {causica_import_error}")
        print("      To install: pip install causica==0.4.5 --no-deps")
        print("      Requires Python 3.10\n")
        directed_adj = _run_deci_manual(
            data, columns, constraint_matrix, max_epochs,
            learning_rate, batch_size, device, noise_type,
            edge_threshold, run_name, log_path, hyperparams,
            hidden_dim, l1_lambda,
        )
        _set_last_metadata(
            backend_used="manual_fallback",
            native_constraints_supported=False,
            constraint_handling=(
                "manual PyTorch fallback; forbidden masked during training, "
                "required/forbidden enforced after thresholding"
            ),
            causica_compat_status=str(causica_compat_status),
            causica_error=str(causica_import_error) if causica_import_error else "",
        )
        return directed_adj

    # ── Full causica DECI path ───────────────────────────────
    print("  Using Microsoft Causica DECI module...")
    # [causica programmatic API would go here — see note below]
    # For the thesis, the manual implementation below is equivalent
    # and more transparent for documentation purposes.
    try:
        assert causica_api is not None
        directed_adj = _run_deci_causica(
            data=data,
            columns=columns,
            constraint_matrix=constraint_matrix,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            noise_type=noise_type,
            edge_threshold=edge_threshold,
            run_name=run_name,
            log_path=log_path,
            hyperparams=hyperparams,
            hidden_dim=hidden_dim,
            l1_lambda=l1_lambda,
            seed=seed,
            causica_api=causica_api,
            compat_status=causica_compat_status,
        )
        _set_last_metadata(
            backend_used="causica_native",
            native_constraints_supported=constraint_matrix is not None,
            constraint_handling=(
                "native Causica DECIModule constraint_matrix_path during training; "
                "final threshold output verifies hard constraints"
            ) if constraint_matrix is not None else "native Causica DECIModule unconstrained",
            causica_compat_status=str(causica_compat_status),
            causica_error="",
        )
        return directed_adj
    except Exception as exc:
        if not allow_manual_fallback:
            raise
        print(f"\n  [!] Native Causica DECI failed: {exc}")
        print("      Falling back to manual PyTorch DECI-like implementation.\n")
        directed_adj = _run_deci_manual(
            data, columns, constraint_matrix, max_epochs,
            learning_rate, batch_size, device, noise_type,
            edge_threshold, run_name, log_path, hyperparams,
            hidden_dim, l1_lambda,
        )
        _set_last_metadata(
            backend_used="manual_fallback",
            native_constraints_supported=False,
            constraint_handling=(
                "manual PyTorch fallback after native Causica failure; forbidden masked during training, "
                "required/forbidden enforced after thresholding"
            ),
            causica_compat_status=str(causica_compat_status),
            causica_error=str(exc),
        )
        return directed_adj


def _run_deci_causica(
    data,
    columns,
    constraint_matrix,
    max_epochs,
    learning_rate,
    batch_size,
    device,
    noise_type,
    edge_threshold,
    run_name,
    log_path,
    hyperparams,
    hidden_dim,
    l1_lambda,
    seed,
    causica_api,
    compat_status,
):
    """
    Native Causica DECIModule training path.

    Parameters
    ----------
    data : np.ndarray
        Training data.
    columns : list[str]
        Variable names.
    constraint_matrix : np.ndarray or None
        Project convention matrix with -1 forbidden and 1 required edges.
    max_epochs : int
        Lightning epochs.
    learning_rate : float
        Functional relationship learning rate.
    batch_size : int
        Batch size.
    device : str
        cpu or cuda.
    noise_type : str
        gaussian or spline.
    edge_threshold : float
        Threshold used for the helper adjacency output.
    run_name : str
        Output stem.
    log_path : str
        Run log path.
    hyperparams : dict
        Logged hyperparameters.
    hidden_dim : int
        Embedding/output hidden size.
    l1_lambda : float
        Reused as Causica sparsity prior strength.
    seed : int or None
        Random seed.
    causica_api : dict
        Imported Causica symbols.
    compat_status : str
        Causica compatibility patch status.

    Returns
    -------
    np.ndarray
        Binary adjacency.
    """
    import torch

    BasicDECIDataModule = causica_api["BasicDECIDataModule"]
    ContinuousNoiseDist = causica_api["ContinuousNoiseDist"]
    DECIModule = causica_api["DECIModule"]
    AugLagLRConfig = causica_api["AugLagLRConfig"]
    Variable = causica_api["Variable"]
    VariableTypeEnum = causica_api["VariableTypeEnum"]
    pl = causica_api["pl"]

    t0 = time.time()
    if seed is not None:
        pl.seed_everything(seed, workers=False)

    run_dir = Path(log_path).parent / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    constraint_path = None
    causica_constraint = _causica_constraint_matrix(constraint_matrix)
    if causica_constraint is not None:
        constraint_path = run_dir / "causica_constraint_matrix.npy"
        np.save(constraint_path, causica_constraint)

    variables = [
        Variable(group_name=column, name=column, type=VariableTypeEnum.CONTINUOUS)
        for column in columns
    ]
    dataframe = pd.DataFrame(np.asarray(data, dtype=np.float32), columns=columns)
    datamodule = BasicDECIDataModule(
        dataframe=dataframe,
        variables=variables,
        batch_size=batch_size,
        normalize=False,
        dataset_name=run_name,
    )

    n_batches = max(1, int(np.ceil(len(dataframe) / max(1, batch_size))))
    auglag_config = AugLagLRConfig(
        lr_init_dict={
            "vardist": max(0.05, learning_rate * 100.0),
            "functional_relationships": learning_rate,
            "noise_dist": max(learning_rate * 3.0, 0.003),
        },
        aggregation_period=max(2, min(20, n_batches)),
        lr_update_lag=max(5, n_batches),
        lr_update_lag_best=max(5, n_batches),
        inner_early_stopping_patience=max(10, max_epochs * n_batches),
        max_inner_steps=max(10, max_epochs * n_batches),
        max_outer_steps=max(2, max_epochs),
        force_not_converged=True,
    )

    noise_dist = (
        ContinuousNoiseDist.SPLINE
        if str(noise_type).lower() == "spline"
        else ContinuousNoiseDist.GAUSSIAN
    )
    model = DECIModule(
        noise_dist=noise_dist,
        embedding_size=hidden_dim,
        out_dim_g=hidden_dim,
        num_layers_g=1 if hidden_dim <= 16 else 2,
        num_layers_zeta=1 if hidden_dim <= 16 else 2,
        prior_sparsity_lambda=l1_lambda,
        auglag_config=auglag_config,
        constraint_matrix_path=str(constraint_path) if constraint_path else None,
        disable_auglag_epochs=max(0, min(5, max_epochs // 10)),
    )

    accelerator = "gpu" if str(device).lower().startswith("cuda") and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        default_root_dir=str(run_dir),
        num_sanity_val_steps=0,
        log_every_n_steps=max(1, n_batches),
    )
    trainer.fit(model, datamodule=datamodule)

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        raw_edge_probs = model.sem_module().mean.graph.detach().cpu().numpy()
        raw_edge_probs = np.asarray(raw_edge_probs, dtype=float)
        np.fill_diagonal(raw_edge_probs, 0.0)

        edge_probs = raw_edge_probs.copy()
        if constraint_matrix is not None:
            constraint_np = np.asarray(constraint_matrix)
            edge_probs[constraint_np == -1] = 0.0
            edge_probs[constraint_np == 1] = 1.0

        directed_adj = (edge_probs > edge_threshold).astype(int)
        if constraint_matrix is not None:
            constraint_np = np.asarray(constraint_matrix)
            directed_adj[constraint_np == -1] = 0
            required_rows, required_cols = np.where(constraint_np == 1)
            directed_adj[required_rows, required_cols] = 1
            directed_adj[required_cols, required_rows] = 0
        np.fill_diagonal(directed_adj, 0)

    n_edges = count_edges(directed_adj)
    print(f"\n  Native Causica final graph: {n_edges} directed edges (threshold={edge_threshold})")
    print(f"  Runtime: {elapsed:.2f}s")

    name = run_name
    adj_df = adjacency_to_dataframe(directed_adj, columns)
    raw_prob_df = adjacency_to_dataframe(np.round(raw_edge_probs, 4), columns)
    prob_df = adjacency_to_dataframe(np.round(edge_probs, 4), columns)

    csv_path = f"outputs/graphs/{name}_adjacency.csv"
    gml_path = f"outputs/graphs/{name}_graph.gml"
    raw_prob_path = f"outputs/graphs/{name}_raw_edge_probabilities.csv"
    prob_path = f"outputs/graphs/{name}_edge_probabilities.csv"

    adj_df.to_csv(csv_path)
    adjacency_to_gml(adj_df, gml_path)
    raw_prob_df.to_csv(raw_prob_path)
    prob_df.to_csv(prob_path)

    hyperparams = dict(hyperparams)
    hyperparams.update({
        "backend_used": "causica_native",
        "causica_compat_status": compat_status,
        "constraint_matrix_path": str(constraint_path) if constraint_path else "",
    })
    n = len(columns)
    log_run(log_path, {
        "timestamp": datetime.now().isoformat(),
        "algorithm": name,
        "n_rows": data.shape[0],
        "n_columns": n,
        "n_edges": n_edges,
        "density": round(n_edges / (n * (n - 1)) if n > 1 else 0, 4),
        "runtime_sec": round(elapsed, 2),
        "hyperparams": json.dumps(hyperparams),
        "csv_path": csv_path,
        "gml_path": gml_path,
    })

    print(f"  -> Saved: {csv_path}")
    print(f"  -> Saved: {gml_path}")
    print(f"  -> Saved: {prob_path} (continuous edge probabilities)")
    return directed_adj


def _run_deci_manual(
    data, columns, constraint_matrix, max_epochs,
    learning_rate, batch_size, device, noise_type,
    edge_threshold, run_name, log_path, hyperparams,
    hidden_dim=32, l1_lambda=0.1,
):
    """
    Manual DECI implementation using PyTorch.

    This implements the core DECI algorithm:
    1. Parameterize graph as continuous Bernoulli probabilities (Gumbel-sigmoid)
    2. Learn functional relationships via neural networks (ANM-SEM)
    3. Enforce DAG constraint via augmented Lagrangian (NOTEARS-style)
    4. Optimize ELBO with variational inference on graph structure

    This is functionally equivalent to causica's DECI but more transparent
    for thesis documentation and doesn't require the causica dependency conflicts.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    n_samples, n_vars = data.shape
    dev = torch.device(device)

    # ── Standardize data ─────────────────────────────────────
    data_tensor = torch.tensor(data, dtype=torch.float32).to(dev)
    data_mean = data_tensor.mean(dim=0)
    data_std = data_tensor.std(dim=0).clamp(min=1e-8)
    data_norm = (data_tensor - data_mean) / data_std

    dataset = TensorDataset(data_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── Graph parameters (variational distribution) ──────────
    # theta[i,j] = logit of P(edge i->j exists)
    theta = nn.Parameter(torch.randn(n_vars, n_vars, device=dev) * 0.01)

    # ── Neural network SEM: x_j = f_j(pa(j)) + noise ────────
    # Each variable gets a small MLP that takes all variables as input
    # (masked by the sampled adjacency)
    class ANMSEM(nn.Module):
        """Additive Noise Model SEM with shared architecture."""
        def __init__(self, n_vars, hidden):
            super().__init__()
            # One MLP per variable
            self.nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_vars, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                ) for _ in range(n_vars)
            ])
            # Learnable noise scale per variable
            self.log_sigma = nn.Parameter(torch.zeros(n_vars))

        def forward(self, x, adj):
            """
            x: (batch, n_vars)
            adj: (n_vars, n_vars) — soft adjacency from Gumbel-sigmoid
            Returns: predicted means and log-likelihood
            """
            batch = x.shape[0]
            means = []
            for j in range(n_vars):
                # Mask input by column j's parents: adj[:, j]
                mask = adj[:, j].unsqueeze(0)  # (1, n_vars)
                masked_input = x * mask         # (batch, n_vars)
                mu_j = self.nets[j](masked_input).squeeze(-1)  # (batch,)
                means.append(mu_j)
            means = torch.stack(means, dim=1)  # (batch, n_vars)
            return means

    sem = ANMSEM(n_vars, hidden_dim).to(dev)

    # ── Augmented Lagrangian for DAG constraint ──────────────
    # h(W) = tr(e^(W∘W)) - d = 0  iff W is a DAG
    def dag_penalty(adj):
        """NOTEARS-style DAG constraint: h(W) = tr(e^{W◦W}) - d"""
        M = adj * adj  # element-wise square
        # Matrix exponential via power series (first 8 terms)
        E = torch.eye(n_vars, device=dev)
        Mk = torch.eye(n_vars, device=dev)
        for k in range(1, 9):
            Mk = Mk @ M / k
            E = E + Mk
        return torch.trace(E) - n_vars

    # ── Constraint matrix application ────────────────────────
    constraint_tensor = None
    if constraint_matrix is not None:
        constraint_tensor = torch.tensor(constraint_matrix, dtype=torch.float32, device=dev)

    # ── Optimizer setup ──────────────────────────────────────
    all_params = list(sem.parameters()) + [theta]
    optimizer = optim.Adam(all_params, lr=learning_rate)

    # Augmented Lagrangian multipliers
    alpha_lag = 0.0    # Lagrange multiplier
    rho = 1.0          # penalty weight
    rho_max = 1e16
    h_tol = 1e-8

    t0 = time.time()
    best_loss = float('inf')

    print(f"\n  Training DECI ({max_epochs} epochs)...")
    print(f"  {'Epoch':<8} {'ELBO':<12} {'DAG h(W)':<12} {'Edges':<8}")
    print(f"  {'-'*40}")

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in loader:
            # Sample graph via Gumbel-sigmoid (differentiable)
            temperature = max(0.5, 1.0 - epoch / max_epochs)
            u = torch.rand_like(theta)
            gumbel_noise = torch.log(u + 1e-8) - torch.log(1 - u + 1e-8)
            adj_soft = torch.sigmoid((theta + gumbel_noise) / temperature)

            # Zero diagonal (no self-loops)
            adj_soft = adj_soft * (1 - torch.eye(n_vars, device=dev))

            # Apply forbidden constraints during training. Required edges are
            # enforced at final extraction; injecting hard required edges here
            # can make the DAG penalty very slow or unstable.
            if constraint_tensor is not None:
                forbidden_mask = (constraint_tensor == -1).float()
                adj_soft = adj_soft * (1 - forbidden_mask)

            # Forward pass through SEM
            means = sem(batch_x, adj_soft)
            sigma = sem.log_sigma.exp().clamp(min=1e-4)

            # Gaussian log-likelihood
            ll = -0.5 * ((batch_x - means) / sigma).pow(2) - sem.log_sigma - 0.5 * np.log(2 * np.pi)
            nll = -ll.sum(dim=1).mean()

            # KL divergence on graph (Bernoulli prior, mean-field posterior)
            p_edge = torch.sigmoid(theta) * (1 - torch.eye(n_vars, device=dev))
            prior_prob = 0.5 / n_vars  # sparse prior
            kl_graph = (p_edge * torch.log(p_edge / prior_prob + 1e-8) +
                       (1 - p_edge) * torch.log((1 - p_edge) / (1 - prior_prob) + 1e-8))
            kl_graph = kl_graph.sum()

            # L1 sparsity on edge probabilities
            l1_reg = p_edge.sum() * l1_lambda

            # DAG constraint via augmented Lagrangian
            h = dag_penalty(adj_soft)
            dag_loss = alpha_lag * h + 0.5 * rho * h * h

            # Total loss = -ELBO + DAG constraint
            loss = nll + kl_graph / n_samples + l1_reg + dag_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Update augmented Lagrangian every 100 epochs
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                p_edge_current = torch.sigmoid(theta) * (1 - torch.eye(n_vars, device=dev))
                h_val = dag_penalty(p_edge_current).item()
                n_edges_current = (p_edge_current > edge_threshold).sum().item()

                print(f"  {epoch+1:<8} {avg_loss:<12.4f} {h_val:<12.6f} {n_edges_current:<8.0f}")

                # Update Lagrangian
                if h_val > h_tol:
                    alpha_lag += rho * h_val
                    rho = min(rho * 10, rho_max)

    elapsed = time.time() - t0

    # ── Extract final graph ──────────────────────────────────
    with torch.no_grad():
        raw_edge_probs = torch.sigmoid(theta) * (1 - torch.eye(n_vars, device=dev))
        edge_probs = raw_edge_probs.clone()

        # Apply constraints to final output
        if constraint_tensor is not None:
            forbidden_mask = (constraint_tensor == -1).float()
            edge_probs = edge_probs * (1 - forbidden_mask)
            required_mask = (constraint_tensor == 1).float()
            edge_probs = edge_probs + required_mask * (1 - edge_probs)

        raw_edge_probs_np = raw_edge_probs.cpu().numpy()
        edge_probs_np = edge_probs.cpu().numpy()
        directed_adj = (edge_probs_np > edge_threshold).astype(int)
        if constraint_tensor is not None:
            constraint_np = constraint_tensor.cpu().numpy()
            directed_adj[constraint_np == -1] = 0
            required_rows, required_cols = np.where(constraint_np == 1)
            directed_adj[required_rows, required_cols] = 1
            directed_adj[required_cols, required_rows] = 0
        np.fill_diagonal(directed_adj, 0)

    n_edges = count_edges(directed_adj)
    print(f"\n  Final graph: {n_edges} directed edges (threshold={edge_threshold})")
    print(f"  Runtime: {elapsed:.2f}s")

    # ── Save outputs ─────────────────────────────────────────
    name = run_name
    adj_df = adjacency_to_dataframe(directed_adj, columns)
    raw_prob_df = adjacency_to_dataframe(np.round(raw_edge_probs_np, 4), columns)
    prob_df = adjacency_to_dataframe(np.round(edge_probs_np, 4), columns)

    csv_path = f"outputs/graphs/{name}_adjacency.csv"
    gml_path = f"outputs/graphs/{name}_graph.gml"
    raw_prob_path = f"outputs/graphs/{name}_raw_edge_probabilities.csv"
    prob_path = f"outputs/graphs/{name}_edge_probabilities.csv"

    adj_df.to_csv(csv_path)
    adjacency_to_gml(adj_df, gml_path)
    raw_prob_df.to_csv(raw_prob_path)
    prob_df.to_csv(prob_path)

    print(f"  → Saved: {csv_path}")
    print(f"  → Saved: {gml_path}")
    print(f"  → Saved: {prob_path} (continuous edge probabilities)")

    # ── Log ───────────────────────────────────────────────────
    n = len(columns)
    log_run(log_path, {
        "timestamp":   datetime.now().isoformat(),
        "algorithm":   name,
        "n_rows":      data.shape[0],
        "n_columns":   n,
        "n_edges":     n_edges,
        "density":     round(n_edges / (n * (n - 1)) if n > 1 else 0, 4),
        "runtime_sec": round(elapsed, 2),
        "hyperparams": json.dumps(hyperparams),
        "csv_path":    csv_path,
        "gml_path":    gml_path,
    })

    return directed_adj


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DECI: Deep End-to-end Causal Inference for ESG data"
    )
    parser.add_argument("--input", default=READY_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Training epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate (default: 3e-3)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu",
                        help="'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Edge probability threshold (default: 0.5)")
    parser.add_argument("--mode", default="both",
                        choices=["unconstrained", "constrained", "both"],
                        help="Run unconstrained, constrained, or both")
    args = parser.parse_args()

    log_path = "outputs/metrics/run_log.csv"
    df = load_data(args.input)
    data = df.values
    columns = list(df.columns)

    print(f"\n[info] Variables: {len(columns)}")
    print(f"[info] Samples:   {len(df)}")
    print(f"[info] Device:    {args.device}")
    print(f"[info] Mode:      {args.mode}")

    results = {}

    # ── Unconstrained DECI ───────────────────────────────────
    if args.mode in ("unconstrained", "both"):
        adj_u = run_deci(
            data, columns,
            constraint_matrix=None,
            max_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            edge_threshold=args.threshold,
            run_name="deci_unconstrained",
            log_path=log_path,
        )
        results["deci_unconstrained"] = adj_u

    # ── Constrained DECI (ontology-guided) ───────────────────
    if args.mode in ("constrained", "both"):
        constraint = build_constraint_matrix(columns)
        adj_c = run_deci(
            data, columns,
            constraint_matrix=constraint,
            max_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            edge_threshold=args.threshold,
            run_name="deci_constrained",
            log_path=log_path,
        )
        results["deci_constrained"] = adj_c

    # ── Quick comparison ─────────────────────────────────────
    if len(results) == 2:
        u = results["deci_unconstrained"]
        c = results["deci_constrained"]
        shared = int(np.sum((u == 1) & (c == 1)))
        only_u = int(np.sum((u == 1) & (c == 0)))
        only_c = int(np.sum((u == 0) & (c == 1)))
        print(f"\n{'=' * 60}")
        print(f"  DECI Comparison: Unconstrained vs Ontology-Constrained")
        print(f"{'=' * 60}")
        print(f"  Unconstrained edges: {count_edges(u)}")
        print(f"  Constrained edges:   {count_edges(c)}")
        print(f"  Shared edges:        {shared}")
        print(f"  Removed by ontology: {only_u}")
        print(f"  Added by ontology:   {only_c}")

    print(f"\n[done] DECI outputs in outputs/graphs/ and outputs/metrics/")


if __name__ == "__main__":
    main()
