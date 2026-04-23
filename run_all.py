import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def format_command(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def relaunch_in_venv_if_available() -> int | None:
    if not VENV_PYTHON.exists():
        return None

    current_python = Path(sys.executable).resolve()
    if current_python == VENV_PYTHON.resolve():
        return None

    if os.environ.get("RUN_ALL_IN_VENV") == "1":
        return None

    env = os.environ.copy()
    env["RUN_ALL_IN_VENV"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    print(f"[run_all] Switching to local venv interpreter: {VENV_PYTHON}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def run_step(step_name: str, cmd: list[str]) -> float:
    print("\n" + "=" * 72, flush=True)
    print(f"[run_all] {step_name}", flush=True)
    print(f"[run_all] Command: {format_command(cmd)}", flush=True)
    print("=" * 72, flush=True)

    started = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    elapsed = time.time() - started
    print(f"[run_all] Completed in {elapsed:.2f}s", flush=True)
    return elapsed


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python_cmd = [sys.executable]
    steps: list[tuple[str, list[str]]] = []

    if args.with_fx:
        steps.append(
            ("Extract ECB exchange rates", python_cmd + ["exchange_rates/extract_ecb_rates.py"])
        )
        steps.append(
            ("Parse and convert workbook monetary fields", python_cmd + ["check_del_after.py"])
        )

    steps.extend(
        [
            ("01 Audit raw workbook", python_cmd + ["01_audit.py"]),
            ("02 Clean dataset", python_cmd + ["02_clean.py"]),
            ("03 Build column mapping", python_cmd + ["03_build_column_mapping.py"]),
            ("04 Build ontology constraints", python_cmd + ["04_forbidden_edges.py"]),
            ("05 Run PC and LiNGAM baselines", python_cmd + ["05_run_baselines.py"]),
            ("06 Run NOTEARS", python_cmd + ["06_run_notears.py"]),
            (
                "07 Run DECI",
                python_cmd + ["07_run_deci.py", "--epochs", str(args.epochs), "--mode", args.mode],
            ),
        ]
    )

    if args.with_gemma:
        gemma_cmd = python_cmd + ["08_gemma_causal_proposals.py"]
        if args.gemma_backend:
            gemma_cmd.extend(["--backend", args.gemma_backend])
        if args.gemma_model:
            gemma_cmd.extend(["--model", args.gemma_model])
        if args.gemma_api_key:
            gemma_cmd.extend(["--api-key", args.gemma_api_key])
        steps.append(("08 Run Gemma causal proposals", gemma_cmd))

    steps.append(("09 Generate figures", python_cmd + ["09_visualize_graphs.py"]))

    if args.with_gemma and args.gemma_backend != "huggingface":
        gemma_eval_cmd = python_cmd + ["10_gemma_evaluate.py"]
        if args.gemma_backend:
            gemma_eval_cmd.extend(["--backend", args.gemma_backend])
        if args.gemma_model:
            gemma_eval_cmd.extend(["--model", args.gemma_model])
        if args.gemma_api_key:
            gemma_eval_cmd.extend(["--api-key", args.gemma_api_key])
        steps.append(("10 Run Gemma edge evaluation", gemma_eval_cmd))

    return steps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full causal discovery project from one file."
    )
    parser.add_argument(
        "--with-fx",
        action="store_true",
        help="Also extract ECB rates and build the EUR-converted helper workbook before the main pipeline.",
    )
    parser.add_argument(
        "--with-gemma",
        action="store_true",
        help="Also run the Gemma-backed steps (08_gemma_causal_proposals.py and 10_gemma_evaluate.py).",
    )
    parser.add_argument(
        "--gemma-backend",
        choices=["ollama", "google", "huggingface"],
        default=None,
        help="Backend for the Gemma-backed steps when --with-gemma is used.",
    )
    parser.add_argument(
        "--gemma-model",
        default=None,
        help="Optional model override for the Gemma-backed steps.",
    )
    parser.add_argument(
        "--gemma-api-key",
        default=None,
        help="Optional API key for the Google backend in the Gemma-backed steps.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="DECI epochs for 07_run_deci.py. Default: 200.",
    )
    parser.add_argument(
        "--mode",
        choices=["unconstrained", "constrained", "both"],
        default="both",
        help="DECI mode for 07_run_deci.py. Default: both.",
    )
    args = parser.parse_args()

    if args.gemma_backend and not args.with_gemma:
        parser.error("--gemma-backend requires --with-gemma")
    if args.gemma_model and not args.with_gemma:
        parser.error("--gemma-model requires --with-gemma")
    if args.gemma_api_key and not args.with_gemma:
        parser.error("--gemma-api-key requires --with-gemma")

    relaunch_code = relaunch_in_venv_if_available()
    if relaunch_code is not None:
        return relaunch_code

    if args.with_gemma and args.gemma_backend == "huggingface":
        print(
            "[run_all] Note: 10_gemma_evaluate.py supports only ollama and google; "
            "that step will be skipped for the huggingface backend.",
            flush=True,
        )

    print(f"[run_all] Project root: {ROOT}", flush=True)
    print(f"[run_all] Python: {sys.executable}", flush=True)
    print("[run_all] Default raw input: data/raw/df_asst_bnk_ecb.xlsx", flush=True)

    steps = build_steps(args)
    timings: list[tuple[str, float]] = []

    for step_name, cmd in steps:
        elapsed = run_step(step_name, cmd)
        timings.append((step_name, elapsed))

    total = sum(elapsed for _, elapsed in timings)
    print("\n" + "=" * 72, flush=True)
    print("[run_all] All requested steps completed.", flush=True)
    print(f"[run_all] Total runtime: {total:.2f}s", flush=True)
    print("[run_all] Timing summary:", flush=True)
    for step_name, elapsed in timings:
        print(f"  - {step_name}: {elapsed:.2f}s", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
