$ErrorActionPreference = "Stop"

$pythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

& $pythonExe 01_audit.py
& $pythonExe 02_clean.py
& $pythonExe 03_build_column_mapping.py
& $pythonExe 04_forbidden_edges.py
& $pythonExe 05_run_baselines.py
& $pythonExe 06_run_notears.py
& $pythonExe 07_run_deci.py --epochs 200 --mode both
& $pythonExe 09_visualize_graphs.py
