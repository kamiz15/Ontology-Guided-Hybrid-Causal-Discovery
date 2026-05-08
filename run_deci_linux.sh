#!/usr/bin/env bash
set -euo pipefail

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "Python:"
python --version

echo "Checking torch/causica:"
python - <<'PY'
import sys
print("Python executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", repr(e))

try:
    import causica
    print("causica import ok")
except Exception as e:
    print("causica import failed:", repr(e))
PY

echo "Running DECI ablation:"
python run_all.py --only-deci --deci-ablation --dataset synthetic

echo "Running full experiment:"
python run_all.py
