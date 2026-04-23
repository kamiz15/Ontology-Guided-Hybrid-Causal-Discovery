$ErrorActionPreference = "Stop"

$venvPath = ".venv"
$pythonExe = Join-Path $venvPath "Scripts\\python.exe"

if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Local environment ready."
Write-Host "Activate it with:"
Write-Host "  .\\.venv\\Scripts\\Activate.ps1"
