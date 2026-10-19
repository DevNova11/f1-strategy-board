# F1 Strategy Board - set up and launch on Windows.
#   .\run.ps1            create the venv if needed, install, launch the dashboard
#   .\run.ps1 -Fetch     also download the real data first (needs network)
#   .\run.ps1 -Test      run the test suite instead of launching
param(
    [switch]$Fetch,
    [switch]$Test,
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv .venv } else { python -m venv .venv }
    & $python -m pip install --upgrade pip --quiet
    & $python -m pip install -r requirements.txt
}
if ($Fetch) { & $python -m src.fetch }
if ($Test) { & $python -m pytest; exit $LASTEXITCODE }
& $python -m streamlit run app/Home.py --server.port $Port
