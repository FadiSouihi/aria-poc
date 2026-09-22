# ARIA-POC one-shot setup for Windows.
#
#   .\setup.ps1                 # venv + deps + CUDA torch + models + selfcheck
#   .\setup.ps1 -CpuOnly        # no NVIDIA GPU (vision/STT will be slow)
#   .\setup.ps1 -SkipModels     # code only, fetch the 1.7 GB later
#   .\setup.ps1 -Dev            # + pytest and optional extras
#   .\setup.ps1 -Python "py -3.12"
#
# Every step is idempotent: re-run it any time.
[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$SkipModels,
    [switch]$Dev,
    [string]$Python = "py",
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)
Write-Host "== ARIA-POC setup ==" -ForegroundColor Cyan

# --- interpreter ------------------------------------------------------------
$pyArgs = @()
if ($Python -eq "py") { $pyArgs = @("-3.12") }
try {
    $version = & $Python @pyArgs -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
} catch {
    Write-Error "Python not found ('$Python'). Install Python 3.12 and re-run, or pass -Python <path>."
}
if ([version]$version -lt [version]"3.10") {
    Write-Error "Python $version is too old; 3.10-3.12 required."
}
Write-Host "-- python $version"

# --- virtualenv -------------------------------------------------------------
if (-not (Test-Path $Venv)) {
    Write-Host "-- creating virtualenv ($Venv)"
    & $Python @pyArgs -m venv $Venv
}
$vpy = Join-Path $Venv "Scripts\python.exe"
& $vpy -m pip install --upgrade pip wheel | Out-Null

# --- dependencies -----------------------------------------------------------
Write-Host "-- runtime dependencies"
& $vpy -m pip install -r requirements.txt
if ($Dev) { & $vpy -m pip install -r requirements-dev.txt }

if ($CpuOnly) {
    Write-Host "-- PyTorch (CPU wheels)"
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    Write-Host "-- PyTorch (CUDA 12.6 wheels)"
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
}

# --- models -----------------------------------------------------------------
if ($SkipModels) {
    Write-Host "-- skipping model download (run: $vpy tools\fetch_models.py)"
} else {
    Write-Host "-- model weights (1.7 GB, hash-verified)"
    & $vpy tools\fetch_models.py
}

# --- verify -----------------------------------------------------------------
Write-Host "-- quick self-check"
& $vpy tools\selfcheck.py

@"

Done. Next:
  $vpy main.py --config configs\laptop.yaml          # camera + mic + HUD + voice
  $vpy main.py --config configs\audio_only.yaml      # conversation only
  $vpy tools\selfcheck.py --full                     # full verification
  $vpy -m pytest -q tests                            # 137 tests

Talk to it: the debug window shows who is tracked; press f to flip the view, q to quit.
"@ | Write-Host