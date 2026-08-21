# Talkbox setup. Creates a virtual environment, installs CUDA-enabled
# PyTorch plus the rest of the stack, and records the XTTS licence agreement.
#
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host ''
Write-Host 'Talkbox setup' -ForegroundColor Cyan
Write-Host '===============' -ForegroundColor Cyan
Write-Host ''

# --- python ----------------------------------------------------------------

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-Host 'Python was not found on PATH. Install Python 3.10-3.12 first.' -ForegroundColor Red
    exit 1
}
$version = & $python -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "Using Python $version at $python"
if ([version]$version -lt [version]'3.10' -or [version]$version -ge [version]'3.13') {
    Write-Host "Python $version is outside the supported 3.10-3.12 range." -ForegroundColor Yellow
    Write-Host 'coqui-tts may fail to install. Continuing anyway.' -ForegroundColor Yellow
}

# --- venv ------------------------------------------------------------------

if (-not (Test-Path '.venv')) {
    Write-Host 'Creating virtual environment in .venv ...'
    & $python -m venv .venv
}
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    Write-Host 'Virtual environment creation failed.' -ForegroundColor Red
    exit 1
}

& $venvPy -m pip install --upgrade pip setuptools wheel

# --- torch with CUDA -------------------------------------------------------

Write-Host ''
Write-Host 'Installing PyTorch with CUDA 12.4 support (~2.5 GB) ...' -ForegroundColor Cyan
& $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host 'CUDA PyTorch install failed.' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'Installing the remaining dependencies ...' -ForegroundColor Cyan
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Dependency install failed.' -ForegroundColor Red
    exit 1
}

# --- model licence ---------------------------------------------------------

Write-Host ''
Write-Host 'XTTS-v2 model licence' -ForegroundColor Cyan
Write-Host '---------------------'
Write-Host 'The voice cloning model is Coqui XTTS-v2, released under the Coqui'
Write-Host 'Public Model License 1.0.0. It permits personal and research use and'
Write-Host 'prohibits commercial use. Full text:'
Write-Host '  https://coqui.ai/cpml' -ForegroundColor Blue
Write-Host ''
Write-Host 'You are also responsible for only cloning voices you have the right'
Write-Host 'to clone - your own, or someone who has agreed to it.'
Write-Host ''
$answer = Read-Host 'Type "agree" to accept and continue'
if ($answer.Trim().ToLower() -ne 'agree') {
    Write-Host 'Not accepted. The app will not be able to synthesise speech.' -ForegroundColor Yellow
    exit 1
}

$configPath = Join-Path $root 'config.json'
if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw | ConvertFrom-Json
} else {
    $config = [pscustomobject]@{}
}
$config | Add-Member -NotePropertyName 'coqui_tos_agreed' -NotePropertyValue $true -Force
$config | ConvertTo-Json -Depth 5 | Out-File -FilePath $configPath -Encoding utf8

# --- virtual cable notice --------------------------------------------------

Write-Host ''
& $venvPy -m talkbox --doctor

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'Launch with:  .\run.ps1' -ForegroundColor Green
Write-Host ''
Write-Host 'If you want the translated voice to appear as a microphone in Discord,'
Write-Host 'Zoom or Teams, install VB-Audio Virtual Cable (free) and reboot:'
Write-Host '  https://vb-audio.com/Cable/' -ForegroundColor Blue
