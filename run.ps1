$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvPy = Join-Path $root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $venvPy)) {
    Write-Host 'No virtual environment found. Run .\setup.ps1 first.' -ForegroundColor Red
    exit 1
}

# pythonw keeps the console from flashing up behind the GUI. Swap to
# python.exe if you want to watch stdout while debugging.
& $venvPy -m talkbox @args
