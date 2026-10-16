$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
}
& ./.venv/Scripts/python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch install failed.' }
& ./.venv/Scripts/python.exe -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Project install failed.' }
& ./.venv/Scripts/python.exe -m pip install -r requirements-windows.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency pinning failed.' }
& ./.venv/Scripts/python.exe -m koikoi.upstream --fetch
if ($LASTEXITCODE -ne 0) { throw 'Pinned upstream setup failed.' }
& ./.venv/Scripts/python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
