$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3.12 -m venv .venv
if ($LASTEXITCODE) { throw 'Python 3.12 environment creation failed' }
& ./.venv/Scripts/python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE) { throw 'CUDA PyTorch install failed' }
& ./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
if ($LASTEXITCODE) { throw 'Dependency install failed' }
