$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv/Scripts/python.exe')) { throw 'Run install.ps1 first.' }
$env:HF_HOME = Join-Path $PSScriptRoot 'data/hf'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
& ./.venv/Scripts/python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
