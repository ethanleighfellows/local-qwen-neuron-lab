$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv/Scripts/python.exe')) { throw 'Run install.ps1 first.' }
& ./.venv/Scripts/python.exe -m pip install 'https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36-cu125/llama_cpp_python-0.3.36-py3-none-win_amd64.whl' 'gguf==0.19.0'
if ($LASTEXITCODE) { throw 'llama.cpp dependencies failed to install.' }
& ./.venv/Scripts/python.exe -m scripts.prepare_llama_libraries
if ($LASTEXITCODE) { throw 'Compatible llama.cpp libraries could not be prepared.' }
& ./.venv/Scripts/python.exe -m scripts.prepare_llama_ui
if ($LASTEXITCODE) { throw 'Official llama.cpp web UI could not be prepared.' }
& ./.venv/Scripts/python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='unsloth/Qwen3.5-4B-GGUF', revision='e87f176479d0855a907a41277aca2f8ee7a09523', filename='Qwen3.5-4B-Q4_K_M.gguf', local_dir='data/gguf'))"
if ($LASTEXITCODE) { throw 'GGUF model download failed.' }

& ./.venv/Scripts/python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive', revision='c09cdbcdb1fefad6d335809d445621b5f5ba0c6e', filename='Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf', local_dir='data/gguf'))"
if ($LASTEXITCODE) { throw 'Unsteered GGUF model download failed.' }
