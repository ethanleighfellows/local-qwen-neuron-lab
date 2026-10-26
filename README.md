# Local Qwen Neuron Lab

Local Qwen3.5-4B text inference with a choice of llama.cpp **Q4_K_M** or
Transformers/bitsandbytes **NF4**. Both backends provide direct MLP neuron
interventions, SQLite profiles, a browser chat UI, and an OpenAI-compatible
endpoint shared with Continue.

## llama.cpp on Windows with NVIDIA CUDA

After the base environment has been installed with `install.ps1`, install and
start the llama.cpp backend:

```powershell
.\install-llama.cmd
.\start.cmd
```

The `.cmd` launchers apply a process-only PowerShell execution policy so the
commands also work when local PowerShell scripts are blocked.
Keep the server terminal open and use <http://127.0.0.1:8000> or the existing
Continue configuration. Stop any other server using port 8000 first.

The llama.cpp backend serves the official b11312 llama.cpp web interface at
that address. The **Steering** control in the lower left selects the shared
profile or opens the original profile editor. Profile changes also apply to
subsequent Continue requests. Your `L14 N8949 +0.4` profile contains only that
one neuron. The old chat interface and its saved conversations remain available
at <http://127.0.0.1:8000/steering>.

The llama.cpp interface saves its new conversations in the browser's IndexedDB;
they persist in that browser but are separate from the original SQLite chat
history. Streaming works, but reconnecting to an interrupted stream and the
llama.cpp server's built-in tools are not available through this adapter.
Temperature, top-p, top-k, min-p, typical-p, repeat penalty and token limit are
supported. Other llama.cpp sampling and reasoning controls are not implemented;
the server returns an error instead of silently applying an unsupported setting.

The installer downloads the Unsloth Qwen3.5-4B Q4_K_M GGUF (2.74 GB), the
official llama-cpp-python 0.3.36 CUDA 12.5 wheel, and the matching official
llama.cpp b11312 CPU archive. Its SHA-256 is checked before the AVX2 CPU
library is copied into a project-local DLL bundle. This combination supports
the RTX 2060 and this machine's AVX2 CPU; the original wheel's CPU DLL uses
unsupported AVX-512 instructions here. The existing CUDA PyTorch installation
supplies CUDA DLLs, but the llama.cpp server does not import PyTorch or run
the Transformers model. The GGUF model contains its own tokenizer and chat
template, so cached local inference needs no network access.
The verified release also supplies the web UI assets; the installer extracts
them with a temporary local server and then stops it.

The llama.cpp loader validates Qwen3.5 and Q4_K_M and uses CUDA offloading.
The GGUF file is about 2.55 GiB on disk; this machine's model load reported
approximately 2.54 GiB of CUDA model buffers plus about 497 MiB of mapped CPU
model data, before context and GPU workspace. Set `LLAMA_GGUF_PATH` to use a
different Qwen3.5 Q4_K_M file, or `LLAMA_CONTEXT_TOKENS` to adjust the 2048
token context limit. The server fails if it cannot load the CUDA build or the
requested quantization; it does not silently switch to NF4.

For existing installations, the same SQLite profiles and conversation history
are used. The two backends produce different output and use separate calibration
data because their quantized weights differ. `start.ps1` continues to select
the original Transformers backend; select llama.cpp with `start.cmd` or `start-llama.cmd`.
Only the Windows NVIDIA build is currently configured for llama.cpp. Its CPU
backend requires AVX2 and Haswell-compatible CPU instructions.

## Transformers backend (Windows)

Requires Python 3.12, a CUDA-capable NVIDIA GPU, and a recent NVIDIA driver.
This machine has Windows 11, RTX 2060 6 GB, and 16 GB RAM. CUDA PyTorch is
installed into `.venv`; system Python packages are unaffected.

```powershell
.\install.ps1
.\start.ps1
```

Open <http://127.0.0.1:8000>. The first load downloads about 9 GB of source
weights/tokenizer into `data/models`, quantizes on load, and calibrates.
Only the text tower is instantiated; image/video inputs are unsupported.
The model does not call external inference services. Downloads require internet;
after caching, set `HF_HUB_OFFLINE=1` for offline restarts.

On Linux with NVIDIA CUDA:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
bash start.sh
```

macOS/CPU-only machines are not supported by this selected backend. It fails
explicitly instead of substituting another model or unquantized weights.
CUDA 12.6 is selected for Turing compatibility. No FlashAttention, Triton,
causal-conv1d, or mamba-ssm installation is required; Transformers uses its
portable Gated DeltaNet implementation, which can be slow.

The packed text linear weights need roughly 2 GB plus about 1.2 GB for tied
FP16 embeddings, quantization metadata, recurrent/KV state, activations and CUDA
workspace. Budget about 4–5 GB VRAM for short contexts; actual measured model
footprint is exposed at `/api/status`. A 6 GB display GPU has limited headroom.
Close GPU-heavy applications on OOM and reduce context/output limits. Source
weights remain full precision on disk; runtime linear weights are real NF4.
No automatic CPU offloading or silent precision fallback is used.

`MAX_INPUT_TOKENS` defaults to 2048; output is limited to 2048 tokens per request
(256 by default). Raising limits can exhaust VRAM. `MODEL_REVISION` defaults
to `main`; use the resolved commit printed in logs to pin subsequent installs.
`DATA_DIR` selects the SQLite/calibration/model-cache directory.

## Steering

Ready-to-select presets ship with new installations:

| Profile | L14 neuron 8949 | L13 neuron 7956 |
| --- | --- | --- |
| Full dose | +1 | -1 |
| Half dose | +0.5 | -0.5 |
| L14 N8949 +0.4 | +0.4 | off |

Select a preset in the steering panel to make it the server's global default
for web chat and Continue. Baseline remains the initial selection on a fresh
installation. Each preset can be edited or duplicated like any other profile.

Create an enabled profile and add one or more codes:

```text
L15 00150 +1
L15 00420 -1
L22 01234 +0.5
```

Indices are zero-based and checked against the loaded model. Each code has an
individual enable toggle and editable signed dose. Zero is a no-op; duplicate
layer/neuron entries are rejected. Create, rename, duplicate and delete profiles
in the panel. The global switch overrides every profile, including conversation
overrides. Baseline is an ordinary disabled profile.

The engine adds `dose * median_residual_norm / down_weight_column_norm` to the
input of the layer's MLP down projection. This changes activations on every
processed token, including prefill and single-token decoding; model parameters
are never edited. Transformers dequantizes the column with bitsandbytes. The
llama.cpp backend dequantizes the relevant GGUF quantization block and applies
offsets through a GGML evaluation callback at the `ffn_swiglu` node immediately
before the down projection. Multiple neurons can be active in a layer.

Calibration runs without steering on three fixed text samples (see
`app/model_config.py`). Each layer's output residual L2 norm is collected at positions
`t >= 1`, excluding the first token of each sample. The median uses interpolated
0.5 quantile over pooled sample tokens. Cache identity includes the resolved
model revision or GGUF SHA-256, quantization, runtime versions, calibration text,
token limit, and method. Calibration statistics, effective shifts, and measured
input differences from the latest inference appear in the inspection panel.

Profiles and chats live in `data/state.sqlite3`. The server restores the active
profile on restart. All clients inherit it unless explicitly selecting a
conversation override or the API's `steering_profile` field. Profile edits are
polled across browser windows every 2.5 seconds. Version checks reject stale
edits. Requests capture a detached profile snapshot at arrival and serialize
GPU execution behind a lock. Profile edits while a request is queued do not
change that request. A disconnected client can leave its bounded generation
running to completion; hooks remain protected until cleanup.

Compare mode sends the same history/settings once with baseline and once with
the selected steering. Set temperature to 0 for deterministic comparisons.

## Continue in VS Code

The provided [`continue/config.yaml`](continue/config.yaml) uses
`http://127.0.0.1:8000/v1`, model `Qwen/Qwen3.5-4B`, and enables `tool_use`.
On this machine Continue is already installed. Its empty configuration is backed
up as `%USERPROFILE%\.continue\config.before-neuron-lab.yaml` and replaced with
the provided local configuration. Reload the VS Code window and select
**Qwen3.5-4B Local Steered** in Continue. For another machine merge this model
entry into `~/.continue/config.yaml` (Windows: `%USERPROFILE%\.continue\config.yaml`).

Select Agent mode, allow its normal read/search tools, and keep confirmation
enabled for edits and terminal commands. Ask it to read a small file and explain
its contents, then make a small approved edit. The model server never executes
tools itself; Continue owns workspace permissions and execution.

The server converts Qwen's native XML or JSON tool calls into OpenAI function
calls, including IDs and JSON arguments. Incoming tool results must reference
a prior call ID. Tool-enabled streams buffer generation until native tool XML
can be parsed safely; ordinary chat streams incrementally. Only `auto` and
`none` tool choice are supported. Nonzero frequency/presence penalties, images,
structured response schemas and embedding endpoints are unsupported. This is
a focused chat-completions API, not the entire OpenAI API.

A 4B model and 2048-token input budget limit agent reliability. Large tool lists,
workspace dumps and long tasks can exceed the context budget. A protocol test
does not establish model tool reliability or a successful Continue UI run;
see `VERIFICATION.md` for the actual acceptance evidence and outstanding checks.
In a test of this Q4_K_M model, the Baseline profile completed a tool call and
continuation, while the Full dose profile produced malformed tool markup and
Half dose answered without calling the tool. Steering can therefore interfere
with Agent mode. The server reports malformed tool markup rather than silently
disabling the selected profile. Select Baseline or a different dose if reliable
tool calling is more important for a task.

Continue reference: <https://docs.continue.dev/reference>.
Model architecture: <https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json>.
NF4 support: <https://huggingface.co/docs/bitsandbytes/main/installation>.

## API and tests

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.verify_model
.\.venv\Scripts\python.exe -m scripts.verify_api
.\.venv\Scripts\python.exe -m scripts.verify_llama
```

`POST /v1/chat/completions` supports standard messages, tools, streaming,
temperature, top_p, max_tokens/max_completion_tokens, seed and stop strings.
Optional `baseline: true` bypasses steering for that request.
`GET/PUT /api/profiles` reads/replaces the versioned profile configuration.
`GET/POST /api/chats` lists/saves conversations.
`GET /api/status` exposes model, calibration and latest intervention statistics.

Hardware verification writes `data/acceptance.json`. Unit/API tests use a tiny
real PyTorch MLP and an explicitly isolated fake inference runtime for protocol
tests; the production app always uses the actual checkpoint.

`requirements-lock.txt` records the versions verified on this Windows machine.
The CUDA PyTorch wheel uses the CUDA index installed first by `install.ps1`.
Live API verification writes `data/api-acceptance.json`. The two-phase
`scripts.verify_persistence prepare` / server restart /
`scripts.verify_persistence finish` checks persistence and restores the original
profile. `tests/vscode/run.cjs` is an optional VS Code extension-host test using
installed Continue's own model client and read-file implementation, restricted
to `tests/vscode/fixture.txt`; it does not change normal tool approvals.

Full plan acceptance is still pending the Continue extension-host check and
rendered browser QA described in `VERIFICATION.md`.

Keep the server bound to loopback. It has no authentication and is intended for
a single local user. Browser cross-origin writes are rejected. SQLite content
and logs may contain conversation/configuration information.

## Two llama.cpp models

The llama.cpp launcher serves both the original Qwen3.5-4B Q4_K_M model and
[HauhauCS Qwen3.5-4B Uncensored Aggressive Q4_K_M](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive).
The installer downloads both GGUF files into `data/gguf`; they are not committed to Git.
Start `start.cmd`, open http://127.0.0.1:8000, and use the **Steering** button
at the bottom left to select a model. The server loads one model at a time;
the first response after switching can take longer while the model loads.
The Uncensored model is always unsteered, regardless of the selected profile.
The original model retains its steering profile.

Continue has separate model entries in `continue/config.yaml`; copy or merge them
into your Continue configuration and select the desired model in Continue. Both
entries use the same local server and can request different models independently.
