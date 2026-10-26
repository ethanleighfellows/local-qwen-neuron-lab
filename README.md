# Local Qwen Neuron Lab

Local Qwen3.5-4B text inference with bitsandbytes **NF4**, double quantization,
FP16 computation, direct MLP neuron interventions, SQLite profiles, a browser
chat UI, and an OpenAI chat-completions endpoint shared with Continue.

## Install and start (Windows)

Requires Python 3.12, a CUDA-capable NVIDIA GPU, and a recent NVIDIA driver.CUDA PyTorch is
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

Two ready-to-select presets ship with new installations:

| Profile | L14 neuron 8949 | L13 neuron 7956 |
| --- | --- | --- |
| Full dose | +1 | -1 |
| Half dose | +0.5 | -0.5 |

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
input of `layer.mlp.down_proj`. This changes activations on every processed
token, including prefill and single-token decoding; model parameters are never
edited. Quantized columns are dequantized using bitsandbytes' quantization
state before computing their L2 norm. Multiple neurons share one hook per layer.
Hooks are removed in `finally`, including failed requests.

Calibration runs without steering on three fixed text samples (see
`app/runtime.py`). Each layer's output residual L2 norm is collected at positions
`t >= 1`, excluding the first token of each sample. The median uses interpolated
0.5 quantile over pooled sample tokens. Cache identity includes the resolved
model commit, quantization, compute dtype, runtime versions, calibration text,
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

Continue reference: <https://docs.continue.dev/reference>.
Model architecture: <https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json>.
NF4 support: <https://huggingface.co/docs/bitsandbytes/main/installation>.

## API and tests

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.verify_model
.\.venv\Scripts\python.exe -m scripts.verify_api
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
