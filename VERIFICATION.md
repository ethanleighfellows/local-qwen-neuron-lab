# Verification on this machine

Windows 11, NVIDIA RTX 2060 (6144 MiB VRAM), 16 GB system RAM, Python 3.12.
Model commit: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Transformers 5.21.0, PyTorch 2.14.1+cu126, bitsandbytes 0.50.2.
Loaded model footprint: **3,057,867,008 bytes (2.85 GiB)**, excluding generation
workspace and CUDA allocator overhead. Native Windows reference DeltaNet kernels
are used. Runtime is genuine NF4 with double quantization and FP16 compute.

| Plan acceptance criterion | Evidence / result |
|---|---|
| 1. Local genuine 4-bit model loading | Passed; all layer MLP down projections asserted `Linear4bit` after loading the actual checkpoint. |
| 2. Unsteered chat | Passed; deterministic response `hello`. |
| 3. `L15 00150 +1` changes the specified activation | Passed; calculated shift 26.9730038, measured prefill mean 26.9713535, decode 26.96875 (FP16 rounding). |
| 4. Negative and simultaneous interventions | Passed; neuron 150 at -1 and neuron 420 at +0.5 both measured with the expected signs. |
| 5. Prefill and generation | Passed; observations include 18-token prefill and single-token decoding. |
| 6. Disabled steering restores deterministic baseline | Passed; baseline responses match after removing hooks; global/profile/per-code disabling also covered by unit/API tests. |
| 7. Restart persistence | Passed across actual server process termination/restart with an active enabled profile. |
| 8. New conversations inherit profile | Passed; fresh persistent conversation and subsequent requests inherited the saved global profile. |
| 9. Web settings affect OpenAI clients | Passed at the backend boundary: the UI's profile API updated server state, and a subsequent OpenAI request inherited its code without steering fields. Browser interaction itself was not exercised. |
| 10. VS Code agent tool cycle | **Pending.** Continue is configured; the actual local model/API tool cycle passed, but the isolated Continue extension host could not start while VS Code's `vscode-updating` mutex was held. No successful VS Code tool cycle is claimed. |
| 11. Concurrent request isolation | Passed with real GPU baseline and steered requests and separate API snapshot tests. GPU inference is serialized. |
| 12. Logging | Model revision, quantization, calibration method/statistics, startup profile, request codes and effective shifts are logged. |

Six automated tests cover bundled presets, parsing, persistence, snapshot isolation, real PyTorch
activation hooks, repeated hook installation, signed/zero/disabled doses, API
streaming, native tool syntax, matching tool results, and concurrent API clients.
`pip check`, Python compilation and JavaScript syntax checks passed.

The live API test completed a genuine model → `read_file` → model continuation,
using only this project's README. Tool calls had IDs, JSON arguments and matching
tool-result messages. This verifies protocol behavior and this simple model
task; it does not establish general 4B agent reliability.

Machine-readable evidence is retained in:

- `data/acceptance.json`: real model activation and deterministic inference checks.
- `data/api-acceptance.json`: live API responses and tool-result continuation.
- `data/persistence-acceptance.json`: actual process-restart checks.
- `data/hardware-test.log`: model-loading and hardware test output.
- `data/server.stderr.log`: model metadata and inference logs.
- `data/vscode-test.stdout.log`: attempted extension-host launch and updater mutex.

The browser UI returned HTTP 200 and its JavaScript parsed successfully. No
browser or native-app control surface was available for rendered UI QA.

## Finish the outstanding Continue acceptance check

After VS Code finishes updating, reload its window, open Continue, select
**Qwen3.5-4B Local Steered**, and enter Agent mode. Ask it to read
`tests/vscode/fixture.txt` with `read_file`, then report the marker. Confirm the
tool result appears and the model continues. Keep existing edit/terminal
permissions and approvals enabled.

The optional automated equivalent is `tests/vscode/run.cjs`, run with VS Code's
`--extensionTestsPath` and `--extensionDevelopmentPath=tests/vscode` options,
`NODE_ENV=test`, and the installed Continue extension enabled. It uses Continue's
actual model client and read-only workspace tool, then writes
`data/continue-acceptance.json`. It reads only the fixed test fixture and does
not execute terminal commands or edit files through Continue.

Until that succeeds, the implementation is installed and the backend is
verified, but the plan's full end-to-end acceptance is not complete.
