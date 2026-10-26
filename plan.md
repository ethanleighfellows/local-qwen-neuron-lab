# Project: Local Qwen3.5-4B Chatbot with Persistent Neuron Steering

## Objective

Build and install a locally hosted Qwen3.5-4B chatbot, quantized to 4-bit, with support for direct MLP neuron activation steering.

I want to use this model in two environments:

1. **Web chat interface:** An interactive chatbot with persistent conversations and an interface for configuring neuron steering.
2. **VS Code coding agent:** Integration with Cline or Continue, allowing the model to act as a coding assistant with file editing, terminal access, and tool calling.

**Critical requirement:** Neuron steering must persist across conversations, chat windows, and VS Code sessions. Steering should operate at the model-inference level, not through system prompts.

## 1. Model Installation and Runtime

- Model: `Qwen/Qwen3.5-4B`
- Run locally with no reliance on external inference APIs.
- Target 4-bit weight quantization.
- Detect available hardware and configure the most practical backend.
- Prefer Hugging Face Transformers, PyTorch, and bitsandbytes NF4 on supported NVIDIA GPUs.
- If using GGUF Q4_K_M instead, recognize that standard Ollama/llama.cpp interfaces do not expose the required PyTorch activation hooks. A compatible internal intervention mechanism must be implemented and tested.
- Do not silently substitute an unquantized model or remove steering functionality.
- Document the actual quantization format and memory requirements.

Use a recent compatible Transformers version supporting Qwen3.5's hybrid Gated DeltaNet architecture.

## 2. Neuron Steering Engine

Implement a system that accepts steering codes using the following syntax:

```text
L15 00150 +1
L15 00420 -1
L22 01234 +0.5
```

Interpretation:

- `L15`: Transformer layer 15, zero-indexed.
- `00150`: MLP intermediate neuron index 150, zero-indexed.
- `+1`: Positive steering dose.
- `-1`: Negative steering dose.

Multiple steering codes must be active simultaneously, including multiple neurons in the same layer.

### Intervention formula

Implement normalized activation steering using:

\[
\Delta a_{L,n} = d \times
\frac{\operatorname{median}_{t\ge1}(\|r_{L,t}\|_2)}
{\|W_{\mathrm{down},L}[:,n]\|_2}
\]

Where:

- `d` is the signed user-specified dose.
- `r` is the residual stream leaving transformer layer L.
- `t >= 1` excludes the first token for calibration.
- `W_down[:, n]` is the down-projection weight column corresponding to neuron n.

Apply the calculated activation offset **to the input of the specified layer's MLP `down_proj`**, at every processed token, including prefill and autoregressive decoding.

Requirements:

- Do not modify model weights permanently.
- Use unsteered calibration to establish residual norms.
- Calculate correct column norms for quantized weights, dequantizing as needed.
- Cache calibration data using model revision, quantization settings, and calibration configuration as identifiers.
- Validate neuron indices against the actual model architecture.
- Permit positive, negative, fractional, and zero doses.
- Support enabling/disabling individual steering codes.
- Avoid accumulating hooks or applying duplicate interventions.
- Expose effective activation shifts and calibration statistics for inspection.

For reference, Qwen3.5-4B has 32 transformer layers and an MLP intermediate dimension of 9,216.

## 3. Persistent Steering Profiles

Create a persistent configuration system that survives application restarts.

Example:

```json
{
  "active_profile": "Experiment A",
  "profiles": {
    "Experiment A": {
      "enabled": true,
      "steering": [
        {"layer": 15, "neuron": 150, "dose": 1.0},
        {"layer": 15, "neuron": 420, "dose": -1.0}
      ]
    },
    "Baseline": {
      "enabled": false,
      "steering": []
    }
  }
}
```

Required functionality:

- Create, rename, duplicate, and delete profiles.
- Save profiles to disk, preferably SQLite.
- Automatically restore the last active profile when the server starts.
- Apply the active profile to all new conversations by default.
- Allow profile changes without restarting the model.
- Synchronize changes across open chat windows.
- Provide a global steering-enabled toggle and a baseline mode.
- Allow optional per-conversation overrides while retaining a global default.

**Important:** Steering persistence must be enforced by the inference server, not depend on remembered chat instructions or frontend session state.

For concurrent requests, capture an immutable snapshot of the active profile at request start. Prevent one request's steering configuration from leaking into another.

## 4. Web Chat Interface

Build a usable local chat interface with:

- Persistent conversation history.
- Multiple conversations and the ability to start new chats.
- Streaming token generation.
- Adjustable temperature, token limits, and generation settings.
- A dedicated neuron steering panel.
- A list of active neuron codes and their doses.
- Add/remove/edit controls and an enable/disable toggle.
- A profile selector.
- A clear indication of the active profile and whether steering is enabled.
- A baseline-versus-steered comparison mode.

Changes made in the steering panel must persist and affect subsequent inference requests, including those from VS Code.

## 5. VS Code Coding Agent Integration

Expose the model through an OpenAI-compatible local API.

Required endpoints include:

```text
GET  /v1/models
POST /v1/chat/completions
```

Support streaming responses and OpenAI-compatible tool calling, including tool-call IDs, arguments, and tool-result messages.

Integrate with **Cline or Continue in VS Code**, using the locally hosted endpoint.

The agent should be able to:

- Read and inspect project files.
- Propose and apply code changes.
- Search the workspace.
- Execute approved terminal commands.
- Maintain context across multi-step coding tasks.

Preserve the extension's normal permissions and approval mechanisms for file changes and command execution.

The VS Code integration must use the same model server and global steering profile as the web chat interface.

Do not implement separate steering configurations for the two clients unless explicitly selected as an override.

If native tool calling is unreliable with this 4B model, investigate and document the limitation rather than claiming that ordinary chat completions constitute functioning agent mode.

## 6. Backend Architecture

Suggested architecture:

```text
           LOCAL QWEN3.5-4B
           4-BIT MODEL
                  |
          STEERING ENGINE
          MLP ACTIVATION HOOKS
                  |
         PERSISTENT PROFILE STORE
                  |
             FASTAPI SERVER
                  |
        OPENAI-COMPATIBLE API
                  |
          +-------+-------+
          |               |
      WEB CHAT        VS CODE
      INTERFACE       CLINE/CONTINUE
```

All inference requests must pass through the steering engine.

Keep the backend modular enough to swap inference runtimes later without rewriting the interface or persistence system.

## 7. Testing and Acceptance Criteria

Do not consider the implementation complete until the following have been demonstrated:

1. Qwen3.5-4B successfully loads locally in genuine 4-bit quantization.
2. Ordinary unsteered chat inference works.
3. Applying `L15 00150 +1` produces a measurable activation change at the intended MLP neuron.
4. Negative doses work and multiple interventions can run simultaneously.
5. The intervention is applied during both prompt processing and generation.
6. Disabling steering restores baseline inference behavior under deterministic generation settings.
7. Profiles persist after server restart.
8. Opening a new conversation retains the active steering profile.
9. Changes from the web interface are reflected in subsequent VS Code requests.
10. A VS Code agent can successfully execute at least one tool-calling cycle, including receiving a tool result and continuing.
11. Concurrent inference requests do not contaminate each other's steering settings.
12. The application logs model version, quantization type, active profile, calibration method, and effective steering doses.

Provide automated tests for code parsing, profile persistence, activation hooks, and API compatibility.

## 8. Deliverables

Produce:

- A working local inference backend.
- A functional browser chat UI.
- A persistent steering configuration system.
- A local OpenAI-compatible API server.
- Working VS Code integration instructions and configuration.
- A `README.md` containing platform-specific installation instructions.
- Startup scripts for launching the entire application.
- A test suite and a brief summary of verified functionality.

Before implementation, inspect the machine's operating system, GPU, VRAM, and installed Python environment. Choose an appropriate quantization and execution backend based on the available hardware.

Prioritize **correct neuron intervention and persistent steering** over inference speed, elaborate interface design, or secondary features.

Build a functioning end-to-end implementation, not a prototype that merely simulates neuron steering.