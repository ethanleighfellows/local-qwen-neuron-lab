"""llama.cpp Qwen3.5 runtime with real GGML activation interventions.

The callback intercepts the pinned build's ffn_swiglu layer outputs, which are
the inputs of the FFN down projection. All requests are serialized because the
callback and llama context belong to one model instance.
"""
import copy
import ctypes
import hashlib
import importlib.util
import json
import logging
import os
import queue
import re
import sys
import threading
from pathlib import Path

import numpy as np
from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from .model_config import CALIBRATION, MODEL, UNSTEERED_MODEL, UNSTEERED_FILENAME
from .steering import Code

LLAMA_VERSION = "0.3.36"
MODEL_FILENAME = "Qwen3.5-4B-Q4_K_M.gguf"
CALLBACK_NAMES = re.compile(r"^(ffn_swiglu|l_out)-(\d+)$")


class GGMLIntervention:
    def __init__(self, llama_cpp, width, hidden, layer_count):
        self.width, self.hidden, self.layer_count = width, hidden, layer_count
        dll = Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "ggml-base.dll"
        if os.name != "nt":
            raise RuntimeError("This prebuilt llama.cpp backend currently supports Windows only")
        self.lib = ctypes.CDLL(str(dll))
        self.lib.ggml_get_name.argtypes = [ctypes.c_void_p]
        self.lib.ggml_get_name.restype = ctypes.c_char_p
        self.lib.ggml_nelements.argtypes = [ctypes.c_void_p]
        self.lib.ggml_nelements.restype = ctypes.c_int64
        self.lib.ggml_nbytes.argtypes = [ctypes.c_void_p]
        self.lib.ggml_nbytes.restype = ctypes.c_size_t
        self.lib.ggml_is_contiguous.argtypes = [ctypes.c_void_p]
        self.lib.ggml_is_contiguous.restype = ctypes.c_bool
        for name in ("ggml_backend_tensor_get", "ggml_backend_tensor_set"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            fn.restype = None
        self.callback = llama_cpp.ggml_backend_sched_eval_callback(self._callback)
        self.calibrating = False
        self.residuals = [[] for _ in range(layer_count)]
        self.shifts = {}
        self.observations = []
        self.error = None

    def _callback(self, ptr, ask, _user_data):
        try:
            name = self.lib.ggml_get_name(ptr).decode("utf-8")
            match = CALLBACK_NAMES.fullmatch(name)
            if not match:
                return False if ask else True
            kind, layer = match[1], int(match[2])
            wanted = (self.calibrating and kind == "l_out") or (kind == "ffn_swiglu" and layer in self.shifts)
            if ask or not wanted:
                return bool(wanted) if ask else True
            if layer >= self.layer_count:
                raise RuntimeError(f"Unexpected GGML layer {layer}")
            width = self.hidden if kind == "l_out" else self.width
            count = self.lib.ggml_nelements(ptr)
            # ggml_tensor begins with enum ggml_type. Refuse unsupported layouts.
            typ = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0]
            if typ != 0 or count % width or self.lib.ggml_nbytes(ptr) != count * 4 or not self.lib.ggml_is_contiguous(ptr):
                raise RuntimeError(f"Unexpected GGML tensor layout for {name}: type={typ}, elements={count}")
            data = np.empty(count, dtype=np.float32)
            self.lib.ggml_backend_tensor_get(ptr, data.ctypes.data, 0, data.nbytes)
            data = data.reshape(-1, width)
            if kind == "l_out":
                self.residuals[layer].extend(np.linalg.norm(data[1:].astype(np.float64), axis=1).tolist())
            else:
                for neuron, shift in self.shifts[layer]:
                    before = data[:, neuron].copy()
                    data[:, neuron] += shift
                    measured = float(np.mean(data[:, neuron] - before, dtype=np.float64))
                    self.observations.append({"layer": layer, "neuron": neuron, "tokens": len(data),
                        "requested_shift": shift, "measured_shift": measured})
                self.lib.ggml_backend_tensor_set(ptr, data.ctypes.data, 0, data.nbytes)
            return True
        except BaseException as error:
            self.error = error
            return False

    def check(self):
        if self.error is not None:
            raise RuntimeError("GGML intervention failed") from self.error


class LlamaRuntime:
    def __init__(self, data="data", model_id=MODEL):
        self.model_id = model_id
        self.data = Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.ready = False

    def load(self):
        import gguf
        if os.name != "nt":
            raise RuntimeError("The provided llama.cpp CUDA build currently supports Windows only")
        if not ctypes.windll.kernel32.IsProcessorFeaturePresent(40):  # PF_AVX2_INSTRUCTIONS_AVAILABLE
            raise RuntimeError("The bundled llama.cpp CPU backend requires an AVX2-capable processor")
        if "torch" in sys.modules:
            raise RuntimeError("Start the llama.cpp backend in a fresh process without importing PyTorch; the OpenMP runtimes conflict")
        libs = self.data.resolve() / "llama-libs-b11312-cu125-avx2"
        if not (libs / "llama.dll").is_file():
            raise RuntimeError("Compatible llama.cpp libraries missing; run install-llama.ps1")
        os.environ["LLAMA_CPP_LIB_PATH"] = str(libs)
        # The CUDA wheel uses CUDA runtime DLLs bundled with the installed
        # PyTorch wheel. Locate them without importing torch: its Intel OpenMP
        # runtime conflicts with llama.cpp's official LLVM OpenMP build.
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None or torch_spec.origin is None:
            raise RuntimeError("CUDA runtime DLLs missing; run install.ps1 first")
        cuda_dlls = str(Path(torch_spec.origin).parent / "lib")
        self._cuda_dll_directory = os.add_dll_directory(cuda_dlls)
        # The wheel loads DLLs with legacy Win32 search semantics, which use
        # PATH in addition to AddDllDirectory.
        os.environ["PATH"] = cuda_dlls + os.pathsep + os.environ.get("PATH", "")
        import llama_cpp
        from llama_cpp import llama_cpp as native

        if llama_cpp.__version__ != LLAMA_VERSION:
            raise RuntimeError(f"Expected llama-cpp-python {LLAMA_VERSION}; install the pinned CUDA wheel")
        if not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError("llama.cpp CUDA backend is unavailable; no silent CPU fallback")
        path = Path(os.environ.get("LLAMA_GGUF_PATH", str(self.data / "gguf" / MODEL_FILENAME)) if self.model_id == MODEL else self.data / "gguf" / UNSTEERED_FILENAME).resolve()
        if not path.is_file():
            raise RuntimeError(f"GGUF model missing: {path}; run install-llama.ps1")
        self.reader = gguf.GGUFReader(path)
        def field(name):
            item = self.reader.get_field(name)
            if item is None:
                raise ValueError(f"GGUF field missing: {name}")
            return item.contents()
        if field("general.architecture") != "qwen35" or field("general.file_type") != int(gguf.LlamaFileType.MOSTLY_Q4_K_M):
            raise ValueError("Expected a Qwen3.5 Q4_K_M GGUF; no unquantized fallback")
        self.width = int(field("qwen35.feed_forward_length"))
        self.hidden = int(field("qwen35.embedding_length"))
        self.layers = range(int(field("qwen35.block_count")))
        self.tensors = {t.name: t for t in self.reader.tensors}
        sha = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                sha.update(chunk)
        model_hash = sha.hexdigest()
        self.intervention = GGMLIntervention(llama_cpp, self.width, self.hidden, len(self.layers))

        # The pinned high-level wrapper does not expose cb_eval. Install it in
        # context defaults only while constructing this one model/context.
        original = native.llama_context_default_params
        def context_params():
            params = original()
            params.cb_eval = self.intervention.callback
            return params
        native.llama_context_default_params = context_params
        try:
            self.model = llama_cpp.Llama(model_path=str(path), n_gpu_layers=-1,
                n_ctx=int(os.environ.get("LLAMA_CONTEXT_TOKENS", "2048")),
                n_batch=128, n_ubatch=128, logits_all=False, verbose=True)
        finally:
            native.llama_context_default_params = original

        jinja = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        def raise_exception(message):
            raise TemplateError(message)
        jinja.globals["raise_exception"] = raise_exception
        self.chat_template = jinja.from_string(field("tokenizer.chat_template"))
        identity = {"model": self.model_id, "gguf_sha256": model_hash, "quantization": "GGUF Q4_K_M",
            "llama_cpp_python": llama_cpp.__version__, "llama_cpp_revision": "0c1e57098bba43ac29e6e3b677cdceebdd22334f",
            "layers": len(self.layers), "intermediate_size": self.width, "hidden_size": self.hidden,
            "calibration": CALIBRATION, "max_length": 128, "method": "median layer-output L2 over pooled t>=1, unsteered"}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_path = self.data / f"llama-calibration-{key}.json"
        if cache_path.exists():
            calibration = json.loads(cache_path.read_text())
            if calibration.get("identity") != identity or len(calibration.get("norms", [])) != len(self.layers):
                raise ValueError("Invalid llama.cpp calibration cache")
        else:
            self.intervention.calibrating = True
            self.intervention.error = None
            self.model._logits_all = True
            try:
                for sample in CALIBRATION:
                    self.model.reset()
                    tokens = self.model.tokenize(sample.encode("utf-8"), add_bos=False, special=False)[:128]
                    if len(tokens) < 2:
                        raise ValueError("Calibration sample has fewer than two tokens")
                    self.model.eval(tokens)
                    self.intervention.check()
            finally:
                self.model._logits_all = False
                self.intervention.calibrating = False
                self.model.reset()
            samples = self.intervention.residuals
            if any(len(row) < len(CALIBRATION) for row in samples):
                raise RuntimeError(f"Missing layer residual callbacks: {[len(row) for row in samples]}")
            calibration = {"identity": identity, "norms": [float(np.quantile(row, .5)) for row in samples],
                "sample_tokens_per_layer": [len(row) for row in samples]}
            cache_path.write_text(json.dumps(calibration, indent=2))
        self.norms = calibration["norms"]
        self.column_norms = {}
        self.info = {**identity, "model_path": str(path), "file_bytes": path.stat().st_size,
            "backend": "llama.cpp CUDA", "calibration": calibration}
        self.ready = True
        logging.info("MODEL %s", json.dumps(self.info))

    def column_norm(self, layer, neuron):
        import gguf
        key = (layer, neuron)
        if key not in self.column_norms:
            tensor = self.tensors[f"blk.{layer}.ffn_down.weight"]
            if tuple(tensor.shape) != (self.width, self.hidden):
                raise ValueError(f"Unexpected down-projection shape at layer {layer}")
            block_size, byte_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            if tensor.data.ndim != 2 or tensor.data.shape[0] != self.hidden:
                raise ValueError("Unexpected GGUF weight layout")
            if block_size == 1:
                values = tensor.data[:, neuron].astype(np.float32)
            else:
                start = (neuron // block_size) * byte_size
                blocks = np.ascontiguousarray(tensor.data[:, start:start + byte_size])
                decoded = gguf.dequantize(blocks, tensor.tensor_type)
                values = decoded[:, neuron % block_size]
            norm = float(np.linalg.norm(values.astype(np.float64)))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid GGUF down-projection column norm")
            self.column_norms[key] = norm
        return self.column_norms[key]

    def encode(self, request):
        messages = [dict(m) for m in request["messages"]]
        system = []
        while messages and messages[0]["role"] in {"system", "developer"}:
            system.append(messages.pop(0).get("content") or "")
        if system:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
        messages = copy.deepcopy(messages)
        for message in messages:
            for call in message.get("tool_calls", []):
                args = call["function"].get("arguments", {})
                if isinstance(args, str):
                    call["function"]["arguments"] = json.loads(args)
        text = self.chat_template.render(messages=messages, tools=request.get("tools") or None,
            add_generation_prompt=True, enable_thinking=False, add_vision_id=False)
        tokens = self.model.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) > int(os.environ.get("MAX_INPUT_TOKENS", "2048")):
            raise ValueError("Input exceeds local context limit; shorten history or MAX_INPUT_TOKENS")
        if len(tokens) >= self.model.n_ctx():
            raise ValueError("Input exceeds llama.cpp context window")
        return tokens

    def stream(self, request, snapshot):
        if not self.ready:
            raise RuntimeError("Model is not ready")
        events = queue.Queue()
        def worker():
            with self.lock:
                try:
                    # Prevent prefix/KV reuse across different profile snapshots.
                    self.model.reset()
                    old_repeat_last_n = self.model.last_n_tokens_size
                    if 'repeat_last_n' in request:
                        self.model.last_n_tokens_size = self.model.n_ctx() if request['repeat_last_n'] == -1 else request['repeat_last_n']
                    tokens = self.encode(request)
                    shifts = {}
                    details = {}
                    for item in snapshot["codes"]:
                        c = Code(**item)
                        if not 0 <= c.layer < len(self.layers) or not 0 <= c.neuron < self.width:
                            raise ValueError("Steering code is outside the GGUF architecture")
                        if c.enabled and c.dose:
                            shift = c.dose * self.norms[c.layer] / self.column_norm(c.layer, c.neuron)
                            shifts.setdefault(c.layer, []).append((c.neuron, shift))
                            details[f"L{c.layer} {c.neuron:05d}"] = shift
                    self.intervention.shifts = shifts
                    self.intervention.observations = []
                    self.intervention.error = None
                    logging.info("INFERENCE profile=%s version=%s codes=%s shifts=%s", snapshot["name"], snapshot["version"], snapshot["codes"], details)
                    chunks = self.model.create_completion(prompt=tokens, max_tokens=request.get("max_tokens", 256),
                        temperature=request.get("temperature", .7), top_p=request.get("top_p", .9),
                        seed=request.get("seed"), stop=request.get("stop") or [], stream=True,
                        top_k=request.get('top_k', 0), min_p=request.get('min_p', 0),
                        typical_p=request.get('typ_p', 1.0), repeat_penalty=request.get('repeat_penalty', 1.0))
                    finish = "stop"
                    for chunk in chunks:
                        self.intervention.check()
                        seen = {o["layer"] for o in self.intervention.observations}
                        if set(shifts) - seen:
                            raise RuntimeError(f"Steering callback missing for layers {sorted(set(shifts) - seen)}")
                        choice = chunk["choices"][0]
                        if choice.get("text"):
                            events.put(choice["text"])
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                    self.intervention.check()
                    # Every active layer must have actually run its intervention.
                    seen = {o["layer"] for o in self.intervention.observations}
                    if set(shifts) - seen:
                        raise RuntimeError(f"Steering callback missing for layers {sorted(set(shifts) - seen)}")
                    generated = max(0, self.model.n_tokens - len(tokens) + 1)
                    request.setdefault("_stats", {}).update(prompt_tokens=len(tokens), completion_tokens=generated,
                        finish_reason=finish)
                    self.last_inspection = {"snapshot": snapshot, "shifts": details,
                        "observations": list(self.intervention.observations)}
                except BaseException as error:
                    logging.exception("llama.cpp inference failed")
                    events.put(error)
                finally:
                    self.intervention.shifts = {}
                    if 'old_repeat_last_n' in locals():
                        self.model.last_n_tokens_size = old_repeat_last_n
                    self.model.reset()
                    events.put(None)
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
