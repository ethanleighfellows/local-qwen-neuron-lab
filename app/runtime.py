import hashlib
import json
import logging
import os
import queue
import threading
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from .steering import Code, SteeringEngine

MODEL = "Qwen/Qwen3.5-4B"
CALIBRATION = [
    "Explain how a computer stores a list of numbers and sorts it into ascending order.",
    "The river flows through the valley. Describe the changing seasons and the water cycle.",
    "Write a Python function that adds two integers, then explain how to test the function.",
]


class Runtime:
    def __init__(self, data="data"):
        self.data = Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.ready = False

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for this NF4 backend; no unquantized fallback")
        from transformers import Qwen3_5ForCausalLM
        revision = os.environ.get("MODEL_REVISION", "main")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        cache = str(self.data / "models")
        from transformers.utils.hub import cached_file
        revision = Path(cached_file(MODEL, "config.json", revision=revision, cache_dir=cache)).parent.name
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, cache_dir=cache)
        # Load only the text tower; checkpoint vision weights are intentionally unused.
        self.model = Qwen3_5ForCausalLM.from_pretrained(MODEL, revision=revision,
            cache_dir=cache, quantization_config=quant, device_map={"": 0},
            dtype=torch.float16, attn_implementation="sdpa").eval()
        self.layers = self.model.model.layers
        self.width = self.model.config.intermediate_size
        from bitsandbytes.nn import Linear4bit
        if not all(isinstance(layer.mlp.down_proj, Linear4bit) for layer in self.layers):
            raise RuntimeError("Model down projections are not genuine 4-bit modules")
        self.info = {"model": MODEL, "revision": revision,
            "quantization": "bitsandbytes NF4, double quantization, FP16 compute",
            "transformers": transformers.__version__, "torch": torch.__version__,
            "layers": len(self.layers), "intermediate_size": self.width,
            "gpu": torch.cuda.get_device_name(0), "footprint_bytes": self.model.get_memory_footprint()}
        key_data = {**self.info, "calibration": CALIBRATION, "max_length": 128,
            "method": "median layer-output L2 over all t>=1 in each unsteered sample; torch.quantile(0.5)"}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        path = self.data / f"calibration-{key}.json"
        if path.exists():
            calibration = json.loads(path.read_text())
        else:
            norms = [[] for _ in self.layers]
            handles = []
            for index, layer in enumerate(self.layers):
                def capture(module, inputs, output, index=index):
                    residual = output[0] if isinstance(output, tuple) else output
                    norms[index].extend(residual[:, 1:].float().norm(dim=-1).flatten().cpu().tolist())
                handles.append(layer.register_forward_hook(capture))
            try:
                with torch.inference_mode():
                    for text in CALIBRATION:
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
                        self.model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            calibration = {"identity": key_data, "norms": [torch.tensor(n).quantile(.5).item() for n in norms],
                "sample_tokens_per_layer": [len(n) for n in norms]}
            path.write_text(json.dumps(calibration, indent=2))
        self.engine = SteeringEngine(self.layers, calibration["norms"])
        self.info["calibration"] = calibration
        self.ready = True
        logging.info("MODEL %s", json.dumps(self.info))

    def encode(self, request):
        messages = [dict(m) for m in request["messages"]]
        system = []
        while messages and messages[0]["role"] in {"system", "developer"}:
            system.append(messages.pop(0).get("content") or "")
        if system:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
        for message in messages:
            # HF templates expect structured tool arguments, OpenAI sends JSON strings.
            if message.get("tool_calls"):
                message["tool_calls"] = json.loads(json.dumps(message["tool_calls"]))
                for call in message["tool_calls"]:
                    args = call["function"].get("arguments", {})
                    if isinstance(args, str):
                        call["function"]["arguments"] = json.loads(args)
        text = self.tokenizer.apply_chat_template(messages, tools=request.get("tools") or None,
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        if inputs.input_ids.shape[1] > int(os.environ.get("MAX_INPUT_TOKENS", "2048")):
            raise ValueError("Input exceeds local context limit; shorten history or MAX_INPUT_TOKENS")
        return inputs

    def stream(self, request, snapshot):
        """Worker thread owns lock, hooks and GPU for the entire generation."""
        if not self.ready:
            raise RuntimeError("Model is not ready")
        events = queue.Queue()
        def worker():
            with self.lock:
                try:
                    inputs = self.encode(request)
                    codes = [Code(**c) for c in snapshot["codes"]]
                    self.engine.install(codes)
                    logging.info("INFERENCE profile=%s version=%s codes=%s shifts=%s", snapshot["name"], snapshot["version"], snapshot["codes"], self.engine.shifts)
                    streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
                    # Generation itself stays on this thread, streamer forwards text into queue.
                    stops = request.get("stop") or []
                    stops = [stops] if isinstance(stops, str) else stops
                    tail, stopped = "", False
                    def emit(text, stream_end=False):
                        nonlocal tail, stopped
                        if stopped:
                            return
                        tail += text
                        matches = [tail.find(s) for s in stops if s and s in tail]
                        if matches:
                            prefix = tail[:min(matches)]
                            if prefix:
                                events.put(prefix)
                            tail, stopped = "", True
                            return
                        keep = max((len(s) - 1 for s in stops), default=0) if not stream_end else 0
                        size = max(0, len(tail) - keep)
                        if size:
                            events.put(tail[:size])
                            tail = tail[size:]
                    streamer.on_finalized_text = emit
                    temperature = request.get("temperature", .7)
                    kwargs = dict(max_new_tokens=request.get("max_tokens", 256), do_sample=temperature > 0,
                        streamer=streamer, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
                    if temperature > 0:
                        kwargs.update(temperature=temperature, top_p=request.get("top_p", .9))
                    with torch.inference_mode():
                        if request.get("seed") is not None:
                            torch.manual_seed(request["seed"])
                        if request.get("stop"):
                            stops = request["stop"]
                            kwargs.update(stop_strings=[stops] if isinstance(stops, str) else stops,
                                tokenizer=self.tokenizer)
                        output = self.model.generate(**inputs, **kwargs)
                    count = output.shape[-1] - inputs.input_ids.shape[-1]
                    stats = request.setdefault("_stats", {})
                    eos = self.model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    ended = output[0, -1].item() in eos
                    stats.update(prompt_tokens=inputs.input_ids.shape[-1], completion_tokens=count,
                        finish_reason="length" if count >= kwargs["max_new_tokens"] and not ended else "stop")
                    self.last_inspection = {"snapshot": snapshot, "shifts": dict(self.engine.shifts),
                        "observations": list(self.engine.observations)}
                except BaseException as error:
                    logging.exception("Inference failed")
                    events.put(error)
                finally:
                    self.engine.remove()
                    events.put(None)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            item = events.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
