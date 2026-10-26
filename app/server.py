import copy
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .model_config import MODEL, UNSTEERED_MODEL
from .steering import Code, parse_codes
from .store import Store
from .tools import parse_answer


class Completion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    messages: list[dict] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=.7, ge=0, le=2)
    top_p: float = Field(default=.9, gt=0, le=1)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=2048)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    stream_options: dict | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float = 0
    presence_penalty: float = 0
    # llama.cpp Web UI transport hints and supported sampler controls.
    return_progress: bool | None = None
    sse_ping_interval: float | None = Field(default=None, ge=0)
    reasoning_format: str | None = None
    reasoning_control: bool | None = None
    chat_template_kwargs: dict | None = None
    backend_sampling: bool | None = None
    timings_per_token: bool | None = None
    top_k: int | None = Field(default=None, ge=0, le=1000)
    min_p: float | None = Field(default=None, ge=0, le=1)
    typ_p: float | None = Field(default=None, gt=0, le=1)
    repeat_penalty: float | None = Field(default=None, gt=0, le=5)
    repeat_last_n: int | None = Field(default=None, ge=-1, le=2048)
    steering_profile: str | None = None
    baseline: bool = False


class SteeringCodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    layer: int = Field(ge=0, strict=True)
    neuron: int = Field(ge=0, strict=True)
    dose: float
    enabled: bool = True


class ProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    steering: list[SteeringCodeInput]


class StateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_profile: str
    enabled: bool
    version: int = Field(ge=0)
    profiles: dict[str, ProfileInput]


def create_app(runtime=None, data=None, ui_dir=None):
    data = data or os.environ.get("DATA_DIR", "data")
    store = Store(Path(data) / "state.sqlite3")
    if runtime is None:
        backend = os.environ.get("MODEL_BACKEND", "transformers").lower()
        if backend == "transformers":
            from .runtime import Runtime
            runtime = Runtime(data)
        elif backend == "llama_cpp":
            from .model_router import ModelRouter
            runtime = ModelRouter(data)
        else:
            raise ValueError(f"Unknown MODEL_BACKEND: {backend}")

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if not runtime.ready:
            runtime.load()
        logging.info("ACTIVE_PROFILE %s", json.dumps(store.snapshot()))
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.store, app.state.runtime = store, runtime
    ui_dir = Path(ui_dir) if ui_dir else Path(__file__).resolve().parent.parent / 'data' / 'llama-ui-b11312'

    def native_ui_available():
        return getattr(runtime, 'info', {}).get('backend') == 'llama.cpp CUDA' and (ui_dir / 'index.html').is_file()

    def native_index():
        content = (ui_dir / 'index.html').read_text(encoding='utf-8')
        content = content.replace('<head>', '<head><base href="/">', 1)
        content = content.replace("base: new URL('.', location).pathname.slice(0, -1)", "base: ''")
        content = content.replace('</head>', '<script defer src="/static/steering-control.js"></script></head>')
        return HTMLResponse(content, headers={'Cache-Control': 'no-cache'})

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # No cross-origin browser writes; bind loopback by default, never execute client tools here.
        origin = request.headers.get("origin")
        parsed = urlsplit(origin) if origin else None
        if parsed and (parsed.scheme != request.url.scheme or parsed.netloc != request.headers.get('host')
                or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.path or parsed.query or parsed.fragment):
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "Origin not allowed"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": {"message": str(error), "type": "invalid_request_error"}}, status_code=400)

    def supported_models():
        return [MODEL, UNSTEERED_MODEL] if hasattr(runtime, "available") else [MODEL]

    def selected_model():
        selected = store.get_setting("active_model", MODEL)
        return selected if selected in supported_models() else MODEL

    @app.get("/v1/models")
    def models():
        ids = supported_models()
        ids.sort(key=lambda name: name != selected_model())
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "local"} for name in ids]}

    @app.get("/api/models")
    def model_options():
        return {"selected": selected_model(), "models": [{"id": name,
            "available": runtime.available(name) if hasattr(runtime, "available") else True,
            "steering": name == MODEL} for name in supported_models()]}

    @app.put("/api/model")
    def select_model(body: dict):
        model = body.get("model")
        if model not in supported_models() or (hasattr(runtime, "available") and not runtime.available(model)):
            raise ValueError("Unknown or unavailable model")
        store.set_setting("active_model", model)
        return model_options()

    @app.get("/api/status")
    def status():
        return {"ready": runtime.ready, "model": getattr(runtime, "info", {}),
            "last_inference": getattr(runtime, "last_inspection", None)}

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/props')
    def llama_props():
        return {'role': 'model', 'model_alias': selected_model(), 'model_path': selected_model(), 'total_slots': 1,
            'modalities': {'vision': False, 'audio': False, 'video': False},
            'chat_template': '', 'bos_token': '', 'eos_token': '', 'ui_settings': {},
            'build_info': 'Local Neuron Lab, llama.cpp-compatible interface',
            'default_generation_settings': {'n_ctx': int(os.environ.get('LLAMA_CONTEXT_TOKENS', '2048')),
                'params': {'temperature': .7, 'top_p': .9, 'max_tokens': 256, 'n_predict': 256}}}

    @app.get('/tools')
    def ui_tools():
        return []

    @app.post('/v1/streams/lookup')
    def ui_stream_lookup():
        # This server does not keep replayable streams after a disconnect.
        return []

    @app.get("/api/profiles")
    def profiles():
        return store.state()

    @app.put("/api/profiles")
    def save_profiles(body: StateInput):
        value = body.model_dump()
        if not isinstance(value.get("enabled"), bool) or value.get("active_profile") not in value.get("profiles", {}):
            raise ValueError("Invalid active profile or global enabled flag")
        for name, profile in value["profiles"].items():
            if not name.strip() or not isinstance(profile.get("enabled"), bool):
                raise ValueError("Invalid profile")
            codes = [Code(**c) for c in profile["steering"]]
            parse_codes("\n".join(f"L{c.layer} {c.neuron} {c.dose:+g}" for c in codes), len(runtime.layers), runtime.width)
            if any(not isinstance(c.enabled, bool) for c in codes):
                raise ValueError("Invalid neuron enabled flag")
        return store.save_state(value)

    @app.get("/api/chats")
    def chats():
        return store.chats()

    @app.post("/api/chats")
    def save_chat(value: dict):
        return store.save_chat(value)

    @app.post("/v1/chat/completions")
    def completion(body: Completion):
        model = body.model or selected_model()
        if model not in supported_models() or (hasattr(runtime, "available") and not runtime.available(model)):
            raise ValueError("Unknown or unavailable model")
        if body.frequency_penalty or body.presence_penalty:
            raise ValueError("Nonzero frequency/presence penalties are not supported")
        if body.backend_sampling:
            raise ValueError('Backend sampling is not supported')
        if body.reasoning_format not in (None, 'auto', 'none'):
            raise ValueError('Unsupported reasoning format')
        if body.chat_template_kwargs and (set(body.chat_template_kwargs) - {'enable_thinking'} or body.chat_template_kwargs.get('enable_thinking')):
            raise ValueError('Thinking mode is not supported by this server')
        if body.repeat_last_n is not None and getattr(runtime, 'info', {}).get('backend') != 'llama.cpp CUDA':
            raise ValueError('repeat_last_n is supported only by the llama.cpp backend')
        request = body.model_dump(exclude_none=True)
        request["model"] = model
        if body.max_completion_tokens:
            request["max_tokens"] = body.max_completion_tokens
        if body.tool_choice not in (None, "auto", "none"):
            raise ValueError("Only auto and none tool_choice are supported")
        if body.tool_choice == "none":
            request.pop("tools", None)
        allowed = {t["function"]["name"] for t in request.get("tools", [])}
        ids = set()
        for message in body.messages:
            if message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
                raise ValueError("Unsupported message role")
            content = message.get("content")
            if isinstance(content, list) and any(item.get("type") != "text" for item in content):
                raise ValueError("This local runtime accepts text messages only")
            if message["role"] == "tool" and message.get("tool_call_id") not in ids:
                raise ValueError("Tool result must reference an earlier tool call ID")
            ids.update(c["id"] for c in message.get("tool_calls", []))
        snapshot = copy.deepcopy(store.snapshot(body.steering_profile))
        if body.baseline or model == UNSTEERED_MODEL:
            snapshot["codes"] = []
        if model == UNSTEERED_MODEL:
            snapshot["name"] = "Unsteered"
        ident, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        stats = request["_stats"] = {}

        def usage():
            prompt, output = stats.get("prompt_tokens", 0), stats.get("completion_tokens", 0)
            return {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}

        def checked_answer(text):
            answer = parse_answer(text)
            content = answer.get("content") or ""
            if request.get("tools") and any(tag in content for tag in ("</tool_call>", "</function>", "</parameter>")):
                raise ValueError("Model generated malformed tool-call markup; try a lower steering dose or Baseline")
            if any(c["function"]["name"] not in allowed for c in answer.get("tool_calls", [])):
                raise ValueError("Model called an undeclared tool")
            return answer

        def envelope(choice, stream=False):
            return {"id": ident, "object": "chat.completion.chunk" if stream else "chat.completion",
                "created": created, "model": model, "choices": [choice], "steering": snapshot}

        if not body.stream:
            answer = checked_answer("".join(runtime.stream(request, snapshot)))
            result = envelope({"index": 0, "message": answer,
                "finish_reason": "tool_calls" if answer.get("tool_calls") else stats.get("finish_reason", "stop")})
            result["usage"] = usage()
            return result

        def events():
            def event(delta, finish=None):
                return "data: " + json.dumps(envelope({"index": 0, "delta": delta, "finish_reason": finish}, True)) + "\n\n"
            yield event({"role": "assistant", "content": ""})
            try:
                full = ""
                # Tool requests buffer native syntax to avoid leaking partial XML to clients.
                for token in runtime.stream(request, snapshot):
                    full += token
                    if not request.get("tools"):
                        yield event({"content": token})
                answer = checked_answer(full)
                if request.get("tools"):
                    if answer.get("content"):
                        yield event({"content": answer["content"]})
                    for index, call in enumerate(answer.get("tool_calls", [])):
                        yield event({"tool_calls": [{"index": index, **call}]})
                yield event({}, "tool_calls" if answer.get("tool_calls") else stats.get("finish_reason", "stop"))
                if (body.stream_options or {}).get("include_usage"):
                    result = envelope({}, True)
                    result.update(choices=[], usage=usage())
                    yield "data: " + json.dumps(result) + "\n\n"
            except Exception as error:
                # The llama.cpp UI renders content deltas but does not inspect
                # an error-only SSE chunk; other clients still get the error field.
                failed = envelope({'index': 0, 'delta': {'content': f'\n[Inference error: {error}]'},
                    'finish_reason': None}, True)
                failed['error'] = {'message': str(error), 'type': 'inference_error'}
                yield "data: " + json.dumps(failed) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        if native_ui_available():
            return native_index()
        return FileResponse(static / "index.html")

    @app.get('/steering')
    def steering_page():
        return FileResponse(static / 'index.html')

    @app.get('/sw.js')
    def disable_stale_service_worker():
        # Avoid caching stale profile/API responses or a previous frontend.
        return Response("self.addEventListener('install',()=>self.skipWaiting());"
            "self.addEventListener('activate',e=>e.waitUntil(self.registration.unregister()));",
            media_type='application/javascript', headers={'Cache-Control': 'no-store'})

    @app.get('/{asset_path:path}')
    def native_asset(asset_path: str):
        if not native_ui_available():
            raise HTTPException(status_code=404)
        if asset_path in {'index.html', 'search'} or asset_path.startswith('chat/'):
            return native_index()
        target = (ui_dir / asset_path).resolve()
        if not target.is_relative_to(ui_dir.resolve()) or not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target)
    return app


app = create_app()
