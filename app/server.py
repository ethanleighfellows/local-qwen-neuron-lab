import copy
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .runtime import MODEL, Runtime
from .steering import Code, parse_codes
from .store import Store
from .tools import parse_answer


class Completion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL
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


def create_app(runtime=None, data=None):
    data = data or os.environ.get("DATA_DIR", "data")
    store = Store(Path(data) / "state.sqlite3")
    runtime = runtime or Runtime(data)

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if not runtime.ready:
            runtime.load()
        logging.info("ACTIVE_PROFILE %s", json.dumps(store.snapshot()))
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.store, app.state.runtime = store, runtime

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # No cross-origin browser writes; bind loopback by default, never execute client tools here.
        origin = request.headers.get("origin")
        if origin and origin not in {"http://127.0.0.1:8000", "http://localhost:8000"}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "Origin not allowed"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": {"message": str(error), "type": "invalid_request_error"}}, status_code=400)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.get("/api/status")
    def status():
        return {"ready": runtime.ready, "model": getattr(runtime, "info", {}),
            "last_inference": getattr(runtime, "last_inspection", None)}

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
        if body.model != MODEL:
            raise ValueError("Unknown model")
        if body.frequency_penalty or body.presence_penalty:
            raise ValueError("Nonzero frequency/presence penalties are not supported")
        request = body.model_dump(exclude_none=True)
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
        if body.baseline:
            snapshot["codes"] = []
        ident, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        stats = request["_stats"] = {}

        def usage():
            prompt, output = stats.get("prompt_tokens", 0), stats.get("completion_tokens", 0)
            return {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}

        def checked_answer(text):
            answer = parse_answer(text)
            if any(c["function"]["name"] not in allowed for c in answer.get("tool_calls", [])):
                raise ValueError("Model called an undeclared tool")
            return answer

        def envelope(choice, stream=False):
            return {"id": ident, "object": "chat.completion.chunk" if stream else "chat.completion",
                "created": created, "model": MODEL, "choices": [choice], "steering": snapshot}

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
                yield "data: " + json.dumps({"error": {"message": str(error), "type": "inference_error"}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")
    return app


app = create_app()
