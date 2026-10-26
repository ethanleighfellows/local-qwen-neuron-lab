"""Load one llama.cpp model at a time, even when requests overlap."""
import gc
import queue
import threading
from pathlib import Path
from .llama_runtime import LlamaRuntime
from .model_config import MODEL, UNSTEERED_MODEL, UNSTEERED_FILENAME


class ModelRouter:
    def __init__(self, data):
        self.data = Path(data)
        self.lock = threading.Lock()
        self.current = None
        self.ready = False
        self.info = {"backend": "llama.cpp CUDA"}
        self.layers = range(32)
        self.width = 9216
        self.last_inspection = None

    def available(self, model):
        return model == MODEL or (model == UNSTEERED_MODEL and (self.data / 'gguf' / UNSTEERED_FILENAME).is_file())

    def load(self):
        self._load(MODEL)
        self.ready = True

    def _load(self, model):
        if self.current and self.current.model_id == model:
            return
        if not self.available(model):
            raise ValueError("Model is not installed; run install-llama.cmd")
        if self.current:
            self.current.model.close()
            self.current = None
            gc.collect()
        runtime = LlamaRuntime(self.data, model)
        runtime.load()
        self.current = runtime
        self.info = runtime.info
        self.layers, self.width = runtime.layers, runtime.width

    def stream(self, request, snapshot):
        events = queue.Queue()
        def worker():
            with self.lock:
                try:
                    model = request['model']
                    self._load(model)
                    # Defense in depth: the alternative never receives activation edits.
                    if model == UNSTEERED_MODEL:
                        snapshot = _safe_snapshot(snapshot_value)
                    else:
                        snapshot = snapshot_value
                    for token in self.current.stream(request, snapshot):
                        events.put(token)
                    self.last_inspection = self.current.last_inspection
                except BaseException as error:
                    events.put(error)
                finally:
                    events.put(None)
        snapshot_value = snapshot
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item


def _safe_snapshot(snapshot):
    return {**snapshot, 'name': 'Unsteered', 'codes': []}
