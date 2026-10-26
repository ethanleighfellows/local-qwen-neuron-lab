import json
import sqlite3
import threading
import uuid
from pathlib import Path
from .presets import initial_state


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, value TEXT)")
            self.db.execute("INSERT OR IGNORE INTO state VALUES (1, ?)", (json.dumps(initial_state()),))

    def state(self):
        with self.lock:
            return json.loads(self.db.execute("SELECT value FROM state WHERE id=1").fetchone()[0])

    def save_state(self, value):
        with self.lock, self.db:
            if value.get("version") != self.state()["version"]:
                raise ValueError("Profiles changed in another window; reload before saving")
            value["version"] += 1
            self.db.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(value),))
        return value

    def snapshot(self, override=None):
        state = self.state()
        name = override if override is not None else state["active_profile"]
        if name not in state["profiles"]:
            raise ValueError("Unknown profile")
        profile = state["profiles"][name]
        return {"name": name, "version": state["version"], "codes": profile["steering"]
                if state["enabled"] and profile["enabled"] else []}

    def get_setting(self, key, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row[0] if row else default

    def set_setting(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))

    def chats(self):
        with self.lock:
            return [json.loads(row[0]) for row in self.db.execute("SELECT value FROM chats ORDER BY rowid DESC")]

    def save_chat(self, chat):
        chat.setdefault("id", uuid.uuid4().hex)
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO chats VALUES (?, ?)", (chat["id"], json.dumps(chat)))
        return chat
