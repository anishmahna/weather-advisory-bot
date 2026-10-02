from __future__ import annotations
"""Per-session memory (in-process; resets on restart by design).
Holds: recent raw history, established facts (location/activity/...), decision log."""
import threading

MAX_HISTORY = 12


class SessionMemory:
    def __init__(self):
        self._s: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, sid: str) -> dict:
        with self._lock:
            s = self._s.setdefault(sid, {"history": [], "facts": {}, "decisions": []})
            return {"history": list(s["history"]), "facts": dict(s["facts"]), "decisions": list(s["decisions"])}

    def update(self, sid: str, user_msg: str, reply: str, facts: dict | None, decision: dict | None):
        with self._lock:
            s = self._s.setdefault(sid, {"history": [], "facts": {}, "decisions": []})
            s["history"] += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}]
            s["history"] = s["history"][-MAX_HISTORY:]
            if facts:
                s["facts"].update({k: v for k, v in facts.items() if v})
            if decision:
                s["decisions"].append(decision)

    def reset(self, sid: str):
        with self._lock:
            self._s.pop(sid, None)
