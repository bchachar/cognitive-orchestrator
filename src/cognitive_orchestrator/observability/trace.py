"""Structured per-run tracing: one entry per executed graph node.

Enabled with CO_TRACE=true (default off). Output goes to
<CO_TRACE_DIR or "traces">/run-<timestamp>.json; render it with tools/view_trace.py.
"""
from __future__ import annotations

import functools
import json
import os
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

_MAX_LEN = 200
_SKIP_KEYS = {"trace"}

# Model calls made while a node is running are collected here by the LLM client.
_model_calls: ContextVar[Optional[list]] = ContextVar("co_model_calls", default=None)


def trace_enabled() -> bool:
    return os.getenv("CO_TRACE", "").lower() == "true"


def record_model_call(model: str, usage: Any = None) -> None:
    """Called by the LLM client; a no-op unless a traced node is running."""
    sink = _model_calls.get()
    if sink is None:
        return
    sink.append({
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    })


def _compact(value: Any, key: str = "") -> str:
    if key == "image_b64":
        return f"<b64 {len(value or '')} chars>"
    try:
        text = json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= _MAX_LEN else text[:_MAX_LEN] + f"... (+{len(text) - _MAX_LEN} chars)"


def summarize(data: dict) -> dict[str, str]:
    return {k: _compact(v, k) for k, v in data.items() if k not in _SKIP_KEYS}


class Trace:
    def __init__(self, command: str = "") -> None:
        self.command = command
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.entries: list[dict] = []

    def append(self, entry: dict) -> None:
        self.entries.append(entry)

    def to_dict(self) -> dict:
        return {"command": self.command, "started_at": self.started_at,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "entries": self.entries}

    def write(self, directory: Optional[str] = None) -> Path:
        out_dir = Path(directory or os.getenv("CO_TRACE_DIR") or "traces")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = out_dir / f"run-{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def traced(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wrap a graph node so it appends an entry to state['trace'] (if present)."""

    @functools.wraps(fn)
    def wrapper(state: dict) -> dict:
        trace: Optional[Trace] = state.get("trace")
        if trace is None:
            return fn(state)

        calls: list = []
        token = _model_calls.set(calls)
        start_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        entry: dict = {"node": name, "start": start_iso, "input": summarize(state),
                       "status": "ok", "error": None}
        try:
            update = fn(state)
        except Exception as e:
            entry.update(status="error", error=f"{type(e).__name__}: {e}", output={})
            raise
        else:
            entry["output"] = summarize(update)
            failure = update.get("failure") or update.get("plan_error")
            if failure:
                entry["status"] = "failure"
                entry["error"] = _compact(failure)
            return update
        finally:
            _model_calls.reset(token)
            entry["end"] = datetime.now(timezone.utc).isoformat()
            entry["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            if calls:
                entry["model_calls"] = calls
            trace.append(entry)

    return wrapper
