"""Small helpers shared across the trainer: atomic IO, logging, interrupts."""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import tempfile
import time
from typing import Any

import numpy as np


# ---------------------------------------------------------------- atomic IO
def atomic_write_bytes(path: str, payload: bytes) -> None:
    """Write ``payload`` to ``path`` so a crash never leaves a torn file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        _silent_unlink(tmp)
        raise


def atomic_write_text(path: str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_with(path: str, writer, suffix: str = ".part") -> None:
    """Atomically produce ``path`` via ``writer(tmp_path)`` (for np.savez etc.).

    ``suffix`` matters for writers such as ``np.savez`` that append ``.npz``
    to a filename that lacks it -- pass the real extension in that case.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".tmp-{os.getpid()}-{time.time_ns()}{suffix}")
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        _silent_unlink(tmp)
        raise


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


# ---------------------------------------------------------------- logging
class JsonlLogger:
    """Append-only structured log; flushed on every record so crashes keep it."""

    def __init__(self, path: str, echo: bool = True) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self.echo = echo
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        record = {"t": round(time.time(), 3), "event": event, **fields}
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()
        if self.echo:
            print(format_record(record), flush=True)

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


def format_record(record: dict[str, Any]) -> str:
    stamp = time.strftime("%H:%M:%S", time.localtime(record.get("t", time.time())))
    parts = []
    for key, value in record.items():
        if key in ("t", "event"):
            continue
        if isinstance(value, float):
            value = f"{value:.4g}"
        parts.append(f"{key}={value}")
    return f"[{stamp}] {record.get('event', '?'):<14} " + " ".join(parts)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- misc
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


class GracefulKiller:
    """Turn SIGINT/SIGTERM into a flag so we can flush and checkpoint first."""

    def __init__(self) -> None:
        self.stop = False
        self._previous: dict[int, Any] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):  # not on the main thread
                pass

    def _handle(self, signum, frame) -> None:  # noqa: ANN001
        if self.stop:  # second Ctrl-C: give up immediately
            log("second interrupt -- exiting now")
            sys.exit(130)
        self.stop = True
        log("interrupt received -- finishing current unit of work, then saving")

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def reset(self) -> float:
        now = time.perf_counter()
        delta = now - self.start
        self.start = now
        return delta


def human_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{sec:02d}s"
