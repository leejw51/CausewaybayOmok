"""A search engine that thinks on a background thread.

The GUI has to keep drawing at 60 fps while a 160-simulation search runs, so
the search cannot happen in the draw loop.  This module owns a worker thread:
the UI posts positions to it and drains results in ``on_update``.  Nothing here
imports arcade, so it stays importable (and testable) without a display.

The search is run in small chunks so a request can be abandoned the moment the
user undoes a move or starts a new game -- otherwise a slow search on a big
board would keep the window unresponsive to anything but the mouse pointer.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .board import Board
from .config import Config
from .mcts import NetworkEvaluator, Tree, run_search

# How many simulations to run between cancellation / progress checks.
CHUNK = 8


@dataclass
class Request:
    job: int
    board: Board
    simulations: int
    temperature: float = 0.0
    kind: str = "move"  # "move" (play it) | "hint" (just show it)


@dataclass
class Result:
    job: int
    kind: str
    move: int
    value: float                      # from the point of view of the side to move
    ply: int                          # board.move_number the search started from
    top: list[tuple[int, float]] = field(default_factory=list)
    simulations: int = 0
    seconds: float = 0.0

    @property
    def nps(self) -> float:
        return self.simulations / self.seconds if self.seconds > 0 else 0.0


class Engine(threading.Thread):
    """Loads the network once, then answers search requests one at a time.

    Only the newest request matters: asking for a new search cancels whatever
    is in flight, and stale results are dropped rather than delivered.
    """

    def __init__(self, cfg: Config, model_path: str | None = None,
                 batch_size: int = 64) -> None:
        super().__init__(name="omok-engine", daemon=True)
        self.cfg = cfg
        self.model_path = model_path
        self.batch_size = batch_size
        self.requests: queue.Queue[Request | None] = queue.Queue()
        self.events: queue.Queue[tuple] = queue.Queue()
        self._lock = threading.Lock()
        self._job = 0

    # -- API used by the UI thread ---------------------------------------
    def submit(self, board: Board, simulations: int, temperature: float = 0.0,
               kind: str = "move") -> int:
        """Queue a search of ``board``, cancelling any earlier one."""
        with self._lock:
            self._job += 1
            job = self._job
        self.requests.put(Request(job, board.copy(), int(simulations),
                                  float(temperature), kind))
        return job

    def cancel(self) -> None:
        """Abandon the running search; its result will be dropped."""
        with self._lock:
            self._job += 1

    def shutdown(self) -> None:
        self.cancel()
        self.requests.put(None)

    def poll(self) -> list[tuple]:
        """Drain every event posted since the last call (never blocks)."""
        out: list[tuple] = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    # -- worker thread ----------------------------------------------------
    def _stale(self, job: int) -> bool:
        with self._lock:
            return job != self._job

    def run(self) -> None:  # pragma: no cover - exercised via the GUI
        from .play import load_play_backend

        try:
            backend, spec, source = load_play_backend(self.cfg, self.model_path)
        except Exception as exc:
            self.events.put(("error", f"cannot load model: {exc}"))
            return
        evaluator = NetworkEvaluator(backend, batch_size=self.batch_size)
        rng = np.random.default_rng()
        info: dict[str, Any] = dict(backend.describe())
        info.update(source=source, spec=spec)
        self.events.put(("ready", info))

        while True:
            request = self.requests.get()
            if request is None:
                return
            if self._stale(request.job):
                continue
            try:
                result = self._search(evaluator, rng, request)
            except Exception as exc:
                self.events.put(("error", f"search failed: {exc}"))
                continue
            if result is not None:
                self.events.put(("result", result))

    def _search(self, evaluator, rng, request: Request) -> Result | None:
        board = request.board
        if board.over:
            return None
        tree = Tree(board, self.cfg.mcts)
        tree.root_noise_applied = True  # no exploration noise when facing a human
        started = time.perf_counter()
        done = 0
        while done < request.simulations:
            if self._stale(request.job):
                return None
            chunk = min(CHUNK, request.simulations - done)
            run_search([tree], evaluator, chunk, rng)
            done += chunk
            self.events.put(("progress", request.job, done, request.simulations))
        if self._stale(request.job):
            return None
        probs = tree.visit_distribution()
        order = np.argsort(-probs)[:6]
        top = [(int(i), float(probs[i])) for i in order if probs[i] > 0.0]
        # Never miss a one-ply win or fail to block one, whatever the search says.
        forced = board.forced_move()
        return Result(
            job=request.job,
            kind=request.kind,
            move=forced if forced is not None else tree.pick_move(rng, request.temperature),
            value=tree.root_value(),
            ply=board.move_number,
            top=top,
            simulations=done,
            seconds=time.perf_counter() - started,
        )
