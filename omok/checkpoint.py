"""Crash-safe checkpointing.

Weights are stored as a plain ``.npz`` of canonical (PyTorch-named, NCHW)
float32 arrays, so a checkpoint written by the MLX backend loads into the
PyTorch backend and vice versa.  Every file is written to a temporary name and
renamed into place, so an interrupted save can never corrupt a run.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .netspec import NetSpec
from .utils import atomic_write_with, log, read_json, write_json

WEIGHTS_SUFFIX = ".npz"
META_SUFFIX = ".json"


def save_weights(path: str, weights: dict[str, np.ndarray]) -> None:
    payload = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
    atomic_write_with(path, lambda tmp: np.savez(tmp, **payload), suffix=".npz")


def load_weights(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: np.array(data[k], dtype=np.float32) for k in data.files}


def weights_nbytes(weights: dict[str, np.ndarray]) -> int:
    return int(sum(v.size * 4 for v in weights.values()))


@dataclass
class Checkpoint:
    tag: str
    weights_path: str
    meta: dict[str, Any]

    @property
    def step(self) -> int:
        return int(self.meta.get("step", 0))

    def weights(self) -> dict[str, np.ndarray]:
        return load_weights(self.weights_path)


class CheckpointManager:
    """Rolling checkpoints plus a separate ``best`` model for self-play."""

    def __init__(self, directory: str, keep_last: int = 5) -> None:
        self.dir = directory
        self.keep_last = max(1, keep_last)
        os.makedirs(self.dir, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def path_for(self, tag: str, ext: str = WEIGHTS_SUFFIX) -> str:
        return os.path.join(self.dir, f"{tag}{ext}")

    @property
    def latest_pointer(self) -> str:
        return os.path.join(self.dir, "latest.json")

    @property
    def best_path(self) -> str:
        return os.path.join(self.dir, "best.npz")

    @property
    def best_meta_path(self) -> str:
        return os.path.join(self.dir, "best.json")

    # -- writing -----------------------------------------------------------
    def save(self, backend, meta: dict[str, Any], tag: str | None = None,
             with_optimizer: bool = True) -> Checkpoint:
        step = int(meta.get("step", 0))
        tag = tag or f"step-{step:08d}"
        weights = backend.get_weights()
        weights_path = self.path_for(tag)
        save_weights(weights_path, weights)
        meta = dict(meta)
        meta.setdefault("spec", backend.spec.to_dict())
        meta["tag"] = tag
        meta["bytes"] = weights_nbytes(weights)
        if with_optimizer:
            opt_ext = ".opt.pt" if backend.name == "torch" else ".opt.npz"
            try:
                if backend.save_optimizer(self.path_for(tag, opt_ext)):
                    meta["optimizer"] = os.path.basename(self.path_for(tag, opt_ext))
            except Exception as exc:  # optimiser state is a nicety, never fatal
                log(f"warning: could not save optimiser state: {exc}")
        write_json(self.path_for(tag, META_SUFFIX), meta)
        write_json(self.latest_pointer, {"tag": tag, "step": step})
        self._rotate()
        return Checkpoint(tag, weights_path, meta)

    def save_best(self, weights: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        save_weights(self.best_path, weights)
        meta = dict(meta)
        meta["bytes"] = weights_nbytes(weights)
        write_json(self.best_meta_path, meta)

    # -- reading -----------------------------------------------------------
    def list_tags(self) -> list[str]:
        tags = []
        for path in glob.glob(os.path.join(self.dir, "step-*" + WEIGHTS_SUFFIX)):
            tag = os.path.basename(path)[: -len(WEIGHTS_SUFFIX)]
            if re.fullmatch(r"step-\d+", tag):
                tags.append(tag)
        return sorted(tags)

    def latest(self) -> Checkpoint | None:
        # Newest step first.  The pointer normally names the newest complete
        # save, but a crash after the weights write and before the pointer
        # update leaves a newer step-N.npz on disk -- that one wins, and the
        # pointer only breaks ties (a non-step tag it names goes first).
        pointer = read_json(self.latest_pointer, default=None)
        steps: dict[str, int] = {}
        if isinstance(pointer, dict) and pointer.get("tag"):
            steps[pointer["tag"]] = int(pointer.get("step", 0))
        for tag in self.list_tags():
            steps.setdefault(tag, int(tag[len("step-"):]))
        candidates = sorted(steps, key=steps.__getitem__, reverse=True)
        for tag in candidates:
            weights_path = self.path_for(tag)
            if not os.path.exists(weights_path):
                continue
            try:  # a half-written file from a hard kill is simply skipped
                load_weights(weights_path)
            except Exception:
                log(f"warning: checkpoint {tag} is unreadable, falling back")
                continue
            meta = read_json(self.path_for(tag, META_SUFFIX), default={}) or {}
            if not meta:
                # A crash between the weights write and the meta write leaves
                # a readable .npz with no .json.  The weights are still the
                # newest trained state -- synthesise enough meta to resume
                # from them instead of falling back to random init.
                match = re.fullmatch(r"step-(\d+)", tag)
                meta = {"step": int(match.group(1)) if match else 0,
                        "tag": tag, "meta_missing": True}
                # save() writes the optimiser file *before* the meta, so it
                # is usually there too: pick it up rather than resuming with
                # fresh optimiser moments.
                for ext in (".opt.pt", ".opt.npz"):
                    if os.path.exists(self.path_for(tag, ext)):
                        meta["optimizer"] = os.path.basename(self.path_for(tag, ext))
                        break
            return Checkpoint(tag, weights_path, meta)
        return None

    def best(self) -> Checkpoint | None:
        if not os.path.exists(self.best_path):
            return None
        try:  # a torn best.npz must not block startup -- the caller re-seeds it
            load_weights(self.best_path)
        except Exception:
            log("warning: best.npz is unreadable -- ignoring it")
            return None
        meta = read_json(self.best_meta_path, default={}) or {}
        return Checkpoint("best", self.best_path, meta)

    def restore(self, backend, checkpoint: Checkpoint | None = None,
                with_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = checkpoint or self.latest()
        if checkpoint is None:
            return {}
        backend.set_weights(checkpoint.weights())
        if with_optimizer and checkpoint.meta.get("optimizer"):
            opt_path = os.path.join(self.dir, checkpoint.meta["optimizer"])
            if backend.load_optimizer(opt_path):
                checkpoint.meta["optimizer_restored"] = True
        return checkpoint.meta

    # -- housekeeping ------------------------------------------------------
    def _rotate(self) -> None:
        tags = self.list_tags()
        for tag in tags[: max(0, len(tags) - self.keep_last)]:
            for ext in (WEIGHTS_SUFFIX, META_SUFFIX, ".opt.pt", ".opt.npz"):
                path = self.path_for(tag, ext)
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


def spec_from_meta(meta: dict[str, Any], fallback: NetSpec) -> NetSpec:
    if isinstance(meta.get("spec"), dict):
        return NetSpec.from_dict(meta["spec"])
    return fallback
