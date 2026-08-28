"""The AlphaZero-style loop: self-play -> train -> gate -> repeat.

Every phase writes its progress to ``state.json`` before moving on, and each
phase is individually resumable, so ``make resume`` after a crash picks up
where the run left off instead of restarting the iteration.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from .arena import evaluate_candidate
from .backends import make_backend
from .checkpoint import CheckpointManager
from .config import Config
from .netspec import NetSpec, init_weights
from .replay import ReplayBuffer, dataset_stats
from .selfplay import run_selfplay
from .train import Trainer
from .utils import (GracefulKiller, JsonlLogger, human_time, read_json,
                    seed_everything, write_json)

PHASES = ("selfplay", "train", "arena")


class Pipeline:
    def __init__(self, cfg: Config, killer: GracefulKiller | None = None,
                 echo: bool = True) -> None:
        self.cfg = cfg
        self.paths = cfg.paths().ensure()
        self.killer = killer or GracefulKiller()
        self.logger = JsonlLogger(os.path.join(self.paths.logs, "run.jsonl"), echo=echo)
        self.spec = NetSpec.from_config(cfg)
        seed_everything(cfg.seed)
        self.state = self._load_state()
        self._save_config()

        self.backend = make_backend(self.spec, cfg.backend, lr=cfg.train.lr,
                                    weight_decay=cfg.train.weight_decay)
        # Announce the compute device before anything else: running on the CPU
        # by accident is the difference between hours and days.
        self.logger.log("backend", **self.backend.describe())
        self.checkpoints = CheckpointManager(self.paths.checkpoints, cfg.train.keep_last_ckpts)
        meta = self.checkpoints.restore(self.backend)
        if not meta:  # fresh run: deterministic init so a restart reproduces it
            self.backend.set_weights(init_weights(self.spec, cfg.seed))
            self.logger.log("init", **self.backend.describe())
        else:
            self.logger.log("resume", step=meta.get("step", 0), tag=meta.get("tag"),
                            **self.backend.describe())
        self.global_step = int(meta.get("step", self.state.get("global_step", 0)))

        # The self-play network is the current *best*, not the latest.
        if self.checkpoints.best() is None:
            self.checkpoints.save_best(self.backend.get_weights(),
                                       {"step": self.global_step, "iteration": 0,
                                        "spec": self.spec.to_dict()})
        self.play_backend = self.backend.clone_for_inference()
        self.play_backend.set_weights(self.checkpoints.best().weights())
        # Where the model that self-play (and `make play` / `make export`) uses
        # actually lives, and whether it is on disk yet.
        best_path = os.path.abspath(self.checkpoints.best_path)
        on_disk = os.path.exists(best_path)
        self.logger.log("model", path=best_path, saved=on_disk,
                        size_mb=round(os.path.getsize(best_path) / 1e6, 2) if on_disk else 0.0,
                        params=self.spec.parameter_count(),
                        checkpoints=os.path.abspath(self.paths.checkpoints))
        self.trainer = Trainer(cfg, self.backend, self.checkpoints, self.logger,
                               global_step=self.global_step)

    # ------------------------------------------------------------- state
    def _load_state(self) -> dict[str, Any]:
        state = read_json(self.paths.state, default=None)
        if not isinstance(state, dict):
            state = {"iteration": 0, "phase": "selfplay", "global_step": 0,
                     "promotions": 0, "started_at": time.time()}
        return state

    def _save_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["global_step"] = self.trainer.global_step if hasattr(self, "trainer") \
            else self.state.get("global_step", 0)
        self.state["updated_at"] = time.time()
        write_json(self.paths.state, self.state)

    def _save_config(self) -> None:
        existing = read_json(self.paths.config, default=None)
        if existing is None:
            self.cfg.save(self.paths.config)
        elif existing != self.cfg.to_dict():
            self.cfg.save(self.paths.config)
            self.logger.log("config.updated", path=self.paths.config)

    # ------------------------------------------------------------- phases
    def _phase_selfplay(self, iteration: int) -> None:
        rng = np.random.default_rng(self.cfg.seed + 1000 * iteration + 1)
        run_selfplay(self.cfg, self.play_backend, iteration,
                     self.cfg.selfplay.games_per_iter, killer=self.killer,
                     logger=self.logger, rng=rng)

    def _phase_train(self, iteration: int) -> None:
        target = self.state.get("train_target_step")
        if target is None:
            target = self.trainer.global_step + self.cfg.train.steps_per_iter
            self._save_state(train_target_step=target)
        steps = max(0, int(target) - self.trainer.global_step)
        if steps == 0:
            return
        buffer = ReplayBuffer(self.cfg.game.board_size, self.cfg.game.win_length,
                              self.cfg.game.allow_overline,
                              self.cfg.train.replay_max_positions)
        buffer.load_dir(self.paths.replay)
        self.logger.log("train.start", iteration=iteration, steps=steps,
                        positions=buffer.size, trainable=buffer.trainable_size)
        self.trainer.train(buffer, steps, killer=self.killer,
                           extra_meta={"iteration": iteration})

    def _phase_arena(self, iteration: int) -> None:
        if self.cfg.arena.games <= 0:  # gating disabled: always take the newest net
            self._promote(iteration, score=None)
            return
        promote, result = evaluate_candidate(self.cfg, self.backend, self.play_backend,
                                             logger=self.logger, killer=self.killer)
        if self.killer.stop and result.games_played < self.cfg.arena.games:
            return  # interrupted -- re-run the arena on resume
        if promote:
            self._promote(iteration, score=result.score_a)
        else:
            self.logger.log("arena.keep_best", iteration=iteration, score=result.score_a)

    def _promote(self, iteration: int, score: float | None) -> None:
        weights = self.backend.get_weights()
        self.checkpoints.save_best(weights, {"step": self.trainer.global_step,
                                             "iteration": iteration,
                                             "score": score,
                                             "spec": self.spec.to_dict()})
        self.play_backend.set_weights(weights)
        self.state["promotions"] = int(self.state.get("promotions", 0)) + 1
        self.logger.log("promote", iteration=iteration, step=self.trainer.global_step,
                        score=score, promotions=self.state["promotions"],
                        path=self.checkpoints.best_path)

    # ------------------------------------------------------------- driver
    def run(self, iterations: int | None = None) -> dict[str, Any]:
        total = iterations if iterations is not None else self.cfg.iterations
        start_iteration = int(self.state.get("iteration", 0))
        end_iteration = start_iteration + total
        self.logger.log("pipeline.start", iteration=start_iteration, until=end_iteration,
                        phase=self.state.get("phase", "selfplay"),
                        run_dir=os.path.abspath(self.paths.root))
        began = time.time()
        recent: list[float] = []  # last few iteration times, for the ETA
        while int(self.state.get("iteration", 0)) < end_iteration:
            if self.killer.stop:
                break
            iteration = int(self.state["iteration"])
            phase = self.state.get("phase", "selfplay")
            iteration_started = time.time()
            for name in PHASES:
                if PHASES.index(name) < PHASES.index(phase):
                    continue
                if self.killer.stop:
                    break
                self._save_state(phase=name)
                getattr(self, f"_phase_{name}")(iteration)
            if self.killer.stop:
                self._save_state()
                break
            iteration_seconds = time.time() - iteration_started
            recent.append(iteration_seconds)
            del recent[:-5]  # average the last few: early iterations are atypical
            per_iteration = sum(recent) / len(recent)
            remaining = max(0, end_iteration - (iteration + 1))
            best_path = os.path.abspath(self.checkpoints.best_path)
            self._save_state(iteration=iteration + 1, phase="selfplay",
                             train_target_step=None,
                             last_iteration_seconds=round(iteration_seconds, 1))
            self.logger.log("iteration.done", iteration=iteration,
                            seconds=round(iteration_seconds, 1),
                            step=self.trainer.global_step,
                            elapsed=human_time(time.time() - began),
                            per_iteration=human_time(per_iteration),
                            remaining=remaining,
                            eta=human_time(per_iteration * remaining),
                            model=best_path,
                            model_saved=os.path.exists(best_path))
        self.trainer.checkpoint({"iteration": int(self.state.get("iteration", 0))})
        self._save_state()
        summary = self.summary()
        self.logger.log("pipeline.stop", **{k: v for k, v in summary.items()
                                            if not isinstance(v, dict)})
        return summary

    def summary(self) -> dict[str, Any]:
        data = dataset_stats(self.paths.replay)
        best = self.checkpoints.best()
        return {
            "run_dir": os.path.abspath(self.paths.root),
            "iteration": int(self.state.get("iteration", 0)),
            "phase": self.state.get("phase", "selfplay"),
            "global_step": self.trainer.global_step,
            "promotions": int(self.state.get("promotions", 0)),
            "games": data["games"],
            "positions": data["positions"],
            "best_step": int(best.meta.get("step", 0)) if best else 0,
            "best_path": os.path.abspath(self.checkpoints.best_path),
            "data": data,
        }

    def close(self) -> None:
        self.logger.close()


def run_pipeline(cfg: Config, iterations: int | None = None) -> dict[str, Any]:
    killer = GracefulKiller()
    pipeline = Pipeline(cfg, killer=killer)
    try:
        return pipeline.run(iterations)
    finally:
        pipeline.close()
        killer.restore()
