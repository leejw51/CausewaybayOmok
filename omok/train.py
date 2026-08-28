"""The supervised part of the loop: fit the network to self-play targets."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .checkpoint import CheckpointManager
from .config import Config
from .replay import ReplayBuffer
from .utils import GracefulKiller, Timer


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup into a cosine decay, floored at ``lr_min``."""
    t = cfg.train
    if t.lr_warmup_steps > 0 and step < t.lr_warmup_steps:
        return t.lr * (step + 1) / t.lr_warmup_steps
    if t.lr_decay_steps <= 0:
        return t.lr
    progress = min(1.0, (step - t.lr_warmup_steps) / max(1, t.lr_decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return t.lr_min + (t.lr - t.lr_min) * cosine


@dataclass
class TrainStats:
    steps: int = 0
    seconds: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0

    def as_dict(self) -> dict[str, float]:
        out = {"steps": self.steps, "seconds": round(self.seconds, 2),
               "steps_per_s": round(self.steps_per_second, 2)}
        out.update({k: round(v, 4) for k, v in self.metrics.items()})
        return out


class Trainer:
    """Runs optimisation steps and checkpoints often enough to be crash-proof."""

    def __init__(self, cfg: Config, backend, checkpoints: CheckpointManager,
                 logger=None, global_step: int = 0) -> None:
        self.cfg = cfg
        self.backend = backend
        self.checkpoints = checkpoints
        self.logger = logger
        self.global_step = global_step
        self.rng = np.random.default_rng(cfg.seed + 7919)
        self._last_ckpt_time = time.time()

    def checkpoint(self, extra: dict | None = None, tag: str | None = None) -> str:
        meta = {"step": self.global_step, "config": self.cfg.to_dict(),
                "backend": self.backend.name, "device": self.backend.device,
                "saved_at": time.time()}
        if extra:
            meta.update(extra)
        checkpoint = self.checkpoints.save(self.backend, meta, tag=tag)
        self._last_ckpt_time = time.time()
        if self.logger is not None:
            self.logger.log("checkpoint", step=self.global_step, path=checkpoint.weights_path,
                            mb=round(checkpoint.meta.get("bytes", 0) / 1e6, 2))
        return checkpoint.weights_path

    def train(self, buffer: ReplayBuffer, steps: int,
              killer: GracefulKiller | None = None,
              extra_meta: dict | None = None) -> TrainStats:
        t = self.cfg.train
        stats = TrainStats()
        timer = Timer()
        running: dict[str, float] = {}
        if buffer.trainable_size < t.replay_min_positions:
            if self.logger is not None:
                self.logger.log("train.skip", have=buffer.trainable_size,
                                need=t.replay_min_positions)
            return stats
        for i in range(steps):
            if killer is not None and killer.stop:
                break
            lr = lr_at(self.global_step, self.cfg)
            planes, pi, z = buffer.sample(t.batch_size, self.rng, augment=t.augment)
            metrics = self.backend.train_step(planes, pi, z, lr, t.value_loss_weight,
                                              t.grad_clip)
            self.global_step += 1
            stats.steps += 1
            for key, value in metrics.items():  # smoothed for readable logs
                running[key] = value if key not in running else running[key] * 0.9 + value * 0.1
            if self.logger is not None and (i + 1) % max(1, steps // 10) == 0:
                self.logger.log("train.step", step=self.global_step, lr=round(lr, 6),
                                **{k: round(v, 4) for k, v in running.items()})
            due = (self.global_step % max(1, t.ckpt_every_steps) == 0
                   or time.time() - self._last_ckpt_time >= t.ckpt_every_seconds)
            if due:
                self.checkpoint(extra_meta)
        stats.seconds = timer.elapsed()
        stats.metrics = running
        self.checkpoint(extra_meta)  # always leave a fresh checkpoint behind
        if self.logger is not None:
            self.logger.log("train.done", **stats.as_dict())
        return stats
