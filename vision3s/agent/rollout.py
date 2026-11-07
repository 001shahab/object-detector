# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Batched multi-turn rollouts and trajectory records."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..env import GroundingEnv
from .policy import Policy


@dataclass
class Trajectory:
    sample: dict
    group: int
    messages: list
    tools: list
    step_rewards: list[float] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    generation_time_s: float = 0.0
    wall_time_s: float = 0.0

    @property
    def total_reward(self) -> float:
        """R(τ) = R_term + Σ R_tool (Eq. 3)."""
        return float(sum(self.step_rewards))

    def record(self) -> dict:
        info = self.info
        tool_counts: dict = {}
        for t in info.get("tool_trace", []):
            tool_counts[t["tool"]] = tool_counts.get(t["tool"], 0) + 1
        return {
            "scene_id": self.sample.get("scene_id"), "query": self.sample.get("query"),
            "dataset": self.sample.get("dataset", "cityrefer"), "object_id": self.sample.get("object_id"),
            "group": self.group, "reason": info.get("reason"), "answer": info.get("answer"),
            "world_bbox": info.get("world_bbox"), "iou_2d": info.get("iou_2d", 0.0),
            "center_dist": _finite(info.get("center_dist")), "assistant_turns": info.get("assistant_turns"),
            "reward": self.total_reward, "terminal": info.get("terminal"), "tool_counts": tool_counts,
            "tool_trace": _json_safe(info.get("tool_trace", [])), "usage": self.usage,
            "generation_time_s": self.generation_time_s, "tool_time_s": sum(
                t.get("latency_s", 0.0) for t in info.get("tool_trace", [])), "wall_time_s": self.wall_time_s,
            "messages": strip_images(self.messages),
        }


def _finite(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def strip_images(messages: list) -> list:
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            content = [{"type": "image"} if c.get("type") == "image" else c for c in content]
        out.append({**msg, "content": content})
    return out


def run_rollouts(policy: Policy, env_factory: Callable[[], GroundingEnv], samples: Iterable[dict],
                 group_size: int = 1, max_concurrent: Optional[int] = None) -> list[Trajectory]:
    """Run ``group_size`` episodes per sample in lock-step batches."""
    jobs = [(s, g) for s in samples for g in range(group_size)]
    out: list[Trajectory] = []
    chunk = max_concurrent or len(jobs) or 1
    for start in range(0, len(jobs), chunk):
        out.extend(_run_batch(policy, env_factory, jobs[start:start + chunk]))
    return out


def _run_batch(policy: Policy, env_factory, jobs) -> list[Trajectory]:
    envs, trajs = [], []
    t0 = time.perf_counter()
    for sample, g in jobs:
        env = env_factory()
        messages = env.reset(sample)
        envs.append(env)
        trajs.append(Trajectory(sample, g, messages, env.schemas))
    active = list(range(len(envs)))
    while active:
        forced = []
        for i in active:
            seq = envs[i].cfg.forced_tool_sequence
            turn = envs[i].turns
            forced.append(seq[turn] if seq and turn < len(seq) else None)
        outputs = policy.act([envs[i].messages for i in active], envs[active[0]].schemas, forced)
        still = []
        for i, out in zip(active, outputs):
            traj = trajs[i]
            for k in traj.usage:
                traj.usage[k] += int(out.usage.get(k, 0))
            traj.generation_time_s += out.latency_s
            result = envs[i].step(out.text)
            traj.step_rewards.append(result.reward)
            if result.done:
                traj.info = result.info
                traj.wall_time_s = time.perf_counter() - t0
            else:
                still.append(i)
        active = still
    return trajs


def write_records(trajectories: list[Trajectory], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in trajectories:
            fh.write(json.dumps(t.record(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
