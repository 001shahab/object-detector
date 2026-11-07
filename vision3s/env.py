# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""The grounding episode as a finite-horizon MDP (Sec. 3.2–3.3).

State: dialogue history, the current set of views and tool feedback. Action:
one assistant turn, either a single ``<tool_call>`` or an ``<answer>`` commit.
Transitions are deterministic given the scene. The episode terminates on an
answer or when the assistant-turn budget ``T_max`` is reached; the final turn
is reserved for answering, so a tool call on it ends the episode unanswered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .landmarks import LandmarkDB, LandmarkResolver
from .oblique.cache import ObliqueRenderer
from .parsing import parse_answer, parse_tool_call, sanitize_action
from .prompts import SYSTEM_PROMPT, TOOL_NAMES, tool_schemas, user_prompt
from .rewards import DEFAULT_REWARD, RewardConfig, terminal_reward
from .tools import CityToolEnv, ToolConfig
from .views import load_bev


@dataclass
class EnvConfig:
    max_assistant_turns: int = 8
    max_tool_calls: int = 7
    protocol: str = "paper"
    enabled_tools: tuple[str, ...] = TOOL_NAMES
    forced_tool_sequence: Optional[tuple[str, ...]] = None
    display_max_side: int = 768
    bbox_norm_max: float = 1000.0
    data_root: Optional[str] = None
    landmark_db: Optional[str] = None
    landmark_mode: str = "clean"
    landmark_noise_m: float = 10.0
    landmark_noise_kind: str = "fixed"
    landmark_wrong_min_distance_m: float = 50.0
    landmark_seed: int = 42
    oblique_cache: Optional[str] = None
    require_cache: bool = False
    raw_data_root: Optional[str] = None
    reward: RewardConfig = field(default_factory=lambda: DEFAULT_REWARD)

    def __post_init__(self):
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)
        elif isinstance(self.reward, str):
            from .rewards import REWARD_PRESETS
            self.reward = REWARD_PRESETS[self.reward]
        self.enabled_tools = tuple(self.enabled_tools)
        if self.forced_tool_sequence is not None:
            self.forced_tool_sequence = tuple(self.forced_tool_sequence)


@dataclass
class StepResult:
    messages: list
    reward: float
    done: bool
    info: dict


class GroundingEnv:
    """One episode at a time; ``reset(sample)`` then ``step(action)`` until done."""

    def __init__(self, cfg: EnvConfig, landmark_db: Optional[LandmarkDB] = None, oblique=None, bev_loader=None):
        self.cfg = cfg
        db = landmark_db if landmark_db is not None else LandmarkDB.load(cfg.landmark_db)
        self._db = db
        resolver = LandmarkResolver(db, cfg.landmark_mode, cfg.landmark_noise_m, cfg.landmark_noise_kind,
                                    cfg.landmark_wrong_min_distance_m, cfg.landmark_seed)
        if oblique is None and (cfg.oblique_cache or cfg.raw_data_root):
            from .data.ply import scene_point_cloud_loader
            loader = scene_point_cloud_loader(cfg.raw_data_root) if cfg.raw_data_root else None
            oblique = ObliqueRenderer(cfg.oblique_cache, cfg.require_cache, loader)
        self._oblique = oblique
        self._bev_loader = bev_loader
        tool_cfg = ToolConfig(max_tool_calls=cfg.max_tool_calls, display_max_side=cfg.display_max_side,
                              bbox_norm_max=cfg.bbox_norm_max, protocol=cfg.protocol,
                              enabled_tools=tuple(cfg.enabled_tools), reward=cfg.reward)
        self.tools = CityToolEnv(tool_cfg, resolver, oblique,
                                 bev_loader or (lambda s: load_bev(s["bev_png"], cfg.data_root)))
        self.schemas = tool_schemas(cfg.protocol, tuple(cfg.enabled_tools))
        self.messages: list = []
        self.sample: dict = {}
        self.turns = 0
        self.trace: list[dict] = []
        self.tool_reward_sum = 0.0
        self._t0 = 0.0

    @classmethod
    def from_data_root(cls, data_root: str | Path, dataset: str = "cityrefer", **overrides) -> "GroundingEnv":
        """Environment over the standard data layout (see README, "Data layout")."""
        root = Path(data_root)
        cfg = EnvConfig(data_root=str(root), landmark_db=str(root / f"landmarks_{dataset}.json"),
                        oblique_cache=str(root / "verify_region_cache" / dataset), **overrides)
        return cls(cfg)

    @classmethod
    def from_config(cls, cfg: EnvConfig, dataset: str = "cityrefer") -> "GroundingEnv":
        """Fill unset data paths from ``cfg.data_root`` using the standard layout."""
        if cfg.data_root:
            root = Path(cfg.data_root)
            if cfg.landmark_db is None and (root / f"landmarks_{dataset}.json").exists():
                cfg.landmark_db = str(root / f"landmarks_{dataset}.json")
            if cfg.oblique_cache is None and (root / "verify_region_cache" / dataset).exists():
                cfg.oblique_cache = str(root / "verify_region_cache" / dataset)
        return cls(cfg)

    def clone(self) -> "GroundingEnv":
        """A fresh episode object sharing the landmark DB, renderer and BEV cache."""
        return GroundingEnv(self.cfg, self._db, self._oblique, self._bev_loader)

    def reset(self, sample: dict) -> list:
        self.sample = sample
        self.turns = 0
        self.trace = []
        self.tool_reward_sum = 0.0
        self._t0 = time.perf_counter()
        glob = self.tools.reset(sample)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt(sample.get("query", ""), self.cfg.forced_tool_sequence)},
                {"type": "image", "image": glob.image},
            ]},
        ]
        return self.messages

    def step(self, action: str) -> StepResult:
        action = sanitize_action(action or "")
        self.turns += 1
        self.messages.append({"role": "assistant", "content": action})
        last_turn = self.turns >= self.cfg.max_assistant_turns
        call = parse_tool_call(action)
        answer = parse_answer(action)
        schedule = self.cfg.forced_tool_sequence
        if schedule is not None:
            expected = schedule[self.turns - 1] if self.turns - 1 < len(schedule) else None
            got = call[0] if call else None
            if (expected is not None and got != expected) or (expected is None and call is not None):
                return self._terminal(None, "schedule_violation")
        if call is not None and not last_turn:
            return self._tool_step(*call)
        if answer is not None:
            return self._terminal(answer, "answer")
        return self._terminal(None, "no_answer")

    def _tool_step(self, name: str, args: dict) -> StepResult:
        text, images, info = self.tools.call(name, args)
        content = [{"type": "text", "text": text}] + [{"type": "image", "image": im} for im in images if im is not None]
        self.messages.append({"role": "tool", "name": name, "content": content})
        rcfg = self.cfg.reward
        reward = 0.0
        if rcfg.use_tool_shaping:
            reward = -rcfg.tool_error_penalty if "error" in info else float(info.get("tool_reward", 0.0))
        self.tool_reward_sum += reward
        record = {"turn": self.turns, "tool": name, "args": args, "reward": reward, "n_images": len(images),
                  **{k: v for k, v in info.items() if k != "tool_reward"}}
        self.trace.append(record)
        return StepResult(self.messages, reward, False, {"tool": name, **record})

    def _terminal(self, answer: Optional[dict], reason: str) -> StepResult:
        if answer is not None:
            score = self.tools.score(answer["bbox_2d"], str(answer.get("view_id", "global")))
        else:
            score = {"world_bbox": None, "iou_2d": 0.0, "center_dist": float("inf")}
        terms = terminal_reward(score["iou_2d"], score["center_dist"], self.turns, self.cfg.max_assistant_turns,
                                self.cfg.reward)
        info = {
            "reason": reason, "answer": answer, "world_bbox": score["world_bbox"], "iou_2d": score["iou_2d"],
            "center_dist": score["center_dist"], "assistant_turns": self.turns, "terminal": terms,
            "tool_reward_sum": self.tool_reward_sum, "return": terms["reward"] + self.tool_reward_sum,
            "tool_trace": self.trace, "episode_time_s": time.perf_counter() - self._t0,
        }
        return StepResult(self.messages, terms["reward"], True, info)
