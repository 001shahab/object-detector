# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Reward functions of Sec. 3.3.

Terminal reward (Eq. 2)::

    R_term = λ_iou · IoU(b̂, b*) + λ_dist · exp(-d(b̂, b*) / τ) - λ_step · T / T_max

Trajectory return (Eq. 3)::

    R(τ) = R_term + Σ_t R_tool(s_t, a_t)

Tool-level shaping (privileged, never shown to the agent):

* Zoom-in: signed progress of the zoom centre towards the target relative to
  the previous zoom, capped at ±``ZOOM_REWARD_CAP``.
* Oblique Rendering: proportional to the overlap between the rendered
  candidate region and the target, capped at ``RENDER_REWARD_CAP``.
* Invalid tool names, malformed arguments, impossible boxes or execution
  errors: ``-TOOL_ERROR_PENALTY``.
* Landmark Query: neutral.

There is no discrete success bonus. Coefficients are those of Appendix A.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    lambda_iou: float = 1.0
    lambda_dist: float = 0.3
    tau_m: float = 80.0
    lambda_step: float = 0.05
    zoom_progress_scale_m: float = 50.0
    zoom_reward_cap: float = 0.05
    render_iou_weight: float = 0.05
    render_reward_cap: float = 0.05
    tool_error_penalty: float = 0.05
    use_distance: bool = True
    use_tool_shaping: bool = True
    use_step_cost: bool = True


DEFAULT_REWARD = RewardConfig()

# Table 4 reward ablation. Every variant keeps the terminal IoU term and the
# step cost; "tool-level" covers zoom progress, render overlap and the
# invalid-call penalty.
REWARD_PRESETS = {
    "iou_only": RewardConfig(use_distance=False, use_tool_shaping=False),
    "iou_distance": RewardConfig(use_distance=True, use_tool_shaping=False),
    "iou_tool": RewardConfig(use_distance=False, use_tool_shaping=True),
    "full": RewardConfig(),
}


def terminal_reward(iou: float, center_distance: float, turns: int, max_turns: int,
                    cfg: RewardConfig = DEFAULT_REWARD) -> dict:
    """Eq. 2. Returns the total and its three terms (for logging)."""
    iou_term = cfg.lambda_iou * float(iou)
    dist_term = 0.0
    if cfg.use_distance and math.isfinite(center_distance):
        dist_term = cfg.lambda_dist * math.exp(-float(center_distance) / cfg.tau_m)
    step_term = 0.0
    if cfg.use_step_cost:
        step_term = cfg.lambda_step * min(int(turns), max_turns) / max(1, max_turns)
    return {
        "reward": iou_term + dist_term - step_term,
        "iou_term": iou_term,
        "dist_term": dist_term,
        "step_term": step_term,
    }


def zoom_progress_reward(prev_dist: float | None, curr_dist: float,
                         cfg: RewardConfig = DEFAULT_REWARD) -> float:
    """Positive when the zoom centre moves closer to the target than the previous zoom."""
    if prev_dist is None:
        return 0.0
    delta = (prev_dist - curr_dist) / cfg.zoom_progress_scale_m
    cap = cfg.zoom_reward_cap
    return max(-cap, min(cap, cap * delta))


def render_reward(candidate_iou: float, cfg: RewardConfig = DEFAULT_REWARD) -> float:
    return min(cfg.render_reward_cap, cfg.render_iou_weight * max(0.0, float(candidate_iou)))
