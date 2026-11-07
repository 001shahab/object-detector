# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""3s-vision: an implementation of Urbex, agentic spatial grounding for city-scale 3D scenes.

Method: Wang et al., "Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes",
arXiv:2610.18356, 2026. Implementation by 3S Holding OÜ.
"""

from .env import EnvConfig, GroundingEnv, StepResult
from .rewards import DEFAULT_REWARD, REWARD_PRESETS, RewardConfig
from .tools import CityToolEnv, ToolConfig

__version__ = "1.0.0"

__all__ = ["CityToolEnv", "DEFAULT_REWARD", "EnvConfig", "GroundingEnv", "REWARD_PRESETS", "RewardConfig",
           "StepResult", "ToolConfig", "__version__"]
