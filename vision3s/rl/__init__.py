# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn GRPO (Sec. 3.3). The trainer requires the ``train`` extra."""

from .grpo import assistant_token_mask, group_advantages, grpo_token_loss, kl_penalty

__all__ = ["assistant_token_mask", "group_advantages", "grpo_token_loss", "kl_penalty"]
