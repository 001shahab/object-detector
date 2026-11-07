# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from vision3s.rewards import REWARD_PRESETS, render_reward, terminal_reward, zoom_progress_reward
from vision3s.rl.grpo import assistant_token_mask, group_advantages, grpo_token_loss, kl_penalty


def test_terminal_reward_eq2():
    r = terminal_reward(0.5, 40.0, 4, 8)
    assert r["reward"] == pytest.approx(0.5 + 0.3 * math.exp(-0.5) - 0.025)
    assert terminal_reward(0.0, float("inf"), 8, 8)["reward"] == pytest.approx(-0.05)
    assert terminal_reward(0.5, 40.0, 4, 8, REWARD_PRESETS["iou_only"])["reward"] == pytest.approx(0.475)


def test_tool_shaping():
    assert zoom_progress_reward(None, 10.0) == 0.0
    assert zoom_progress_reward(100.0, 75.0) == pytest.approx(0.025)
    assert zoom_progress_reward(100.0, 0.0) == pytest.approx(0.05)
    assert zoom_progress_reward(0.0, 500.0) == pytest.approx(-0.05)
    assert render_reward(0.4) == pytest.approx(0.02)
    assert render_reward(2.0) == pytest.approx(0.05)


def test_group_advantages_eq4():
    adv = group_advantages([1.0, 0.0, 0.5, 0.5, 2.0, 2.0], ["a", "a", "a", "a", "b", "b"])
    a = torch.tensor(adv[:4])
    assert a.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert a.std().item() == pytest.approx(1.0, rel=1e-4)
    assert adv[4:] == [0.0, 0.0]


def test_grpo_loss_eq5():
    logp = torch.log(torch.tensor([0.5, 0.4, 0.3])).requires_grad_()
    loss, stats = grpo_token_loss(logp, logp.detach(), logp.detach(), advantage=1.0, kl_coef=0.001)
    loss.backward()
    assert loss.item() == pytest.approx(-1.0)
    assert stats["kl"] == pytest.approx(0.0)
    assert (logp.grad < 0).all()            # minimising -J increases likelihood for A > 0
    old = logp.detach() - 1.0               # ratio e^1 > 1 + eps: clipped for A > 0
    _, stats = grpo_token_loss(logp.detach(), old, None, advantage=1.0)
    assert stats["clipfrac"] == 1.0


def test_k3_kl_nonnegative():
    a, b = torch.randn(100), torch.randn(100)
    assert (kl_penalty(a, b) >= 0).all()


def test_assistant_mask():
    S, E, A, NL = 1, 2, 7, 9
    ids = torch.tensor([S, 5, NL, 3, 3, E, S, A, NL, 40, 41, E, S, 6, NL, 50, E, S, A, NL, 60, E])
    m = assistant_token_mask(ids, S, [A, NL], E)
    assert ids[m].tolist() == [40, 41, E, 60, E]
