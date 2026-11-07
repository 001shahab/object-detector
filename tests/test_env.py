# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from conftest import CARS, LANDMARK, TARGET
from vision3s.agent import ScriptedPolicy, run_rollouts
from vision3s.env import GroundingEnv
from vision3s.geometry import center_to_xyxy
from vision3s.oblique import ObliqueRenderer
from vision3s.parsing import format_tool_call
from vision3s.prompts import FIXED_WORKFLOW, tool_schemas
from vision3s.rewards import REWARD_PRESETS


def make_env(data_root, scene, **kw):
    points, colors, _ = scene
    env = GroundingEnv.from_data_root(data_root, **kw)
    env.tools.oblique = ObliqueRenderer(None, False, lambda s, d: (points, colors))
    env._oblique = env.tools.oblique
    return env


def answer(bbox, view_id):
    return f'<answer>{{"bbox_2d": {[round(v) for v in bbox]}, "view_id": "{view_id}"}}</answer>'


def lzo_script(env, sample):
    """Landmark Query -> Zoom-in around the target -> Oblique Rendering -> answer in Z1."""
    gt = center_to_xyxy(sample["gt_bbox"])

    def fn(conv):
        turn = env.turns
        if turn == 0:
            return format_tool_call("get_landmark_xy", {"names": [LANDMARK["name"]]})
        if turn == 1:
            g = env.tools.views["global"]
            region = g.world_xyxy_to_norm([gt[0] - 25, gt[1] - 25, gt[2] + 25, gt[3] + 25])
            return format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [round(v) for v in region]})
        z1 = env.tools.views["Z1"]
        if turn == 2:
            return format_tool_call("render_oblique", {"view_id": "Z1", "bbox_2d": [round(v) for v in z1.world_xyxy_to_norm(gt)],
                                                       "direction": "south"})
        return "The blue car is confirmed. " + answer(z1.world_xyxy_to_norm(gt), "Z1")

    return fn


def test_lzo_episode_rewards(data_root, scene, sample):
    env = make_env(data_root, scene)
    env.reset(sample)
    policy = ScriptedPolicy(lzo_script(env, sample))
    rewards, result = [], None
    while True:
        result = env.step(policy.act([env.messages], env.schemas)[0].text)
        rewards.append(result.reward)
        if result.done:
            break
    info = result.info
    assert info["reason"] == "answer"
    assert [t["tool"] for t in info["tool_trace"]] == list(FIXED_WORKFLOW)
    assert info["iou_2d"] > 0.6
    assert rewards[0] == 0.0                      # landmark lookup is neutral
    assert rewards[1] == 0.0                      # first zoom has no progress reference
    assert 0.0 < rewards[2] <= 0.05               # render reward min(0.05, 0.05 * IoU)
    expected = info["iou_2d"] + 0.3 * math.exp(-info["center_dist"] / 80.0) - 0.05 * 4 / 8
    assert rewards[3] == pytest.approx(expected)
    assert info["return"] == pytest.approx(sum(rewards))
    tool_msgs = [m for m in env.messages if m["role"] == "tool"]
    assert "world (60.0,140.0)" in tool_msgs[0]["content"][0]["text"]
    assert sum(c["type"] == "image" for c in tool_msgs[1]["content"]) == 1
    assert sum(c["type"] == "image" for c in tool_msgs[2]["content"]) == 1


def test_tool_call_on_last_turn_is_no_answer(data_root, scene, sample):
    env = make_env(data_root, scene)
    env.reset(sample)
    call = format_tool_call("get_landmark_xy", {"names": ["Red Tower"]})
    for _ in range(7):
        assert not env.step(call).done
    res = env.step(call)
    assert res.done and res.info["reason"] == "no_answer" and res.info["iou_2d"] == 0.0
    assert res.info["assistant_turns"] == 8 and len(res.info["tool_trace"]) == 7


def test_tool_errors_are_penalised(data_root, scene, sample):
    env = make_env(data_root, scene)
    env.reset(sample)
    res = env.step(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [500, 500, 400, 600]}))
    assert res.reward == pytest.approx(-0.05) and res.info["error"] == "invalid_bbox"
    res = env.step(format_tool_call("zoom_in", {"view_id": "nope", "bbox_2d": [0, 0, 10, 10]}))
    assert res.info["error"] == "unknown_view"
    res = env.step(format_tool_call("teleport", {}))
    assert res.info["error"] == "unknown_tool"


def test_zoom_limits(data_root, scene, sample):
    env = make_env(data_root, scene)
    env.reset(sample)
    env.step(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [200, 200, 800, 800]}))
    res = env.step(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [205, 205, 805, 805]}))
    assert res.info["error"] == "duplicate_zoom"
    env.step(format_tool_call("zoom_in", {"view_id": "Z1", "bbox_2d": [100, 100, 900, 900]}))
    res = env.step(format_tool_call("zoom_in", {"view_id": "Z2", "bbox_2d": [100, 100, 900, 900]}))
    assert res.info["error"] == "max_depth"
    res = env.step(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [500, 500, 520, 520]}))
    assert res.info["error"] == "too_small"


def test_zoom_progress_reward(data_root, scene, sample):
    env = make_env(data_root, scene)
    env.reset(sample)
    g = env.tools.views["global"]
    cx, cy = CARS[TARGET]
    far = g.world_xyxy_to_norm([cx + 30, cy + 60, cx + 70, cy + 100])
    near = g.world_xyxy_to_norm([cx - 30, cy - 30, cx + 30, cy + 30])
    env.step(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [round(v) for v in far]}))
    res = env.step(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [round(v) for v in near]}))
    assert res.reward == pytest.approx(0.05)


def test_fixed_workflow_and_violation(data_root, scene, sample):
    env = make_env(data_root, scene, forced_tool_sequence=FIXED_WORKFLOW)
    env.reset(sample)
    assert "get_landmark_xy -> zoom_in -> render_oblique" in env.messages[1]["content"][0]["text"]
    res = env.step(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [0, 0, 500, 500]}))
    assert res.done and res.info["reason"] == "schedule_violation"


def test_disabled_tools_and_presets(data_root, scene, sample):
    env = make_env(data_root, scene, enabled_tools=("zoom_in",), reward=REWARD_PRESETS["iou_only"])
    env.reset(sample)
    assert [t["function"]["name"] for t in env.schemas] == ["zoom_in"]
    res = env.step(format_tool_call("get_landmark_xy", {"names": ["Red Tower"]}))
    assert res.info["error"] == "unknown_tool" and res.reward == 0.0


def test_release_protocol_schema():
    paper = {t["function"]["name"]: t for t in tool_schemas("paper")}
    release = {t["function"]["name"]: t for t in tool_schemas("release")}
    assert set(paper) == set(release)
    assert "landmark_type" in paper["get_landmark_xy"]["function"]["parameters"]["properties"]
    assert "landmark_type" not in release["get_landmark_xy"]["function"]["parameters"]["properties"]


def test_evaluation_runner(data_root, scene, sample, tmp_path):
    from vision3s.eval.hit_k import load_box3d_cache
    from vision3s.eval.runner import evaluate

    points = scene[0]
    base = make_env(data_root, scene)
    envs = []

    def factory():
        envs.append(base.clone())
        return envs[-1]

    def fn(conv):
        env = next(e for e in envs if e.messages is conv)
        return lzo_script(env, sample)(conv)

    summary = evaluate(lambda run: ScriptedPolicy(fn), factory, [sample], tmp_path, num_runs=2,
                       candidates=load_box3d_cache(data_root / "box3d"), load_points=lambda s: points)
    agg = summary["aggregate"]
    assert agg["Hit@1"]["mean"] == 100.0 and agg["Hit@1"]["n"] == 2
    assert agg["iou_bev_pct"]["mean"] > 60
    assert (tmp_path / "run_02" / "records.jsonl").exists() and (tmp_path / "summary.json").exists()


def test_batched_rollouts(data_root, scene, sample):
    base = make_env(data_root, scene)
    envs = []

    def factory():
        e = base.clone()
        envs.append(e)
        return e

    def fn(conv):
        env = next(e for e in envs if e.messages is conv)
        return lzo_script(env, sample)(conv)

    trajs = run_rollouts(ScriptedPolicy(fn), factory, [sample], group_size=3)
    assert len(trajs) == 3
    assert all(t.info["reason"] == "answer" for t in trajs)
    rec = trajs[0].record()
    assert rec["tool_counts"] == {"get_landmark_xy": 1, "zoom_in": 1, "render_oblique": 1}
    assert all(c.get("type") != "image" or "image" not in c for m in rec["messages"]
               if isinstance(m["content"], list) for c in m["content"])
