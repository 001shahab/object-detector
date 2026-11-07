# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from conftest import SCENE
from vision3s.data.cityrefer import split_records
from vision3s.data.stpls3d_refer import dbscan_instances, filter_expressions, fit_georegistration
from vision3s.landmarks import LandmarkDB, LandmarkResolver
from vision3s.oblique import ObliqueRenderer, SceneCache, build_scene_cache, render_views
from vision3s.oblique.camera import backproject, look_at, project
from vision3s.oblique.renderer import PointGrid
from vision3s.parsing import parse_answer, parse_tool_call


def test_parsing():
    assert parse_tool_call('x <tool_call>{"name": "zoom_in", "arguments": {"view_id": "global"}}</tool_call>') == (
        "zoom_in", {"view_id": "global"})
    assert parse_tool_call('<tool_call>{"name": "zoom_in", "arguments": "{\\"a\\": 1}"}</tool_call>') == ("zoom_in", {"a": 1})
    assert parse_tool_call("<tool_call>{bad}</tool_call>") is None
    assert parse_answer('<answer>{"bbox_2d": [1, 2, 3, 4], "view_id": "Z1"}</answer>')["view_id"] == "Z1"
    assert parse_answer('<answer>{"bbox_2d": [1, 2, 3]}</answer>') is None


def test_landmark_modes():
    db = LandmarkDB({"s": [{"name": "Red Tower", "xy": [0, 0]}, {"name": "Blue Hall", "xy": [80, 0]},
                           {"name": "Green Park", "xy": [20, 0]}]})
    assert LandmarkResolver(db).resolve("s", "red tower").xy == (0.0, 0.0)
    assert LandmarkResolver(db).resolve("s", "tower").name == "Red Tower"
    assert LandmarkResolver(db, "missing").resolve("s", "Red Tower") is None
    noisy = LandmarkResolver(db, "noisy").resolve("s", "Red Tower", sample_query="q")
    assert math.hypot(*noisy.xy) == pytest.approx(10.0)
    again = LandmarkResolver(db, "noisy").resolve("s", "Red Tower", sample_query="q")
    assert again.xy == noisy.xy
    wrong = LandmarkResolver(db, "wrong").resolve("s", "Red Tower")
    assert wrong.name == "Blue Hall"
    assert db.mentioned_in("s", "the car near red tower.")[0]["name"] == "Red Tower"


def test_camera_roundtrip():
    ext, K = look_at(np.array([0.0, -50.0, 30.0]), np.array([0.0, 0.0, 0.0]), 384, 40.0)
    pts = np.array([[1.0, 2.0, 0.5], [-3.0, 1.0, 2.0]])
    uv, z = project(pts, ext, K)
    depth = np.zeros((384, 384), np.float32)
    for (u, v), d in zip(np.round(uv).astype(int), z):
        depth[v, u] = d
    back, valid = backproject(depth, ext, K, (384, 384))
    got = back[valid]
    assert len(got) == 2
    assert np.allclose(got[np.argsort(got[:, 0])], pts[np.argsort(pts[:, 0])], atol=0.2)


def test_render_and_cache(scene, tmp_path):
    points, colors, boxes = scene
    grid = PointGrid(points, colors)
    out = render_views(grid, [95, 50, 130, 70])
    assert len(out["views"]) == 4
    arr = np.asarray(out["views"][3])
    assert (arr < 250).any(axis=2).mean() > 0.3
    sub = (points[:, 0] < 60) & (points[:, 1] < 60)
    info = build_scene_cache(SCENE, "cityrefer", points[sub], colors[sub], tmp_path, region_size=40, step=20,
                             output_size=128)
    assert info["status"] == "built" and info["regions"] > 0
    image = SceneCache.get(tmp_path, SCENE).lookup([20, 20, 45, 40], "north")
    assert image is not None and image.size == (128, 128)
    renderer = ObliqueRenderer(str(tmp_path), require_cache=True)
    img, meta = renderer.render(SCENE, "cityrefer", [20, 20, 45, 40], "west")
    assert meta["source"] == "cache"
    img, meta = renderer.render(SCENE, "cityrefer", [180, 180, 190, 190], "west")
    assert img is None and meta["source"] == "cache_miss"


def test_cityrefer_split():
    records = [{"scene_id": f"s{i}", "query": f"q{i}_{j}"} for i in range(20) for j in range(5)]
    splits = split_records(records, seed=0)
    no = {r["scene_id"] for r in splits["test_NO"]}
    assert len(no) == 7
    seen = {r["scene_id"] for k in ("train", "val", "test_ND") for r in splits[k]}
    assert not (no & seen)
    n = sum(len(splits[k]) for k in ("train", "val", "test_ND"))
    assert len(splits["train"]) == pytest.approx(0.625 * n, abs=2)
    assert len(splits["val"]) == pytest.approx(0.125 * n, abs=2)


def test_stpls3d_tools():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 1], 0.3, (400, 3))
    b = rng.normal([10, 0, 1], 0.3, (400, 3))
    labels = dbscan_instances(np.r_[a, b], eps=0.6, min_samples=30, dims=3)
    assert len(set(labels[labels >= 0])) == 2
    gcps = [{"x": x, "y": y, "lat": 34.0 + y / 111_000, "lon": -118.0 + x / 92_000}
            for x, y in [(0, 0), (100, 0), (0, 100), (100, 100), (50, 30)]]
    reg = fit_georegistration(gcps)
    back = reg.wgs84_to_local(reg.local_to_wgs84(np.array([[40.0, 60.0]])))
    assert np.allclose(back, [[40.0, 60.0]], atol=1e-6) and reg.rms_m < 5
    ctx = {"category": "building"}
    res = filter_expressions(["The tall white building just north of Doheny Library with a flat roof.",
                              "The building marked in red in the image near Doheny Library.",
                              "The tall white building just north of Doheny Library, with a flat roof.",
                              "The grey building beside Unknown Plaza and the main road."],
                             ctx, {"Doheny Library"})
    assert [r["ok"] for r in res] == [True, False, False, False]
    assert "meta_language" in res[1]["reasons"] and "duplicate" in res[2]["reasons"]
