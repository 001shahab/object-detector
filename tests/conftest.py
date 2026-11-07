# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Synthetic city block: ground, three buildings (one named landmark) and a row of cars."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vision3s.data.bev import build_scene_bev

SCENE = "synthetic_block_1"
LANDMARK = {"name": "Red Tower", "xy": [60.0, 140.0], "type": "building",
            "footprint": [[50, 130], [70, 130], [70, 150], [50, 150]]}
CARS = [(100.0 + 6.0 * i, 60.0) for i in range(5)]
TARGET = 2


def _box_points(rng, x0, y0, x1, y1, z0, z1, color, n):
    pts = np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)]
    top = np.c_[rng.uniform(x0, x1, n // 2), rng.uniform(y0, y1, n // 2), np.full(n // 2, z1)]
    pts = np.r_[pts, top]
    return pts, np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1))


def make_scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(0, 200, 0.5), np.arange(0, 200, 0.5))
    ground = np.c_[gx.ravel(), gy.ravel(), rng.normal(0, 0.02, gx.size)]
    parts = [(ground, np.tile(np.array([120, 120, 120], np.uint8), (len(ground), 1)))]
    parts.append(_box_points(rng, 50, 130, 70, 150, 0, 30, (200, 30, 30), 6000))
    parts.append(_box_points(rng, 130, 130, 160, 160, 0, 12, (200, 200, 200), 6000))
    parts.append(_box_points(rng, 20, 20, 45, 40, 0, 8, (180, 160, 120), 4000))
    boxes = []
    for i, (cx, cy) in enumerate(CARS):
        color = (30, 60, 220) if i == TARGET else (235, 235, 235)
        parts.append(_box_points(rng, cx - 1.0, cy - 2.2, cx + 1.0, cy + 2.2, 0.2, 1.6, color, 800))
        boxes.append({"object_id": i, "object_name": "car", "bbox": [cx, cy, 0.9, 2.0, 4.4, 1.4, 1, i]})
    boxes.append({"object_id": 10, "object_name": "building", "bbox": [60, 140, 15, 20, 20, 30, 0, 10]})
    boxes.append({"object_id": 11, "object_name": "building", "bbox": [145, 145, 6, 30, 30, 12, 0, 11]})
    boxes.append({"object_id": 12, "object_name": "building", "bbox": [32.5, 30, 4, 25, 20, 8, 0, 12]})
    points = np.concatenate([p for p, _ in parts]).astype(np.float32)
    colors = np.concatenate([c for _, c in parts])
    return points, colors, boxes


@pytest.fixture(scope="session")
def scene():
    return make_scene()


@pytest.fixture(scope="session")
def data_root(tmp_path_factory, scene):
    points, colors, boxes = scene
    root = tmp_path_factory.mktemp("data") / "cityrefer"
    build_scene_bev(SCENE, points, colors, root / "bev_cityrefer", resolution=0.1)
    (root / "landmarks_cityrefer.json").write_text(json.dumps({SCENE: [LANDMARK]}))
    (root / "box3d").mkdir(parents=True)
    (root / "box3d" / f"{SCENE}_bbox.json").write_text(json.dumps({"scene_id": SCENE, "bboxes": boxes}))
    sample = {"dataset": "cityrefer", "scene_id": SCENE, "object_id": str(TARGET), "object_name": "car",
              "query": "The blue car in the middle of the row of five cars south-east of the Red Tower.",
              "gt_bbox": boxes[TARGET]["bbox"], "bev_png": f"bev_cityrefer/{SCENE}.png"}
    (root / "meta").mkdir()
    (root / "meta" / "cityrefer_test_NO.jsonl").write_text(json.dumps(sample) + "\n")
    return root


@pytest.fixture()
def sample(data_root):
    return json.loads((data_root / "meta" / "cityrefer_test_NO.jsonl").read_text().splitlines()[0])
