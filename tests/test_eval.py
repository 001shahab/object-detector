# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from conftest import CARS, TARGET
from vision3s.eval.diagnostics import aggregate_runs, follows_lzo, search_diagnostics
from vision3s.eval.hit_k import compute_hit_k, load_box3d_cache, overlap_and_area_summary
from vision3s.eval.metrics_3d import evaluate_3d, evaluate_one
from vision3s.geometry import center_to_xyxy
from vision3s.landmarks import LandmarkDB


def _rows(data_root):
    return [json.loads(line) for line in (data_root / "meta" / "cityrefer_test_NO.jsonl").read_text().splitlines()]


def test_hit_k_rules(data_root, sample):
    cands = load_box3d_cache(data_root / "box3d")
    gt = center_to_xyxy(sample["gt_bbox"])
    nx, ny = CARS[TARGET + 1]
    neighbour = [gt[2] - 0.5, ny - 2.2, nx + 1.0, ny + 2.2]   # mostly on the next car, touching the target
    preds = [{**sample, "world_bbox": gt}, {**sample, "world_bbox": neighbour}, {**sample, "world_bbox": None}]
    res = compute_hit_k(preds, _rows(data_root), cands)
    assert res["total"] == 3 and res["no_answer"] == 1
    assert res["hits"][1] == 1 and res["hits"][3] == 2
    assert res["hit1_eta"]["Hit@1_0.25"] == pytest.approx(100 / 3)
    loose = [g + d for g, d in zip(gt, (-3, -3, 3, 3))]
    res = compute_hit_k([{**sample, "world_bbox": loose}], _rows(data_root), cands)
    assert res["hits"][1] == 1 and res["hit1_eta"]["Hit@1_0.5"] == 0.0
    summary = overlap_and_area_summary(res["stats"])
    assert summary["area_ratio_median"] > 3
    assert compute_hit_k(preds, _rows(data_root), cands, score_rule="ciou")["hits"][1] == 1
    assert compute_hit_k(preds, _rows(data_root), cands, require_positive_overlap=True)["hits"][1] == 1


def test_metrics_3d(scene, sample, data_root):
    points, _, _ = scene
    gt = center_to_xyxy(sample["gt_bbox"])
    res = evaluate_one(gt, sample["gt_bbox"], points)
    assert res["iou_bev"] == pytest.approx(1.0)
    assert res["point_f1"] > 0.6 and res["iou_3d"] > 0.5   # ground points under the car lower precision
    agg = evaluate_3d([{**sample, "world_bbox": gt}, {**sample, "world_bbox": None}], _rows(data_root),
                      lambda s: points)
    assert agg["summary"]["evaluated"] == 1 and agg["summary"]["no_world_bbox"] == 1


def test_aggregate_t_interval():
    agg = aggregate_runs([{"Hit@1": v} for v in (25.3, 25.9, 25.4, 26.0, 25.5)])["Hit@1"]
    assert agg["mean"] == pytest.approx(25.62)
    half = 2.776 * agg["std"] / 5 ** 0.5
    assert agg["ci95"][0] == pytest.approx(25.62 - half, rel=1e-3)


def test_lzo_and_search_diagnostics(data_root, sample):
    assert follows_lzo(("get_landmark_xy", "zoom_in", "render_oblique", "render_oblique"))
    assert not follows_lzo(("zoom_in", "get_landmark_xy", "render_oblique"))
    cands = load_box3d_cache(data_root / "box3d")
    gt = center_to_xyxy(sample["gt_bbox"])
    rec = {**sample, "world_bbox": gt, "tool_trace": [
        {"tool": "zoom_in", "crop_world": [gt[0] - 20, gt[1] - 20, gt[2] + 20, gt[3] + 20]}]}
    hk = compute_hit_k([rec], _rows(data_root), cands)
    db = LandmarkDB.load(data_root / "landmarks_cityrefer.json")
    res = search_diagnostics([rec], _rows(data_root), cands, db, hk["stats"])
    q = res["per_query"][0]
    assert q["regional_success"] and q["local_success"]
    assert q["same_class_distractors"] == 4
    assert 80 < q["landmark_distance_m"] < 100
    assert res["regional_search_by_landmark_distance"]["50–100m"] == {"success": 100.0, "n": 1}
    assert res["local_disambiguation_by_distractors"]["4–9"] == {"success": 100.0, "n": 1}
