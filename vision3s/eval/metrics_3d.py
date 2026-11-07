# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# The 2D-to-3D lifting protocol follows the Urbex reference release
# (https://github.com/AMAP-ML/Urbex, Apache-2.0).
"""Candidate-free localization metrics: 2D (BEV) IoU, 3D IoU and point F1.

The model outputs a BEV box. It is lifted to 3D by cropping scene points whose
XY lies inside the box and taking their axis-aligned bounds; the vertical
extent therefore comes from observed geometry. Point precision / recall
compare the points inside the lifted box with those inside the GT box.
When several GT rows share a query, the best BEV-IoU match is reported.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from ..geometry import center_to_minmax3d, iou_minmax3d, iou_xyxy


def points_in_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    return ((points[:, 0] >= box[0]) & (points[:, 0] <= box[3]) & (points[:, 1] >= box[1]) & (points[:, 1] <= box[4])
            & (points[:, 2] >= box[2]) & (points[:, 2] <= box[5]))


def giou_3d(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.maximum(0.0, np.minimum(a[3:], b[3:]) - np.maximum(a[:3], b[:3])).prod()
    va, vb = np.maximum(0.0, a[3:] - a[:3]).prod(), np.maximum(0.0, b[3:] - b[:3]).prod()
    union = va + vb - inter
    if union < 1e-9:
        return 0.0
    enc = max(1e-9, np.maximum(0.0, np.maximum(a[3:], b[3:]) - np.minimum(a[:3], b[:3])).prod())
    return float(inter / union - (enc - union) / enc)


def evaluate_one(world_bbox, gt_bbox, points: np.ndarray) -> dict:
    pred2d = [float(v) for v in world_bbox[:4]]
    gt3d = np.asarray(center_to_minmax3d(gt_bbox[:6]), dtype=np.float64)
    gt2d = [gt3d[0], gt3d[1], gt3d[3], gt3d[4]]
    bev = iou_xyxy(pred2d, gt2d)
    gcx, gcy = (gt3d[0] + gt3d[3]) / 2, (gt3d[1] + gt3d[4]) / 2
    cd2 = math.hypot((pred2d[0] + pred2d[2]) / 2 - gcx, (pred2d[1] + pred2d[3]) / 2 - gcy)
    xy = points[:, :2]
    sel = (xy[:, 0] >= pred2d[0]) & (xy[:, 0] <= pred2d[2]) & (xy[:, 1] >= pred2d[1]) & (xy[:, 1] <= pred2d[3])
    cropped = points[sel]
    if len(cropped) == 0:
        return {"iou_bev": bev, "iou_3d": 0.0, "iou_3d_zclip": 0.0, "giou_3d": -1.0, "center_dist_2d": cd2,
                "center_dist_3d": float("inf"), "point_precision": 0.0, "point_recall": 0.0, "point_f1": 0.0,
                "n_points_cropped": 0, "pred_3d_bbox": None}
    pred3d = np.concatenate([cropped.min(0), cropped.max(0)]).astype(np.float64)
    zclip = pred3d.copy()
    zclip[2], zclip[5] = gt3d[2], gt3d[5]
    in_gt = points_in_box(points, gt3d)
    n_gt = int(in_gt.sum())
    recall = float(points_in_box(points[in_gt], pred3d).sum()) / n_gt if n_gt else 0.0
    precision = float(points_in_box(cropped, gt3d).sum()) / len(cropped)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 1e-9 else 0.0
    return {"iou_bev": bev, "iou_3d": iou_minmax3d(pred3d, gt3d), "iou_3d_zclip": iou_minmax3d(zclip, gt3d),
            "giou_3d": giou_3d(pred3d, gt3d), "center_dist_2d": cd2,
            "center_dist_3d": float(np.linalg.norm((pred3d[:3] + pred3d[3:]) / 2 - (gt3d[:3] + gt3d[3:]) / 2)),
            "point_precision": precision, "point_recall": recall, "point_f1": f1,
            "n_points_cropped": int(len(cropped)), "pred_3d_bbox": pred3d.tolist()}


METRICS = ("iou_bev", "iou_3d", "iou_3d_zclip", "giou_3d", "center_dist_2d", "center_dist_3d",
           "point_precision", "point_recall", "point_f1")


def evaluate_3d(preds: list[dict], dataset: list[dict], load_points: Callable[[str], Optional[np.ndarray]],
                no_answer_as_zero: bool = False, iou_thresholds=(0.25, 0.5)) -> dict:
    """Aggregate metrics; values in ``[0, 1]`` (multiply by 100 for the paper's tables).

    By default unanswered predictions are skipped, as in the reference
    evaluation; ``no_answer_as_zero`` counts them as zero overlap instead.
    """
    gts = defaultdict(list)
    for r in dataset:
        gts[(r["scene_id"], r.get("query") or r.get("description"))].append(r)
    per_sample, n_no_box, n_no_gt, n_no_points = [], 0, 0, 0
    for pred in preds:
        rows = gts.get((pred.get("scene_id"), pred.get("query")))
        if not rows:
            n_no_gt += 1
            continue
        wb = pred.get("world_bbox")
        if not wb or len(wb) < 4:
            n_no_box += 1
            if no_answer_as_zero:
                per_sample.append({"scene_id": pred["scene_id"], "query": pred["query"], "iou_bev": 0.0, "iou_3d": 0.0,
                                   "iou_3d_zclip": 0.0, "giou_3d": -1.0, "point_precision": 0.0, "point_recall": 0.0,
                                   "point_f1": 0.0, "center_dist_2d": float("inf"), "center_dist_3d": float("inf")})
            continue
        points = load_points(pred["scene_id"])
        if points is None or len(points) == 0:
            n_no_points += 1
            continue
        best = None
        for gt in rows:
            res = evaluate_one(wb, gt["gt_bbox"], points)
            if best is None or res["iou_bev"] > best["iou_bev"]:
                best = {**res, "matched_object_id": gt.get("object_id")}
        per_sample.append({"scene_id": pred["scene_id"], "query": pred["query"], "pred_world_bbox": list(wb), **best})
    summary = {"evaluated": len(per_sample), "no_world_bbox": n_no_box, "no_gt": n_no_gt, "no_points": n_no_points}
    for m in METRICS:
        vals = [s[m] for s in per_sample if m in s and math.isfinite(s[m])]
        if vals:
            summary[m] = statistics.mean(vals)
            summary[f"{m}_median"] = statistics.median(vals)
    for thr in iou_thresholds:
        summary[f"acc_bev@{thr}"] = (sum(s["iou_bev"] >= thr for s in per_sample) / len(per_sample)) if per_sample else 0.0
        summary[f"acc_3d@{thr}"] = (sum(s["iou_3d"] >= thr for s in per_sample) / len(per_sample)) if per_sample else 0.0
    for d in (5, 10, 20):
        cds = [s["center_dist_2d"] for s in per_sample]
        summary[f"acc_dist2d@{d}m"] = (sum(c <= d for c in cds) / len(cds)) if cds else 0.0
    return {"summary": summary, "per_sample": per_sample}
