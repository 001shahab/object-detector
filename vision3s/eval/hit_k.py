# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# Candidate sources, ranking rules and denominator semantics follow the Urbex
# reference release (https://github.com/AMAP-ML/Urbex, Apache-2.0).
"""With-GT evaluation of free-form predictions (Sec. 4.1, Appendix D.5).

For a free-form prediction ``b̂`` and GT candidate boxes ``{b_i}``, candidates
are ranked in descending order of ``IoU_BEV(b̂, b_i)``; a hit at ``k`` occurs
when the target identity is among the top ``k``. GT boxes are used only for
this post-inference scoring. Variants: positive-overlap filtering and CIoU
ranking (Table 12), the target-IoU-thresholded ``Hit@1_η`` of Eq. 6 (Table 13),
candidate-overlap counts (Table 14) and predicted-to-target area ratio
(Table 15). Predictions without a committed box count as misses and stay in
the denominator; pipeline gaps (no GT row, no candidates) are reported
separately.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Optional

from ..geometry import area_xyxy, ciou_xyxy, iou_xyxy

SCORE_RULES = ("iou", "ciou", "center_distance")


def box3d_to_xyxy(b) -> list[float]:
    cx, cy, dx, dy = float(b[0]), float(b[1]), float(b[3]), float(b[4])
    return [cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2]


def load_box3d_cache(box3d_dir: str | Path) -> dict[str, list[dict]]:
    cache = {}
    for path in sorted(Path(box3d_dir).glob("*_bbox.json")):
        data = json.loads(path.read_text())
        cache[data["scene_id"]] = data["bboxes"]
    return cache


def load_pth_cache(pth_dir: str | Path) -> dict[str, list[dict]]:
    """CityAnchor-style ``.pth`` instances: ``d[6]`` boxes, ``d[9]`` = [offset_x, offset_y, scale]."""
    import torch

    cache = {}
    for path in sorted(Path(pth_dir).glob("*.pth")):
        d = torch.load(path, map_location="cpu", weights_only=False)
        ox, oy = float(d[9][0]), float(d[9][1])
        cache[path.stem] = [{"object_id": int(b[7]), "object_name": "",
                             "bbox": [float(b[0]) + ox, float(b[1]) + oy, *map(float, b[2:8])]}
                            for b in d[6].values()]
    return cache


def load_proposal_cache(path: str | Path) -> dict[str, list[dict]]:
    """Predicted instance proposals as ``{scene_id: [{"bbox": [cx,cy,cz,dx,dy,dz], "score"?}]}``."""
    raw = json.loads(Path(path).read_text())
    return {s: [{"object_id": f"p{i}", "object_name": "", **p} for i, p in enumerate(v)] for s, v in raw.items()}


def _score(pred, cand, rule: str) -> float:
    if rule == "ciou":
        return ciou_xyxy(pred, cand)
    if rule == "center_distance":
        return -math.hypot(0.5 * (pred[0] + pred[2] - cand[0] - cand[2]), 0.5 * (pred[1] + pred[3] - cand[1] - cand[3]))
    return iou_xyxy(pred, cand)


def gt_lookup(dataset: Iterable[dict]) -> dict:
    out = {}
    for s in dataset:
        out[(s["scene_id"], s["query"])] = s
        if s.get("object_id") is not None:
            out[(s["scene_id"], str(s["object_id"]), s["query"])] = s
    return out


def compute_hit_k(preds: list[dict], dataset: list[dict], candidates: dict[str, list[dict]],
                  k_values=(1, 3, 5), score_rule: str = "iou", require_positive_overlap: bool = False,
                  etas=(0.25, 0.5), geometric_match_iou: Optional[float] = None) -> dict:
    """Hit@k plus per-sample detail.

    ``geometric_match_iou`` switches to proposal matching (Table 11): a ranked
    candidate counts as the target if its BEV IoU with the GT box reaches the
    threshold, since predicted proposals carry no GT identity.
    """
    if score_rule not in SCORE_RULES:
        raise ValueError(f"score_rule must be one of {SCORE_RULES}")
    lookup = gt_lookup(dataset)
    hits = {k: 0 for k in k_values}
    hit1_eta = {eta: 0 for eta in etas}
    total = no_answer = skipped_no_gt = skipped_no_cands = 0
    stats = []
    for pred in preds:
        gt = None
        if pred.get("object_id") is not None:
            gt = lookup.get((pred["scene_id"], str(pred["object_id"]), pred["query"]))
        gt = gt or lookup.get((pred["scene_id"], pred["query"]))
        if gt is None:
            skipped_no_gt += 1
            continue
        cands = candidates.get(pred["scene_id"])
        if not cands:
            skipped_no_cands += 1
            continue
        total += 1
        gt_id = str(gt["object_id"])
        gt_xyxy = box3d_to_xyxy(gt["gt_bbox"])
        row = {"scene_id": pred["scene_id"], "query": pred["query"], "object_id": gt_id,
               "object_name": gt.get("object_name", ""), "reason": pred.get("reason"), "rank": None,
               "top_ids": [], "pred_gt_iou": 0.0, "area_ratio": None, "n_overlap_pos": 0, "n_overlap_025": 0,
               "score_rule": score_rule}
        wb = pred.get("world_bbox")
        if not wb:
            no_answer += 1
            stats.append(row)
            continue
        ranked = []
        for c in cands:
            cb = box3d_to_xyxy(c["bbox"])
            iou = iou_xyxy(wb, cb)
            if iou > 0:
                row["n_overlap_pos"] += 1
            if iou >= 0.25:
                row["n_overlap_025"] += 1
            if require_positive_overlap and iou <= 0.0:
                continue
            if geometric_match_iou is not None:
                is_target = iou_xyxy(cb, gt_xyxy) >= geometric_match_iou
            else:
                is_target = str(c["object_id"]) == gt_id
            ranked.append((_score(wb, cb, score_rule), str(c["object_id"]), is_target))
        ranked.sort(key=lambda r: r[0], reverse=True)
        rank = next((i + 1 for i, r in enumerate(ranked) if r[2]), None)
        for k in k_values:
            if rank is not None and rank <= k:
                hits[k] += 1
        pred_iou = iou_xyxy(wb, gt_xyxy)
        for eta in etas:
            if rank == 1 and pred_iou >= eta:
                hit1_eta[eta] += 1
        gt_area = area_xyxy(gt_xyxy)
        row.update(rank=rank, top_ids=[r[1] for r in ranked[:max(k_values)]], pred_gt_iou=pred_iou,
                   area_ratio=area_xyxy(wb) / gt_area if gt_area > 0 else None)
        stats.append(row)
    pct = lambda n: 100.0 * n / total if total else 0.0  # noqa: E731
    return {
        "total": total, "no_answer": no_answer, "skipped_no_gt": skipped_no_gt,
        "skipped_no_candidates": skipped_no_cands, "score_rule": score_rule,
        "require_positive_overlap": require_positive_overlap,
        "hits": hits, "hit_k": {f"Hit@{k}": pct(v) for k, v in hits.items()},
        "hit1_eta": {f"Hit@1_{eta}": pct(v) for eta, v in hit1_eta.items()},
        "candidates_per_scene": statistics.mean(len(v) for v in candidates.values()) if candidates else 0.0,
        "stats": stats,
    }


def overlap_and_area_summary(stats: list[dict]) -> dict:
    """Tables 14 and 15 over answered predictions."""
    answered = [s for s in stats if s.get("area_ratio") is not None]
    if not answered:
        return {}
    pos = [s["n_overlap_pos"] for s in answered]
    q25 = [s["n_overlap_025"] for s in answered]
    ratios = sorted(s["area_ratio"] for s in answered)

    def pctl(p):
        idx = (len(ratios) - 1) * p
        lo, hi = math.floor(idx), math.ceil(idx)
        return ratios[lo] + (ratios[hi] - ratios[lo]) * (idx - lo)

    return {"overlap_pos_mean": statistics.mean(pos), "overlap_pos_median": statistics.median(pos),
            "overlap_025_mean": statistics.mean(q25), "overlap_025_median": statistics.median(q25),
            "area_ratio_median": pctl(0.5), "area_ratio_p75": pctl(0.75), "area_ratio_p90": pctl(0.9)}


def proposal_recall(proposals: dict[str, list[dict]], dataset: list[dict], iou_threshold: float = 0.25) -> float:
    """Fraction of targets covered by a proposal with 3D IoU ≥ threshold (Appendix D.4)."""
    from ..geometry import iou_3d

    hit = n = 0
    for s in dataset:
        n += 1
        if any(iou_3d(p["bbox"][:6], s["gt_bbox"][:6]) >= iou_threshold for p in proposals.get(s["scene_id"], [])):
            hit += 1
    return 100.0 * hit / n if n else 0.0


def format_report(result: dict, source: str = "box3d") -> str:
    """Plain-text report in the reference release's log format."""
    total = result["total"]
    lines = [f"Evaluated:  {total} samples (denominator includes no-answer attempts)",
             f"Ranking:    {result['score_rule']}" + (" (positive-overlap candidates only)"
                                                       if result["require_positive_overlap"] else ""),
             f"  no_answer: {result['no_answer']}  (model never committed -> 0 hit, still in denom)",
             f"  scored:    {total - result['no_answer']}"]
    if result["skipped_no_gt"] or result["skipped_no_candidates"]:
        lines.append(f"  skipped:   no_gt={result['skipped_no_gt']}  no_{source}={result['skipped_no_candidates']}")
    lines.append("")
    for k, n in sorted(result["hits"].items()):
        lines.append(f"  Hit@{k}: {n}/{total} = {100.0 * n / total if total else 0.0:.2f}%")
    for name, v in result["hit1_eta"].items():
        lines.append(f"  {name}: {v:.2f}%")
    return "\n".join(lines)
