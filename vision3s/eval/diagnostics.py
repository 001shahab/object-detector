# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Controlled experiments and diagnostic analyses (Appendix D).

* :func:`aggregate_runs` – repeated decoding runs: mean ± sample std and a
  Student-t 95% interval (Table 8).
* :func:`policy_summary` – tokens/query, tool usage and L-Z-O adherence (Table 9).
* :func:`landmark_subsets` – expressions with / without named landmarks (Table 10).
* :func:`search_diagnostics` – regional-search success by target–landmark
  distance and local-disambiguation success by same-class distractors within
  50 m (Table 16, Fig. 9).
* :func:`latency_summary` – per-tool and per-query latency (Table 17).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Iterable, Optional

from ..geometry import point_in_xyxy
from ..landmarks import LandmarkDB
from ..prompts import FIXED_WORKFLOW
from .hit_k import box3d_to_xyxy, gt_lookup

DISTANCE_BINS = ((0, 50, "<50m"), (50, 100, "50–100m"), (100, 200, "100–200m"), (200, math.inf, "≥200m"))
DISTRACTOR_BINS = ((0, 0, "0"), (1, 3, "1–3"), (4, 9, "4–9"), (10, math.inf, "≥10"))

_T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
          15: 2.131, 20: 2.086, 30: 2.042}


def t_quantile_975(dof: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, dof))
    except ImportError:
        if dof in _T_975:
            return _T_975[dof]
        return 1.96 if dof > 30 else _T_975[min(k for k in _T_975 if k >= dof)]


def aggregate_runs(per_run: list[dict]) -> dict:
    """Mean, sample std (ddof=1), min, max and the t-interval ``mean ± t_{n-1,0.975}·s/√n`` per metric."""
    keys = sorted({k for r in per_run for k, v in r.items() if isinstance(v, (int, float))})
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in per_run if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
        if not vals:
            continue
        n = len(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        half = t_quantile_975(n - 1) * std / math.sqrt(n) if n > 1 else 0.0
        out[k] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "n": n,
                  "ci95": [mean - half, mean + half], "values": vals}
    return out


def tool_sequence(record: dict) -> tuple[str, ...]:
    return tuple(t["tool"] for t in record.get("tool_trace", []))


def follows_lzo(seq: tuple[str, ...]) -> bool:
    """Landmark Query → Zoom-in → Oblique Rendering, allowing repeated calls within a stage."""
    collapsed = tuple(name for i, name in enumerate(seq) if i == 0 or name != seq[i - 1])
    return collapsed == FIXED_WORKFLOW


def policy_summary(records: list[dict]) -> dict:
    n = len(records) or 1
    seqs = [tool_sequence(r) for r in records]
    usage = [r.get("usage", {}).get("total_tokens", 0) for r in records]
    per_tool = Counter(name for s in seqs for name in s)
    return {
        "queries": len(records), "tokens_per_query": sum(usage) / n,
        "prompt_tokens_per_query": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in records) / n,
        "completion_tokens_per_query": sum(r.get("usage", {}).get("completion_tokens", 0) for r in records) / n,
        "tool_calls_per_query": sum(len(s) for s in seqs) / n,
        "calls_per_query": {k: v / n for k, v in per_tool.items()},
        "lzo_fraction": sum(follows_lzo(s) for s in seqs) / n,
        "exact_lzo_fraction": sum(s == FIXED_WORKFLOW for s in seqs) / n,
        "no_answer_fraction": sum(r.get("reason") != "answer" for r in records) / n,
        "top_sequences": [(" → ".join(s) or "(none)", c) for s, c in Counter(seqs).most_common(10)],
    }


def landmark_subsets(dataset: list[dict], db: LandmarkDB) -> tuple[list[dict], list[dict]]:
    """Split expressions into those naming a landmark of the scene and those that do not."""
    with_lm, without = [], []
    for row in dataset:
        (with_lm if db.mentioned_in(row["scene_id"], row["query"]) else without).append(row)
    return with_lm, without


def _bin(value: float, bins) -> Optional[str]:
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return None


def _distractor_bin(n: int) -> str:
    for lo, hi, label in DISTRACTOR_BINS:
        if lo <= n <= hi:
            return label
    return DISTRACTOR_BINS[-1][2]


def zoom_regions(record: dict) -> list[list[float]]:
    return [t["crop_world"] for t in record.get("tool_trace", [])
            if t.get("tool") == "zoom_in" and "error" not in t and t.get("crop_world")]


def search_diagnostics(records: list[dict], dataset: list[dict], candidates: dict[str, list[dict]],
                       db: LandmarkDB, hit_stats: list[dict], region_rule: str = "last",
                       distractor_radius_m: float = 50.0) -> dict:
    """Table 16.

    Regional search succeeds when the selected Zoom-in region contains the GT
    centre (``region_rule``: ``last`` selected region, ``first``, or ``any``).
    Local disambiguation succeeds when, among regionally successful queries,
    the target ranks first under the IoU matching rule. Target–landmark
    distance uses the nearest landmark named in the expression; queries
    without a named landmark are excluded from that panel.
    """
    lookup = gt_lookup(dataset)
    rank1 = {(s["scene_id"], s["query"]): s.get("rank") == 1 for s in hit_stats}
    dist_bins: dict = defaultdict(lambda: [0, 0])
    distr_bins: dict = defaultdict(lambda: [0, 0])
    per_query = []
    for rec in records:
        gt = None
        if rec.get("object_id") is not None:
            gt = lookup.get((rec["scene_id"], str(rec["object_id"]), rec["query"]))
        gt = gt or lookup.get((rec["scene_id"], rec["query"]))
        if gt is None:
            continue
        gx, gy = float(gt["gt_bbox"][0]), float(gt["gt_bbox"][1])
        regions = zoom_regions(rec)
        if region_rule == "any":
            regional = any(point_in_xyxy(gx, gy, r) for r in regions)
        elif region_rule == "first":
            regional = bool(regions) and point_in_xyxy(gx, gy, regions[0])
        else:
            regional = bool(regions) and point_in_xyxy(gx, gy, regions[-1])
        named = db.mentioned_in(rec["scene_id"], rec["query"])
        lm_dist = min((math.hypot(float(e["xy"][0]) - gx, float(e["xy"][1]) - gy) for e in named), default=None)
        cls = gt.get("object_name", "")
        n_distr = 0
        for c in candidates.get(rec["scene_id"], []):
            if str(c["object_id"]) == str(gt["object_id"]) or (cls and c.get("object_name", "") != cls):
                continue
            cx, cy = 0.5 * sum(box3d_to_xyxy(c["bbox"])[0::2]), 0.5 * sum(box3d_to_xyxy(c["bbox"])[1::2])
            if math.hypot(cx - gx, cy - gy) <= distractor_radius_m:
                n_distr += 1
        local = rank1.get((rec["scene_id"], rec["query"]), False)
        if lm_dist is not None:
            label = _bin(lm_dist, DISTANCE_BINS)
            dist_bins[label][0] += int(regional)
            dist_bins[label][1] += 1
        if regional:
            label = _distractor_bin(n_distr)
            distr_bins[label][0] += int(local)
            distr_bins[label][1] += 1
        per_query.append({"scene_id": rec["scene_id"], "query": rec["query"], "regional_success": regional,
                          "local_success": bool(regional and local), "landmark_distance_m": lm_dist,
                          "same_class_distractors": n_distr})

    def table(bins, counts):
        return {label: {"success": 100.0 * counts[label][0] / counts[label][1] if counts[label][1] else None,
                        "n": counts[label][1]} for _, _, label in bins}

    return {"regional_search_by_landmark_distance": table(DISTANCE_BINS, dist_bins),
            "local_disambiguation_by_distractors": table(DISTRACTOR_BINS, distr_bins),
            "regional_success_overall": 100.0 * sum(q["regional_success"] for q in per_query) / max(1, len(per_query)),
            "region_rule": region_rule, "per_query": per_query}


def latency_summary(records: Iterable[dict]) -> dict:
    per_tool: dict = defaultdict(list)
    wall, gen, tool = [], [], []
    for r in records:
        for t in r.get("tool_trace", []):
            if "latency_s" in t:
                per_tool[t["tool"]].append(t["latency_s"])
        wall.append(r.get("wall_time_s", 0.0))
        gen.append(r.get("generation_time_s", 0.0))
        tool.append(r.get("tool_time_s", 0.0))
    n = max(1, len(wall))
    return {"per_tool_ms": {k: 1000 * statistics.mean(v) for k, v in per_tool.items()},
            "tool_time_per_query_s": sum(tool) / n, "generation_time_per_query_s": sum(gen) / n,
            "wall_time_per_query_s": sum(wall) / n}
