# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""End-to-end evaluation: rollouts over a split, then scoring.

Each run writes ``run_XX/records.jsonl``; when candidates are available it is
scored with Hit@k (IoU, positive-overlap IoU, CIoU) and Hit@1_η, and when raw
point clouds are available with 2D/3D IoU and point F1. Repeated runs are
aggregated with mean ± sample std and a 95% t-interval (Table 8).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

from ..agent.policy import Policy
from ..agent.rollout import read_jsonl, run_rollouts, write_records
from ..env import GroundingEnv
from .diagnostics import aggregate_runs, latency_summary, policy_summary
from .hit_k import compute_hit_k, format_report, overlap_and_area_summary


def score_records(records: list[dict], dataset: list[dict], candidates: Optional[dict] = None,
                  load_points: Optional[Callable] = None) -> dict:
    out: dict = {"policy": policy_summary(records), "latency": latency_summary(records)}
    flat: dict = {"tokens_per_query": out["policy"]["tokens_per_query"],
                  "tool_calls_per_query": out["policy"]["tool_calls_per_query"],
                  "no_answer_pct": 100.0 * out["policy"]["no_answer_fraction"],
                  "mean_iou_2d_pct": 100.0 * sum(r.get("iou_2d", 0.0) for r in records) / max(1, len(records))}
    if candidates:
        for rule, pos in (("iou", False), ("iou", True), ("ciou", False)):
            res = compute_hit_k(records, dataset, candidates, score_rule=rule, require_positive_overlap=pos)
            tag = f"{rule}{'_pos' if pos else ''}"
            out[f"hit_k_{tag}"] = {k: v for k, v in res.items() if k != "stats"}
            for k, v in res["hit_k"].items():
                flat[k if tag == "iou" else f"{k}_{tag}"] = v
            if tag == "iou":
                flat.update(res["hit1_eta"])
                out["overlap_area"] = overlap_and_area_summary(res["stats"])
                out["hit_k_stats"] = res["stats"]
                out["hit_k_report"] = format_report(res)
    if load_points is not None:
        from .metrics_3d import evaluate_3d

        m3 = evaluate_3d(records, dataset, load_points)
        out["metrics_3d"] = m3["summary"]
        out["metrics_3d_per_sample"] = m3["per_sample"]
        for k in ("iou_bev", "iou_3d", "point_f1"):
            if k in m3["summary"]:
                flat[f"{k}_pct"] = 100.0 * m3["summary"][k]
    out["flat"] = flat
    return out


def evaluate(policy_factory: Callable[[int], Policy], env_factory: Callable[[], GroundingEnv], dataset: list[dict],
             out_dir: str | Path, num_runs: int = 1, batch_size: int = 16, candidates: Optional[dict] = None,
             load_points: Optional[Callable] = None, limit: Optional[int] = None, seed: int = 42,
             resume: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = dataset[:limit] if limit else dataset
    per_run = []
    for run in range(1, num_runs + 1):
        run_dir = out / f"run_{run:02d}"
        rec_path = run_dir / "records.jsonl"
        if resume and rec_path.exists() and len(read_jsonl(rec_path)) == len(rows):
            records = read_jsonl(rec_path)
        else:
            random.seed(seed + run)
            try:
                import torch

                torch.manual_seed(seed + run)
            except ImportError:
                pass
            policy = policy_factory(run)
            trajs = run_rollouts(policy, env_factory, rows, group_size=1, max_concurrent=batch_size)
            write_records(trajs, rec_path)
            records = read_jsonl(rec_path)
        scored = score_records(records, rows, candidates, load_points)
        stats = scored.pop("hit_k_stats", None)
        per_sample_3d = scored.pop("metrics_3d_per_sample", None)
        if stats is not None:
            with (run_dir / "hit_k_detail.jsonl").open("w", encoding="utf-8") as fh:
                for s in stats:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        if per_sample_3d is not None:
            with (run_dir / "metrics_3d_detail.jsonl").open("w", encoding="utf-8") as fh:
                for s in per_sample_3d:
                    fh.write(json.dumps(s, ensure_ascii=False, default=float) + "\n")
        (run_dir / "scores.json").write_text(json.dumps(scored, indent=2, ensure_ascii=False, default=float))
        per_run.append(scored["flat"])
    summary = {"num_runs": num_runs, "queries": len(rows), "per_run": per_run, "aggregate": aggregate_runs(per_run)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
