# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Staged STPLS3D-Refer build over the standard data layout (Appendix B.2).

``instances`` → (manual instance review) → ``landmarks`` → ``generate`` →
(manual expression review) → ``finalize``. Every stage reads and writes files
under ``out_root`` so the two human-review steps can happen between runs.

STPLS3D RealWorld semantic ids: 0 ground, 1 building, 2–4 vegetation,
5 vehicle, 6 truck, … The defaults use building ``1`` and vehicles ``(5, 6)``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

from .ply import read_ply
from .stpls3d_refer import (Instance, apply_review, extract_instances, filter_expressions, fit_georegistration,
                            fetch_osm, generate_expressions, match_landmarks_to_instances, osm_to_landmarks,
                            overpass_query, render_marked_views, target_context, to_benchmark_rows,
                            write_review_sheet)


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def stage_instances(ply_path: str, scene_id: str, out_root: str, building_label: int = 1,
                    vehicle_labels: tuple[int, ...] = (5, 6)) -> dict:
    out = Path(out_root)
    (out / "instances").mkdir(parents=True, exist_ok=True)
    points, colors, labels = read_ply(ply_path, with_labels=True)
    if labels is None:
        raise ValueError(f"{ply_path} has no semantic label field")
    insts = extract_instances(points, colors, labels, building_label=building_label, vehicle_labels=vehicle_labels)
    (out / "instances" / f"{scene_id}.json").write_text(json.dumps([i.record() for i in insts], indent=1))
    review = out / "instances" / f"{scene_id}_review.json"
    if not review.exists():
        review.write_text(json.dumps({str(i.instance_id): i.status != "rejected" for i in insts}, indent=1))
    kept = sum(i.status != "rejected" for i in insts)
    return {"scene_id": scene_id, "instances": len(insts), "auto_kept": kept, "review_file": str(review)}


def _reviewed_instances(out: Path, scene_id: str) -> list[dict]:
    recs = json.loads((out / "instances" / f"{scene_id}.json").read_text())
    review = _load_json(out / "instances" / f"{scene_id}_review.json", {})
    return [r for r in recs if review.get(str(r["object_id"]), r["status"] != "rejected")]


def _as_instances(recs: list[dict]) -> list[Instance]:
    return [Instance(int(r["object_id"]), r["object_name"], r["bbox"], r["footprint_m2"], r["n_points"],
                     r["mean_color"], np.empty(0, dtype=np.int64)) for r in recs]


def stage_landmarks(scene_id: str, gcps_path: str, out_root: str, extent: Optional[list[float]] = None,
                    osm_json: Optional[str] = None, endpoint: str = "https://overpass-api.de/api/interpreter") -> dict:
    """Geo-register with control points, query OSM and write ``landmarks_stpls3d.json`` + box3d."""
    out = Path(out_root)
    gcps = json.loads(Path(gcps_path).read_text())
    reg = fit_georegistration(gcps)
    recs = _reviewed_instances(out, scene_id)
    if extent is None:
        xs = [r["bbox"][0] for r in recs]
        ys = [r["bbox"][1] for r in recs]
        extent = [min(xs) - 100, min(ys) - 100, max(xs) + 100, max(ys) + 100]
    if osm_json:
        osm = json.loads(Path(osm_json).read_text())
    else:
        corners = reg.local_to_wgs84(np.array([[extent[0], extent[1]], [extent[2], extent[3]],
                                               [extent[0], extent[3]], [extent[2], extent[1]]]))
        south, west = corners[:, 0].min(), corners[:, 1].min()
        north, east = corners[:, 0].max(), corners[:, 1].max()
        osm = fetch_osm(overpass_query(south, west, north, east), endpoint)
        (out / "osm").mkdir(parents=True, exist_ok=True)
        (out / "osm" / f"{scene_id}.json").write_text(json.dumps(osm))
    landmarks = osm_to_landmarks(osm, reg, extent)
    names = match_landmarks_to_instances(landmarks, _as_instances(recs))
    db_path = out / "landmarks_stpls3d.json"
    db = _load_json(db_path, {})
    db[scene_id] = landmarks
    db_path.write_text(json.dumps(db, ensure_ascii=False, indent=1))
    (out / "box3d").mkdir(parents=True, exist_ok=True)
    bboxes = [{"object_id": r["object_id"], "object_name": r["object_name"],
               "bbox": [*r["bbox"], 0 if r["object_name"] == "building" else 1, r["object_id"]],
               **({"osm_name": names[r["object_id"]]} if r["object_id"] in names else {})} for r in recs]
    (out / "box3d" / f"{scene_id}_bbox.json").write_text(json.dumps({"scene_id": scene_id, "bboxes": bboxes}))
    return {"scene_id": scene_id, "landmarks": len(landmarks), "gcp_rms_m": reg.rms_m,
            "named_buildings": len(names), "candidates": len(bboxes)}


def stage_generate(scene_id: str, ply_path: str, out_root: str, model: str = "qwen3.6-plus",
                   base_url: Optional[str] = None, api_key: Optional[str] = None, max_targets: Optional[int] = None,
                   seed: int = 42, distractor_radius_m: float = 50.0) -> dict:
    from openai import OpenAI

    from ..oblique.renderer import PointGrid
    from ..views import load_bev

    out = Path(out_root)
    recs = _reviewed_instances(out, scene_id)
    insts = _as_instances(recs)
    landmarks = _load_json(out / "landmarks_stpls3d.json", {}).get(scene_id, [])
    names = {lm["name"] for lm in landmarks}
    bev, mapping = load_bev(out / f"bev_stpls3d/{scene_id}.png")
    points, colors = read_ply(ply_path)
    grid = PointGrid(points, colors)
    client = OpenAI(base_url=base_url, api_key=api_key)
    targets = list(insts)
    random.Random(seed).shuffle(targets)
    if max_targets:
        targets = targets[:max_targets]
    img_dir = out / "review" / f"{scene_id}_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out / "review" / f"{scene_id}_candidates.jsonl"
    candidates = []
    for t in targets:
        ctx = target_context(t, insts, landmarks, radius_m=distractor_radius_m)
        distractors = [i for i in insts if i.instance_id in {d["id"] for d in ctx["distractors"]}]
        images = render_marked_views(t, distractors, bev, mapping, grid)
        exprs = generate_expressions(client, model, images, ctx)
        bx = t.bbox
        inside = (mapping["x_min"] <= bx[0] <= mapping["x_max"]) and (mapping["y_min"] <= bx[1] <= mapping["y_max"])
        for k, f in enumerate(filter_expressions(exprs, ctx, names, projection_valid=inside)):
            rid = f"{scene_id}_{t.instance_id}_{k}"
            for i, im in enumerate(images):
                im.convert("RGB").save(img_dir / f"{rid}_{i}.jpg", quality=85)
            candidates.append({"review_id": rid, "scene_id": scene_id, "object_id": t.instance_id,
                               "object_name": t.category, "gt_bbox": [*bx, 0 if t.category == "building" else 1,
                                                                      t.instance_id],
                               "n_images": len(images), "context": ctx, **f})
    with cand_path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    auto_ok = [c for c in candidates if c["ok"]]
    write_review_sheet(auto_ok, out / "review" / f"{scene_id}_review.html", img_dir)
    review = out / "review" / f"{scene_id}_accept.json"
    if not review.exists():
        review.write_text(json.dumps({c["review_id"]: None for c in auto_ok}, indent=1))
    return {"scene_id": scene_id, "targets": len(targets), "expressions": len(candidates), "auto_ok": len(auto_ok),
            "accept_file": str(review)}


def stage_finalize(out_root: str, scenes: list[str]) -> dict:
    """Merge human-accepted expressions into ``meta/stpls3d_test.jsonl``."""
    out = Path(out_root)
    rows = []
    for scene in scenes:
        cands = [json.loads(line) for line in (out / "review" / f"{scene}_candidates.jsonl").read_text().splitlines()
                 if line.strip()]
        review = _load_json(out / "review" / f"{scene}_accept.json", {})
        rows.extend(to_benchmark_rows(apply_review([c for c in cands if c["ok"]], review)))
    (out / "meta").mkdir(parents=True, exist_ok=True)
    with (out / "meta" / "stpls3d_test.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"accepted": len(rows), "scenes": scenes}
