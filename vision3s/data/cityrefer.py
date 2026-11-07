# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""CityRefer preparation (Appendix B.1).

1. Load CityRefer referring expressions and per-scene instance boxes.
2. Remove duplicate ``(scene, query)`` rows and low-quality annotations
   (a reviewer-maintained exclusion list of visibly incorrect labels/boxes).
3. Reserve seven scenes for the novel-object (NO) split; split the queries of
   the remaining scenes into train / val / novel-description (ND) with a fixed
   seed. With the paper's 12 scenes this yields 2,744 / 549 / 1,098 / 1,396.
4. Write ``meta/cityrefer_{train,val,test_ND,test_NO}.jsonl``, the Hit@k
   candidate pool ``box3d/<scene>_bbox.json`` and per-scene statistics (Table 6).

Annotation records need ``scene_id``, ``object_id`` and ``description`` (or
``query``); ``object_name`` is kept when present. Box files follow
``{"scene_id": ..., "bboxes": [{"object_id", "object_name", "bbox": [cx, cy, cz, dx, dy, dz, ...]}]}``.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

SPLIT_RATIOS = {"train": 0.625, "val": 0.125, "test_ND": 0.25}
NUM_NO_SCENES = 7


def _read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("annotations", [])


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def load_boxes(box_dir: str | Path) -> dict[str, dict[str, dict]]:
    """``{scene_id: {object_id: box_record}}`` from ``*_bbox.json`` files."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(box_dir).glob("*_bbox.json")):
        data = json.loads(path.read_text())
        out[data["scene_id"]] = {str(b["object_id"]): b for b in data["bboxes"]}
    return out


def load_exclusions(path: Optional[str | Path]) -> tuple[set, set]:
    """Exclusion list rows ``{"scene_id", "object_id"}`` (bad instance) or ``{"scene_id", "query"}`` (bad text)."""
    objects, queries = set(), set()
    if path:
        for row in _read_records(Path(path)):
            if row.get("query"):
                queries.add((row["scene_id"], normalize_query(row["query"]).lower()))
            elif row.get("object_id") is not None:
                objects.add((row["scene_id"], str(row["object_id"])))
    return objects, queries


def build_records(annotation_files: Iterable[str | Path], boxes: dict, exclusions: tuple[set, set]) -> tuple[list[dict], Counter]:
    bad_objects, bad_queries = exclusions
    stats: Counter = Counter()
    seen: set = set()
    records = []
    for path in annotation_files:
        for row in _read_records(Path(path)):
            stats["raw"] += 1
            scene_id, object_id = str(row["scene_id"]), str(row["object_id"])
            query = normalize_query(row.get("description") or row.get("query") or "")
            key = (scene_id, query.lower())
            if not query or key in seen:
                stats["duplicate_or_empty"] += 1
                continue
            seen.add(key)
            if (scene_id, object_id) in bad_objects or key in bad_queries:
                stats["low_quality"] += 1
                continue
            box = boxes.get(scene_id, {}).get(object_id)
            if box is None:
                stats["missing_box"] += 1
                continue
            records.append({
                "dataset": "cityrefer", "scene_id": scene_id, "object_id": object_id,
                "object_name": row.get("object_name") or box.get("object_name", ""), "query": query,
                "gt_bbox": [float(v) for v in box["bbox"][:6]] + list(box["bbox"][6:]),
                "bev_png": f"bev_cityrefer/{scene_id}.png",
            })
    stats["kept"] = len(records)
    return records, stats


def split_records(records: list[dict], seed: int = 42, no_scenes: Optional[list[str]] = None,
                  num_no_scenes: int = NUM_NO_SCENES) -> dict[str, list[dict]]:
    scenes = sorted({r["scene_id"] for r in records})
    rng = random.Random(seed)
    if no_scenes is None:
        no_scenes = sorted(rng.sample(scenes, min(num_no_scenes, max(0, len(scenes) - 1))))
    no_set = set(no_scenes)
    splits = {"train": [], "val": [], "test_ND": [], "test_NO": [r for r in records if r["scene_id"] in no_set]}
    seen_pool = [r for r in records if r["scene_id"] not in no_set]
    rng.shuffle(seen_pool)
    n = len(seen_pool)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_val = round(n * SPLIT_RATIOS["val"])
    splits["train"] = seen_pool[:n_train]
    splits["val"] = seen_pool[n_train:n_train + n_val]
    splits["test_ND"] = seen_pool[n_train + n_val:]
    return splits


def scene_statistics(records: list[dict], boxes: dict, landmarks: dict, point_counts: Optional[dict] = None,
                     areas: Optional[dict] = None) -> dict:
    """Table 6: area, points, GT candidates, OSM landmarks and test queries per scene."""
    tests = Counter(r["scene_id"] for r in records if r.get("split") in ("test_ND", "test_NO"))
    per_scene = {}
    for scene in sorted({r["scene_id"] for r in records}):
        per_scene[scene] = {
            "area_km2": (areas or {}).get(scene), "points_millions": (point_counts or {}).get(scene),
            "gt_candidates": len(boxes.get(scene, {})), "osm_landmarks": len(landmarks.get(scene, [])),
            "test_queries": tests.get(scene, 0),
        }

    def agg(key):
        vals = [v[key] for v in per_scene.values() if v[key] is not None]
        return {"total": sum(vals), "min": min(vals), "max": max(vals)} if vals else None

    return {"scenes": per_scene, "summary": {k: agg(k) for k in
                                             ("area_km2", "points_millions", "gt_candidates", "osm_landmarks",
                                              "test_queries")}}


def prepare_cityrefer(annotation_files: list[str], box_dir: str, out_root: str, *, landmarks: Optional[str] = None,
                      exclusions: Optional[str] = None, no_scenes: Optional[list[str]] = None, seed: int = 42,
                      raw_root: Optional[str] = None) -> dict:
    out = Path(out_root)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "box3d").mkdir(parents=True, exist_ok=True)
    boxes = load_boxes(box_dir)
    records, stats = build_records(annotation_files, boxes, load_exclusions(exclusions))
    splits = split_records(records, seed=seed, no_scenes=no_scenes)
    for name, rows in splits.items():
        with (out / "meta" / f"cityrefer_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                r["split"] = name
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for scene, objs in boxes.items():
        (out / "box3d" / f"{scene}_bbox.json").write_text(
            json.dumps({"scene_id": scene, "bboxes": list(objs.values())}))
    lm = json.loads(Path(landmarks).read_text()) if landmarks else {}
    if landmarks:
        (out / "landmarks_cityrefer.json").write_text(json.dumps(lm, ensure_ascii=False, indent=1))
    point_counts, areas = None, None
    if raw_root:
        from .ply import read_ply, resolve_ply
        point_counts, areas = {}, {}
        for scene in {r["scene_id"] for r in records}:
            path = resolve_ply(scene, "cityrefer", raw_root)
            if path:
                pts, _ = read_ply(path)
                point_counts[scene] = len(pts) / 1e6
                span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
                areas[scene] = float(span[0] * span[1]) / 1e6
    table6 = scene_statistics([r for rows in splits.values() for r in rows], boxes, lm, point_counts, areas)
    summary = {"filter": dict(stats), "splits": {k: len(v) for k, v in splits.items()},
               "no_scenes": sorted({r["scene_id"] for r in splits["test_NO"]}), "seed": seed, "table6": table6}
    (out / "meta" / "cityrefer_prepare_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
