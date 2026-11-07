# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Point-cloud IO and raw-data path resolution.

Raw layout (``$URBEX_RAW_DATA``)::

    sensaturban/ply/{train,test}/<scene>.ply      CityRefer scenes
    STPLS3D/RealWorldData/<scene>[_points].ply    STPLS3D-Refer scenes
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

LABEL_FIELDS = ("class", "label", "scalar_Label", "semantic", "sem_label", "scalar_label")


def read_ply(path: str | Path, with_labels: bool = False):
    """Return ``(points float32 (N,3), colors uint8 (N,3)[, labels int (N,) or None])``."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    names = vertex.data.dtype.names
    pts = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([np.asarray(vertex[c], dtype=np.float32) for c in ("red", "green", "blue")], axis=1)
        if rgb.max() <= 1.0:
            rgb *= 255.0
        colors = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(pts), 3), 180, dtype=np.uint8)
    if not with_labels:
        return pts, colors
    field = next((f for f in LABEL_FIELDS if f in names), None)
    labels = np.asarray(vertex[field]).astype(np.int64) if field else None
    return pts, colors, labels


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
    from plyfile import PlyData, PlyElement

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if labels is not None:
        fields.append(("class", "i4"))
    data = np.empty(len(points), dtype=fields)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    if labels is not None:
        data["class"] = labels
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def ply_candidates(scene_id: str, dataset: str, root: str | Path) -> list[Path]:
    root = Path(root).expanduser()
    if dataset == "cityrefer":
        rel = [Path(f"{scene_id}.ply")] + [Path(*p, f"{scene_id}.ply") for p in (
            ("train",), ("test",), ("ply", "train"), ("ply", "test"),
            ("sensaturban", "ply", "train"), ("sensaturban", "ply", "test"))]
    elif dataset == "stpls3d":
        rel = [d / f for f in (f"{scene_id}_points.ply", f"{scene_id}.ply")
               for d in (Path(), Path("RealWorldData"), Path("STPLS3D") / "RealWorldData")]
    else:
        rel = [Path(f"{scene_id}.ply"), Path(dataset) / f"{scene_id}.ply"]
    return [root / r for r in rel]


def resolve_ply(scene_id: str, dataset: str, root: str | Path) -> Optional[Path]:
    return next((p for p in ply_candidates(scene_id, dataset, root) if p.is_file()), None)


def scene_point_cloud_loader(raw_root: str | Path):
    """Callable ``(scene_id, dataset) -> (points, colors) | None`` with a per-process cache."""

    @lru_cache(maxsize=4)
    def load(scene_id: str, dataset: str):
        path = resolve_ply(scene_id, dataset, raw_root)
        return read_ply(path) if path else None

    return load
