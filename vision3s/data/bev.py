# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Global bird's-eye-view construction (Sec. 3.1, Appendix B.1).

Each pixel shows the colour of the highest point that projects into it
(top-down orthographic z-buffer); empty pixels are filled from their nearest
occupied neighbour. The high-resolution image (long side typically above
10,000 px) is written next to its mapping JSON; the environment downsamples
it for the global observation and crops it for Zoom-in.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..views import world_to_pixel


def render_bev(points: np.ndarray, colors: np.ndarray, *, resolution: float = 0.05, fill: bool = True,
               fill_radius: Optional[float] = None, subsample: Optional[int] = None,
               seed: int = 0) -> tuple[Image.Image, dict]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    finite = np.isfinite(points[:, :3]).all(axis=1)
    points, colors = points[finite, :3], np.asarray(colors, dtype=np.uint8)[finite]
    if not len(points):
        raise ValueError("point cloud contains no finite points")
    lo, hi = points.min(axis=0), points.max(axis=0)
    x_span = max(float(hi[0] - lo[0]), resolution)
    y_span = max(float(hi[1] - lo[1]), resolution)
    width, height = int(np.ceil(x_span / resolution)), int(np.ceil(y_span / resolution))
    mapping = {"x_min": float(lo[0]), "x_max": float(lo[0]) + x_span, "y_min": float(lo[1]),
               "y_max": float(lo[1]) + y_span, "width": width, "height": height,
               "res_x": x_span / width, "res_y": y_span / height, "requested_resolution": float(resolution)}
    if subsample is not None and len(points) > subsample:
        keep = np.sort(np.random.default_rng(seed).choice(len(points), subsample, replace=False))
        points, colors = points[keep], colors[keep]
    px, py = world_to_pixel(mapping, points[:, 0], points[:, 1])
    px = np.clip(np.floor(px).astype(np.int64), 0, width - 1)
    py = np.clip(np.floor(py).astype(np.int64), 0, height - 1)
    linear = py * width + px
    order = np.lexsort((np.arange(len(points)), -points[:, 2], linear))
    winners = order[np.unique(linear[order], return_index=True)[1]]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=bool)
    image[py[winners], px[winners]] = colors[winners]
    occupied[py[winners], px[winners]] = True
    if fill and occupied.any() and not occupied.all():
        from scipy.ndimage import distance_transform_edt

        distances, (iy, ix) = distance_transform_edt(~occupied, return_indices=True)
        mask = ~occupied if fill_radius is None else (~occupied & (distances <= fill_radius))
        image[mask] = image[iy[mask], ix[mask]]
    return Image.fromarray(image, mode="RGB"), mapping


def build_scene_bev(scene_id: str, points: np.ndarray, colors: np.ndarray, out_dir: str | Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png, meta = out_dir / f"{scene_id}.png", out_dir / f"{scene_id}.json"
    if png.exists() and meta.exists():
        return {"scene_id": scene_id, "status": "skipped", "output": str(png)}
    image, mapping = render_bev(points, colors, **kwargs)
    with tempfile.TemporaryDirectory(dir=out_dir) as stage:
        image.save(Path(stage) / png.name, "PNG", optimize=True)
        (Path(stage) / meta.name).write_text(json.dumps(mapping, indent=2) + "\n")
        os.replace(Path(stage) / meta.name, meta)
        os.replace(Path(stage) / png.name, png)
    return {"scene_id": scene_id, "status": "built", "output": str(png), "width": image.width, "height": image.height}
