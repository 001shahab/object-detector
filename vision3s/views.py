# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# The view-to-world mapping convention and the patch-aligned display sizing
# follow the Urbex reference release (https://github.com/AMAP-ML/Urbex,
# Copyright 2026 The Urbex Authors, Apache-2.0) so released BEV assets load
# unchanged.
"""BEV-aligned views and their view-to-world transforms ``T_v`` (Sec. 3.1).

A mapping is a dict ``{x_min, x_max, y_min, y_max, res_x, res_y}``: pixel
column ``px`` grows with world X, pixel row ``py`` grows towards *smaller*
world Y (north up). Agents address a view through a normalised
``[0, norm_max]`` integer frame (origin top-left, x right, y down); the
environment resolves those coordinates through the view's mapping.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image

# City-scale BEVs exceed PIL's decompression-bomb guard; these are our own files.
Image.MAX_IMAGE_PIXELS = 1_000_000_000

DATA_ROOT_ENV = "CITY_AGENT_DATA"


def world_to_pixel(mapping: dict, wx, wy):
    return (wx - mapping["x_min"]) / mapping["res_x"], (mapping["y_max"] - wy) / mapping["res_y"]


def pixel_to_world(mapping: dict, px, py):
    return mapping["x_min"] + px * mapping["res_x"], mapping["y_max"] - py * mapping["res_y"]


def compute_display_size(width: int, height: int, *, max_side: int = 768, patch: int = 32,
                         max_pixels: int = 589_824) -> tuple[int, int]:
    """Aspect-preserving, patch-aligned size for an image shown to the VLM."""
    r = min(1.0, max_side / max(width, height))
    dw = max(patch, round(width * r / patch) * patch)
    dh = max(patch, round(height * r / patch) * patch)
    if dw * dh > max_pixels:
        shrink = (max_pixels / (dw * dh)) ** 0.5
        dw = max(patch, (int(dw * shrink) // patch) * patch)
        dh = max(patch, (int(dh * shrink) // patch) * patch)
    return dw, dh


def resolve_data_path(path: str | Path, root: Optional[str] = None) -> Path:
    """Resolve an annotation path that may be relative to ``$CITY_AGENT_DATA``."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = root or os.environ.get(DATA_ROOT_ENV)
    if not root:
        raise RuntimeError(f"{path!r} is relative but {DATA_ROOT_ENV} is unset")
    return Path(root) / p


@lru_cache(maxsize=64)
def _load_bev(resolved: str) -> tuple[Image.Image, dict]:
    png = Path(resolved)
    image = Image.open(png).convert("RGB")
    mapping = json.loads(png.with_suffix(".json").read_text())
    return image, mapping


def load_bev(path: str | Path, root: Optional[str] = None) -> tuple[Image.Image, dict]:
    """Load the full-resolution BEV and its mapping JSON (cached per process)."""
    return _load_bev(str(resolve_data_path(path, root)))


def crop_bev(image: Image.Image, mapping: dict, world_xyxy, *, max_side: int = 768, patch: int = 32,
             max_pixels: int = 589_824) -> tuple[Image.Image, dict]:
    """Crop a world region from the high-resolution BEV and resize it for display.

    Returns the crop and a mapping describing exactly the returned canvas.
    """
    x0, y0, x1, y1 = (float(v) for v in world_xyxy)
    left, top = world_to_pixel(mapping, x0, y1)
    right, bottom = world_to_pixel(mapping, x1, y0)
    width, height = image.size
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(width, int(right) + 1), min(height, int(bottom) + 1)
    if right <= left or bottom <= top:
        dw, dh = compute_display_size(width, height, max_side=max_side, patch=patch, max_pixels=max_pixels)
        return image.resize((dw, dh), Image.Resampling.LANCZOS), {
            "x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1,
            "res_x": (x1 - x0) / max(1, dw), "res_y": (y1 - y0) / max(1, dh),
        }
    cropped = image.crop((left, top, right, bottom))
    dw, dh = compute_display_size(*cropped.size, max_side=max_side, patch=patch, max_pixels=max_pixels)
    resized = cropped.resize((dw, dh), Image.Resampling.LANCZOS)
    sx0, sy1 = pixel_to_world(mapping, left, top)
    sx1, sy0 = pixel_to_world(mapping, right, bottom)
    return resized, {
        "x_min": sx0, "x_max": sx1, "y_min": sy0, "y_max": sy1,
        "res_x": (sx1 - sx0) / max(1, dw), "res_y": (sy1 - sy0) / max(1, dh),
    }


@dataclass
class View:
    """One BEV-aligned observation ``(I_v, T_v)`` in the view registry."""

    view_id: str
    image: Image.Image
    mapping: dict
    depth: int = 0
    parent_id: Optional[str] = None
    bbox_in_parent: Optional[list[float]] = None
    kind: str = "global"
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def world_bounds(self) -> list[float]:
        m = self.mapping
        return [m["x_min"], m["y_min"], m["x_max"], m["y_max"]]

    def norm_to_world(self, bbox, norm_max: float = 1000.0) -> list[float]:
        """Project a normalised ``[x0, y0, x1, y1]`` box to a world ``xyxy`` box."""
        w, h = self.size
        x0, y0, x1, y1 = (float(v) for v in bbox)
        wx0, wy0 = pixel_to_world(self.mapping, x0 / norm_max * w, y0 / norm_max * h)
        wx1, wy1 = pixel_to_world(self.mapping, x1 / norm_max * w, y1 / norm_max * h)
        return [min(wx0, wx1), min(wy0, wy1), max(wx0, wx1), max(wy0, wy1)]

    def world_to_norm(self, wx: float, wy: float, norm_max: float = 1000.0) -> Optional[list]:
        """Project a world point into the view frame; ``None`` if off-canvas."""
        w, h = self.size
        px, py = world_to_pixel(self.mapping, wx, wy)
        if px < 0 or py < 0 or px > w or py > h:
            return None
        nx, ny = px / w * norm_max, py / h * norm_max
        if norm_max >= 100.0:
            return [round(nx), round(ny)]
        return [round(nx, 4), round(ny, 4)]

    def world_xyxy_to_norm(self, box, norm_max: float = 1000.0) -> list[float]:
        w, h = self.size
        px0, py0 = world_to_pixel(self.mapping, float(box[0]), float(box[3]))
        px1, py1 = world_to_pixel(self.mapping, float(box[2]), float(box[1]))
        return [px0 / w * norm_max, py0 / h * norm_max, px1 / w * norm_max, py1 / h * norm_max]
