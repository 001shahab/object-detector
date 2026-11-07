# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Headless software renderer for oblique point-cloud views.

Points are projected through the cameras of :mod:`vision3s.oblique.camera`,
resolved with a per-pixel z-buffer and splatted to ``point_size`` pixels. The
renderer returns colour images and camera-space depth maps in metres (0 where
empty), the same quantities an OpenGL point renderer would capture, without
needing a display, OpenGL or Xvfb.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .camera import DIRECTION_INDEX, region_cameras


class PointGrid:
    """Uniform XY bucket index for fast rectangular crops of large point clouds."""

    def __init__(self, points: np.ndarray, colors: np.ndarray, cell: float = 10.0):
        self.cell = float(cell)
        self.origin = points[:, :2].min(axis=0)
        ij = np.floor((points[:, :2] - self.origin) / self.cell).astype(np.int64)
        self.shape = ij.max(axis=0) + 1
        keys = ij[:, 0] * self.shape[1] + ij[:, 1]
        order = np.argsort(keys, kind="stable")
        self.points = np.ascontiguousarray(points[order, :3], dtype=np.float32)
        self.colors = np.ascontiguousarray(colors[order], dtype=np.uint8)
        self._keys = keys[order]

    def crop(self, x0: float, y0: float, x1: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
        i0, j0 = np.floor((np.array([x0, y0]) - self.origin) / self.cell).astype(int)
        i1, j1 = np.floor((np.array([x1, y1]) - self.origin) / self.cell).astype(int)
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, self.shape[0] - 1), min(j1, self.shape[1] - 1)
        if i1 < i0 or j1 < j0:
            return self.points[:0], self.colors[:0]
        chunks = []
        for i in range(i0, i1 + 1):
            lo = np.searchsorted(self._keys, i * self.shape[1] + j0, side="left")
            hi = np.searchsorted(self._keys, i * self.shape[1] + j1, side="right")
            if hi > lo:
                chunks.append(np.arange(lo, hi))
        if not chunks:
            return self.points[:0], self.colors[:0]
        idx = np.concatenate(chunks)
        pts = self.points[idx]
        mask = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        return pts[mask], self.colors[idx][mask]


def splat(points: np.ndarray, colors: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
          size: int, point_size: float = 5.0, background=(255, 255, 255),
          near: float = 0.1) -> tuple[Image.Image, np.ndarray]:
    """Z-buffered point splatting. Returns ``(RGB image, depth (size, size) in metres)``."""
    cam = points.astype(np.float64) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = cam[:, 2]
    front = z > near
    cam, z, col = cam[front], z[front], colors[front]
    u = np.round(cam[:, 0] / z * intrinsic[0, 0] + intrinsic[0, 2]).astype(np.int64)
    v = np.round(cam[:, 1] / z * intrinsic[1, 1] + intrinsic[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v, z, col = u[inside], v[inside], z[inside], col[inside]

    depth = np.full((size, size), np.inf)
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:] = background
    if len(z):
        linear = v * size + u
        order = np.lexsort((z, linear))
        first = np.unique(linear[order], return_index=True)[1]
        win = order[first]
        depth.flat[linear[win]] = z[win]
        rgb.reshape(-1, 3)[linear[win]] = col[win]

    radius = max(0, int(round(point_size)) // 2)
    if radius:
        out_d, out_c = depth.copy(), rgb.copy()
        pad_d = np.pad(depth, radius, constant_values=np.inf)
        pad_c = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                sd = pad_d[radius + dy:radius + dy + size, radius + dx:radius + dx + size]
                better = sd < out_d
                out_d[better] = sd[better]
                out_c[better] = pad_c[radius + dy:radius + dy + size, radius + dx:radius + dx + size][better]
        depth, rgb = out_d, out_c
    depth[~np.isfinite(depth)] = 0.0
    return Image.fromarray(rgb, mode="RGB"), depth.astype(np.float32)


def view_radius(box_size: float, camera_dist_factor: float, elevation_deg: float) -> float:
    """Ground radius around the target that can appear in a framed oblique view."""
    cam_dist = camera_dist_factor * box_size
    height = cam_dist * np.sin(np.radians(elevation_deg))
    half_fov = np.arctan(0.5 * box_size / cam_dist)
    depression = max(np.radians(elevation_deg) - half_fov, np.radians(3.0))
    return float(cam_dist * np.cos(np.radians(elevation_deg)) + height / np.tan(depression))


def render_views(grid: PointGrid, world_xyxy, *, target_z: Optional[float] = None, output_size: int = 384,
                 scale: float = 3.0, min_size: float = 20.0, max_size: float = 100.0,
                 elevation_deg: float = 30.0, camera_dist_factor: float = 2.0, point_size: float = 4.0,
                 only_direction: Optional[str] = None, min_points: int = 5) -> Optional[dict]:
    """Render the four (or one) cardinal oblique views over an XY region.

    Returns ``{"views": [...4], "depths": [...4], "cameras": [...4], "target_z"}``
    with ``None`` in non-requested slots, or ``None`` if the region is empty.
    """
    x0, y0, x1, y1 = (float(v) for v in world_xyxy)
    local, _ = grid.crop(x0, y0, x1, y1)
    if len(local) < min_points:
        pad = 0.3 * max(x1 - x0, y1 - y0) / 2.0
        local, _ = grid.crop(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        if len(local) < min_points:
            return None
    if target_z is None:
        target_z = float((local[:, 2].min() + local[:, 2].max()) / 2.0)
    cameras = region_cameras([x0, y0, x1, y1], target_z, output_size=output_size, scale=scale,
                             min_size=min_size, max_size=max_size, elevation_deg=elevation_deg,
                             camera_dist_factor=camera_dist_factor)
    reach = view_radius(cameras[0]["box_size"], camera_dist_factor, elevation_deg)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    pts, cols = grid.crop(cx - reach, cy - reach, cx + reach, cy + reach)
    views: list = [None] * 4
    depths: list = [None] * 4
    wanted = range(4) if only_direction is None else [DIRECTION_INDEX[only_direction]]
    for i in wanted:
        views[i], depths[i] = splat(pts, cols, cameras[i]["extrinsic"], cameras[i]["intrinsic"],
                                    output_size, point_size)
    return {"views": views, "depths": depths, "cameras": cameras, "target_z": target_z}
