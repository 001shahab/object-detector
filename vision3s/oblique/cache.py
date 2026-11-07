# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# The on-disk cache layout and its parameters follow the Urbex reference
# release (https://github.com/AMAP-ML/Urbex, Copyright 2026 The Urbex Authors,
# Apache-2.0) so caches built here and the published caches are interchangeable.
"""Oblique-view cache (Sec. 4.2 / Appendix D.7) and the Oblique Rendering backend.

Layout per scene::

    <root>/<scene_id>/
        scene_meta.json   parameters, ground_z, region_count
        regions.npy       float32 (N, 4) XY world regions [x0, y0, x1, y1]
        index.pkl         {"tree": cKDTree, "feats": (N, 2) region centres}
        imgs/{id:06d}_{e,n,w,s}.jpg
        cams/{id:06d}.npz extrinsics (4,4,4), intrinsics (4,3,3), targets (4,3), box_sizes (4,)
        depths/{id:06d}.npz depths uint16 (4, H, W), metres = value * depth_scale

At query time the region nearest to the candidate centre is selected, the
candidate box's 3D extent is inferred from the cached depth inside its
footprint, and the cuboid is drawn as a red wireframe with depth-tested
occlusion. Rendering on demand from the raw point cloud is the fallback when
no cached region is close enough (or the only path when no cache exists).
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from .camera import DIRECTION_INDEX, DIRECTION_NAMES, backproject, project, region_cameras
from .renderer import PointGrid, render_views

logger = logging.getLogger(__name__)

CARDINALS = ("e", "n", "w", "s")
_CUBOID_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def cuboid_corners(world_xyxy, z_min: float, z_max: float) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in world_xyxy[:4])
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z_min, z_max)], dtype=np.float64)


def _sample_depth(depth: Optional[np.ndarray], u: float, v: float, image_size: tuple[int, int],
                  window: int = 1) -> float:
    if depth is None:
        return 0.0
    h, w = depth.shape
    iu, iv = int(round(u * w / max(1, image_size[0]))), int(round(v * h / max(1, image_size[1])))
    if iu < 0 or iu >= w or iv < 0 or iv >= h:
        return 0.0
    patch = depth[max(0, iv - window):iv + window + 1, max(0, iu - window):iu + window + 1]
    pos = patch[patch > 0]
    return float(pos.min()) if pos.size else 0.0


def draw_cuboid(image: Image.Image, corners: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
                depth: Optional[np.ndarray], color=(255, 0, 0), width: int = 3, samples: int = 32,
                occlusion_eps_m: float = 0.8) -> Image.Image:
    """Draw the 12 cuboid edges, skipping sub-segments hidden behind scene geometry."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    ts = np.linspace(0.0, 1.0, samples + 1)[:, None]
    for a, b in _CUBOID_EDGES:
        uv, zc = project(corners[a] * (1 - ts) + corners[b] * ts, extrinsic, intrinsic)
        for i in range(samples):
            if zc[i] <= 0 and zc[i + 1] <= 0:
                continue
            (ua, va), (ub, vb) = uv[i], uv[i + 1]
            if max(ua, ub) < -w or min(ua, ub) > 2 * w or max(va, vb) < -h or min(va, vb) > 2 * h:
                continue
            if depth is not None:
                scene_z = _sample_depth(depth, 0.5 * (ua + ub), 0.5 * (va + vb), out.size)
                if scene_z > 0 and 0.5 * (zc[i] + zc[i + 1]) > scene_z + occlusion_eps_m:
                    continue
            draw.line([(float(ua), float(va)), (float(ub), float(vb))], fill=color, width=width)
    return out


def draw_marker(image: Image.Image, world_xyxy, z: float, extrinsic: np.ndarray, intrinsic: np.ndarray,
                color=(255, 0, 0), radius: int = 10) -> Image.Image:
    """Centre dot used when no depth is available to infer the cuboid height."""
    x0, y0, x1, y1 = (float(v) for v in world_xyxy[:4])
    uv, zc = project(np.array([[0.5 * (x0 + x1), 0.5 * (y0 + y1), z]]), extrinsic, intrinsic)
    out = image.copy()
    if zc[0] <= 0:
        return out
    cx, cy = float(uv[0, 0]), float(uv[0, 1])
    draw = ImageDraw.Draw(out)
    draw.ellipse([cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2], fill=(255, 255, 255))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    return out


def infer_z_range(depths: list, extrinsics, intrinsics, world_xyxy, image_size,
                  min_height: float = 0.5, max_height: float = 60.0) -> Optional[tuple[float, float]]:
    """Local ground (min over views) and top (median over views) of points inside the footprint."""
    x0, y0, x1, y1 = (float(v) for v in world_xyxy[:4])
    grounds, tops = [], []
    for depth, ext, K in zip(depths, extrinsics, intrinsics):
        if depth is None:
            continue
        world, valid = backproject(depth, ext, K, image_size)
        m = valid & (world[..., 0] >= x0) & (world[..., 0] <= x1) & (world[..., 1] >= y0) & (world[..., 1] <= y1)
        if m.any():
            zs = world[..., 2][m]
            grounds.append(float(zs.min()))
            tops.append(float(zs.max()))
    if not grounds:
        return None
    base = float(np.min(grounds))
    height = float(np.clip(np.median(tops) - base, min_height, max_height))
    return base, base + height


# ── cache construction ───────────────────────────────────────────────────────

def _axis_centers(lower: float, upper: float, size: float, step: float) -> np.ndarray:
    if upper - lower <= size:
        return np.array([(lower + upper) / 2.0])
    centers = np.arange(lower + size / 2.0, upper - size / 2.0 + 1e-9, step)
    if centers.size == 0 or (upper - size / 2.0) - centers[-1] > 1e-6:
        centers = np.append(centers, upper - size / 2.0)
    return centers


def sample_regions(x_min, y_min, x_max, y_max, *, region_size: float, step: float) -> np.ndarray:
    half = region_size / 2.0
    return np.array([[cx - half, cy - half, cx + half, cy + half]
                     for cx in _axis_centers(x_min, x_max, region_size, step)
                     for cy in _axis_centers(y_min, y_max, region_size, step)], dtype=np.float32)


def _min_pool(depth: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return depth
    h, w = depth.shape[0] // factor, depth.shape[1] // factor
    blocks = depth[:h * factor, :w * factor].reshape(h, factor, w, factor)
    pooled = np.where(blocks > 0, blocks, np.inf).min(axis=(1, 3))
    pooled[~np.isfinite(pooled)] = 0.0
    return pooled


def build_scene_cache(scene_id: str, dataset: str, points: np.ndarray, colors: np.ndarray, out_root: str | Path, *,
                      region_size: float = 40.0, step: float = 15.0, output_size: int = 512,
                      jpeg_quality: int = 82, elevation_deg: float = 30.0, camera_dist_factor: float = 1.3,
                      point_size: float = 5.0, default_bbox_height: float = 3.0, depth_scale: float = 0.05,
                      depth_downsample: int = 4, progress: bool = False) -> dict:
    """Pre-render four oblique views for every sampled region of a scene (atomic publish)."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    final = out_root / scene_id
    if final.exists():
        return {"scene_id": scene_id, "status": "skipped", "output": str(final)}
    finite = np.isfinite(points[:, :3]).all(axis=1)
    points, colors = points[finite, :3], colors[finite]
    grid = PointGrid(points, colors)
    lo, hi = points.min(axis=0), points.max(axis=0)
    scene_z_mid = float((lo[2] + hi[2]) / 2.0)
    regions = sample_regions(lo[0], lo[1], hi[0], hi[1], region_size=region_size, step=step)
    stage = Path(tempfile.mkdtemp(prefix=f".{scene_id}.building-", dir=out_root))
    for sub in ("imgs", "cams", "depths"):
        (stage / sub).mkdir()
    kept = []
    try:
        iterator = enumerate(regions)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(regions), desc=scene_id)
        for _, region in iterator:
            pad = 0.3 * region_size / 2.0
            local, _ = grid.crop(region[0] - pad, region[1] - pad, region[2] + pad, region[3] + pad)
            target_z = float(np.percentile(local[:, 2], 5.0)) + 3.0 if len(local) >= 5 else scene_z_mid
            rendered = render_views(grid, region.tolist(), target_z=target_z, output_size=output_size, scale=1.0,
                                    min_size=region_size, max_size=region_size, elevation_deg=elevation_deg,
                                    camera_dist_factor=camera_dist_factor, point_size=point_size)
            if rendered is None:
                continue
            entry = len(kept)
            for card, view in zip(CARDINALS, rendered["views"]):
                view.save(stage / "imgs" / f"{entry:06d}_{card}.jpg", "JPEG", quality=jpeg_quality, optimize=True)
            cams = rendered["cameras"]
            np.savez_compressed(stage / "cams" / f"{entry:06d}.npz",
                                extrinsics=np.stack([c["extrinsic"] for c in cams]),
                                intrinsics=np.stack([c["intrinsic"] for c in cams]),
                                targets=np.stack([c["target"] for c in cams]),
                                box_sizes=np.array([c["box_size"] for c in cams]))
            depth = np.stack([_min_pool(d, depth_downsample) for d in rendered["depths"]])
            np.savez_compressed(stage / "depths" / f"{entry:06d}.npz",
                                depths=np.clip(depth / depth_scale, 0, 65535).astype(np.uint16))
            kept.append(region)
        if not kept:
            raise RuntimeError(f"scene {scene_id} produced no cache regions")
        kept_arr = np.asarray(kept, dtype=np.float32)
        feats = np.stack([[0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])] for r in kept_arr]).astype(np.float32)
        np.save(stage / "regions.npy", kept_arr)
        with (stage / "index.pkl").open("wb") as fh:
            pickle.dump({"tree": cKDTree(feats), "feats": feats}, fh)
        meta = {
            "scene_id": scene_id, "dataset": dataset, "region_size": region_size, "step": step,
            "output_size": output_size, "elevation_deg": elevation_deg, "camera_dist_factor": camera_dist_factor,
            "point_size": point_size, "jpeg_quality": jpeg_quality, "save_depth": True, "depth_scale": depth_scale,
            "depth_downsample": depth_downsample, "depth_format": "npz", "default_bbox_height": default_bbox_height,
            "ground_z": scene_z_mid, "region_count": len(kept_arr), "sampled_region_count": len(regions),
            "failed_region_count": len(regions) - len(kept_arr), "renderer": "vision3s.numpy_splat",
        }
        (stage / "scene_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        stage.rename(final)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return {"scene_id": scene_id, "status": "built", "output": str(final), "regions": len(kept)}


# ── runtime lookup ───────────────────────────────────────────────────────────

class SceneCache:
    """Nearest-region lookup over one scene's cache (lazy, process-wide registry)."""

    _registry: dict[tuple[str, str], "SceneCache"] = {}

    def __init__(self, root: str | Path, scene_id: str, dist_threshold: float = 60.0):
        self.root = Path(root) / scene_id
        self.scene_id = scene_id
        self.dist_threshold = float(dist_threshold)
        self._loaded = False
        self._tree: Optional[cKDTree] = None
        self.meta: dict = {}

    @classmethod
    def get(cls, root: str | Path, scene_id: str) -> "SceneCache":
        key = (str(root), scene_id)
        if key not in cls._registry:
            cls._registry[key] = cls(root, scene_id)
        return cls._registry[key]

    def _load(self) -> bool:
        if not self._loaded:
            self._loaded = True
            regions_path, meta_path = self.root / "regions.npy", self.root / "scene_meta.json"
            if regions_path.exists() and meta_path.exists():
                regions = np.load(regions_path)
                self._tree = cKDTree(0.5 * (regions[:, :2] + regions[:, 2:4]))
                self.meta = json.loads(meta_path.read_text())
        return self._tree is not None

    def _depths(self, entry: int) -> Optional[list]:
        if not self.meta.get("save_depth"):
            return None
        fmt, scale = str(self.meta.get("depth_format", "npz")), float(self.meta.get("depth_scale", 0.05))
        try:
            if fmt == "npz":
                return list(np.load(self.root / "depths" / f"{entry:06d}.npz")["depths"].astype(np.float32) * scale)
            out = []
            for card in CARDINALS:
                im = Image.open(self.root / "depths" / f"{entry:06d}_{card}.{fmt}")
                if fmt == "png":
                    out.append(np.asarray(im, dtype=np.float32) * scale)
                else:
                    out.append(np.asarray(im.convert("L"), dtype=np.float32) * (150.0 / 255.0))
            return out
        except (OSError, KeyError, ValueError):
            return None

    def lookup(self, world_xyxy, direction: str) -> Optional[Image.Image]:
        """Annotated cached view from ``direction`` for a candidate world box, or ``None``."""
        if not self._load():
            return None
        x0, y0, x1, y1 = (float(v) for v in world_xyxy[:4])
        dists, idxs = self._tree.query([0.5 * (x0 + x1), 0.5 * (y0 + y1)], k=min(3, self._tree.n))
        for d, entry in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
            if d > self.dist_threshold:
                break
            entry = int(entry)
            paths = [self.root / "imgs" / f"{entry:06d}_{c}.jpg" for c in CARDINALS]
            cam_path = self.root / "cams" / f"{entry:06d}.npz"
            if not all(p.exists() for p in paths) or not cam_path.exists():
                continue
            slot = DIRECTION_INDEX[direction]
            image = Image.open(paths[slot]).convert("RGB")
            cams = np.load(cam_path)
            exts, Ks = cams["extrinsics"], cams["intrinsics"]
            depths = self._depths(entry)
            if depths is not None:
                zr = infer_z_range(depths, exts, Ks, [x0, y0, x1, y1], image.size,
                                   float(self.meta.get("min_bbox_height", 0.5)),
                                   float(self.meta.get("max_bbox_height", 60.0)))
                if zr is None:
                    base = float(self.meta.get("ground_z", 0.0))
                    zr = (base, base + float(self.meta.get("default_bbox_height", 3.0)))
                return draw_cuboid(image, cuboid_corners([x0, y0, x1, y1], *zr), exts[slot], Ks[slot], depths[slot])
            return draw_marker(image, [x0, y0, x1, y1], float(self.meta.get("ground_z", 0.0)), exts[slot], Ks[slot])
        return None


class ObliqueRenderer:
    """Oblique Rendering backend: cached views first, on-demand rendering as fallback."""

    def __init__(self, cache_root: Optional[str] = None, require_cache: bool = False,
                 point_cloud_loader=None, output_size: int = 384, elevation_deg: float = 30.0):
        self.cache_root = cache_root
        self.require_cache = require_cache
        self.point_cloud_loader = point_cloud_loader
        self.output_size = output_size
        self.elevation_deg = elevation_deg
        self._grids: dict[str, PointGrid] = {}

    def _grid(self, scene_id: str, dataset: str) -> Optional[PointGrid]:
        if scene_id not in self._grids:
            loaded = self.point_cloud_loader(scene_id, dataset) if self.point_cloud_loader else None
            if loaded is None:
                return None
            self._grids[scene_id] = PointGrid(*loaded)
        return self._grids[scene_id]

    def render(self, scene_id: str, dataset: str, world_xyxy, direction: str,
               margin: Optional[float] = None) -> tuple[Optional[Image.Image], dict]:
        if direction not in DIRECTION_NAMES:
            raise ValueError(f"direction must be one of {DIRECTION_NAMES}")
        if self.cache_root and margin is None:
            image = SceneCache.get(self.cache_root, scene_id).lookup(world_xyxy, direction)
            if image is not None:
                return image, {"source": "cache"}
        if self.require_cache and margin is None:
            return None, {"source": "cache_miss"}
        grid = self._grid(scene_id, dataset)
        if grid is None:
            return None, {"source": "no_point_cloud"}
        x0, y0, x1, y1 = (float(v) for v in world_xyxy[:4])
        frame = [x0, y0, x1, y1]
        if margin:
            mx, my = margin * (x1 - x0), margin * (y1 - y0)
            frame = [x0 - mx, y0 - my, x1 + mx, y1 + my]
        rendered = render_views(grid, frame, output_size=self.output_size, elevation_deg=self.elevation_deg,
                                only_direction=direction, **({"scale": 1.0} if margin else {}))
        if rendered is None:
            return None, {"source": "empty_region"}
        slot = DIRECTION_INDEX[direction]
        inside, _ = grid.crop(x0, y0, x1, y1)
        if len(inside):
            zr = (float(inside[:, 2].min()), float(inside[:, 2].max()))
        else:
            zr = (rendered["target_z"], rendered["target_z"] + 3.0)
        cam = rendered["cameras"][slot]
        image = draw_cuboid(rendered["views"][slot], cuboid_corners([x0, y0, x1, y1], *zr), cam["extrinsic"],
                            cam["intrinsic"], rendered["depths"][slot])
        return image, {"source": "live"}


__all__ = ["ObliqueRenderer", "SceneCache", "build_scene_cache", "region_cameras", "sample_regions"]
