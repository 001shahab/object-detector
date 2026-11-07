# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# Camera conventions (azimuth order, extrinsic row layout and focal-length
# rule) match the Urbex reference cache format (https://github.com/AMAP-ML/Urbex,
# Copyright 2026 The Urbex Authors, Apache-2.0) so cached views are
# interchangeable with the public release.
"""Pinhole cameras for cardinal-direction oblique views.

Azimuths: east 0°, north 90°, west 180°, south 270°. A camera named "east"
sits east of the target and looks west. Extrinsics map world to camera
coordinates with rows ``(right, down, forward)``; intrinsics place the target
region of side ``box_size`` across the full image at the target distance.
"""

from __future__ import annotations

import numpy as np

DIRECTIONS: tuple[tuple[str, float], ...] = (("east", 0.0), ("north", 90.0), ("west", 180.0), ("south", 270.0))
DIRECTION_NAMES = tuple(name for name, _ in DIRECTIONS)
DIRECTION_INDEX = {name: i for i, name in enumerate(DIRECTION_NAMES)}


def eye_from_azimuth(target, distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    offset = distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(target, dtype=np.float64) + offset


def look_at(eye, target, output_size: int, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(extrinsic 4x4, intrinsic 3x3)`` for a camera at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    d = float(np.linalg.norm(forward))
    forward /= d + 1e-9
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    extrinsic = np.eye(4)
    extrinsic[0, :3], extrinsic[1, :3], extrinsic[2, :3] = right, down, forward
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye
    f = output_size * d / box_size
    c = output_size / 2.0 - 0.5
    intrinsic = np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])
    return extrinsic, intrinsic


def region_cameras(world_xyxy, target_z: float, *, output_size: int, scale: float = 1.0,
                   min_size: float = 60.0, max_size: float = 100.0, elevation_deg: float = 30.0,
                   camera_dist_factor: float = 2.0) -> list[dict]:
    """The four cardinal cameras framing an XY region."""
    x0, y0, x1, y1 = (float(v) for v in world_xyxy)
    target = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, float(target_z)])
    box_size = float(np.clip(max(x1 - x0, y1 - y0) * scale, min_size, max_size))
    cameras = []
    for name, az in DIRECTIONS:
        eye = eye_from_azimuth(target, camera_dist_factor * box_size, az, elevation_deg)
        extrinsic, intrinsic = look_at(eye, target, output_size, box_size)
        cameras.append({"direction": name, "eye": eye, "target": target, "box_size": box_size,
                        "output_size": int(output_size), "extrinsic": extrinsic, "intrinsic": intrinsic})
    return cameras


def project(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points ``(N, 3)`` to pixel coordinates ``(N, 2)`` and camera depth ``(N,)``."""
    cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = (cam / safe[:, None]) @ intrinsic.T
    return uv[:, :2], z


def backproject(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
                image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Depth map (possibly downsampled) to world points ``(H, W, 3)`` and a validity mask."""
    h, w = depth.shape
    img_w, img_h = image_size
    u, v = np.meshgrid(np.arange(w) * (img_w / max(1, w)), np.arange(h) * (img_h / max(1, h)))
    z = depth.astype(np.float64)
    valid = z > 0
    xc = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    yc = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    cam = np.stack([xc, yc, z], axis=-1)
    world = (cam - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    return world, valid
