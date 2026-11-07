# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Box geometry shared by the environment, rewards and evaluation.

Two box representations are used throughout:

* ``center`` boxes ``[cx, cy, cz, dx, dy, dz, ...]`` with *full* dimensions in
  metres. Ground-truth boxes use this layout; trailing class / object metadata
  is ignored.
* ``xyxy`` world boxes ``[x_min, y_min, x_max, y_max]`` on the ground plane.
  Predictions projected from a BEV-aligned view use this layout.

Conversions are explicit; no function guesses the layout of its input.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Box = Sequence[float]


def center_to_xyxy(box: Box) -> list[float]:
    cx, cy, dx, dy = float(box[0]), float(box[1]), float(box[3]), float(box[4])
    return [cx - dx / 2.0, cy - dy / 2.0, cx + dx / 2.0, cy + dy / 2.0]


def xyxy_to_center(box: Box, z: float = 0.0, dz: float = 1.0) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    return [0.5 * (x0 + x1), 0.5 * (y0 + y1), z, max(0.0, x1 - x0), max(0.0, y1 - y0), dz]


def center_to_minmax3d(box: Box) -> np.ndarray:
    c = np.asarray(box[:3], dtype=np.float64)
    d = np.asarray(box[3:6], dtype=np.float64)
    return np.concatenate([c - d / 2.0, c + d / 2.0])


def area_xyxy(box: Box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def iou_xyxy(a: Box, b: Box) -> float:
    iw = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    ih = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = iw * ih
    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / union if union > 1e-6 else 0.0


def ciou_xyxy(a: Box, b: Box) -> float:
    """Complete IoU: IoU minus centre-distance and aspect-ratio penalties."""
    iou = iou_xyxy(a, b)
    aw, ah = max(0.0, a[2] - a[0]), max(0.0, a[3] - a[1])
    bw, bh = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
    rho_sq = (0.5 * (a[0] + a[2] - b[0] - b[2])) ** 2 + (0.5 * (a[1] + a[3] - b[1] - b[3])) ** 2
    enc_w = max(a[2], b[2]) - min(a[0], b[0])
    enc_h = max(a[3], b[3]) - min(a[1], b[1])
    diag_sq = enc_w ** 2 + enc_h ** 2
    distance_penalty = rho_sq / diag_sq if diag_sq > 1e-12 else 0.0
    if min(aw, ah, bw, bh) <= 1e-12:
        return iou - distance_penalty
    v = 4.0 / math.pi ** 2 * (math.atan(bw / bh) - math.atan(aw / ah)) ** 2
    alpha = v / max(1e-12, 1.0 - iou + v)
    return iou - distance_penalty - alpha * v


def iou_2d(a: Box, b: Box) -> float:
    """BEV IoU between two ``center`` boxes (Z ignored)."""
    return iou_xyxy(center_to_xyxy(a), center_to_xyxy(b))


def iou_3d(a: Box, b: Box) -> float:
    """Axis-aligned 3D IoU between two ``center`` boxes."""
    return iou_minmax3d(center_to_minmax3d(a), center_to_minmax3d(b))


def iou_minmax3d(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.maximum(0.0, np.minimum(a[3:], b[3:]) - np.maximum(a[:3], b[:3])).prod()
    vol_a = np.maximum(0.0, a[3:] - a[:3]).prod()
    vol_b = np.maximum(0.0, b[3:] - b[:3]).prod()
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def center_dist(a: Box, b: Box) -> float:
    """Ground-plane distance between the centres of two ``center`` boxes."""
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def xyxy_center(box: Box) -> tuple[float, float]:
    return 0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3]))


def point_in_xyxy(x: float, y: float, box: Box) -> bool:
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])
