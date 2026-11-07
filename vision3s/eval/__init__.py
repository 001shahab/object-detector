# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""With-GT Hit@k, candidate-free 2D/3D metrics and diagnostics (Sec. 4, Appendix D)."""

from .hit_k import compute_hit_k
from .metrics_3d import evaluate_3d

__all__ = ["compute_hit_k", "evaluate_3d"]
