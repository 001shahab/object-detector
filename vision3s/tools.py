# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# Tool semantics, validation limits and response wording follow the Urbex
# reference release (https://github.com/AMAP-ML/Urbex, Copyright 2026 The Urbex
# Authors, Apache-2.0) for checkpoint compatibility.
"""City-aware tools over an interactive multi-view environment (Sec. 3.1, Table 5).

The environment keeps a registry of BEV-aligned views ``v -> (I_v, T_v)``.
Tools are pure functions of their arguments from the agent's perspective:
they address views explicitly by ``view_id`` and every created view is
announced in the tool response, so the dialogue history is the single source
of truth. Each tool returns ``(text, images, info)``; ``info`` carries
privileged shaping rewards and diagnostics that never enter the observation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

from .geometry import center_dist, iou_2d, iou_xyxy, xyxy_to_center
from .landmarks import LandmarkResolver
from .oblique.camera import DIRECTION_NAMES
from .prompts import LANDMARK_TOOL, OBLIQUE_TOOL, TOOL_NAMES, ZOOM_TOOL
from .rewards import DEFAULT_REWARD, RewardConfig, render_reward, zoom_progress_reward
from .views import View, compute_display_size, crop_bev


@dataclass
class ToolConfig:
    max_tool_calls: int = 7
    max_zoom_depth: int = 2
    zoom_dedup_iou: float = 0.8
    min_zoom_side_m: float = 10.0
    max_zoom_side_m: float = 1200.0
    display_max_side: int = 768
    display_patch: int = 32
    display_max_pixels: int = 589_824
    bbox_norm_max: float = 1000.0
    protocol: str = "paper"
    enabled_tools: tuple[str, ...] = TOOL_NAMES
    reward: RewardConfig = field(default_factory=lambda: DEFAULT_REWARD)


class CityToolEnv:
    """View registry plus the three city-aware tools."""

    def __init__(self, cfg: ToolConfig, landmarks: LandmarkResolver, oblique=None,
                 bev_loader: Optional[Callable[[dict], tuple[Image.Image, dict]]] = None):
        self.cfg = cfg
        self.landmarks = landmarks
        self.oblique = oblique
        self.bev_loader = bev_loader
        self.sample: dict = {}
        self.views: dict[str, View] = {}
        self.tool_calls_used = 0
        self._zoom_counter = 0
        self._last_zoom_dist: Optional[float] = None
        self._bev: Optional[Image.Image] = None
        self._bev_mapping: dict = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self, sample: dict) -> View:
        self.sample = sample
        self.views = {}
        self.tool_calls_used = 0
        self._zoom_counter = 0
        self._last_zoom_dist = None
        self._bev, self._bev_mapping = self.bev_loader(sample)
        dw, dh = compute_display_size(*self._bev.size, max_side=self.cfg.display_max_side,
                                      patch=self.cfg.display_patch, max_pixels=self.cfg.display_max_pixels)
        m = self._bev_mapping
        mapping = {"x_min": m["x_min"], "x_max": m["x_max"], "y_min": m["y_min"], "y_max": m["y_max"],
                   "res_x": (m["x_max"] - m["x_min"]) / dw, "res_y": (m["y_max"] - m["y_min"]) / dh}
        glob = View("global", self._bev.resize((dw, dh), Image.Resampling.LANCZOS), mapping)
        self.views["global"] = glob
        return glob

    @property
    def gt_bbox(self) -> Optional[list]:
        return self.sample.get("gt_bbox")

    # ── helpers ──────────────────────────────────────────────────────────────

    def budget_str(self) -> str:
        used, cap = self.tool_calls_used, self.cfg.max_tool_calls
        base = f"budget {used}/{cap}"
        answer = "<answer>{\"bbox_2d\":[x0,y0,x1,y1],\"view_id\":\"...\"}</answer>"
        if cap - used <= 0:
            return (f"{base} — BUDGET EXHAUSTED. Your next assistant turn MUST emit {answer} with NO tool_call. "
                    f"Pick the best candidate from the evidence gathered so far; an empty or missing answer "
                    f"ends the episode without a predicted box.")
        if cap - used == 1:
            return (f"{base} — ONE TOOL CALL LEFT. You may inspect once more, then use the following assistant "
                    f"turn to emit {answer} with NO tool_call. You may also submit the answer now.")
        return base

    def valid_bbox(self, bbox) -> bool:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return False
        m = self.cfg.bbox_norm_max
        return x0 < x1 and y0 < y1 and 0 <= x0 and 0 <= y0 and x1 <= m and y1 <= m

    def project(self, view_id: str, bbox) -> Optional[list[float]]:
        view = self.views.get(view_id) if isinstance(view_id, str) else None
        if view is None or not self.valid_bbox(bbox):
            return None
        return view.norm_to_world([float(v) for v in bbox], self.cfg.bbox_norm_max)

    def score(self, bbox, view_id: str) -> dict:
        """Project a committed answer to world coordinates and score it against the target."""
        world = self.project(view_id, bbox)
        gt = self.gt_bbox
        if world is None or not gt:
            return {"world_bbox": world, "iou_2d": 0.0, "center_dist": float("inf")}
        pred = xyxy_to_center(world)
        return {"world_bbox": world, "iou_2d": float(iou_2d(pred, gt)), "center_dist": float(center_dist(pred, gt))}

    def _error(self, text: str, code: str, **extra) -> tuple[str, list, dict]:
        return f"error: {text} {self.budget_str()}", [], {"error": code, **extra}

    def call(self, name: str, args: dict) -> tuple[str, list, dict]:
        """Validate and execute a tool. Unknown or disabled tools are errors."""
        start = time.perf_counter()
        if name not in self.cfg.enabled_tools:
            valid = sorted(self.cfg.enabled_tools)
            text, images, info = f"error: unknown tool: {name!r}. Valid tools: {valid}", [], {"error": "unknown_tool"}
        else:
            text, images, info = {LANDMARK_TOOL: self.landmark_query, ZOOM_TOOL: self.zoom_in,
                                  OBLIQUE_TOOL: self.oblique_render}[name](args)
        info["latency_s"] = time.perf_counter() - start
        return text, images, info

    # ── Landmark Query ───────────────────────────────────────────────────────

    def landmark_query(self, args: dict) -> tuple[str, list, dict]:
        self.tool_calls_used += 1
        raw = args.get("names", args.get("name", args.get("landmark_name")))
        if isinstance(raw, str):
            names = [raw.strip()] if raw.strip() else []
        elif isinstance(raw, list):
            names = [str(n).strip() for n in raw if str(n).strip()]
        else:
            names = []
        if not names:
            return self._error("get_landmark_xy needs non-empty 'names'.", "empty_names")
        view_id = args.get("view_id", args.get("source_view_id", "global"))
        view = self.views.get(view_id) if isinstance(view_id, str) else None
        if view is None:
            view = self.views["global"]
        landmark_type = args.get("landmark_type") if isinstance(args.get("landmark_type"), str) else None
        scene_id = str(self.sample.get("scene_id", ""))
        lines, entries = [], []
        for name in names:
            hit = self.landmarks.resolve(scene_id, name, landmark_type, sample_query=str(self.sample.get("query", "")))
            if hit is None:
                lines.append(f"  - '{name}': no match")
                continue
            wx, wy = hit.xy
            nxy = view.world_to_norm(wx, wy, self.cfg.bbox_norm_max)
            entry = {**hit.as_dict(), "in_view": view.view_id if nxy else None, "norm_xy": nxy}
            entries.append(entry)
            where = f"on view {view.view_id} at ({nxy[0]},{nxy[1]})" if nxy else f"OFF view {view.view_id}"
            if self.cfg.protocol == "paper":
                kind = f" [{hit.type}]" if hit.type else ""
                extent = ""
                if hit.footprint:
                    xs = [p[0] for p in hit.footprint]
                    ys = [p[1] for p in hit.footprint]
                    extent = f"; footprint x∈[{min(xs):.1f},{max(xs):.1f}] y∈[{min(ys):.1f},{max(ys):.1f}]m"
                lines.append(f"  - '{name}' → matched '{hit.name}'{kind}: world ({wx:.1f},{wy:.1f}){extent}; {where}")
            else:
                lines.append(f"  - '{name}' → world ({wx:.1f},{wy:.1f}); {where}")
        text = "get_landmark_xy results:\n" + "\n".join(lines) + f"\n{self.budget_str()}"
        return text, [], {"landmarks": entries, "n_resolved": len(entries), "tool_reward": 0.0}

    # ── Zoom-in ──────────────────────────────────────────────────────────────

    def zoom_in(self, args: dict) -> tuple[str, list, dict]:
        self.tool_calls_used += 1
        view_id = args.get("view_id", args.get("source_view_id", "global"))
        bbox = args.get("bbox_2d", args.get("crop_box", args.get("bbox")))
        view = self.views.get(view_id) if isinstance(view_id, str) else None
        if view is None:
            return self._error(f"zoom_in unknown view_id={view_id!r}. Known: {list(self.views)}.", "unknown_view")
        if not self.valid_bbox(bbox):
            return self._error("zoom_in needs bbox_2d=[x0,y0,x1,y1] with x0<x1 and y0<y1, expressed in view_id's "
                               "coordinate frame.", "invalid_bbox")
        bbox = [float(v) for v in bbox]
        if view.depth >= self.cfg.max_zoom_depth:
            return self._error(f"zoom_in cannot nest deeper than depth {self.cfg.max_zoom_depth} (view {view_id} is "
                               f"at depth {view.depth}). Pick a different parent view_id (e.g. 'global') and try "
                               f"again.", "max_depth")
        for vid, other in self.views.items():
            if other.parent_id == view_id and other.bbox_in_parent is not None:
                overlap = iou_xyxy(bbox, other.bbox_in_parent)
                if overlap >= self.cfg.zoom_dedup_iou:
                    return self._error(f"zoom region too similar to existing view {vid} (IoU={overlap:.2f}); pick a "
                                       f"different area or reuse {vid} via view_id arg.", "duplicate_zoom")
        world = view.norm_to_world(bbox, self.cfg.bbox_norm_max)
        side = max(world[2] - world[0], world[3] - world[1])
        if side < self.cfg.min_zoom_side_m:
            return self._error(f"zoom region too small ({side:.1f}m < {self.cfg.min_zoom_side_m:.0f}m).", "too_small")
        if side > self.cfg.max_zoom_side_m:
            return self._error(f"zoom region too large ({side:.1f}m > {self.cfg.max_zoom_side_m:.0f}m).", "too_large")
        max_side = self.cfg.display_max_side
        requested = args.get("output_max_side", args.get("scale"))
        if self.cfg.protocol == "paper" and isinstance(requested, (int, float)) and requested > 0:
            max_side = int(min(self.cfg.display_max_side, max(self.cfg.display_patch, requested)))
        crop, mapping = crop_bev(self._bev, self._bev_mapping, world, max_side=max_side,
                                 patch=self.cfg.display_patch, max_pixels=self.cfg.display_max_pixels)
        self._zoom_counter += 1
        new_id = f"Z{self._zoom_counter}"
        new_view = View(new_id, crop, mapping, depth=view.depth + 1, parent_id=view_id,
                        bbox_in_parent=[round(v) for v in bbox], kind="zoom")
        self.views[new_id] = new_view
        wb = new_view.world_bounds
        cw, ch = crop.size
        m = self.cfg.bbox_norm_max
        frame = "[0,1] floats" if m == 1.0 else f"[0,{int(m)}] integers"
        text = (f"zoom_in ok → new view '{new_id}' (parent={view_id}, depth={new_view.depth})\n"
                f"  world_bounds: x∈[{wb[0]:.1f},{wb[2]:.1f}]m  y∈[{wb[1]:.1f},{wb[3]:.1f}]m  (side ≈ {side:.1f}m)\n"
                f"  image frame: {cw}×{ch} px; coord frame: {frame}\n")
        if self.cfg.protocol == "paper":
            text += (f"  crop in {view_id}: [{', '.join(str(round(v)) for v in bbox)}]; view-to-world: "
                     f"x = {wb[0]:.2f} + {mapping['res_x']:.4f}·px, y = {wb[3]:.2f} − {mapping['res_y']:.4f}·py "
                     f"(metres per pixel)\n")
        text += self.budget_str()

        reward = 0.0
        gt = self.gt_bbox
        if gt:
            curr = ((0.5 * (world[0] + world[2]) - float(gt[0])) ** 2 + (0.5 * (world[1] + world[3]) - float(gt[1])) ** 2) ** 0.5
            reward = zoom_progress_reward(self._last_zoom_dist, curr, self.cfg.reward)
            self._last_zoom_dist = curr
        return text, [crop], {"view_id": new_id, "parent_id": view_id, "world_bbox": wb, "crop_world": world,
                              "tool_reward": reward}

    # ── Oblique Rendering ────────────────────────────────────────────────────

    def oblique_render(self, args: dict) -> tuple[str, list, dict]:
        self.tool_calls_used += 1
        view_id = args.get("view_id", args.get("source_view_id", "global"))
        bbox = args.get("bbox_2d", args.get("candidate_box", args.get("bbox")))
        direction = str(args.get("direction") or "").lower()
        view = self.views.get(view_id) if isinstance(view_id, str) else None
        if view is None:
            return self._error(f"render_oblique unknown view_id={view_id!r}.", "unknown_view")
        if not self.valid_bbox(bbox):
            return self._error("render_oblique needs bbox_2d=[x0,y0,x1,y1] with x0<x1 and y0<y1, in view_id's "
                               "coordinate frame.", "invalid_bbox")
        if direction not in DIRECTION_NAMES:
            return self._error(f"render_oblique needs `direction` to be one of {sorted(DIRECTION_NAMES)}, got "
                               f"{direction!r}.", "invalid_direction")
        world = view.norm_to_world([float(v) for v in bbox], self.cfg.bbox_norm_max)
        margin = args.get("margin") if self.cfg.protocol == "paper" else None
        margin = float(margin) if isinstance(margin, (int, float)) and margin > 0 else None
        region = f"[{world[0]:.1f},{world[1]:.1f},{world[2]:.1f},{world[3]:.1f}]"
        image, meta = (None, {"source": "disabled"})
        if self.oblique is not None:
            image, meta = self.oblique.render(str(self.sample.get("scene_id", "")),
                                              str(self.sample.get("dataset", "cityrefer")), world, direction, margin)
        if image is None:
            return self._error(f"render_oblique: no oblique views available for region {region} (cache miss / no "
                               f"scene point cloud). Decide from BEV alone.", "no_oblique_views", world_bbox=world)
        text = (f"render_oblique ok → 1 oblique view from the {direction.upper()} over world {region} "
                f"(view_id={view_id}). The image shows the proposed bbox as a RED 3D wireframe. If it lands on the "
                f"queried object, commit by emitting an assistant turn with NO tool_call and an "
                f"<answer>{{\"bbox_2d\":[...], \"view_id\":\"{view_id}\"}}</answer> tag. To inspect another angle, "
                f"call render_oblique again with a different `direction` (each call costs 1 tool budget). Otherwise "
                f"revise via zoom_in / new bbox.\n{self.budget_str()}")
        overlap = float(iou_2d(xyxy_to_center(world), self.gt_bbox)) if self.gt_bbox else 0.0
        return text, [image], {"world_bbox": world, "direction": direction, "view_id": view_id,
                               "render_source": meta.get("source"), "margin": margin, "candidate_iou": overlap,
                               "tool_reward": render_reward(overlap, self.cfg.reward)}
