# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Figures: training dynamics (Fig. 7), search diagnostics (Fig. 9) and
trajectory overlays (Figs. 8 and 10: green GT, red prediction, dashed blue zoom)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from .views import world_to_pixel


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training(metrics_jsonl: str | Path, out_png: str | Path) -> None:
    rows = [json.loads(line) for line in Path(metrics_jsonl).read_text().splitlines() if line.strip()]
    plt = _plt()
    panels = [("train/reward", "Reward"), ("train/iou", "Train IoU"), ("val/iou", "Val IoU"),
              ("train/tool_calls", "Tool calls / trajectory"), ("train/response_tokens", "Response tokens"),
              ("train/kl", "KL")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (key, title) in zip(axes.flat, panels):
        pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
        if pts:
            ax.plot(*zip(*pts), marker="o" if key.startswith("val/") else None, lw=1.4)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    tool_keys = sorted({k for r in rows for k in r if k.startswith("train/calls/")})
    if tool_keys:
        ax = axes.flat[3]
        for k in tool_keys:
            pts = [(r["step"], r[k]) for r in rows if k in r]
            ax.plot(*zip(*pts), lw=1, ls="--", label=k.split("/")[-1])
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_search_diagnostics(results: dict[str, dict], out_png: str | Path) -> None:
    """``results``: ``{dataset_label: search_diagnostics(...)}``."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, res in results.items():
        for ax, key in ((a, "regional_search_by_landmark_distance"), (b, "local_disambiguation_by_distractors")):
            table = res[key]
            xs = list(table)
            ys = [table[x]["success"] if table[x]["success"] is not None else float("nan") for x in xs]
            ax.plot(xs, ys, marker="o", label=label)
    a.set_title("a  Landmark distance")
    a.set_xlabel("Target–landmark distance (m)")
    b.set_title("b  Same-class distractors")
    b.set_xlabel("Same-class distractors within 50 m")
    for ax in (a, b):
        ax.set_ylabel("success (%)")
        ax.set_ylim(0, 70)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _rect(draw, img_size, mapping, xyxy, color, width=3, dashed=False):
    x0, y1 = world_to_pixel(mapping, xyxy[0], xyxy[1])
    x1, y0 = world_to_pixel(mapping, xyxy[2], xyxy[3])
    box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    if not dashed:
        draw.rectangle(box, outline=color, width=width)
        return
    step = 10
    for a, b, horizontal, fixed in ((box[0], box[2], True, box[1]), (box[0], box[2], True, box[3]),
                                    (box[1], box[3], False, box[0]), (box[1], box[3], False, box[2])):
        p = a
        while p < b:
            q = min(b, p + step)
            draw.line([(p, fixed), (q, fixed)] if horizontal else [(fixed, p), (fixed, q)], fill=color, width=width)
            p += 2 * step


def draw_trajectory(bev: Image.Image, mapping: dict, record: dict, gt_xyxy: Optional[list] = None,
                    crop_margin_m: Optional[float] = 60.0) -> Image.Image:
    """BEV overlay of one trajectory: zooms dashed blue, prediction red, GT green."""
    img = bev.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    zooms = [t["crop_world"] for t in record.get("tool_trace", []) if t.get("tool") == "zoom_in" and t.get("crop_world")]
    for z in zooms:
        _rect(draw, img.size, mapping, z, (40, 90, 255), width=3, dashed=True)
    if gt_xyxy:
        _rect(draw, img.size, mapping, gt_xyxy, (0, 200, 0), width=3)
    if record.get("world_bbox"):
        _rect(draw, img.size, mapping, record["world_bbox"], (255, 0, 0), width=3)
    if crop_margin_m is None:
        return img
    boxes = zooms + [b for b in (gt_xyxy, record.get("world_bbox")) if b]
    if not boxes:
        return img
    x0 = min(b[0] for b in boxes) - crop_margin_m
    y0 = min(b[1] for b in boxes) - crop_margin_m
    x1 = max(b[2] for b in boxes) + crop_margin_m
    y1 = max(b[3] for b in boxes) + crop_margin_m
    px0, py1 = world_to_pixel(mapping, x0, y0)
    px1, py0 = world_to_pixel(mapping, x1, y1)
    w, h = img.size
    return img.crop((max(0, int(px0)), max(0, int(py0)), min(w, int(px1)), min(h, int(py1))))
