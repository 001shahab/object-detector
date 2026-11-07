# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""STPLS3D-Refer construction (Appendix B.2).

A diagnostic, semi-automatic zero-shot benchmark built from STPLS3D RealWorld
photogrammetry scenes (USC and WMSC). Stages, each exposed as a function and a
CLI step:

1. ``extract_instances`` – DBSCAN over semantic masks. Buildings: planar XY,
   ``eps=2.0 m``, ``min_samples=80``, footprint in ``[4, 5000] m²``.
   Vehicles: XYZ, ``eps=0.6 m``, ``min_samples=30``, footprint in
   ``[1.5, 30] m²``. Geometric filters remove fragments, merged objects and
   ground/noise artifacts; survivors go to manual review.
2. ``fit_georegistration`` – first-order affine transform from 5–7 ground
   control points (local metres -> WGS-84), reporting the RMS residual.
3. ``query_osm_landmarks`` – OpenStreetMap features tagged building, amenity,
   highway, leisure, tourism, office or natural, projected to local
   coordinates and matched to building instances.
4. ``generate_expressions`` – for each target, a centred BEV crop and four
   cardinal oblique views with the target and nearby same-class distractors
   marked, plus target colour, size, distractor relations and nearest
   landmarks, are given to a VLM (Qwen3.6-plus in the paper) that writes three
   concise CityRefer-style expressions.
5. ``filter_expressions`` – automatic checks for length, meta-language,
   landmark-name validity, projection validity and token-level duplication,
   followed by manual acceptance (``apply_review``).
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

BUILDING_PARAMS = {"eps": 2.0, "min_samples": 80, "footprint": (4.0, 5000.0), "dims": 2}
VEHICLE_PARAMS = {"eps": 0.6, "min_samples": 30, "footprint": (1.5, 30.0), "dims": 3}
OSM_TAGS = ("building", "amenity", "highway", "leisure", "tourism", "office", "natural")


# ── 1. instance extraction ───────────────────────────────────────────────────

def _footprint_area(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError

    try:
        return float(ConvexHull(xy).volume)
    except QhullError:
        return 0.0


def dbscan_instances(points: np.ndarray, *, eps: float, min_samples: int, dims: int,
                     voxel: Optional[float] = None) -> np.ndarray:
    """DBSCAN labels for ``points``; ``-1`` is noise.

    Large masks are voxelised (``voxel`` << ``eps``) and clustered with
    per-voxel point counts as ``sample_weight``, which preserves the
    density criterion of point-level DBSCAN at a fraction of the memory.
    """
    from sklearn.cluster import DBSCAN

    coords = points[:, :dims].astype(np.float64)
    if voxel is None:
        voxel = eps / 4.0
    keys = np.floor(coords / voxel).astype(np.int64)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    centers = np.zeros((len(uniq), dims))
    np.add.at(centers, inverse.reshape(-1), coords)
    centers /= counts[:, None]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(centers, sample_weight=counts).labels_
    return labels[inverse.reshape(-1)]


@dataclass
class Instance:
    instance_id: int
    category: str
    bbox: list[float]
    footprint_m2: float
    n_points: int
    mean_color: list[float]
    point_index: np.ndarray = field(repr=False)
    status: str = "candidate"
    reject_reason: str = ""

    def record(self) -> dict:
        return {"object_id": self.instance_id, "object_name": self.category, "bbox": self.bbox,
                "footprint_m2": self.footprint_m2, "n_points": self.n_points, "mean_color": self.mean_color,
                "status": self.status, "reject_reason": self.reject_reason}


def geometric_filter(inst: Instance) -> str:
    """Reject fragments, merged objects and ground/noise artifacts; ``""`` if kept."""
    dx, dy, dz = inst.bbox[3:6]
    long_side, short_side = max(dx, dy), max(1e-6, min(dx, dy))
    if inst.category == "vehicle":
        if dz < 0.8 or dz > 4.5:
            return "height"
        if long_side > 13.0 or long_side / short_side > 5.0:
            return "merged_or_elongated"
        if inst.n_points < 60:
            return "fragment"
    else:
        if dz < 2.0:
            return "ground_or_noise"
        if long_side / short_side > 12.0:
            return "elongated_fragment"
        if inst.n_points < 400:
            return "fragment"
    return ""


def extract_instances(points: np.ndarray, colors: np.ndarray, labels: np.ndarray, *, building_label: int,
                      vehicle_labels: tuple[int, ...]) -> list[Instance]:
    out: list[Instance] = []
    groups = (("building", np.isin(labels, [building_label]), BUILDING_PARAMS),
              ("vehicle", np.isin(labels, list(vehicle_labels)), VEHICLE_PARAMS))
    for category, mask, params in groups:
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        cl = dbscan_instances(points[idx], eps=params["eps"], min_samples=params["min_samples"], dims=params["dims"])
        for c in np.unique(cl[cl >= 0]):
            member = idx[cl == c]
            pts = points[member]
            area = _footprint_area(pts[:, :2])
            lo_area, hi_area = params["footprint"]
            if not lo_area <= area <= hi_area:
                continue
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            bbox = [*((lo + hi) / 2).tolist(), *(hi - lo).tolist()]
            inst = Instance(len(out), category, [float(v) for v in bbox], area, len(member),
                            colors[member].mean(axis=0).round(1).tolist(), member)
            reason = geometric_filter(inst)
            if reason:
                inst.status, inst.reject_reason = "rejected", reason
            out.append(inst)
    return out


# ── 2. geo-registration ──────────────────────────────────────────────────────

_EARTH_R = 6_378_137.0


@dataclass
class GeoRegistration:
    """Affine local metres -> local tangent plane (east, north) around ``lat0, lon0``."""

    matrix: list[list[float]]
    lat0: float
    lon0: float
    rms_m: float

    def _A(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def local_to_wgs84(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        en = np.c_[xy, np.ones(len(xy))] @ self._A().T
        lat = self.lat0 + np.degrees(en[:, 1] / _EARTH_R)
        lon = self.lon0 + np.degrees(en[:, 0] / (_EARTH_R * math.cos(math.radians(self.lat0))))
        return np.c_[lat, lon]

    def wgs84_to_local(self, latlon: np.ndarray) -> np.ndarray:
        latlon = np.atleast_2d(latlon)
        en = _latlon_to_en(latlon, self.lat0, self.lon0)
        A = self._A()
        return np.linalg.solve(A[:, :2], (en - A[:, 2]).T).T


def _latlon_to_en(latlon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    e = np.radians(latlon[:, 1] - lon0) * _EARTH_R * math.cos(math.radians(lat0))
    n = np.radians(latlon[:, 0] - lat0) * _EARTH_R
    return np.c_[e, n]


def fit_georegistration(gcps: list[dict]) -> GeoRegistration:
    """Least-squares affine fit from GCPs ``{"x", "y", "lat", "lon"}`` (5–7 points in the paper)."""
    if len(gcps) < 3:
        raise ValueError("an affine transform needs at least 3 ground-control points")
    local = np.array([[g["x"], g["y"]] for g in gcps], dtype=np.float64)
    latlon = np.array([[g["lat"], g["lon"]] for g in gcps], dtype=np.float64)
    lat0, lon0 = latlon.mean(axis=0)
    en = _latlon_to_en(latlon, lat0, lon0)
    X = np.c_[local, np.ones(len(local))]
    sol, *_ = np.linalg.lstsq(X, en, rcond=None)
    rms = float(np.sqrt(np.mean(np.sum((X @ sol - en) ** 2, axis=1))))
    return GeoRegistration(sol.T.tolist(), float(lat0), float(lon0), rms)


# ── 3. OpenStreetMap landmarks ───────────────────────────────────────────────

def overpass_query(south: float, west: float, north: float, east: float, tags=OSM_TAGS) -> str:
    bbox = f"{south},{west},{north},{east}"
    parts = "".join(f'nwr["{t}"]["name"]({bbox});' for t in tags)
    return f"[out:json][timeout:120];({parts});out geom;"


def fetch_osm(query: str, endpoint: str = "https://overpass-api.de/api/interpreter") -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    with urllib.request.urlopen(urllib.request.Request(endpoint, data=data), timeout=180) as resp:
        return json.loads(resp.read())


def osm_to_landmarks(osm: dict, reg: GeoRegistration, extent: Optional[list[float]] = None,
                     tags=OSM_TAGS) -> list[dict]:
    out, seen = [], set()
    for el in osm.get("elements", []):
        name = (el.get("tags") or {}).get("name")
        if not name:
            continue
        kind = next((t for t in tags if t in el.get("tags", {})), "")
        if "geometry" in el:
            latlon = np.array([[p["lat"], p["lon"]] for p in el["geometry"]])
        elif "lat" in el:
            latlon = np.array([[el["lat"], el["lon"]]])
        elif "center" in el:
            latlon = np.array([[el["center"]["lat"], el["center"]["lon"]]])
        elif "bounds" in el:
            b = el["bounds"]
            latlon = np.array([[0.5 * (b["minlat"] + b["maxlat"]), 0.5 * (b["minlon"] + b["maxlon"])]])
        else:
            continue
        local = reg.wgs84_to_local(latlon)
        center = local.mean(axis=0)
        if extent is not None and not (extent[0] <= center[0] <= extent[2] and extent[1] <= center[1] <= extent[3]):
            continue
        key = (name.lower(), kind)
        if key in seen:
            continue
        seen.add(key)
        entry = {"name": name, "xy": [float(center[0]), float(center[1])], "type": kind, "source": "osm",
                 "osm_id": f"{el.get('type')}/{el.get('id')}"}
        if len(local) >= 3:
            entry["footprint"] = [[float(x), float(y)] for x, y in local]
        out.append(entry)
    return out


def _point_in_polygon(x: float, y: float, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def match_landmarks_to_instances(landmarks: list[dict], instances: list[Instance], max_dist_m: float = 15.0) -> dict:
    """``{instance_id: landmark_name}`` for buildings whose centre lies in (or near) a named footprint."""
    out = {}
    for inst in instances:
        if inst.category != "building" or inst.status == "rejected":
            continue
        cx, cy = inst.bbox[0], inst.bbox[1]
        best, best_d = None, max_dist_m
        for lm in landmarks:
            if lm.get("type") != "building":
                continue
            if lm.get("footprint") and _point_in_polygon(cx, cy, lm["footprint"]):
                best = lm["name"]
                break
            d = math.hypot(lm["xy"][0] - cx, lm["xy"][1] - cy)
            if d < best_d:
                best, best_d = lm["name"], d
        if best:
            out[inst.instance_id] = best
    return out


# ── 4. expression generation ─────────────────────────────────────────────────

_COLORS = {"white": (235, 235, 235), "black": (30, 30, 30), "gray": (128, 128, 128), "red": (190, 40, 40),
           "orange": (230, 140, 40), "yellow": (220, 200, 60), "green": (60, 140, 60), "blue": (50, 80, 180),
           "brown": (130, 90, 60), "beige": (210, 190, 150), "silver": (190, 190, 200)}


def color_name(rgb) -> str:
    rgb = np.asarray(rgb, dtype=np.float64)
    return min(_COLORS, key=lambda k: np.sum((np.asarray(_COLORS[k]) - rgb) ** 2))


def compass(dx: float, dy: float) -> str:
    ang = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    return ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"][int((ang + 22.5) // 45) % 8]


def target_context(target: Instance, instances: list[Instance], landmarks: list[dict], radius_m: float = 50.0,
                   n_landmarks: int = 3) -> dict:
    cx, cy = target.bbox[0], target.bbox[1]
    distractors = []
    for other in instances:
        if other.instance_id == target.instance_id or other.category != target.category or other.status == "rejected":
            continue
        d = math.hypot(other.bbox[0] - cx, other.bbox[1] - cy)
        if d <= radius_m:
            distractors.append({"id": other.instance_id, "distance_m": round(d, 1), "color": color_name(other.mean_color),
                                "direction_from_target": compass(other.bbox[0] - cx, other.bbox[1] - cy)})
    distractors.sort(key=lambda r: r["distance_m"])
    near = sorted(landmarks, key=lambda lm: math.hypot(lm["xy"][0] - cx, lm["xy"][1] - cy))[:n_landmarks]
    return {
        "category": target.category, "color": color_name(target.mean_color),
        "size_m": [round(v, 1) for v in target.bbox[3:6]], "distractors": distractors[:8],
        "nearest_landmarks": [{"name": lm["name"], "type": lm.get("type", ""),
                               "distance_m": round(math.hypot(lm["xy"][0] - cx, lm["xy"][1] - cy), 1),
                               "target_is_to_the": compass(cx - lm["xy"][0], cy - lm["xy"][1])} for lm in near],
    }


def render_marked_views(target: Instance, distractors: list[Instance], bev: Image.Image, mapping: dict, grid,
                        crop_side_m: float = 80.0) -> list[Image.Image]:
    """Centred BEV crop and four oblique views with the target (red) and distractors (blue) marked."""
    from ..oblique.cache import cuboid_corners, draw_cuboid
    from ..oblique.renderer import render_views
    from ..views import crop_bev, world_to_pixel

    cx, cy = target.bbox[0], target.bbox[1]
    half = crop_side_m / 2.0
    crop, cmap = crop_bev(bev, mapping, [cx - half, cy - half, cx + half, cy + half])
    draw = ImageDraw.Draw(crop)
    for inst, color in [(d, (40, 90, 255)) for d in distractors] + [(target, (255, 0, 0))]:
        x0, y0 = world_to_pixel(cmap, inst.bbox[0] - inst.bbox[3] / 2, inst.bbox[1] + inst.bbox[4] / 2)
        x1, y1 = world_to_pixel(cmap, inst.bbox[0] + inst.bbox[3] / 2, inst.bbox[1] - inst.bbox[4] / 2)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    out = [crop]
    tb = target.bbox
    region = [tb[0] - tb[3] / 2, tb[1] - tb[4] / 2, tb[0] + tb[3] / 2, tb[1] + tb[4] / 2]
    rendered = render_views(grid, region, output_size=512)
    if rendered:
        for view, depth, cam in zip(rendered["views"], rendered["depths"], rendered["cameras"]):
            for inst, color in [(d, (40, 90, 255)) for d in distractors] + [(target, (255, 0, 0))]:
                b = inst.bbox
                corners = cuboid_corners([b[0] - b[3] / 2, b[1] - b[4] / 2, b[0] + b[3] / 2, b[1] + b[4] / 2],
                                         b[2] - b[5] / 2, b[2] + b[5] / 2)
                view = draw_cuboid(view, corners, cam["extrinsic"], cam["intrinsic"], depth, color=color)
            out.append(view)
    return out


GENERATION_PROMPT = (
    "You write referring expressions for a city-scale 3D grounding benchmark in the style of CityRefer.\n"
    "The images show a top-down crop and oblique views from the east, north, west and south. The TARGET is "
    "marked in red; other {category}s of the same class nearby are marked in blue.\n"
    "Target facts: {facts}\n"
    "Write three different concise expressions (one sentence each) that uniquely identify the target among the "
    "blue distractors. Use visual attributes (colour, size, roof or body shape, height) and spatial relations to "
    "the distractors, roads and the named landmarks listed above, using landmark names exactly as given. Never "
    "mention images, views, cameras, markings, colours of the markers, or boxes.\n"
    "Return JSON only: {{\"expressions\": [\"...\", \"...\", \"...\"]}}"
)


def _image_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_expressions(client, model: str, images: list[Image.Image], context: dict,
                         temperature: float = 0.7) -> list[str]:
    """Ask an OpenAI-compatible VLM for three expressions for one target."""
    content = [{"type": "text", "text": GENERATION_PROMPT.format(category=context["category"],
                                                                 facts=json.dumps(context, ensure_ascii=False))}]
    content += [{"type": "image_url", "image_url": {"url": _image_url(im)}} for im in images]
    resp = client.chat.completions.create(model=model, temperature=temperature,
                                          messages=[{"role": "user", "content": content}])
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        exprs = json.loads(m.group(0))["expressions"] if m else []
    except (json.JSONDecodeError, KeyError, TypeError):
        exprs = []
    return [str(e).strip() for e in exprs if str(e).strip()][:3]


# ── 5. filtering and review ──────────────────────────────────────────────────

META_LANGUAGE = re.compile(
    r"\b(image|images|picture|photo|view|views|camera|render(ed|ing)?|marked|marker|highlight(ed)?|bounding|"
    r"box(es)?|wireframe|red outline|blue outline|outlined|annotation|frame)\b", re.IGNORECASE)
_CAPITALISED = re.compile(r"\b(?:[A-Z][a-zA-Z'’.-]+)(?:\s+(?:of|the|and|[A-Z][a-zA-Z'’.-]+))*")
_STOP = {"The", "A", "An", "It", "This", "That", "Its", "On", "In", "Near", "Next", "North", "South", "East", "West"}


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def filter_expressions(expressions: list[str], context: dict, landmark_names: set[str], *,
                       projection_valid: bool = True, min_words: int = 6, max_words: int = 60,
                       dup_jaccard: float = 0.8, require_landmark: bool = True) -> list[dict]:
    """Automatic filters; returns ``[{"query", "ok", "reasons"}]``."""
    lower_names = {n.lower() for n in landmark_names}
    kept_tokens: list[set[str]] = []
    out = []
    for expr in expressions:
        reasons = []
        words = expr.split()
        if not min_words <= len(words) <= max_words:
            reasons.append("length")
        if META_LANGUAGE.search(expr):
            reasons.append("meta_language")
        mentioned = [n for n in landmark_names if n.lower() in expr.lower()]
        if require_landmark and not mentioned:
            reasons.append("no_landmark")
        body = re.sub(r"(^|[.!?]\s+)([A-Z])", lambda m: m.group(1) + m.group(2).lower(), expr)
        for phrase in _CAPITALISED.findall(body):
            phrase = phrase.strip(" .,'’")
            if phrase in _STOP or len(phrase) < 3:
                continue
            if not any(phrase.lower() in n or n in phrase.lower() for n in lower_names):
                reasons.append(f"unknown_name:{phrase}")
                break
        if not projection_valid:
            reasons.append("projection")
        toks = _token_set(expr)
        if any(len(toks & t) / max(1, len(toks | t)) >= dup_jaccard for t in kept_tokens):
            reasons.append("duplicate")
        if not reasons:
            kept_tokens.append(toks)
        out.append({"query": expr, "ok": not reasons, "reasons": reasons, "landmarks": mentioned})
    return out


def apply_review(candidates: list[dict], review: dict[str, bool]) -> list[dict]:
    """Keep candidates whose ``review_id`` a human accepted (``{review_id: true}``)."""
    return [c for c in candidates if review.get(c["review_id"]) is True]


def write_review_sheet(candidates: list[dict], out_html: str | Path, image_dir: str | Path) -> None:
    rows = []
    for c in candidates:
        imgs = "".join(f'<img src="{Path(image_dir).name}/{c["review_id"]}_{i}.jpg" height="160">'
                       for i in range(c.get("n_images", 0)))
        rows.append(f"<tr><td>{c['review_id']}</td><td>{c['scene_id']}/{c['object_id']} ({c['object_name']})</td>"
                    f"<td>{c['query']}</td><td>{imgs}</td></tr>")
    Path(out_html).write_text("<html><body><table border=1><tr><th>id</th><th>target</th><th>expression</th>"
                              "<th>views</th></tr>" + "".join(rows) + "</table></body></html>", encoding="utf-8")


def to_benchmark_rows(accepted: list[dict]) -> list[dict]:
    return [{"dataset": "stpls3d", "scene_id": c["scene_id"], "object_id": str(c["object_id"]),
             "object_name": c["object_name"], "query": c["query"], "gt_bbox": c["gt_bbox"],
             "bev_png": f"bev_stpls3d/{c['scene_id']}.png", "split": "test"} for c in accepted]
