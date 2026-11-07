# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Landmark database and the Landmark Query resolver (Sec. 3.1, Table 5).

The database is a JSON object ``{scene_id: [entry, ...]}`` where each entry is::

    {"name": "Merton House", "xy": [x, y], "type": "building",
     "footprint": [[x, y], ...], "source": "osm"}

``type``, ``footprint`` and ``source`` are optional. Coordinates are world
metres in the scene frame.

Evaluation-only robustness modes (Appendix D.3):

* ``clean``   – the resolver's best match.
* ``missing`` – every query returns no match.
* ``noisy``   – the correct coordinate is perturbed by ``noise_m`` metres
  (``fixed``: exactly ``noise_m`` in a random direction; ``gaussian``: an
  isotropic Gaussian with that standard deviation per axis).
* ``wrong``   – a same-scene landmark at least ``wrong_min_distance_m`` away.

Perturbations are seeded per (scene, query, name) so they are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LANDMARK_MODES = ("clean", "missing", "noisy", "wrong")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _PUNCT_RE.sub(" ", text.lower()).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in text.split() if len(t) >= 2}


@dataclass
class LandmarkHit:
    query: str
    name: str
    xy: tuple[float, float]
    score: float
    type: str = ""
    footprint: Optional[list] = None
    perturbation: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "query": self.query, "name": self.name, "world_xy": list(self.xy), "score": self.score,
            "type": self.type, "footprint": self.footprint, "perturbation": self.perturbation,
        }


class LandmarkDB:
    def __init__(self, entries: dict[str, list[dict]]):
        self._entries = entries

    @classmethod
    def load(cls, path: str | Path | None) -> "LandmarkDB":
        if path is None or not Path(path).exists():
            return cls({})
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def scene(self, scene_id: str) -> list[dict]:
        return [e for e in self._entries.get(scene_id, []) if e.get("name") and len(e.get("xy") or []) >= 2]

    def scenes(self) -> list[str]:
        return sorted(self._entries)

    def find(self, scene_id: str, query: str, landmark_type: Optional[str] = None,
             top_k: int = 1) -> list[LandmarkHit]:
        """Substring matches score 1.0; otherwise token overlap / larger token set."""
        q = (query or "").strip().lower()
        if not q:
            return []
        q_norm = _norm(q)
        q_tokens = _tokens(q_norm)
        hits = []
        for entry in self.scene(scene_id):
            name = str(entry["name"])
            if landmark_type and str(entry.get("type", "")).lower() != landmark_type.lower():
                continue
            name_norm = _norm(name)
            if q in name.lower() or (q_norm and q_norm in name_norm):
                score = 1.0
            else:
                lm_tokens = _tokens(name_norm)
                overlap = len(q_tokens & lm_tokens)
                if not overlap:
                    continue
                score = overlap / max(len(q_tokens), len(lm_tokens))
            hits.append(LandmarkHit(query, name, (float(entry["xy"][0]), float(entry["xy"][1])), score,
                                    str(entry.get("type", "")), entry.get("footprint")))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def mentioned_in(self, scene_id: str, text: str) -> list[dict]:
        """Landmarks whose full normalised name occurs in ``text`` (used by diagnostics)."""
        t = f" {_norm(text)} "
        return [e for e in self.scene(scene_id) if f" {_norm(str(e['name']))} " in t]


class LandmarkResolver:
    """Landmark Query backend with optional robustness perturbations."""

    def __init__(self, db: LandmarkDB, mode: str = "clean", noise_m: float = 10.0,
                 noise_kind: str = "fixed", wrong_min_distance_m: float = 50.0, seed: int = 42):
        if mode not in LANDMARK_MODES:
            raise ValueError(f"landmark mode must be one of {LANDMARK_MODES}, got {mode!r}")
        if noise_kind not in ("fixed", "gaussian"):
            raise ValueError("noise_kind must be 'fixed' or 'gaussian'")
        self.db = db
        self.mode = mode
        self.noise_m = float(noise_m)
        self.noise_kind = noise_kind
        self.wrong_min_distance_m = float(wrong_min_distance_m)
        self.seed = int(seed)

    def _rng(self, scene_id: str, sample_query: str, name: str) -> random.Random:
        key = "\x1f".join([str(self.seed), scene_id, sample_query, name]).encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def resolve(self, scene_id: str, name: str, landmark_type: Optional[str] = None,
                sample_query: str = "") -> Optional[LandmarkHit]:
        if self.mode == "missing":
            return None
        matches = self.db.find(scene_id, name, landmark_type=landmark_type, top_k=1)
        hit = matches[0] if matches else None
        if hit is None or self.mode == "clean":
            return hit
        rng = self._rng(scene_id, sample_query, name)
        if self.mode == "noisy":
            if self.noise_kind == "fixed":
                theta = rng.uniform(0.0, 2.0 * math.pi)
                dx, dy = self.noise_m * math.cos(theta), self.noise_m * math.sin(theta)
            else:
                dx, dy = rng.gauss(0.0, self.noise_m), rng.gauss(0.0, self.noise_m)
            hit.xy = (hit.xy[0] + dx, hit.xy[1] + dy)
            hit.footprint = None
            hit.perturbation = {"mode": "noisy", "kind": self.noise_kind, "noise_m": self.noise_m}
            return hit
        far, near = [], []
        for entry in self.db.scene(scene_id):
            if _norm(str(entry["name"])) == _norm(hit.name):
                continue
            d = math.hypot(float(entry["xy"][0]) - hit.xy[0], float(entry["xy"][1]) - hit.xy[1])
            (far if d >= self.wrong_min_distance_m else near).append((d, entry))
        pool = sorted(far or near, key=lambda item: item[0])[:5]
        if not pool:
            return None
        _, entry = rng.choice(pool)
        return LandmarkHit(name, str(entry["name"]), (float(entry["xy"][0]), float(entry["xy"][1])), 1.0,
                           str(entry.get("type", "")), entry.get("footprint"),
                           {"mode": "wrong", "correct_match": hit.name})
