# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# Prompt wording reproduces Figure 5 of Wang et al. (2026), "Urbex: Agentic
# Spatial Grounding for City-Scale 3D Scenes", arXiv:2610.18356. Tool
# descriptions for the ``release`` protocol reproduce the public Urbex release
# (https://github.com/AMAP-ML/Urbex, Copyright 2026 The Urbex Authors,
# Apache-2.0) so that its checkpoint sees an identical interface.
"""System / user prompts (Fig. 5) and OpenAI function schemas (Table 5).

Two tool protocols are provided:

* ``paper``   – every argument listed in Table 5, including the optional
  ``landmark_type`` (Landmark Query), ``output_max_side`` (Zoom-in output
  resolution) and ``margin`` (Oblique Rendering).
* ``release`` – the exact schema the public Urbex-4B checkpoint was trained
  with (no optional arguments). Use it when evaluating that checkpoint.

Both protocols share function names, required arguments, the coordinate frame
and the answer format, so trajectories are interchangeable.
"""

from __future__ import annotations

import copy

SYSTEM_PROMPT = "You are a helpful visual grounding assistant for 3D city scenes."

USER_PROMPT_TEMPLATE = (
    "Query: {query}\n\n"
    "This is the Bird's-Eye View (BEV) of the scene; the initial view's id "
    "is \"global\". Coordinates throughout are in the SOURCE view's "
    "[0,1000] x [0,1000] integer frame, origin top-left, x->right, y->down "
    "(\"above\"=smaller y, \"below\"=larger y, \"left\"=smaller x).\n\n"
    "For each step, you must conduct a thought section to reason about the "
    "visual clues and plan your next move before calling any tools or "
    "committing the final answer.\n\n"
    "Commit by emitting ONE assistant turn with NO tool_call containing "
    "an <answer> tag with the final bbox and the view_id you committed "
    "on, e.g.:\n"
    "  <answer>{{\"bbox_2d\":[855,605,985,705], \"view_id\":\"Z1\"}}</answer>"
)

# Appendix D.2: the Landmark Query -> Zoom-in -> Oblique Rendering -> Answer
# workflow extracted from the learned policy and enforced at inference.
FIXED_WORKFLOW = ("get_landmark_xy", "zoom_in", "render_oblique")
FIXED_WORKFLOW_SUFFIX = (
    "\n\nControlled fixed-policy evaluation: call exactly one tool per "
    "turn in this order: {schedule}. After the final scheduled tool, "
    "submit the answer immediately without another tool call. You must "
    "still infer every tool argument from the query and observations."
)

LANDMARK_TOOL = "get_landmark_xy"
ZOOM_TOOL = "zoom_in"
OBLIQUE_TOOL = "render_oblique"
TOOL_NAMES = (LANDMARK_TOOL, ZOOM_TOOL, OBLIQUE_TOOL)
DIRECTIONS = ("east", "north", "west", "south")

_BBOX_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
}

_RELEASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": LANDMARK_TOOL,
            "description": (
                "Resolve proper-noun landmarks (road/building/park names) to world XY metres. Optionally "
                "projects each result onto a known view's [0,1000] integer frame. Pure lookup — no view is "
                "modified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Proper-noun landmark names to resolve, e.g. [\"Castle House\", \"Bragg Road\"]. "
                            "Generic descriptors (\"red car\") rarely match."
                        ),
                    },
                    "view_id": {
                        "type": "string",
                        "description": (
                            "A known view to project onto. Use \"global\" (the initial BEV) or any \"Z<n>\" "
                            "returned by a previous zoom_in. Defaults to \"global\"."
                        ),
                    },
                },
                "required": ["names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ZOOM_TOOL,
            "description": (
                "Crop view_id's image by bbox_2d to create a new view named \"Z<n>\" (the response reports "
                "the new view_id). Returns the new view's id, world-metre bounds, and the cropped image. Use "
                "\"global\" (the initial BEV) or any prior \"Z<k>\" as view_id; nesting capped at depth 2 from "
                "\"global\". Cropped world side must be 10–1200 m."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "view_id": {
                        "type": "string",
                        "description": (
                            "Source view to crop from. Must be \"global\" (the initial BEV) or a \"Z<n>\" id "
                            "returned by an earlier zoom_in. To re-zoom a different region of the BEV, pass "
                            "\"global\" again — every prior view's image stays in this conversation history."
                        ),
                    },
                    "bbox_2d": {
                        **_BBOX_SCHEMA,
                        "description": (
                            "[x0,y0,x1,y1] integer coords in the source view's [0,1000] × [0,1000] frame. "
                            "Origin top-left; x→right, y→down. Must satisfy x0<x1 and y0<y1."
                        ),
                    },
                },
                "required": ["view_id", "bbox_2d"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": OBLIQUE_TOOL,
            "description": (
                "Render ONE oblique 30°-elevation view from a chosen cardinal direction (east / north / west / "
                "south) over a candidate bbox on view_id. The image shows the proposed bbox as a red 3D "
                "wireframe so you can visually verify the box actually lands on the queried object before "
                "committing. To inspect another angle, call render_oblique again with a different `direction`; "
                "each call costs 1 tool budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "view_id": {
                        "type": "string",
                        "description": (
                            "The view whose [0,1000] frame bbox_2d is expressed in (\"global\" or any \"Z<n>\" "
                            "from a previous zoom_in)."
                        ),
                    },
                    "bbox_2d": {
                        **_BBOX_SCHEMA,
                        "description": "[x0,y0,x1,y1] integer coords in view_id's [0,1000] × [0,1000] frame. "
                                       "x0<x1 and y0<y1.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": list(DIRECTIONS),
                        "description": (
                            "Camera position relative to the bbox center. \"east\" puts the camera east of the "
                            "bbox looking west; \"south\" puts it south looking north; etc. Pick the direction "
                            "that best exposes the queried object's distinguishing feature."
                        ),
                    },
                },
                "required": ["view_id", "bbox_2d", "direction"],
            },
        },
    },
]

_PAPER_EXTRA = {
    LANDMARK_TOOL: {
        "landmark_type": {
            "type": "string",
            "description": (
                "Optional landmark type used to disambiguate a name shared by several landmarks, e.g. "
                "\"building\", \"road\", \"amenity\", \"leisure\"."
            ),
        },
    },
    ZOOM_TOOL: {
        "output_max_side": {
            "type": "integer",
            "description": (
                "Optional requested long side (pixels) of the returned crop; clamped to the display budget. "
                "Defaults to the environment's display resolution."
            ),
        },
    },
    OBLIQUE_TOOL: {
        "margin": {
            "type": "number",
            "description": (
                "Optional fractional context margin added around the candidate region (e.g. 0.5 adds half "
                "the region size on every side). Defaults to the environment's framing."
            ),
        },
    },
}


def tool_schemas(protocol: str = "paper", enabled: tuple[str, ...] = TOOL_NAMES) -> list[dict]:
    """OpenAI function schemas for the enabled tools under ``protocol``."""
    if protocol not in ("paper", "release"):
        raise ValueError(f"unknown tool protocol {protocol!r}")
    schemas = []
    for schema in copy.deepcopy(_RELEASE_TOOLS):
        name = schema["function"]["name"]
        if name not in enabled:
            continue
        if protocol == "paper":
            schema["function"]["parameters"]["properties"].update(copy.deepcopy(_PAPER_EXTRA[name]))
        schemas.append(schema)
    return schemas


def user_prompt(query: str, forced_tool_sequence: tuple[str, ...] | None = None) -> str:
    text = USER_PROMPT_TEMPLATE.format(query=query)
    if forced_tool_sequence:
        text += FIXED_WORKFLOW_SUFFIX.format(schedule=" -> ".join(forced_tool_sequence))
    return text
