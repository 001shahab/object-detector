# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Parsing of assistant turns: ``<tool_call>`` actions and ``<answer>`` commits."""

from __future__ import annotations

import json
import re
from typing import Optional

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)
# Vision-token literals that a chat template would expand into image slots with
# no matching image, and a trailing end-of-turn token that would be duplicated
# when the conversation is re-rendered.
_VISION_LITERAL_RE = re.compile(r"<image>|<\|vision_start\|>|<\|vision_end\|>|<\|image_pad\|>|<\|video_pad\|>")
_TAIL_RE = re.compile(r"\s*(?:<\|im_end\|>|<\|endoftext\|>)\s*$")


def sanitize_action(text: str) -> str:
    if not text:
        return ""
    return _TAIL_RE.sub("", _VISION_LITERAL_RE.sub("", text))


def parse_tool_call(text: str) -> Optional[tuple[str, dict]]:
    """Return ``(name, arguments)`` of the first ``<tool_call>`` or ``None``."""
    m = _TOOL_CALL_RE.search(text or "")
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    args = payload.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return payload["name"], args if isinstance(args, dict) else {}


def parse_answer(text: str) -> Optional[dict]:
    """Return ``{"bbox_2d": [...], "view_id": ...}`` or ``None`` if missing/malformed."""
    m = _ANSWER_RE.search(text or "")
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "bbox_2d" not in payload or "view_id" not in payload:
        return None
    bbox = payload["bbox_2d"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return payload


def format_tool_call(name: str, arguments: dict) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}</tool_call>"
