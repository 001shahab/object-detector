# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Policies ``π_θ(a_t | o_t, h_t)`` (Eq. 1).

* :class:`HFPolicy` – a local VLM (Qwen3-VL-4B-Instruct by default) driven
  through its chat template with ``tools=``; used for RL rollouts and local
  evaluation.
* :class:`OpenAIPolicy` – any OpenAI-compatible endpoint (GPT-5.4,
  Qwen3.6-plus, or a ``vllm serve`` deployment of a trained checkpoint) with
  the same tools, prompt, coordinate convention and answer format (Appendix A.3).
* :class:`ScriptedPolicy` – deterministic callable, for tests and scripted
  workflows.

Every policy maps a batch of conversations to one assistant turn each. Tool
calls are returned in the ``<tool_call>{...}</tool_call>`` text form the
environment parses.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from PIL import Image

from ..parsing import format_tool_call

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 60

    @classmethod
    def greedy(cls, max_new_tokens: int = 1024) -> "GenerationConfig":
        return cls(max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1)


@dataclass
class TurnOutput:
    text: str
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0


def collect_images(messages: list) -> list[Image.Image]:
    """Images in document order, matching the order of the template's vision placeholders."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            out.extend(item["image"] for item in content
                       if isinstance(item, dict) and item.get("type") == "image" and item.get("image") is not None)
    return out


class Policy:
    def act(self, conversations: Sequence[list], tools: list[dict],
            forced_tools: Optional[Sequence[Optional[str]]] = None) -> list[TurnOutput]:
        raise NotImplementedError


class ScriptedPolicy(Policy):
    def __init__(self, fn: Callable[[list], str]):
        self.fn = fn

    def act(self, conversations, tools, forced_tools=None):
        return [TurnOutput(self.fn(conv)) for conv in conversations]


class HFPolicy(Policy):
    """Local Hugging Face VLM. Shares the model object with the GRPO trainer."""

    def __init__(self, model, processor, gen: GenerationConfig, max_prompt_tokens: int = 15000):
        self.model = model
        self.processor = processor
        self.gen = gen
        self.max_prompt_tokens = max_prompt_tokens

    @classmethod
    def from_pretrained(cls, path: str, gen: Optional[GenerationConfig] = None, dtype: str = "bfloat16",
                        device_map: Optional[str] = "auto", attn_implementation: Optional[str] = None) -> "HFPolicy":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs = {"dtype": getattr(torch, dtype)}
        if device_map:
            kwargs["device_map"] = device_map
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
        processor = AutoProcessor.from_pretrained(path)
        processor.tokenizer.padding_side = "left"
        return cls(model.eval(), processor, gen or GenerationConfig.greedy())

    def render(self, conversation: list, tools: list[dict], add_generation_prompt: bool = True) -> str:
        return self.processor.apply_chat_template(conversation, tools=tools or None, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)

    def act(self, conversations, tools, forced_tools=None):
        import time

        import torch

        if not conversations:
            return []
        texts = [self.render(c, tools) for c in conversations]
        images = [im for c in conversations for im in collect_images(c)]
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        inputs = self.processor(text=texts, images=images or None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        sample = self.gen.temperature > 0
        kwargs = {"max_new_tokens": self.gen.max_new_tokens, "do_sample": sample,
                  "pad_token_id": tok.pad_token_id or tok.eos_token_id}
        if sample:
            kwargs.update(temperature=self.gen.temperature, top_p=self.gen.top_p)
            if self.gen.top_k and self.gen.top_k > 0:
                kwargs["top_k"] = self.gen.top_k
        start = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        elapsed = (time.perf_counter() - start) / len(texts)
        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for row, mask in zip(out[:, prompt_len:], inputs["attention_mask"]):
            text = tok.decode(row, skip_special_tokens=False)
            for stop in ("<|im_end|>", "<|endoftext|>"):
                text = text.split(stop)[0]
            if tok.pad_token:
                text = text.replace(tok.pad_token, "")
            n_gen = int((row != (tok.pad_token_id or -1)).sum())
            results.append(TurnOutput(text.strip(), {"prompt_tokens": int(mask.sum()), "completion_tokens": n_gen,
                                                     "total_tokens": int(mask.sum()) + n_gen}, elapsed))
        return results


def _image_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def to_openai_messages(conversation: list) -> list[dict]:
    """Environment messages to chat-completions messages.

    Tool results are sent as user turns wrapped in ``<tool_response>`` because
    the ``tool`` role cannot carry images; previous tool calls stay in the
    assistant text, as in the training trajectories.
    """
    out = []
    for msg in conversation:
        role, content = msg["role"], msg.get("content")
        if role in ("system", "assistant"):
            out.append({"role": role, "content": content if isinstance(content, str) else ""})
            continue
        parts = []
        items = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for item in items:
            if item.get("type") == "text":
                text = item.get("text", "")
                parts.append({"type": "text", "text": f"<tool_response>\n{text}\n</tool_response>" if role == "tool" else text})
            elif item.get("type") == "image" and item.get("image") is not None:
                parts.append({"type": "image_url", "image_url": {"url": _image_url(item["image"]), "detail": "high"}})
        out.append({"role": "user", "content": parts})
    return out


class OpenAIPolicy(Policy):
    """Prompted VLM baseline over an OpenAI-compatible chat-completions API."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 gen: Optional[GenerationConfig] = None, reasoning_effort: Optional[str] = None,
                 enable_thinking: Optional[bool] = None, bbox_order: str = "xyxy", max_retries: int = 5,
                 timeout: float = 180.0, max_workers: int = 8):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout)
        self.max_workers = max_workers
        self.model = model
        self.gen = gen or GenerationConfig.greedy()
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.bbox_order = bbox_order

    def _swap(self, args: dict) -> dict:
        if self.bbox_order != "yxyx":
            return args
        out = dict(args)
        for k in ("bbox_2d", "bbox"):
            v = out.get(k)
            if isinstance(v, list) and len(v) == 4:
                out[k] = [v[1], v[0], v[3], v[2]]
        return out

    def _one(self, conversation: list, tools: list[dict], forced: Optional[str]) -> TurnOutput:
        import time

        turn_tools = tools
        kwargs = {"model": self.model, "messages": to_openai_messages(conversation),
                  "max_tokens": self.gen.max_new_tokens, "temperature": self.gen.temperature, "top_p": self.gen.top_p}
        if forced:
            turn_tools = [t for t in tools if t["function"]["name"] == forced]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": forced}}
        elif tools:
            kwargs["tool_choice"] = "auto"
        if turn_tools:
            kwargs["tools"] = turn_tools
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
        start = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start
        if not resp.choices:
            raise RuntimeError("API returned no choices")
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        usage = {k: int(getattr(usage, k, 0) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        parts = [p for p in ((getattr(msg, "reasoning_content", None) or "").strip(), (msg.content or "").strip()) if p]
        text = "\n".join(parts)
        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            text = (text + "\n" + format_tool_call(tc.function.name, self._swap(args))).strip()
        elif self.bbox_order == "yxyx" and "<answer>" in text:
            from ..parsing import parse_answer

            ans = parse_answer(text)
            if ans:
                ans = self._swap(ans)
                text = text.split("<answer>")[0] + "<answer>" + json.dumps(ans) + "</answer>"
        return TurnOutput(text, usage, latency)

    def _safe(self, conv, tools, forced) -> TurnOutput:
        try:
            return self._one(conv, tools, forced)
        except Exception as exc:  # network / provider errors end the episode unanswered
            logger.warning("API call failed: %s", exc)
            return TurnOutput("", {"error": str(exc)})

    def act(self, conversations, tools, forced_tools=None):
        from concurrent.futures import ThreadPoolExecutor

        forced_tools = forced_tools or [None] * len(conversations)
        if self.max_workers <= 1 or len(conversations) <= 1:
            return [self._safe(c, tools, f) for c, f in zip(conversations, forced_tools)]
        with ThreadPoolExecutor(min(self.max_workers, len(conversations))) as pool:
            return list(pool.map(lambda cf: self._safe(cf[0], tools, cf[1]), zip(conversations, forced_tools)))
