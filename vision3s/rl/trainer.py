# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn GRPO training of the grounding policy (Sec. 3.3, Appendix A.1).

The loop per step is: sample ``rollout_batch_size / group_size`` training
queries, roll out ``group_size`` trajectories each with the current policy in
the tool environment, compute Eq. 4 advantages from the trajectory returns
(Eq. 3), and take one policy-gradient pass over the assistant tokens with the
clipped objective and k3 KL to the frozen reference (Eq. 5). The vision
encoder (patch embedding and transformer blocks) stays frozen. Validation and
checkpointing happen every ``eval_steps`` / ``save_steps``.

Multi-GPU runs use ``accelerate launch``; each rank rolls out its share of the
query groups and gradients are averaged by the accelerator.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from ..agent.policy import GenerationConfig, HFPolicy
from ..agent.rollout import Trajectory, run_rollouts, write_records
from ..env import EnvConfig, GroundingEnv
from ..rewards import REWARD_PRESETS
from .grpo import assistant_token_mask, group_advantages, grpo_token_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    data_root: str = "data"
    dataset: str = "cityrefer"
    train_file: str = "meta/cityrefer_train.jsonl"
    val_file: str = "meta/cityrefer_val.jsonl"
    output_dir: str = "runs/urbex_grpo"
    seed: int = 42

    max_steps: int = 300
    rollout_batch_size: int = 256
    group_size: int = 8
    ppo_epochs: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    lr_scheduler: str = "constant"
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.001
    kl_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    advantage_clip: Optional[float] = 10.0
    reward_clip: Optional[float] = 20.0
    max_sequence_length: int = 15000

    response_length: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 60
    rollout_micro_batch: int = 32

    eval_steps: int = 40
    save_steps: int = 40
    val_batch_size: int = 16
    logging_steps: int = 1

    freeze_patterns: list = field(default_factory=lambda: ["visual.blocks", "visual.patch_embed"])
    gradient_checkpointing: bool = True
    bf16: bool = True
    attn_implementation: Optional[str] = None

    env: dict = field(default_factory=dict)
    reward_preset: str = "full"

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[dict] = None) -> "TrainConfig":
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        raw.update(overrides or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        return cls(**raw)


class QuerySampler:
    """Shuffled epochs over training queries."""

    def __init__(self, samples: list[dict], seed: int):
        self.samples = samples
        self.rng = random.Random(seed)
        self.order: list[int] = []

    def take(self, n: int) -> list[dict]:
        out = []
        while len(out) < n:
            if not self.order:
                self.order = list(range(len(self.samples)))
                self.rng.shuffle(self.order)
            out.append(self.samples[self.order.pop()])
        return out


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def freeze_vision(model, patterns) -> int:
    frozen = 0
    for name, p in model.named_parameters():
        if any(pat in name for pat in patterns):
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen


class GRPOTrainer:
    def __init__(self, cfg: TrainConfig):
        from accelerate import Accelerator
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.accelerator = Accelerator(mixed_precision="bf16" if cfg.bf16 else "no")
        self.out = Path(cfg.output_dir)
        if self.accelerator.is_main_process:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        torch.manual_seed(cfg.seed + self.accelerator.process_index)

        kwargs = {"dtype": torch.float32}
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        self.processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)
        self.processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(cfg.model_name_or_path, **kwargs)
        n_frozen = freeze_vision(model, cfg.freeze_patterns)
        logger.info("froze %.1fM vision parameters", n_frozen / 1e6)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
        ref = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name_or_path, dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
            **({"attn_implementation": cfg.attn_implementation} if cfg.attn_implementation else {}))
        ref.requires_grad_(False).eval()

        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.LambdaLR(optim, self._lr_lambda)
        self.model, self.optim, self.sched = self.accelerator.prepare(model, optim, sched)
        self.ref = ref.to(self.accelerator.device)

        tok = self.processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.assistant_ids = tok.encode("assistant\n", add_special_tokens=False)

        root = Path(cfg.data_root)
        self.train = load_jsonl(root / cfg.train_file)
        self.val = load_jsonl(root / cfg.val_file) if (root / cfg.val_file).exists() else []
        self.sampler = QuerySampler(self.train, cfg.seed)
        self.val_rng = random.Random(cfg.seed + 1)
        env_kw = dict(cfg.env)
        env_kw.setdefault("reward", REWARD_PRESETS[cfg.reward_preset])
        self.env_cfg = EnvConfig(data_root=cfg.data_root, **env_kw)
        self._shared = GroundingEnv.from_config(self.env_cfg, dataset=cfg.dataset)

    def _lr_lambda(self, step: int) -> float:
        cfg = self.cfg
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.lr_scheduler == "cosine":
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return 1.0

    def env_factory(self) -> GroundingEnv:
        return self._shared.clone()

    def policy(self, greedy: bool) -> HFPolicy:
        cfg = self.cfg
        gen = GenerationConfig.greedy(cfg.response_length) if greedy else GenerationConfig(
            cfg.response_length, cfg.temperature, cfg.top_p, cfg.top_k)
        return HFPolicy(self.accelerator.unwrap_model(self.model), self.processor, gen, cfg.max_sequence_length)

    # ---- log-probabilities ----------------------------------------------
    def _encode(self, traj: Trajectory):
        processor = self.processor
        text = processor.apply_chat_template(traj.messages, tools=traj.tools or None, tokenize=False,
                                             add_generation_prompt=False)
        from ..agent.policy import collect_images

        images = collect_images(traj.messages)
        enc = processor(text=[text], images=images or None, return_tensors="pt")
        ids = enc["input_ids"][0]
        if ids.numel() > self.cfg.max_sequence_length:
            return None
        mask = assistant_token_mask(ids, self.im_start, self.assistant_ids, self.im_end)
        if not mask.any():
            return None
        return enc.to(self.accelerator.device), mask.to(self.accelerator.device)

    def _logprobs(self, model, enc, mask) -> torch.Tensor:
        target = mask[1:]
        positions = torch.nonzero(target, as_tuple=False).squeeze(-1)
        labels = enc["input_ids"][0, 1:][positions]
        try:
            out = model(**enc, use_cache=False, logits_to_keep=positions)
            logits = out.logits[0]
            if logits.shape[0] != positions.numel():
                logits = logits[positions]
        except TypeError:
            logits = model(**enc, use_cache=False).logits[0, :-1][positions]
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # ---- training --------------------------------------------------------
    def _rollout(self, step: int) -> list[Trajectory]:
        cfg = self.cfg
        acc = self.accelerator
        n_groups = cfg.rollout_batch_size // cfg.group_size
        queries = self.sampler.take(n_groups)
        local = queries[acc.process_index::acc.num_processes]
        self.model.eval()
        trajs = run_rollouts(self.policy(greedy=False), self.env_factory, local, cfg.group_size,
                             cfg.rollout_micro_batch)
        for i, t in enumerate(trajs):
            t.info["group_key"] = (step, i // cfg.group_size)
        return trajs

    def train_step(self, step: int) -> dict:
        cfg = self.cfg
        acc = self.accelerator
        t0 = time.perf_counter()
        trajs = self._rollout(step)
        t_roll = time.perf_counter() - t0
        rewards = [t.total_reward for t in trajs]
        if cfg.reward_clip is not None:
            rewards = [max(-cfg.reward_clip, min(cfg.reward_clip, r)) for r in rewards]
        advs = group_advantages(rewards, [t.info["group_key"] for t in trajs], clip=cfg.advantage_clip)

        encoded = [self._encode(t) for t in trajs]
        items = [(e, a) for e, a in zip(encoded, advs) if e is not None]
        skipped = len(trajs) - len(items)
        n_items = torch.tensor(float(len(items)), device=acc.device)
        n_total = acc.gather(n_items).sum().item()

        with torch.no_grad():
            refs = [self._logprobs(self.ref, enc, mask) for (enc, mask), _ in items]
            olds = None
            if cfg.ppo_epochs > 1:
                self.model.eval()
                olds = [self._logprobs(self.model, enc, mask) for (enc, mask), _ in items]

        stats: dict = {"kl": [], "clipfrac": [], "loss": []}
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        distributed = acc.num_processes > 1
        self.model.train()
        for _ in range(cfg.ppo_epochs):
            self.optim.zero_grad(set_to_none=True)
            # Ranks may hold different numbers of trajectories (over-length ones are dropped), so
            # gradients are accumulated locally and summed once instead of syncing per backward.
            with acc.no_sync(self.model) if distributed else nullcontext():
                for k, ((enc, mask), adv) in enumerate(items):
                    with acc.autocast():
                        logp = self._logprobs(self.model, enc, mask)
                    old = olds[k] if olds is not None else logp.detach()
                    loss, s = grpo_token_loss(logp, old, refs[k], adv, cfg.clip_eps, cfg.kl_coef, cfg.kl_type)
                    acc.backward(loss / max(1.0, n_total))
                    stats["loss"].append(loss.item())
                    stats["clipfrac"].append(s["clipfrac"])
                    if "kl" in s:
                        stats["kl"].append(s["kl"])
            if distributed:
                import torch.distributed as dist

                for p in trainable:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad)
            grad_norm = acc.clip_grad_norm_(trainable, cfg.max_grad_norm) if cfg.max_grad_norm else None
            self.optim.step()
        self.sched.step()

        metrics = summarise(trajs, prefix="train/")
        metrics.update({
            "step": step, "lr": self.sched.get_last_lr()[0], "time/rollout_s": t_roll,
            "time/step_s": time.perf_counter() - t0, "train/skipped_overlength": skipped,
            "train/advantage_abs": sum(abs(a) for a in advs) / max(1, len(advs)),
            "train/grad_norm": float(grad_norm) if grad_norm is not None else None,
        })
        for k, v in stats.items():
            if v:
                metrics[f"train/{k}"] = sum(v) / len(v)
        return self._reduce(metrics)

    def _reduce(self, metrics: dict) -> dict:
        acc = self.accelerator
        if acc.num_processes == 1:
            return metrics
        from accelerate.utils import gather_object

        per_rank = gather_object([metrics])
        out = {"step": metrics.get("step")}
        for k in sorted({k for m in per_rank for k in m} - {"step"}):
            vals = [m[k] for m in per_rank if isinstance(m.get(k), (int, float))]
            out[k] = sum(vals) / len(vals) if vals else None
        return out

    def validate(self, step: int) -> dict:
        cfg = self.cfg
        acc = self.accelerator
        if not self.val:
            return {}
        rng = random.Random(cfg.seed + step)
        batch = rng.sample(self.val, min(cfg.val_batch_size, len(self.val)))
        local = batch[acc.process_index::acc.num_processes]
        self.model.eval()
        trajs = run_rollouts(self.policy(greedy=True), self.env_factory, local, 1, cfg.rollout_micro_batch)
        if acc.is_main_process:
            write_records(trajs, self.out / "val" / f"step_{step:04d}.jsonl")
        m = summarise(trajs, prefix="val/")
        m["step"] = step
        return self._reduce(m)

    def save(self, step: int) -> None:
        acc = self.accelerator
        acc.wait_for_everyone()
        if acc.is_main_process:
            path = self.out / f"checkpoint-{step}"
            model = acc.unwrap_model(self.model)
            model.save_pretrained(path, safe_serialization=True, state_dict=acc.get_state_dict(self.model))
            self.processor.save_pretrained(path)
            logger.info("saved %s", path)

    def fit(self) -> None:
        cfg = self.cfg
        log = (self.out / "metrics.jsonl").open("a", encoding="utf-8") if self.accelerator.is_main_process else None
        try:
            for step in range(1, cfg.max_steps + 1):
                m = self.train_step(step)
                if step % cfg.eval_steps == 0 or step == cfg.max_steps:
                    m.update(self.validate(step))
                if log is not None and (step % cfg.logging_steps == 0):
                    log.write(json.dumps(m) + "\n")
                    log.flush()
                    logger.info("step %d reward %.3f iou %.3f", step, m.get("train/reward", 0), m.get("train/iou", 0))
                if step % cfg.save_steps == 0 or step == cfg.max_steps:
                    self.save(step)
        finally:
            if log is not None:
                log.close()


def summarise(trajs: list[Trajectory], prefix: str = "") -> dict:
    """Training-dynamics metrics of Fig. 7: reward, IoU, tool calls and response length."""
    if not trajs:
        return {}
    n = len(trajs)
    tool_calls = [len(t.info.get("tool_trace", [])) for t in trajs]
    per_tool: dict = {}
    for t in trajs:
        for c in t.info.get("tool_trace", []):
            per_tool[c["tool"]] = per_tool.get(c["tool"], 0) + 1
    out = {
        f"{prefix}reward": sum(t.total_reward for t in trajs) / n,
        f"{prefix}iou": sum(t.info.get("iou_2d", 0.0) for t in trajs) / n,
        f"{prefix}acc@0.25": sum(t.info.get("iou_2d", 0.0) >= 0.25 for t in trajs) / n,
        f"{prefix}acc@0.5": sum(t.info.get("iou_2d", 0.0) >= 0.5 for t in trajs) / n,
        f"{prefix}tool_calls": sum(tool_calls) / n,
        f"{prefix}turns": sum(t.info.get("assistant_turns", 0) for t in trajs) / n,
        f"{prefix}response_tokens": sum(t.usage.get("completion_tokens", 0) for t in trajs) / n,
        f"{prefix}no_answer": sum(t.info.get("reason") != "answer" for t in trajs) / n,
    }
    for name, c in per_tool.items():
        out[f"{prefix}calls/{name}"] = c / n
    return out
