# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""Smoke test of the Hugging Face policy and one GRPO update with a tiny random Qwen3-VL.

Downloads only the Qwen3-VL processor/tokenizer; the model is randomly initialised
from a shrunken config, so this checks plumbing (chat template with tools and
image-bearing tool turns, batched generation, assistant masks, log-probs and
gradients), not accuracy.

    python scripts/smoke_hf.py --data-root <synthetic or real data root>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def tiny_model(name: str):
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(name)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.rope_scaling = {**(t.rope_scaling or {}), "mrope_section": [2, 3, 3]} if getattr(t, "rope_scaling", None) else None
    if getattr(t, "rope_parameters", None):
        t.rope_parameters = {**t.rope_parameters, "mrope_section": [2, 3, 3]}
    v.hidden_size, v.intermediate_size, v.depth, v.num_heads = 32, 64, 2, 2
    v.out_hidden_size = 64
    v.deepstack_visual_indexes = [0]
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(cfg).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--workdir", default="/tmp/vision3s_smoke")
    ap.add_argument("--train", action="store_true", help="also run two GRPOTrainer steps on a saved tiny model")
    args = ap.parse_args()

    from transformers import AutoProcessor

    from conftest import make_scene, SCENE, LANDMARK, TARGET
    from vision3s.agent import GenerationConfig, HFPolicy, run_rollouts
    from vision3s.data.bev import build_scene_bev
    from vision3s.env import GroundingEnv
    from vision3s.oblique import ObliqueRenderer
    from vision3s.rl.grpo import assistant_token_mask, group_advantages, grpo_token_loss
    import json

    root = Path(args.workdir) / "cityrefer"
    points, colors, boxes = make_scene()
    build_scene_bev(SCENE, points, colors, root / "bev_cityrefer", resolution=0.2)
    (root / "landmarks_cityrefer.json").write_text(json.dumps({SCENE: [LANDMARK]}))
    sample = {"dataset": "cityrefer", "scene_id": SCENE, "object_id": str(TARGET), "query": "the blue car near Red Tower",
              "gt_bbox": boxes[TARGET]["bbox"], "bev_png": f"bev_cityrefer/{SCENE}.png"}

    processor = AutoProcessor.from_pretrained(args.model)
    model = tiny_model(args.model)
    policy = HFPolicy(model, processor, GenerationConfig(max_new_tokens=24, temperature=1.0, top_p=0.95, top_k=60))

    base = GroundingEnv.from_data_root(root, display_max_side=256, max_assistant_turns=3, max_tool_calls=2)
    base._oblique = base.tools.oblique = ObliqueRenderer(None, False, lambda s, d: (points, colors), output_size=128)

    # Inject one valid tool call per episode so tool turns (with images) are exercised.
    from vision3s.agent.policy import TurnOutput
    from vision3s.parsing import format_tool_call

    class Mixed(HFPolicy):
        def act(self, convs, tools, forced=None):
            outs = super().act(convs, tools, forced)
            fixed = []
            for c, o in zip(convs, outs):
                if sum(m["role"] == "assistant" for m in c) == 0:
                    o = TurnOutput(format_tool_call("zoom_in", {"view_id": "global", "bbox_2d": [300, 500, 700, 900]}),
                                   o.usage, o.latency_s)
                fixed.append(o)
            return fixed

    mixed = Mixed(model, processor, policy.gen)
    trajs = run_rollouts(mixed, base.clone, [sample], group_size=2)
    print("rollouts:", [(t.info["reason"], len(t.info["tool_trace"]), round(t.total_reward, 3)) for t in trajs])

    tok = processor.tokenizer
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    asst = tok.encode("assistant\n", add_special_tokens=False)
    advs = group_advantages([t.total_reward + 0.1 * i for i, t in enumerate(trajs)], [0] * len(trajs))
    model.train()
    from vision3s.agent.policy import collect_images

    for t, adv in zip(trajs, advs):
        text = processor.apply_chat_template(t.messages, tools=t.tools, tokenize=False, add_generation_prompt=False)
        enc = processor(text=[text], images=collect_images(t.messages) or None, return_tensors="pt")
        mask = assistant_token_mask(enc["input_ids"][0], im_start, asst, im_end)
        n_img = sum(1 for m in t.messages if isinstance(m["content"], list) for c in m["content"] if c.get("type") == "image")
        print(f"tokens={enc['input_ids'].shape[1]} assistant_tokens={int(mask.sum())} images={n_img}")
        assert "<tool_response>" in text and mask.any()
        pos = torch.nonzero(mask[1:]).squeeze(-1)
        out = model(**enc, use_cache=False, logits_to_keep=pos)
        logits = out.logits[0]
        assert logits.shape[0] == pos.numel(), (logits.shape, pos.numel())
        logp = torch.log_softmax(logits.float(), -1).gather(-1, enc["input_ids"][0, 1:][pos].unsqueeze(-1)).squeeze(-1)
        loss, stats = grpo_token_loss(logp, logp.detach(), logp.detach(), adv)
        loss.backward()
    grads = [p.grad.abs().sum().item() for n, p in model.named_parameters() if p.grad is not None]
    print("loss ok; params with grad:", len(grads))

    if args.train:
        from vision3s.rl.trainer import GRPOTrainer, TrainConfig

        ckpt = Path(args.workdir) / "tiny_qwen3vl"
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        (root / "meta").mkdir(exist_ok=True)
        for split in ("train", "val"):
            (root / "meta" / f"cityrefer_{split}.jsonl").write_text(json.dumps(sample) + "\n")
        cfg = TrainConfig(model_name_or_path=str(ckpt), data_root=str(root), output_dir=str(Path(args.workdir) / "run"),
                          max_steps=2, rollout_batch_size=4, group_size=2, response_length=24, eval_steps=2,
                          save_steps=2, val_batch_size=1, bf16=False, gradient_checkpointing=True,
                          env={"max_assistant_turns": 3, "max_tool_calls": 2, "display_max_side": 256,
                               "raw_data_root": None})
        trainer = GRPOTrainer(cfg)
        trainer._shared._oblique = trainer._shared.tools.oblique = ObliqueRenderer(
            None, False, lambda s, d: (points, colors), output_size=128)
        import vision3s.rl.trainer as trainer_mod

        # The random tiny model earns identical rewards, so force non-zero advantages to exercise updates.
        trainer_mod.group_advantages = lambda rewards, groups, **kw: [(-1.0) ** i for i in range(len(rewards))]
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
        trainer.fit()
        changed = sum(not torch.equal(before[n], p) for n, p in trainer.model.named_parameters() if n in before)
        frozen = [n for n, p in trainer.model.named_parameters() if "visual.blocks" in n and p.requires_grad]
        checksum = sum(p.detach().float().cpu().double().sum().item() for p in trainer.model.parameters())
        from accelerate.utils import gather_object

        sums = gather_object([checksum])
        print("trainer: params changed", changed, "| frozen-violations", len(frozen),
              "| ranks in sync", max(sums) - min(sums) < 1e-9)
        assert changed > 0 and max(sums) - min(sums) < 1e-9
        print((Path(cfg.output_dir) / "metrics.jsonl").read_text().strip().splitlines()[-1][:400])
        assert (Path(cfg.output_dir) / "checkpoint-2").exists() and not frozen
    print("SMOKE OK")


if __name__ == "__main__":
    main()
