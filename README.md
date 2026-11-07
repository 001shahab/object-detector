# 3s-vision — Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes

An implementation of **Urbex**, which grounds free-form referring expressions in
city-scale 3D scenes. A vision-language model (VLM) is treated as an agent that
actively searches a bird's-eye view (BEV) of the city with three city-aware tools.
The agent is trained with multi-turn GRPO and outputs a world-space box directly,
without a candidate pool.

**Developed by** Prof. Shahab Anbarjafari, **3S Holding OÜ**.

> **The method is not ours.** Urbex was proposed by Wang et al. This repository is an
> independent implementation by 3S Holding OÜ. If you use it, please cite the original
> paper:
>
> Wang, Y., Dai, R., Chen, W., Wang, Y., Dang, B., Liu, K., Chu, X. and Li, Y., 2026.
> Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes. arXiv preprint arXiv:2610.18356.

```bibtex
@article{wang2026urbex,
  title   = {Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes},
  author  = {Wang, Yu and Dai, Rui and Chen, Wei and Wang, Yi and Dang, Bo and
             Liu, Kaikui and Chu, Xiangxiang and Li, Yansheng},
  journal = {arXiv preprint arXiv:2610.18356},
  year    = {2026}
}
```

The distribution and command are named `3s-vision`. The Python import package is
`vision3s`, because Python module names cannot start with a digit.

---

## Contents

- [Method in brief](#method-in-brief)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation and reproduction of every table](#evaluation-and-reproduction-of-every-table)
- [Python API](#python-api)
- [Paper-to-code map](#paper-to-code-map)
- [Implementation notes](#implementation-notes)
- [Tests](#tests)
- [License and attribution](#license-and-attribution)

## Method in brief

**Interactive multi-view environment (Sec. 3.1).** The environment keeps a registry of
BEV-aligned views `v → (I_v, T_v)`, where `T_v` is the view-to-world transform. The
agent always addresses a view by its `view_id` and works in a normalised `[0, 1000]`
frame: origin at the top left, x to the right, y downwards. The episode starts from a
downsampled global BEV of the whole scene.

| Tool | Name in schema | Purpose | Output |
|---|---|---|---|
| Landmark Query | `get_landmark_xy` | resolve a named building, road or facility to an anchor | match, name/type, world centre or footprint, projection into the requested view |
| Zoom-in | `zoom_in` | crop a region from the high-resolution BEV | new local view `Z<k>` with an updated `T_v` and crop extents |
| Oblique Rendering | `render_oblique` | render a candidate region from a cardinal direction, with a red 3D wireframe | oblique image for verification (never used as the answer frame) |

**Agent and episode (Sec. 3.2).** Each assistant turn is either one
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` or a final
`<answer>{"bbox_2d": [x0,y0,x1,y1], "view_id": "..."}</answer>`. An episode allows at
most 8 assistant turns: up to 7 tool calls plus 1 answer turn. A tool call on the last
turn ends the episode with no answer. The answer box is projected to world coordinates
through `T_v` of the chosen view.

**Rewards (Eqs. 2–3).**

- Terminal reward: `R_term = λ_iou·IoU + λ_dist·exp(−d/τ) − λ_step·T/T_max`, with
  λ_iou = 1.0, λ_dist = 0.3, τ = 80 m and λ_step = 0.05.
- Zoom-in shaping: `clip(0.05·Δd/50 m, ±0.05)`, rewarding progress of the zoom centre
  towards the target. The first zoom receives 0.
- Oblique Rendering shaping: `min(0.05, 0.05·IoU(candidate, target))`.
- An invalid tool call costs −0.05. Landmark Query is reward-neutral.
- Trajectory return: `R(τ) = R_term + Σ R_tool`.

**Multi-turn GRPO (Eqs. 4–5).** For each query, G = 8 trajectories are sampled. Their
advantages are group-normalised, and the clipped surrogate with a token-level k3 KL
penalty (β = 0.001) is averaged over each trajectory's policy tokens, then over the
group. System, user and tool-response tokens are masked out.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # model + API + data + plotting
pip install -e ".[dev]"        # tests
```

Extras:

- `model`: torch, torchvision, transformers ≥ 4.57, accelerate.
- `api`: openai.
- `data`: scikit-learn, huggingface_hub, zstandard.
- `viz`: matplotlib.

The core environment and the evaluators only need NumPy, Pillow, SciPy, plyfile and
PyYAML.

## Data

### Layout

Every component reads the same layout. It matches the official release, so released
assets work unchanged:

```text
data/cityrefer/                         # --data-root (also $CITY_AGENT_DATA)
├── meta/cityrefer_{train,val,test_ND,test_NO}.jsonl
├── landmarks_cityrefer.json            # {scene_id: [{name, xy, type, footprint?}]}
├── bev_cityrefer/<scene>.png + <scene>.json   # high-res BEV + mapping {x_min,x_max,y_min,y_max,res_x,res_y}
├── verify_region_cache/cityrefer/<scene>/     # oblique cache: scene_meta.json, regions.npy, imgs/, cams/, depths/
└── box3d/<scene>_bbox.json             # GT candidate pool, used only for Hit@k scoring
raw_data/sensaturban/ply/{train,test}/<scene>.ply   # optional: on-demand oblique views, 3D metrics
```

### Download the released assets

```bash
3s-vision download cityrefer --all --candidates      # → data/cityrefer
3s-vision download stpls3d --all --point-clouds      # → data/stpls3d (+ raw_data/)
3s-vision download model --output model/Urbex-4B     # official checkpoint
```

Every file is pinned to one Hub revision and checked against the manifest's size and
SHA-256. SensatUrban point clouds are not redistributed; obtain them from
<https://github.com/QingyongHu/SensatUrban>.

### Build the assets yourself (Appendix B)

```bash
# CityRefer: filtering, 7 held-out NO scenes, 0.625/0.125/0.25 split of the rest,
# candidate pool and per-scene statistics (Table 6)
3s-vision prepare-cityrefer --annotations CityRefer_train.json CityRefer_val.json \
    --box-dir cityrefer/box3d --landmarks landmarks.json --exclusions exclusions.jsonl \
    --out data/cityrefer --raw-root raw_data

# High-resolution BEV (top-down z-buffer, hole filling) and oblique region cache
# (40 m regions, 15 m stride, 4 cardinal views, depth)
3s-vision build-bev   --dataset cityrefer --raw-root raw_data --data-root data/cityrefer
3s-vision build-cache --dataset cityrefer --raw-root raw_data --data-root data/cityrefer
```

STPLS3D-Refer is built in stages, with two human-review steps between them
(Appendix B.2):

1. `instances`: DBSCAN over the semantic masks. Buildings use XY with ε = 2.0 m and
   80 minimum samples, footprint 4–5000 m². Vehicles use XYZ with ε = 0.6 m and 30
   minimum samples, footprint 1.5–30 m². Geometric filters run next, and you then
   review `instances/<scene>_review.json`.
2. `landmarks`: affine geo-registration from ground-control points, with the RMS
   residual reported, then an OpenStreetMap query and matching to instances.
3. `generate`: a centred BEV crop and four oblique views with the target and
   same-class distractors marked, plus a facts sheet, are sent to a VLM. It writes
   three expressions per target, which are then filtered automatically.
4. You review the expressions (`review/<scene>_review.html`) and accept them in
   `review/<scene>_accept.json`.
5. `finalize`: writes `meta/stpls3d_test.jsonl`.

```bash
3s-vision stpls3d-refer instances --out data/stpls3d --scene USC --ply raw_data/STPLS3D/RealWorldData/USC_points.ply
3s-vision stpls3d-refer landmarks --out data/stpls3d --scene USC --gcps gcps_usc.json
3s-vision build-bev --dataset stpls3d --raw-root raw_data --data-root data/stpls3d --scenes USC
3s-vision stpls3d-refer generate  --out data/stpls3d --scene USC --ply ... --model qwen3.6-plus --base-url <endpoint>
3s-vision stpls3d-refer finalize  --out data/stpls3d --scenes USC WMSC
```

## Training

The paper's setting (Appendix A.1) is: Qwen3-VL-4B-Instruct, rollout batch 256,
group size 8, 300 steps, one PPO epoch, learning rate 1e-6, KL 0.001, bf16, frozen
vision encoder, 16 GPUs. All of it is in [`configs/grpo.yaml`](configs/grpo.yaml).

```bash
accelerate launch --num_processes 16 -m vision3s.cli train --config configs/grpo.yaml
# override any key:
3s-vision train --config configs/grpo.yaml --set max_steps=10 output_dir=runs/debug
```

Each step samples 32 queries, rolls out 8 trajectories per query in the tool
environment, and computes the advantages and the update:

- Rollout sampling uses temperature 1.0, top-p 0.95 and top-k 60, with at most 1024
  new tokens per turn.
- Sequences longer than 15,000 tokens are dropped from the update.
- Validation (16 queries, greedy) and checkpoints happen every 40 steps.
- `runs/<name>/metrics.jsonl` logs the Fig. 7 quantities: reward, train and validation
  IoU, tool calls per trajectory (also per tool), response length and KL.

To plot them:

```bash
3s-vision plot training runs/urbex_grpo/metrics.jsonl --output fig7.png
```

The tool and reward ablations (Tables 3–4) are listed in
[`configs/ablations.yaml`](configs/ablations.yaml).

## Evaluation and reproduction of every table

`3s-vision evaluate` rolls out a split and scores it in one go:

- **Hit@k with GT candidates:** IoU ranking, plus the positive-overlap and CIoU
  variants and Hit@1 thresholded at η = 0.25 and 0.5. This needs `box3d/`.
- **Candidate-free metrics:** 2D (BEV) IoU, 3D IoU and point F1. This needs
  `--raw-root`.
- **Policy statistics:** tokens per query, tool usage and how often trajectories
  follow the Landmark → Zoom-in → Oblique (L-Z-O) order.
- **Per-tool latency.**

With `--num-runs N` it also reports the mean ± sample standard deviation and a 95%
Student-t interval.

```bash
# Urbex checkpoint, local Hugging Face inference
3s-vision evaluate --data-root data/cityrefer --split meta/cityrefer_test_NO.jsonl \
    --model model/Urbex-4B --protocol release --raw-root raw_data --output output/urbex_NO

# Any OpenAI-compatible endpoint: GPT-5.4, Qwen3.6-plus, or `vllm serve model/Urbex-4B`
3s-vision evaluate --data-root data/cityrefer --split meta/cityrefer_test_NO.jsonl \
    --api-model gpt-5.4 --output output/gpt54_NO
```

| Paper | Command / options |
|---|---|
| Table 1 (CityRefer ND/NO, with and without GT) | `evaluate` on `cityrefer_test_ND.jsonl` / `cityrefer_test_NO.jsonl` with `--raw-root` |
| Table 2 (STPLS3D-Refer zero-shot) | `evaluate --dataset stpls3d --data-root data/stpls3d --split meta/stpls3d_test.jsonl` |
| Table 3 (tools and turn budget) | train and evaluate with `configs/ablations.yaml` (`--tools ...`, `--max-turns 7/8/9`, `--tools` with nothing = direct answer) |
| Table 4 (reward design) | `train --set reward_preset=iou_only` / `iou_distance` / `iou_tool` / `full` |
| Table 8 (five decoding runs) | `evaluate --num-runs 5 --decoding sample` → `summary.json` (mean, std, `ci95`) |
| Table 9 (free vs. fixed L-Z-O) | `evaluate --fixed-workflow` (the tool order is enforced; arguments and box stay free) |
| Table 10 (landmark robustness) | `landmark-subsets`, then `evaluate --landmark-mode clean/missing/noisy/wrong` on each subset |
| Table 11 (predicted proposals) | `hit-k --proposals proposals.json --geometric-match-iou 0.25`; proposal recall via `vision3s.eval.hit_k.proposal_recall` |
| Table 12 (matching rules) | `hit-k --score-rule iou` / `--positive-overlap` / `--score-rule ciou` |
| Table 13 (Hit@1_η, Eq. 6) | reported by `hit-k` and `evaluate` as `Hit@1_0.25`, `Hit@1_0.5` |
| Tables 14–15 (candidate overlaps, area ratio) | `hit-k` summary (`overlap_*`, `area_ratio_*`) |
| Table 16 / Fig. 9 (search vs. disambiguation) | `diagnostics --pred records.jsonl --dataset ... --box3d-dir ... --landmarks ...`; `plot diagnostics` |
| Table 17 (latency) | `latency` block of `scores.json` (per-tool ms, generation and wall time per query; use `--batch-size 1` for per-query timing) |
| Figs. 8 and 10 (trajectories) | `vision3s.viz.draw_trajectory(bev, mapping, record, gt_xyxy)` |

The landmark perturbation modes (Table 10) are:

- `missing`: the landmark returns no match.
- `noisy`: the correct coordinate moves exactly 10 m in a random direction, seeded per
  query. `--noise-kind gaussian` gives per-axis Gaussian noise instead.
- `wrong`: returns a different landmark from the same scene, at least 50 m away.

Single query:

```bash
3s-vision infer --data-root data/cityrefer --model model/Urbex-4B --protocol release \
    --scene birmingham_block_12 --query "The white car ... near the West Midlands Police Custody Suite"
```

## Python API

```python
from vision3s import GroundingEnv
from vision3s.agent import HFPolicy, GenerationConfig, run_rollouts

env = GroundingEnv.from_data_root("data/cityrefer", protocol="release")
policy = HFPolicy.from_pretrained("model/Urbex-4B", GenerationConfig.greedy())
sample = {"scene_id": "birmingham_block_12", "query": "...", "gt_bbox": None,
          "bev_png": "bev_cityrefer/birmingham_block_12.png"}
traj = run_rollouts(policy, env.clone, [sample])[0]
print(traj.info["answer"], traj.info["world_bbox"])     # world-space [x_min, y_min, x_max, y_max]
```

`GroundingEnv` is a plain `reset(sample) → messages` / `step(text) → StepResult` loop.
You can drive it with any policy: a local VLM, an API, or a script.

## Paper-to-code map

| Paper | Code |
|---|---|
| Sec. 3.1 views, `T_v`, normalised frame | `vision3s/views.py` |
| Sec. 3.1 / Table 5 tools | `vision3s/tools.py`, schemas in `vision3s/prompts.py` |
| Oblique Rendering (cache + on-demand) | `vision3s/oblique/` (NumPy z-buffer point splatting, no OpenGL) |
| Sec. 3.2 MDP, turn budget, answer projection | `vision3s/env.py`, `vision3s/parsing.py` |
| Fig. 5 system / user prompt | `vision3s/prompts.py` |
| Eqs. 2–3 rewards and tool shaping | `vision3s/rewards.py` |
| Eqs. 4–5 GRPO | `vision3s/rl/grpo.py`, `vision3s/rl/trainer.py` |
| Appendix A.1 hyperparameters | `configs/grpo.yaml` |
| Appendix A.3 prompted VLM baselines | `vision3s/agent/policy.py` (`OpenAIPolicy`) |
| Appendix B.1 CityRefer preparation, BEV | `vision3s/data/cityrefer.py`, `vision3s/data/bev.py` |
| Appendix B.2 STPLS3D-Refer | `vision3s/data/stpls3d_refer.py`, `vision3s/data/stpls3d_pipeline.py` |
| Sec. 4.1 / Appendix D.5 Hit@k, Eq. 6 | `vision3s/eval/hit_k.py` |
| Without-GT 2D/3D IoU, point F1 | `vision3s/eval/metrics_3d.py` |
| Appendix D diagnostics | `vision3s/eval/diagnostics.py`, `vision3s/eval/runner.py` |
| Figs. 7, 9, 8/10 | `vision3s/viz.py` |

## Implementation notes

The following points are not fully specified in the paper. Each decision below is
configurable.

- **Tool protocol.** `--protocol paper` (the default) exposes the Table 5 optional
  arguments: `landmark_type`, a requested output resolution (`output_max_side`/`scale`)
  and `margin`. Tool responses also carry the matched landmark type and footprint and
  the explicit view-to-world transform. `--protocol release` reproduces the released
  checkpoint's schema exactly; use it with `vvangfaye/Urbex-4B`. Both protocols use the
  same function names and accept the Table 5 argument names as aliases
  (`landmark_name`, `source_view_id`, `crop_box`, `candidate_box`).
- **Frozen vision encoder.** The patch embedding and vision transformer blocks are
  frozen. The vision-language merger stays trainable, as in the released training
  configuration.
- **Unspecified GRPO constants.** The paper does not state these, so common defaults
  are used, all exposed in the config:
  - PPO clip ε = 0.2, k3 KL estimator.
  - Advantage normalisation ε = 1e-6, with sample std inside each group.
  - Advantage clip 10, reward clip 20, gradient-norm clip 1.0.
  - Entropy bonus 0.
- **Precision and parallelism.** Training keeps fp32 master weights under bf16
  autocast, because updates at a learning rate of 1e-6 are below bf16 resolution.
  Multi-GPU training is data-parallel through `accelerate`: each rank rolls out its
  share of the query groups, and gradients are summed once per step. For the 4B model
  this needs roughly 70 GB per GPU (weights, Adam state, a bf16 reference copy and
  activations with gradient checkpointing), which fits the paper's 96 GB H20s.
- **Repeated decoding runs (Table 8).** The paper does not give the decoding
  temperature. `--decoding sample` reuses the training sampling settings
  (temperature 1.0, top-p 0.95, top-k 60), seeded per run. Single-run evaluation is
  greedy, like the reference evaluation.
- **3D metrics.** Unanswered queries are skipped, as in the reference evaluation;
  `--no-answer-as-zero` counts them as zero instead. Hit@k always keeps unanswered
  queries in the denominator.
- **Diagnostics (Table 16).**
  - Regional-search success means the selected zoom region (the last one by default;
    `--region-rule first|any`) contains the target centre.
  - Local disambiguation is Hit@1 among regionally successful queries.
  - Target–landmark distance uses the nearest scene landmark named in the expression.
  - Distractors are same-class GT candidates within 50 m.
- **Oblique renderer.** The renderer is a pure-NumPy point splatter with the same
  camera conventions and cache format as the reference Open3D cache. Its images are
  similar but not pixel-identical. Cached views from the release load unchanged.

## Tests

```bash
pytest -q                          # synthetic city block: env, tools, rewards, GRPO, renderer, cache, eval, data
python scripts/smoke_hf.py --train # tiny random Qwen3-VL: chat template, generation, masks, two trainer steps
```

## License and attribution

Copyright 2026 3S Holding OÜ, licensed under the [Apache License 2.0](LICENSE).

The Urbex method belongs to its authors; cite their paper (above). Prompts, the
release tool schemas, the data layout and the evaluation protocols are adapted from
the Urbex reference release (<https://github.com/AMAP-ML/Urbex>, Apache-2.0) for
compatibility with its checkpoint and data; see [NOTICE](NOTICE). Datasets and model
weights are not distributed here. CityRefer, SensatUrban and STPLS3D remain under
their original terms.
