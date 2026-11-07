# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
"""``3s-vision`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .prompts import FIXED_WORKFLOW, TOOL_NAMES


def _jsonl(path):
    from .agent.rollout import read_jsonl

    return read_jsonl(path)


def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


# ── data preparation ─────────────────────────────────────────────────────────

def cmd_download(a):
    from .data.download import download_dataset, download_model

    if a.what == "model":
        print(download_model(a.output or "model/Urbex-4B", a.repo_id, a.revision))
        return
    if not (a.scenes or a.all or a.metadata_only):
        sys.exit("choose --scenes, --all or --metadata-only")
    print(download_dataset(a.what, a.output or "data", a.scenes, a.all, a.candidates, a.point_clouds, a.repo_id,
                           a.revision))


def _scenes(a, dataset):
    from .data.ply import resolve_ply

    if a.scenes:
        return a.scenes
    root = Path(a.raw_root)
    return sorted({p.stem.replace("_points", "") for p in root.rglob("*.ply")
                   if resolve_ply(p.stem.replace("_points", ""), dataset, root)})


def cmd_build_bev(a):
    from .data.bev import build_scene_bev
    from .data.ply import read_ply, resolve_ply

    out = Path(a.data_root) / f"bev_{a.dataset}"
    for scene in _scenes(a, a.dataset):
        path = resolve_ply(scene, a.dataset, a.raw_root)
        if path is None:
            print(f"[skip] {scene}: no PLY")
            continue
        pts, cols = read_ply(path)
        _print(build_scene_bev(scene, pts, cols, out, resolution=a.resolution))


def cmd_build_cache(a):
    from .data.ply import read_ply, resolve_ply
    from .oblique.cache import build_scene_cache

    out = Path(a.data_root) / "verify_region_cache" / a.dataset
    for scene in _scenes(a, a.dataset):
        path = resolve_ply(scene, a.dataset, a.raw_root)
        if path is None:
            print(f"[skip] {scene}: no PLY")
            continue
        pts, cols = read_ply(path)
        _print(build_scene_cache(scene, a.dataset, pts, cols, out, region_size=a.region_size, step=a.step,
                                 output_size=a.output_size, progress=True))


def cmd_prepare_cityrefer(a):
    from .data.cityrefer import prepare_cityrefer

    _print(prepare_cityrefer(a.annotations, a.box_dir, a.out, landmarks=a.landmarks, exclusions=a.exclusions,
                             no_scenes=a.no_scenes, seed=a.seed, raw_root=a.raw_root))


def cmd_stpls3d(a):
    from .data import stpls3d_pipeline as p

    if a.stage == "instances":
        _print(p.stage_instances(a.ply, a.scene, a.out, a.building_label, tuple(a.vehicle_labels)))
    elif a.stage == "landmarks":
        _print(p.stage_landmarks(a.scene, a.gcps, a.out, osm_json=a.osm_json))
    elif a.stage == "generate":
        _print(p.stage_generate(a.scene, a.ply, a.out, a.model, a.base_url, a.api_key, a.max_targets, a.seed))
    else:
        _print(p.stage_finalize(a.out, a.scenes))


# ── training ─────────────────────────────────────────────────────────────────

def _overrides(pairs):
    import yaml

    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def cmd_train(a):
    from .rl.trainer import GRPOTrainer, TrainConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainConfig.from_yaml(a.config, _overrides(a.set))
    GRPOTrainer(cfg).fit()


# ── evaluation ───────────────────────────────────────────────────────────────

def _env_factory(a):
    from .env import EnvConfig, GroundingEnv

    cfg = EnvConfig(data_root=a.data_root, protocol=a.protocol, landmark_mode=a.landmark_mode,
                    max_assistant_turns=a.max_turns, max_tool_calls=max(0, a.max_turns - 1),
                    landmark_noise_kind=a.noise_kind, enabled_tools=tuple(a.tools),
                    forced_tool_sequence=FIXED_WORKFLOW if a.fixed_workflow else None,
                    raw_data_root=a.raw_root, require_cache=a.require_cache)
    if a.no_cache:
        cfg.oblique_cache = ""
    base = GroundingEnv.from_config(cfg, a.dataset)
    return base.clone


def _policy_factory(a):
    from .agent.policy import GenerationConfig, HFPolicy, OpenAIPolicy

    sample = a.decoding == "sample"
    gen = GenerationConfig(a.max_new_tokens, a.temperature, a.top_p, a.top_k) if sample \
        else GenerationConfig.greedy(a.max_new_tokens)
    if a.api_model:
        return lambda run: OpenAIPolicy(a.api_model, a.base_url, a.api_key, gen, a.reasoning_effort,
                                        max_workers=a.workers)
    cache = {}

    def make(run):
        if "p" not in cache:
            cache["p"] = HFPolicy.from_pretrained(a.model, gen, attn_implementation=a.attn)
        return cache["p"]

    return make


def _candidates(a):
    from .eval.hit_k import load_box3d_cache, load_proposal_cache, load_pth_cache

    if getattr(a, "pth_dir", None):
        return load_pth_cache(a.pth_dir)
    if getattr(a, "proposals", None):
        return load_proposal_cache(a.proposals)
    box_dir = getattr(a, "box3d_dir", None) or (Path(a.data_root) / "box3d" if getattr(a, "data_root", None) else None)
    if box_dir and Path(box_dir).exists():
        return load_box3d_cache(box_dir)
    return None


def _points_loader(a, dataset):
    if not a.raw_root:
        return None
    from .data.ply import scene_point_cloud_loader

    load = scene_point_cloud_loader(a.raw_root)

    def points(scene):
        res = load(scene, dataset)
        return res[0] if res else None

    return points


def _split_rows(a):
    path = Path(a.split)
    if not path.exists():
        path = Path(a.data_root) / a.split
    return _jsonl(path)


def cmd_evaluate(a):
    from .eval.runner import evaluate

    logging.basicConfig(level=logging.INFO)
    rows = _split_rows(a)
    summary = evaluate(_policy_factory(a), _env_factory(a), rows, a.output, a.num_runs, a.batch_size,
                       _candidates(a), _points_loader(a, a.dataset), a.limit, a.seed)
    _print({k: {"mean": v["mean"], "std": v["std"], "ci95": v["ci95"]} for k, v in summary["aggregate"].items()})


def cmd_infer(a):
    from .agent.rollout import run_rollouts

    sample = {"scene_id": a.scene, "dataset": a.dataset, "query": a.query,
              "bev_png": f"bev_{a.dataset}/{a.scene}.png", "gt_bbox": None}
    traj = run_rollouts(_policy_factory(a)(1), _env_factory(a), [sample])[0]
    rec = traj.record()
    for m in rec["messages"][2:]:
        content = m["content"] if isinstance(m["content"], str) else " ".join(
            c.get("text", "[image]") for c in m["content"])
        print(f"--- {m['role']}\n{content}")
    _print({"reason": rec["reason"], "answer": rec["answer"], "world_bbox": rec["world_bbox"]})


def cmd_hit_k(a):
    from .eval.hit_k import compute_hit_k, format_report, overlap_and_area_summary

    res = compute_hit_k(_jsonl(a.pred), _jsonl(a.dataset), _candidates(a), score_rule=a.score_rule,
                        require_positive_overlap=a.positive_overlap,
                        geometric_match_iou=a.geometric_match_iou)
    print(format_report(res))
    _print(overlap_and_area_summary(res["stats"]))
    if a.save_detail:
        with open(a.save_detail, "w", encoding="utf-8") as fh:
            for s in res["stats"]:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def cmd_metrics_3d(a):
    from .eval.metrics_3d import evaluate_3d

    res = evaluate_3d(_jsonl(a.pred), _jsonl(a.dataset), _points_loader(a, a.scene_dataset),
                      no_answer_as_zero=a.no_answer_as_zero)
    _print(res["summary"])
    if a.output:
        with open(a.output, "w", encoding="utf-8") as fh:
            for s in res["per_sample"]:
                fh.write(json.dumps(s, ensure_ascii=False, default=float) + "\n")


def cmd_diagnostics(a):
    from .eval.diagnostics import latency_summary, policy_summary, search_diagnostics
    from .eval.hit_k import compute_hit_k
    from .landmarks import LandmarkDB

    records, rows = _jsonl(a.pred), _jsonl(a.dataset)
    cands = _candidates(a)
    hk = compute_hit_k(records, rows, cands)
    res = search_diagnostics(records, rows, cands, LandmarkDB.load(a.landmarks), hk["stats"], a.region_rule)
    out = {"policy": policy_summary(records), "latency": latency_summary(records),
           **{k: v for k, v in res.items() if k != "per_query"}}
    _print(out)
    if a.output:
        Path(a.output).write_text(json.dumps({**out, "per_query": res["per_query"]}, indent=2, default=float))


def cmd_landmark_subsets(a):
    from .eval.diagnostics import landmark_subsets
    from .landmarks import LandmarkDB

    with_lm, without = landmark_subsets(_jsonl(a.dataset), LandmarkDB.load(a.landmarks))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.dataset).stem
    for name, rows in (("with_landmarks", with_lm), ("without_landmarks", without)):
        with (out / f"{stem}_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _print({"with_landmarks": len(with_lm), "without_landmarks": len(without)})


def cmd_aggregate(a):
    from .eval.diagnostics import aggregate_runs

    runs = [json.loads(Path(p).read_text()).get("flat", {}) for p in a.scores]
    _print(aggregate_runs(runs))


def cmd_plot(a):
    from . import viz

    if a.kind == "training":
        viz.plot_training(a.inputs[0], a.output)
    else:
        results = {Path(p).stem: json.loads(Path(p).read_text()) for p in a.inputs}
        viz.plot_search_diagnostics(results, a.output)
    print(a.output)


# ── parser ───────────────────────────────────────────────────────────────────

def _env_args(p):
    p.add_argument("--data-root", required=True, help="dataset root (meta/, bev_*/, landmarks_*.json, ...)")
    p.add_argument("--dataset", default="cityrefer", choices=["cityrefer", "stpls3d"])
    p.add_argument("--protocol", default="paper", choices=["paper", "release"],
                   help="tool schema: 'paper' (Table 5) or 'release' (exact released checkpoint schema)")
    p.add_argument("--landmark-mode", default="clean", choices=["clean", "missing", "noisy", "wrong"])
    p.add_argument("--noise-kind", default="fixed", choices=["fixed", "gaussian"])
    p.add_argument("--tools", nargs="*", default=list(TOOL_NAMES), help="enabled tools (none = direct answer)")
    p.add_argument("--max-turns", type=int, default=8, help="assistant-turn budget T_max")
    p.add_argument("--fixed-workflow", action="store_true", help="enforce Landmark → Zoom-in → Oblique (Table 9)")
    p.add_argument("--raw-root", help="raw point clouds (on-demand oblique views and 3D metrics)")
    p.add_argument("--no-cache", action="store_true", help="ignore the oblique cache; render on demand")
    p.add_argument("--require-cache", action="store_true")


def _policy_args(p):
    p.add_argument("--model", default="vvangfaye/Urbex-4B", help="HF model path for local inference")
    p.add_argument("--api-model", help="use an OpenAI-compatible endpoint with this model name instead")
    p.add_argument("--base-url")
    p.add_argument("--api-key")
    p.add_argument("--reasoning-effort")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--attn")
    p.add_argument("--decoding", default="greedy", choices=["greedy", "sample"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=60)
    p.add_argument("--max-new-tokens", type=int, default=1024)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="3s-vision", description="Urbex agentic city-scale 3D grounding")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download released data or the Urbex-4B checkpoint")
    p.add_argument("what", choices=["cityrefer", "cityrefer_ND", "cityrefer_NO", "stpls3d", "model"])
    p.add_argument("--output")
    p.add_argument("--repo-id")
    p.add_argument("--revision", default="main")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scenes", nargs="+")
    g.add_argument("--all", action="store_true")
    g.add_argument("--metadata-only", action="store_true")
    p.add_argument("--candidates", action="store_true")
    p.add_argument("--point-clouds", action="store_true")
    p.set_defaults(fn=cmd_download)

    for name, fn, help_ in (("build-bev", cmd_build_bev, "render high-resolution BEV images + mappings"),
                            ("build-cache", cmd_build_cache, "pre-render the oblique-view region cache")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--dataset", default="cityrefer", choices=["cityrefer", "stpls3d"])
        p.add_argument("--raw-root", required=True)
        p.add_argument("--data-root", required=True)
        p.add_argument("--scenes", nargs="+")
        if name == "build-bev":
            p.add_argument("--resolution", type=float, default=0.05, help="metres per pixel")
        else:
            p.add_argument("--region-size", type=float, default=40.0)
            p.add_argument("--step", type=float, default=15.0)
            p.add_argument("--output-size", type=int, default=512)
        p.set_defaults(fn=fn)

    p = sub.add_parser("prepare-cityrefer", help="filter, split and lay out CityRefer (Appendix B.1)")
    p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--box-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--landmarks")
    p.add_argument("--exclusions")
    p.add_argument("--no-scenes", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-root")
    p.set_defaults(fn=cmd_prepare_cityrefer)

    p = sub.add_parser("stpls3d-refer", help="build STPLS3D-Refer stage by stage (Appendix B.2)")
    p.add_argument("stage", choices=["instances", "landmarks", "generate", "finalize"])
    p.add_argument("--out", required=True)
    p.add_argument("--scene")
    p.add_argument("--scenes", nargs="+")
    p.add_argument("--ply")
    p.add_argument("--building-label", type=int, default=1)
    p.add_argument("--vehicle-labels", type=int, nargs="+", default=[5, 6])
    p.add_argument("--gcps", help="JSON list of {x, y, lat, lon} ground-control points")
    p.add_argument("--osm-json", help="pre-fetched Overpass response")
    p.add_argument("--model", default="qwen3.6-plus")
    p.add_argument("--base-url")
    p.add_argument("--api-key")
    p.add_argument("--max-targets", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_stpls3d)

    p = sub.add_parser("train", help="multi-turn GRPO training (run under `accelerate launch` for multi-GPU)")
    p.add_argument("--config", default="configs/grpo.yaml")
    p.add_argument("--set", nargs="*", help="override config keys, e.g. max_steps=10")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate", help="roll out a split and score it (Hit@k, 2D/3D IoU, point F1)")
    _env_args(p)
    _policy_args(p)
    p.add_argument("--split", required=True, help="jsonl path, absolute or relative to --data-root")
    p.add_argument("--output", required=True)
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--box3d-dir")
    p.add_argument("--pth-dir")
    p.add_argument("--proposals")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("infer", help="ground one free-form query in one scene")
    _env_args(p)
    _policy_args(p)
    p.add_argument("--scene", required=True)
    p.add_argument("--query", required=True)
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("hit-k", help="Hit@k against a GT candidate pool")
    p.add_argument("--pred", required=True)
    p.add_argument("--dataset", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--box3d-dir")
    src.add_argument("--pth-dir")
    src.add_argument("--proposals")
    p.add_argument("--score-rule", default="iou", choices=["iou", "ciou", "center_distance"])
    p.add_argument("--positive-overlap", action="store_true")
    p.add_argument("--geometric-match-iou", type=float, help="proposal matching threshold (Table 11)")
    p.add_argument("--save-detail")
    p.set_defaults(fn=cmd_hit_k)

    p = sub.add_parser("metrics-3d", help="2D IoU, 3D IoU and point F1 from raw point clouds")
    p.add_argument("--pred", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--scene-dataset", default="cityrefer", choices=["cityrefer", "stpls3d"])
    p.add_argument("--no-answer-as-zero", action="store_true")
    p.add_argument("--output")
    p.set_defaults(fn=cmd_metrics_3d)

    p = sub.add_parser("diagnostics", help="trajectory diagnostics (Tables 9, 16, 17)")
    p.add_argument("--pred", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--box3d-dir", required=True)
    p.add_argument("--landmarks", required=True)
    p.add_argument("--region-rule", default="last", choices=["last", "first", "any"])
    p.add_argument("--output")
    p.set_defaults(fn=cmd_diagnostics)

    p = sub.add_parser("landmark-subsets", help="split a test set by named-landmark presence (Table 10)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--landmarks", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_landmark_subsets)

    p = sub.add_parser("aggregate", help="mean ± std and 95%% t-interval over runs' scores.json")
    p.add_argument("scores", nargs="+")
    p.set_defaults(fn=cmd_aggregate)

    p = sub.add_parser("plot", help="training curves (Fig. 7) or search diagnostics (Fig. 9)")
    p.add_argument("kind", choices=["training", "diagnostics"])
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_plot)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
