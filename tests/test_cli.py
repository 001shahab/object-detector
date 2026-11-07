# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
import json

from vision3s.cli import build_parser, main
from vision3s.geometry import center_to_xyxy


def test_parser_has_all_commands():
    sub = next(a for a in build_parser()._actions if a.dest == "cmd")
    assert {"download", "build-bev", "build-cache", "prepare-cityrefer", "stpls3d-refer", "train", "evaluate",
            "infer", "hit-k", "metrics-3d", "diagnostics", "landmark-subsets", "aggregate", "plot"} <= set(sub.choices)


def test_hit_k_and_subsets_cli(data_root, sample, tmp_path, capsys):
    pred = tmp_path / "records.jsonl"
    pred.write_text(json.dumps({**sample, "world_bbox": center_to_xyxy(sample["gt_bbox"])}) + "\n")
    split = data_root / "meta" / "cityrefer_test_NO.jsonl"
    main(["hit-k", "--pred", str(pred), "--dataset", str(split), "--box3d-dir", str(data_root / "box3d")])
    assert "Hit@1: 1/1 = 100.00%" in capsys.readouterr().out
    main(["landmark-subsets", "--dataset", str(split), "--landmarks", str(data_root / "landmarks_cityrefer.json"),
          "--out-dir", str(tmp_path)])
    assert '"with_landmarks": 1' in capsys.readouterr().out
