# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
#
# Manifest format and archive validation follow the Urbex reference release
# (https://github.com/AMAP-ML/Urbex, Copyright 2026 The Urbex Authors,
# Apache-2.0) so the official Hugging Face assets can be used unchanged.
"""Checksum-verified download of the released data and checkpoint.

Data lands under ``<output>/<dataset>/`` (``cityrefer`` or ``stpls3d``), which
is the ``data_root`` used by the environment. Every file is pinned to one Hub
revision and verified against the manifest's size and SHA-256; scene archives
(``.tar.zst``) are validated in a temporary tree before replacing the target.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional

DEFAULT_DATASET_ID = "vvangfaye/Urbex-Data"
DEFAULT_MODEL_ID = "vvangfaye/Urbex-4B"
DATASETS = {"cityrefer_ND": "cityrefer", "cityrefer_NO": "cityrefer", "cityrefer": "cityrefer", "stpls3d": "stpls3d"}
MARKER = ".urbex-download.json"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe release path: {value!r}")
    return path


def describe(path: str | Path, root: str | Path) -> dict:
    """Manifest entry for a local file (used when publishing derived assets)."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def fetch(repo_id: str, entry: dict, destination: str | Path, revision: str, install: bool = True) -> Path:
    from huggingface_hub import hf_hub_download

    rel = safe_relative(entry["path"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo_id, str(rel), repo_type="dataset", revision=revision,
                                  cache_dir=str(destination / ".hf-cache")))
    if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
        raise ValueError(f"checksum mismatch: {rel}")
    if not install:
        return source
    target = destination / rel
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"destination escapes output directory: {target}")
    if target.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == entry["bytes"] and sha256(target) == entry["sha256"]:
        return target
    partial = target.with_name(target.name + ".partial")
    shutil.copyfile(source, partial)
    os.replace(partial, target)
    return target


def load_manifest(repo_id: str, output: str | Path, revision: str = "main") -> tuple[dict, str]:
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(repo_id, revision=revision).sha
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id, "manifest.json", repo_type="dataset", revision=revision,
                           cache_dir=str(output / ".hf-cache"))
    manifest = json.loads(Path(path).read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported release manifest")
    return manifest, revision


def unpack(archive: str | Path, destination: str | Path, prefix: str, expected: dict) -> Path:
    import zstandard

    destination = Path(destination)
    prefix_path = safe_relative(prefix)
    target = destination / prefix_path
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"destination escapes output directory: {target}")
    marker = target / MARKER
    if marker.is_file():
        saved = json.loads(marker.read_text())
        files = [p for p in target.rglob("*") if p.is_file() and p != marker]
        if (saved.get("sha256") == expected["sha256"] and len(files) == expected["file_count"]
                and sum(p.stat().st_size for p in files) == expected["unpacked_bytes"]):
            return target
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination) as tmp:
        stage = Path(tmp)
        names: set = set()
        size = 0
        with Path(archive).open("rb") as src, zstandard.ZstdDecompressor().stream_reader(src) as reader, \
                tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                path = safe_relative(member.name)
                if not path.is_relative_to(prefix_path) or not member.isfile() or str(path) in names:
                    raise ValueError(f"invalid archive member: {member.name}")
                names.add(str(path))
                size += member.size
                if len(names) > expected["file_count"] or size > expected["unpacked_bytes"]:
                    raise ValueError("archive exceeds declared contents")
                tar.extract(member, path=stage, filter="data")
        if len(names) != expected["file_count"] or size != expected["unpacked_bytes"]:
            raise ValueError("archive contents do not match the manifest")
        new = stage / prefix_path
        (new / MARKER).write_text(json.dumps({"sha256": expected["sha256"]}) + "\n")
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = stage / "previous"
        if target.exists():
            target.rename(backup)
        try:
            new.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
    return target


def download_dataset(dataset: str, output: str | Path = "data", scenes: Optional[list[str]] = None,
                     all_scenes: bool = False, candidates: bool = False, point_clouds: bool = False,
                     repo_id: Optional[str] = None, revision: str = "main") -> Path:
    name = DATASETS[dataset]
    repo_id = repo_id or os.getenv("HF_DATASET_ID") or DEFAULT_DATASET_ID
    manifest, revision = load_manifest(repo_id, output, revision)
    data = manifest["datasets"][name]
    selected = sorted(data["scenes"]) if all_scenes else list(dict.fromkeys(scenes or []))
    clouds = manifest.get("point_clouds", {}).get(name, {})
    unknown = set(selected) - (set(data["scenes"]) | (set(clouds) if point_clouds else set()))
    if unknown:
        raise ValueError(f"unknown scenes for {name}: {sorted(unknown)}")
    for entry in data["files"]:
        fetch(repo_id, entry, output, revision)
    for scene in selected:
        if scene not in data["scenes"]:
            continue
        entry = data["scenes"][scene]
        for f in entry["files"]:
            fetch(repo_id, f, output, revision)
        archive = fetch(repo_id, entry["archive"], output, revision, install=False)
        unpack(archive, output, entry["extract_prefix"], entry["archive"])
    if point_clouds:
        requested = sorted(clouds) if all_scenes else selected
        if not clouds or any(s not in clouds for s in requested):
            hint = (" SensatUrban point clouds are not redistributed; obtain them from "
                    "https://github.com/QingyongHu/SensatUrban and place the PLY files under "
                    "raw_data/sensaturban/ply/{train,test}/.") if name == "cityrefer" else ""
            raise ValueError("original point clouds are not released for the requested scenes." + hint)
        for scene in requested:
            for entry in clouds[scene]:
                fetch(repo_id, entry, Path(output).resolve().parent, revision)
    if candidates:
        if name != "cityrefer":
            raise ValueError("candidates are only separate for CityRefer; STPLS3D box3d is included")
        entry = manifest["optional"]["cityrefer_candidates"]
        archive = fetch(repo_id, entry, output, revision, install=False)
        unpack(archive, output, "cityrefer_candidates", entry)
    return Path(output) / name


def download_model(output: str | Path = "model/Urbex-4B", repo_id: Optional[str] = None,
                   revision: str = "main") -> Path:
    from huggingface_hub import HfApi, snapshot_download

    repo_id = repo_id or os.getenv("HF_MODEL_ID") or DEFAULT_MODEL_ID
    revision = HfApi().model_info(repo_id, revision=revision).sha
    return Path(snapshot_download(repo_id, revision=revision, local_dir=str(output)))
