# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Persistent, content-addressed collision preparation for reconstructed scenes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

__all__ = []


def _cook(
    vertices: np.ndarray, triangles: np.ndarray, recipe: dict[str, Any], directory: Path
) -> Path:
    """Isolate native CUDA/OptiX work and its interpreter lifetime from the host."""
    source = directory / "input.npz"
    np.savez(source, vertices=vertices, triangles=triangles)
    output = directory / "output.obj"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodichain.toolkits.skill_fitting._collision_worker",
            str(source),
            str(output),
            str(recipe["max_hulls"]),
            recipe["algorithm"],
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"碰撞网格分割失败（exit {result.returncode}）："
            + (result.stdout + result.stderr)[-4000:]
        )
    return output


@contextmanager
def _entry_lock(path: Path) -> Iterator[None]:
    """Serialize one entry across preparation threads and independent processes."""
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resolve_scene_path(scene: str, joints: str) -> Path | None:
    """Resolve an explicit scene or the existing conventional sibling export."""
    if scene:
        return Path(scene).expanduser().resolve(strict=True)
    if joints:
        inferred = (
            Path(joints).expanduser().resolve().parent
            / "blender_scene"
            / "physical_export"
        )
        if (inferred / "scene_assets.yaml").is_file():
            return inferred
    return None


def prepare_scene_collision(
    scene: Path,
    *,
    cache_root: Path | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Prepare each declared decomposition once, retaining source files unchanged.

    Keys include source mesh bytes, effective local geometry, cooking parameters
    and DexSim version. A checksum-validated complete entry is published only
    after successful cooking and part validation. Pose changes reuse local parts.
    """
    import yaml
    import dexsim
    from embodichain.data_pipeline.skill_fitting import file_digest
    from embodichain.lab.sim.shapes import MeshCollisionCfg
    from embodichain.lab.sim.spawn._convex_parts import load_convex_parts
    from ._scene import load_scene

    objects, source_files = load_scene(scene, include_ungraspable=True)
    source = (scene / "scene_assets.yaml" if scene.is_dir() else scene).resolve()
    declarations = yaml.safe_load(source.read_text())["objects"]
    root = cache_root or Path.home() / ".cache" / "embodichain" / "scene_collision"
    root.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for obj in objects:
        shape = declarations[obj.identifier]["shape"]
        cfg = MeshCollisionCfg.from_dict(shape.get("collision") or {})
        if cfg.approximation != "convex_decomposition":
            continue
        # Decompose before body scaling, matching the runtime collision adapter.
        scale = np.asarray(shape.get("scale", [1, 1, 1]), dtype=np.float64)
        vertices = np.ascontiguousarray(obj.vertices / scale, dtype=np.float32)
        triangles = np.ascontiguousarray(obj.triangles, dtype=np.int32)
        supplied = None
        if cfg.decomposition_fpath is not None:
            supplied = (source.parent / cfg.decomposition_fpath).resolve(strict=True)
            source_files[str(supplied)] = file_digest(supplied)
        recipe = dict(
            schema=1,
            source_mesh_sha256=source_files[str(obj.mesh_path)],
            geometry_sha256=hashlib.sha256(
                vertices.tobytes() + triangles.tobytes()
            ).hexdigest(),
            max_hulls=cfg.max_hulls,
            vertex_limit=64,
            algorithm=cfg.acd_method or "visacd",
            dexsim_version=dexsim.__version__,
            supplied_sha256=None if supplied is None else source_files[str(supplied)],
        )
        key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        entry = root / key
        with _entry_lock(root / f"{key}.lock"):
            metadata = None
            try:
                candidate = json.loads((entry / "manifest.json").read_text())
                if (
                    candidate["recipe"] == recipe
                    and file_digest(entry / "convex_parts.obj") == candidate["sha256"]
                ):
                    parts = load_convex_parts(entry / "convex_parts.obj")
                    if (
                        len(parts) == candidate["hull_count"]
                        and len(parts) <= cfg.max_hulls
                    ):
                        metadata = candidate
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                pass
            hit = metadata is not None
            if not hit:
                progress(f"正在准备 {obj.identifier} 的碰撞网格（首次处理需要等待）…")
                with tempfile.TemporaryDirectory(
                    prefix=f".{key}.", dir=root
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    cooked = supplied or _cook(
                        vertices, triangles, recipe, Path(temporary)
                    )
                    destination = staging / "convex_parts.obj"
                    shutil.copyfile(cooked, destination)
                    parts = load_convex_parts(destination)
                    if len(parts) > cfg.max_hulls:
                        raise ValueError(
                            f"{obj.identifier}: decomposition exceeds max_hulls"
                        )
                    metadata = dict(
                        recipe=recipe,
                        sha256=file_digest(destination),
                        hull_count=len(parts),
                    )
                    (staging / "manifest.json").write_text(
                        json.dumps(metadata, indent=2)
                    )
                    for path, digest in source_files.items():
                        if file_digest(path) != digest:
                            raise ValueError(
                                "碰撞网格处理期间重建场景发生变化，请重新加载。"
                            )
                    if entry.exists():
                        shutil.rmtree(entry)
                    staging.rename(entry)
            prepared[obj.identifier] = dict(
                **metadata,
                fpath=str(entry / "convex_parts.obj"),
                cache_key=key,
                cache_hit=hit,
            )
            progress(
                f"{obj.identifier} 碰撞网格已就绪：{len(parts)} 个凸块"
                + ("（复用缓存）" if hit else "")
            )
    for path, digest in source_files.items():
        if file_digest(path) != digest:
            raise ValueError("碰撞网格处理期间重建场景发生变化，请重新加载。")
    return dict(
        scene_path=str(source),
        objects=prepared,
        object_count=len(objects),
        prepared_count=len(prepared),
        hull_count=sum(item["hull_count"] for item in prepared.values()),
        cache_hits=sum(item["cache_hit"] for item in prepared.values()),
        source_files_sha256=source_files,
    )
