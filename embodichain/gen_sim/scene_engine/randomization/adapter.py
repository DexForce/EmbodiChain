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

"""Scene Engine exports adapted to the source-independent layout service."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import yaml

from embodichain.lab.sim.scene.expansion import (
    SceneLayoutInput,
    SceneRandomizationJobCfg,
    SceneRandomizationPolicyCfg,
    SceneRandomizationBatch,
    generate_scene_variants,
    prepare_scene_randomization,
    save_scene_variants,
)
from ._geometry import _Y_UP_TO_Z_UP
from .baseline import _entry_pose, _rigid_entries, _support_groups, _source_digest
from .cfg import _UniqueKeyLoader

__all__ = [
    "scene_layout_input_from_export",
    "randomize_scene_export",
    "convert_layouts",
]


def _export_digest(output_root: Path) -> str:
    export = (
        output_root
        if (output_root / "scene_config.json").is_file()
        else output_root / "scene_export"
    )
    config_bytes = (export / "scene_config.json").read_bytes()
    graph_bytes = (export / "scene_graph.json").read_bytes()
    config = json.loads(config_bytes)
    return _source_digest(
        export,
        config_bytes,
        graph_bytes,
        [export / entry["shape"]["fpath"] for entry in _rigid_entries(config)],
    )


def scene_layout_input_from_export(
    output_root: str | Path,
    *,
    scene_id: str | None = None,
) -> SceneLayoutInput:
    """Read scaled local Z-up geometry without modifying its source export.

    Args:
        output_root: Workspace or scene_export directory.
        scene_id: Optional preserved legacy proposal parent identity.

    Returns:
        Owned source-independent geometry and support tree.
    """
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
        SceneExportImporter,
    )
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
        load_glb_mesh,
    )

    root = Path(output_root).expanduser().resolve()
    export = root if (root / "scene_config.json").is_file() else root / "scene_export"
    config = json.loads((export / "scene_config.json").read_text())
    entries = _rigid_entries(config)
    _, graph = SceneExportImporter(
        output_root=export.parent, scene_export_root=export
    ).read_scene_and_graph()
    parents = {}
    for uid, members in _support_groups(graph).items():
        parents[uid] = "table"
        parents.update(dict(members))
    vertices, faces, poses = {}, {}, {}
    for entry in entries:
        uid = entry["uid"]
        mesh = load_glb_mesh(export / entry["shape"]["fpath"])
        vertices[uid] = (
            np.asarray(mesh.vertices) * entry.get("body_scale", [1, 1, 1])
        ) @ _Y_UP_TO_Z_UP.T
        faces[uid] = np.asarray(mesh.faces)
        poses[uid] = _entry_pose(entry)
    return SceneLayoutInput(
        config["scene_id"] if scene_id is None else scene_id,
        "table",
        poses,
        vertices,
        faces,
        parents,
    )


def _job_config(
    value: SceneRandomizationJobCfg | Mapping | str | Path,
) -> SceneRandomizationJobCfg:
    if isinstance(value, (str, Path)):
        value = yaml.load(Path(value).expanduser().read_text(), Loader=_UniqueKeyLoader)
    if isinstance(value, Mapping):
        raw = dict(value)
        version = raw.pop("schema_version", None)
        if version is not None and (
            type(version) is not int or version not in (1, 2, 3, 4)
        ):
            raise ValueError("Unsupported scene randomization configuration version.")
        if "policy" not in raw:
            job_fields = {
                key: raw.pop(key)
                for key in ("seed", "candidate_count", "max_attempts", "layout_storage")
                if key in raw
            }
            raw.setdefault("backend", "batched")
            value = SceneRandomizationJobCfg(
                policy=SceneRandomizationPolicyCfg(**raw), **job_fields
            )
        else:
            if isinstance(raw["policy"], Mapping):
                raw["policy"] = SceneRandomizationPolicyCfg(**dict(raw["policy"]))
            value = SceneRandomizationJobCfg(**raw)
    if not isinstance(value, SceneRandomizationJobCfg):
        raise TypeError(
            "scene_randomization must be a job config, mapping or YAML path"
        )
    value.validate()
    return value.copy()


def randomize_scene_export(
    output_root: str | Path,
    config: SceneRandomizationJobCfg | Mapping | str | Path,
) -> Path:
    """Generate a v4 run after the final Scene Engine export.

    Args:
        output_root: Existing workspace containing scene_export.
        config: Shared job settings or legacy/structured policy YAML path.

    Returns:
        Completed run under scene_layouts, retaining the original export.
    """
    cfg = _job_config(config)
    root = Path(output_root).expanduser().resolve()
    if root.name == "scene_export":
        root = root.parent
    original_digest = _export_digest(root)
    source = scene_layout_input_from_export(root)
    prepared = prepare_scene_randomization(
        source, cfg.policy, cache_dir=root / "scene_layouts" / "geometry_cache"
    )
    result = generate_scene_variants(
        prepared,
        seed=cfg.seed,
        candidate_count=cfg.candidate_count,
        max_attempts=cfg.max_attempts,
    )
    runs = root / "scene_layouts"
    runs.mkdir(exist_ok=True)
    index = 1
    while (runs / f"run_{index:03d}").exists():
        index += 1
    if _export_digest(root) != original_digest:
        raise ValueError("Scene Engine source changed during layout generation.")
    destination = runs / f"run_{index:03d}"
    return save_scene_variants(
        result,
        destination,
        storage=cfg.layout_storage,
        source_context={
            "scene_engine_output_root": os.path.relpath(root, destination),
            "scene_engine_source_sha256": original_digest,
        },
    )


def convert_layouts(
    layout_dir: str | Path, output_dir: str | Path, *, storage: str = "packed"
) -> Path:
    """Convert validated legacy layouts without resampling or changing identity.

    Args:
        layout_dir: Completed legacy v1/v2/v3 layouts directory.
        output_dir: New v4 run directory.
        storage: JSON or packed storage.

    Returns:
        Completed source-independent run.
    """
    from .loading import load_layouts

    loaded = load_layouts(layout_dir)
    if not len(loaded.variants):
        raise ValueError("No legacy layouts to convert.")
    run = Path(layout_dir).resolve().parent
    manifest = json.loads((run / "run_result.json").read_text())
    policy = yaml.load(
        (run / manifest["policy_file"]).read_text(), Loader=_UniqueKeyLoader
    )
    source = scene_layout_input_from_export(
        loaded.output_root, scene_id=loaded.variants[0].parent_scene_id
    )
    cfg = SceneRandomizationPolicyCfg(
        randomize_groups=policy.get("randomize_groups", True),
        intra_group=policy.get("intra_group"),
        collision_margin_m=policy["collision_margin_m"],
    )
    prepared = prepare_scene_randomization(source, cfg)
    from .baseline import prepare_scene_baseline
    from .cfg import load_randomization_config

    baseline = prepare_scene_baseline(
        load_randomization_config(
            run / manifest["policy_file"], output_root=loaded.output_root
        )
    )
    table = baseline.data["table"]
    center = np.asarray(table["center_xy"])
    half = np.asarray(table["size_xy"]) / 2
    geometry = deepcopy(prepared._geometry)
    if not policy.get("intra_group", {}).get("targets"):
        geometry["fixed_orientation_group_proxies"] = {
            group["root_id"]: (
                np.asarray(group["aabb_2d"]["center_xy"])[None]
                + np.array([-1, 1])[:, None]
                * np.asarray(group["aabb_2d"]["size_xy"])[None]
                / 2
            ).tolist()
            for group in baseline.data["groups"]
        }
    from embodichain.lab.sim.scene.expansion.layout import _digest

    prepared = replace(
        prepared,
        _geometry=geometry,
        geometry_sha256=_digest(geometry),
        _table=tuple(tuple(row) for row in (center - half, center + half)),
    )
    result = SceneRandomizationBatch(
        prepared,
        loaded.variants,
        manifest["attempts"],
        manifest["status"],
        policy["seed"],
        0,
        manifest["attempts"],
        manifest.get("requested_count", len(loaded.variants)),
        manifest["max_attempts"],
    )
    return save_scene_variants(
        result,
        output_dir,
        storage=storage,
        source_context={
            "scene_engine_output_root": os.path.relpath(
                loaded.output_root, Path(output_dir).resolve()
            ),
            "scene_engine_source_sha256": baseline.data["source_scene"][
                "content_sha256"
            ],
        },
    )
