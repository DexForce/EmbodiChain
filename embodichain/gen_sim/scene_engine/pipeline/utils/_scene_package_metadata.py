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

"""Project an editable Scene Engine export into AssetRevision metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.usd_scene import UsdSceneIndex

__all__: list[str] = []

_SCENE_EXTENSION = "embodichain:scene_export"
_FRAME_ID = "asset-stage"
_GLTF_CONVENTION = {
    "unit": "meter",
    "meters_per_unit": 1.0,
    "up_axis": "Y",
    "handedness": "right",
    "frame_id": "gltf-y-up",
}
_STABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_TAXONOMY = {
    "namespace": "embodichain",
    "taxonomy": "scene-graph",
    "taxonomy_version": "1",
}


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stable_id(value: str, prefix: str) -> str:
    return value if _STABLE_ID.fullmatch(value) else f"{prefix}-{_digest(value)[7:]}"


def _local_file(export: Path, path: Path) -> Path:
    if not path.is_relative_to(export) or path.resolve(strict=True) != path:
        raise ValueError(f"Editable scene dependency escapes its export: {path}")
    if not path.is_file():
        raise ValueError(f"Editable scene dependency is not a regular file: {path}")
    return path


def _glb_dependencies(glb: Path, export: Path) -> set[Path]:
    # Read the JSON chunk only, so embedded meshes and textures stay on disk.
    with glb.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError(f"Invalid GLB header: {glb}")
        magic, version, size, length, kind = struct.unpack("<4sIIII", header)
        if (
            magic != b"glTF"
            or version != 2
            or size != glb.stat().st_size
            or kind != 0x4E4F534A
            or length > size - 20
        ):
            raise ValueError(f"Invalid GLB JSON chunk: {glb}")
        document = json.loads(stream.read(length))
    dependencies = set()
    for item in document.get("buffers", []) + document.get("images", []):
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        parsed = urlsplit(uri)
        path = Path(unquote(parsed.path))
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or path.is_absolute()
        ):
            raise ValueError(f"Editable GLB dependency must be relative: {uri!r}")
        dependencies.add(_local_file(export, Path(os.path.abspath(glb.parent / path))))
    return dependencies


def _usd_dependencies(path: Path, export: Path) -> set[Path]:
    """Resolve USD dependencies within the editable export."""
    from pxr import UsdUtils

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    if unresolved:
        raise ValueError(
            f"Editable articulation has unresolved dependencies: {unresolved} (in {path})"
        )
    return {
        _local_file(export, Path(dependency))
        for dependency in [*(layer.realPath for layer in layers), *assets]
    }


def _asset_files(
    obj: SceneObject, export: Path, entry: dict[str, Any]
) -> tuple[Path, Path, set[Path]]:
    """Resolve the runtime file, editable proxy and all their local dependencies."""
    assert obj.simready_glb_path is not None  # SceneExportImporter requires a GLB.
    proxy = _local_file(export, Path(obj.simready_glb_path))
    dependencies = _glb_dependencies(proxy, export)
    if obj.articulated_usdc_path is None:
        return proxy, proxy, dependencies

    # The imported object can hold the original USD for editing. The config
    # identifies which USD the simulator actually loads.
    runtime = _local_file(export, export / entry["fpath"])
    usd_files = {runtime}
    if "source_usdc_fpath" in entry:
        usd_files.add(_local_file(export, export / entry["source_usdc_fpath"]))
    dependencies.update(usd_files)
    for path in sorted(usd_files):
        dependencies.update(_usd_dependencies(path, export))
    dependencies.discard(runtime)
    return runtime, proxy, dependencies


def _entries_by_uid(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        entry["uid"]: entry
        for section in ("background", "rigid_object", "articulation")
        for entry in config.get(section, [])
    }


def _stage_scene_export(source: Path, destination: Path) -> None:
    """Stage the v1 config, graph and referenced assets for the SDK inventory."""
    from pxr import Sdf, UsdUtils

    importer = SceneExportImporter(output_root=source.parent, scene_export_root=source)
    scene, _ = importer.read_scene_and_graph()
    entries = _entries_by_uid(
        json.loads(importer.scene_config_path.read_text(encoding="utf-8"))
    )
    files = {
        _local_file(source, source / "scene_config.json"),
        _local_file(source, source / "scene_graph.json"),
    }
    for obj in scene.objects:
        runtime, proxy, dependencies = _asset_files(obj, source, entries[obj.id])
        files.update((runtime, proxy, *dependencies))
    for path in sorted(files):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        references = UsdUtils.ExtractExternalReferences(str(path))
        if any(Path(dep).is_absolute() for group in references for dep in group):

            def rebase(dependency: str) -> str:
                local = Path(dependency)
                if local.is_absolute():
                    return os.path.relpath(_local_file(source, local), path.parent)
                return dependency

            # Older editable USDCs can use absolute texture paths inside the
            # workspace. Only the private copy needs portable authoring.
            layer = Sdf.Layer.FindOrOpen(str(target))
            UsdUtils.ModifyAssetPaths(layer, rebase)
            layer.Save()


def _usd_convention(path: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if up_axis not in ("Y", "Z"):
        raise ValueError(f"Unsupported editable USD up axis: {up_axis}")
    convention = {
        "unit": "usd_stage_unit",
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "kilograms_per_unit": float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
        "up_axis": up_axis,
        "handedness": "right",
    }
    return {**convention, "frame_id": f"source-usd-{_digest(convention)[7:]}"}


def _transform(entry: dict[str, Any], source_up_axis: str) -> dict[str, Any]:
    translation = np.asarray(entry.get("init_pos", [0, 0, 0]), dtype=float)
    angles = np.asarray(entry.get("init_rot", [0, 0, 0]), dtype=float)
    scale = np.asarray(entry.get("body_scale", [1, 1, 1]), dtype=float)
    if any(
        value.shape != (3,) or not np.isfinite(value).all()
        for value in (translation, angles, scale)
    ) or np.any(scale <= 0):
        raise ValueError(
            f"Invalid packaged transform for scene object {entry['uid']!r}"
        )
    return {
        "translation": translation.tolist(),
        "rotation_xyzw": Rotation.from_euler("XYZ", angles, degrees=True)
        .as_quat()
        .tolist(),
        # V1 scales the source asset before axis conversion.
        "scale": (scale[[0, 2, 1]] if source_up_axis == "Y" else scale).tolist(),
        "frame_id": _FRAME_ID,
    }


def _semantic_id(kind: str, *parts: str) -> str:
    return f"{kind}-{_digest(parts)[7:]}"


def _same_scene_layout(layout: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    """Compare placement identities with tolerance for quaternion roundoff."""
    if layout is None or any(
        layout[key] != expected[key] for key in ("scene_id", "layout_revision")
    ):
        return False
    if len(layout["instances"]) != len(expected["instances"]):
        return False
    for actual, reference in zip(
        layout["instances"], expected["instances"], strict=True
    ):
        actual_pose, reference_pose = actual["transform"], reference["transform"]
        rotation = np.asarray(reference_pose["rotation_xyzw"])
        # Both signs describe the same rotation. Translation and scale come
        # directly from the inventoried config and can be compared exactly.
        if not any(
            np.allclose(
                actual_pose["rotation_xyzw"], sign * rotation, rtol=0, atol=1e-12
            )
            for sign in (1, -1)
        ):
            return False
        if {
            **actual_pose,
            "rotation_xyzw": reference_pose["rotation_xyzw"],
        } != reference_pose:
            return False
        if {**actual, "transform": reference_pose} != reference:
            return False
    return True


def _scene_package_metadata(
    root: Path, asset: dict[str, Any], *, legacy_floor_semantics: bool = False
) -> dict[str, Any]:
    """Build the standard layout and semantics from the inventoried scene files."""
    export = root / "scene_export"
    importer = SceneExportImporter(output_root=root, scene_export_root=export)
    scene, graph = importer.read_scene_and_graph()
    config = json.loads(importer.scene_config_path.read_text(encoding="utf-8"))
    scene_id = config.get("scene_id")
    if (
        config.get("format") != "embodichain.scene-export/v1"
        or not isinstance(scene_id, str)
        or not scene_id.strip()
    ):
        raise ValueError("Scene package requires a v1 scene export with a scene_id")
    entries = _entries_by_uid(config)
    representations = {r["relative_path"]: r for r in asset["representations"]}
    frame = representations["asset.usda"]["convention"]
    if (
        frame["frame_id"] != _FRAME_ID
        or frame["up_axis"] != "Z"
        or frame["meters_per_unit"] != 1.0
    ):
        raise ValueError(
            "Scene package layout requires a metre, Z-up asset-stage frame"
        )

    used_files: set[str] = set()

    def representation(path: Path) -> dict[str, Any]:
        relative = path.relative_to(root).as_posix()
        if relative not in representations:
            raise ValueError(f"Scene package omits editable file {relative!r}")
        used_files.add(relative)
        return representations[relative]

    ids = {obj.id: _stable_id(obj.id, "instance") for obj in scene.objects}
    if len(set(ids.values())) != len(ids):
        raise ValueError("Scene object IDs collide in the packaged instance namespace")
    stage_index = UsdSceneIndex.load(root / "asset.usda", require_schema=True)
    if {entity.uid: entity.kind for entity in stage_index.entities} != {
        obj.id: "articulation" if obj.is_articulated else "rigid"
        for obj in scene.objects
    }:
        raise ValueError("Scene package USD entities disagree with the editable export")

    def subject(uid: str) -> dict[str, str]:
        return {"kind": "instance", "instance_id": ids[uid]}

    instances = []
    references = {}
    conventions = {}
    labels = []
    descriptions = []
    attributes = []
    relations = []
    origin = {"origin": "unknown", "evidence_refs": []}
    for obj in scene.objects:
        entry = entries[obj.id]
        runtime, proxy, dependencies = _asset_files(obj, export, entry)
        runtime_rep, proxy_rep = representation(runtime), representation(proxy)
        runtime_convention = (
            _GLTF_CONVENTION if runtime == proxy else _usd_convention(runtime)
        )
        conventions[runtime_rep["representation_id"]] = runtime_convention
        conventions[proxy_rep["representation_id"]] = _GLTF_CONVENTION
        dependency_reps = [representation(path) for path in sorted(dependencies)]
        artifact_digest = runtime_rep["digest"]
        revision = _digest(
            [
                artifact_digest,
                proxy_rep["digest"],
                sorted(r["digest"] for r in dependency_reps),
            ]
        )
        instances.append(
            {
                "instance_id": ids[obj.id],
                "asset_ref": {
                    "asset_id": str(
                        uuid5(NAMESPACE_URL, f"embodichain:{artifact_digest}")
                    ),
                    "revision_id": revision,
                    "artifact_digest": artifact_digest,
                },
                "transform": _transform(entry, runtime_convention["up_axis"]),
                "usage": "unspecified",
                "collision_binding_ids": [],
                "interaction_region_ids": [],
            }
        )
        references[ids[obj.id]] = {
            "uid": obj.id,
            "runtime_representation_id": runtime_rep["representation_id"],
            "edit_proxy_representation_id": proxy_rep["representation_id"],
            "dependency_representation_ids": [
                r["representation_id"] for r in dependency_reps
            ],
        }
        labels.append(
            {
                "label_id": _semantic_id("category", obj.id),
                "subject": subject(obj.id),
                "value": obj.category,
                **_TAXONOMY,
                **origin,
            }
        )
        descriptions.append(
            {
                "description_id": _semantic_id("description", obj.id),
                "subject": subject(obj.id),
                "language": "und",
                "text": obj.description,
                "conditions": [],
                **origin,
            }
        )
    edges = [(r.source_id, r.relation, r.target_id) for r in graph.relations]
    for node in graph.nodes:
        on_floor = (
            not legacy_floor_semantics
            and entries[node.object_id].get("placement_support") == "floor"
        )
        if (
            not on_floor
            and node.parent_id is not None
            and node.parent_relation is not None
        ):
            edges.append((node.object_id, node.parent_relation, node.parent_id))
        for name, value in (
            ("table_region", None if on_floor else node.table_region),
            ("pose_description", node.pose_description),
            ("placement_support", "floor" if on_floor else None),
        ):
            if value is not None:
                attributes.append(
                    {
                        "attribute_id": _semantic_id(name, node.object_id),
                        "subject": subject(node.object_id),
                        "name": name,
                        "value": value,
                        "unit": None,
                        "conditions": [],
                        **_TAXONOMY,
                        **origin,
                    }
                )
    for source, relation, target in edges:
        relations.append(
            {
                "relation_id": _semantic_id("relation", source, relation, target),
                "subject": subject(source),
                "object": subject(target),
                "relation": relation,
                "conditions": [],
                **_TAXONOMY,
                **origin,
            }
        )
    semantics = {
        "descriptions": descriptions,
        "labels": labels,
        "attributes": attributes,
        "affordances": [],
        "relations": relations,
    }
    config_rep = representation(importer.scene_config_path)
    graph_rep = representation(importer.scene_graph_path)
    extension = {
        "version": 1,
        "instances": references,
        "config_representation_id": config_rep["representation_id"],
        "graph_representation_id": graph_rep["representation_id"],
    }
    if {
        path for path in representations if path.startswith("scene_export/")
    } != used_files:
        raise ValueError("Scene package contains unreferenced editable files")
    return {
        "representations": [
            (
                {**rep, "convention": conventions[rep["representation_id"]]}
                if rep["representation_id"] in conventions
                else rep
            )
            for rep in asset["representations"]
        ],
        "scene_layout": {
            "scene_id": _stable_id(scene_id, "scene"),
            "layout_revision": _digest(
                [
                    config_rep["digest"],
                    graph_rep["digest"],
                    [instance["asset_ref"] for instance in instances],
                ]
            ),
            "instances": instances,
        },
        "semantics": semantics,
        _SCENE_EXTENSION: extension,
    }
