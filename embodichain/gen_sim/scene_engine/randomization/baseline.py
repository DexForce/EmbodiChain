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

"""Persist read-only, source-identified geometry and support-group baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraph
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)

from .cfg import SceneRandomizationCfg, _UniqueKeyLoader
from ._geometry import SceneGeometry, measure_scene_geometry

__all__ = ["PreparedSceneBaseline", "prepare_scene_baseline"]

_EXTRACTION_VERSION = "rigid_support_groups/v1"
_FRAME = "local_arena_z_up"


@dataclass(frozen=True)
class PreparedSceneBaseline:
    """Result of preparing a persisted scene randomization baseline.

    Attributes:
        path: Absolute path to the baseline YAML file.
        cache_hit: Whether an existing baseline was validated and reused.
        data: Validated baseline payload, including its source identity.
    """

    path: Path
    cache_hit: bool
    data: dict[str, Any]


def prepare_scene_baseline(
    cfg: SceneRandomizationCfg, *, rebuild: bool = False
) -> PreparedSceneBaseline:
    """Validate a portable scene and persist its reusable grouped baseline.

    The source configuration, graph and final GLBs are never written. Cached
    geometry is reused only when its source digest, extraction version,
    structure and payload checksum agree. Digest checks read asset bytes but
    do not parse meshes. No model service, simulator or USD package is needed.

    Args:
        cfg: Resolved paths from the mandatory input configuration.
        rebuild: Recompute geometry even when a valid baseline already exists.

    Returns:
        The owned baseline data and its destination.

    Raises:
        OSError: Source files cannot be read or the baseline cannot be written.
        ValueError: Configuration, scene geometry, support hierarchy or output
            destination does not satisfy the supported baseline contract.
    """
    cfg.validate()
    if cfg.schema_version != 1 or type(cfg.schema_version) is not int:
        raise ValueError("Randomization configuration schema_version must be 1.")
    if type(rebuild) is not bool:
        raise ValueError("rebuild must be a boolean.")
    export_root = Path(cfg.scene_path).expanduser().resolve()
    if not (export_root / "scene_config.json").is_file():
        export_root = export_root / "scene_export"
    config_bytes, config = _read_json(export_root / "scene_config.json")
    graph_bytes, _ = _read_json(export_root / "scene_graph.json")
    entries = _rigid_entries(config)
    _, graph = SceneExportImporter(
        output_root=export_root.parent, scene_export_root=export_root
    ).read_scene_and_graph()
    groups = _support_groups(graph)
    files = [export_root / entry["shape"]["fpath"] for entry in entries]
    path = Path(os.path.abspath(Path(cfg.base_data_path).expanduser()))
    _check_destination(path, export_root, files)
    digest = _source_digest(export_root, config_bytes, graph_bytes, files)
    source = {
        "scene_id": config["scene_id"],
        "export_dir": Path(os.path.relpath(export_root, path.parent)).as_posix(),
        "content_sha256": digest,
    }
    if not rebuild and path.is_file():
        cached = _cached_base(path, source, entries, groups)
        if cached is not None:
            return PreparedSceneBaseline(path, True, cached)

    geometry = measure_scene_geometry(entries, export_root)
    data: dict[str, Any] = {
        "schema_version": 1,
        "coordinate_frame": _FRAME,
        "extraction_version": _EXTRACTION_VERSION,
        "source_scene": source,
        "table": _plain_data(geometry.table),
        "groups": _group_data(groups, geometry),
    }
    data["data_sha256"] = _data_digest(data)
    current_config = (export_root / "scene_config.json").read_bytes()
    current_graph = (export_root / "scene_graph.json").read_bytes()
    if _source_digest(export_root, current_config, current_graph, files) != digest:
        raise ValueError(
            "Source scene changed during extraction; run preparation again."
        )
    _write_atomic(path, data)
    return PreparedSceneBaseline(path, False, data)


def _read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path.name}.")
            result[key] = value
        return result

    contents = path.read_bytes()
    try:
        value = json.loads(contents, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid scene JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Scene JSON must contain an object: {path}.")
    return contents, value


def _rigid_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("format") != "embodichain.scene-export/v1":
        raise ValueError("Input must use the embodichain.scene-export/v1 format.")
    if set(config) - {
        "format",
        "scene_id",
        "background",
        "rigid_object",
        "articulation",
    }:
        raise ValueError("Scene export contains unsupported sections.")
    if not isinstance(config.get("scene_id"), str) or not config["scene_id"].strip():
        raise ValueError("Scene export must contain a non-empty scene_id.")
    background = config.get("background")
    if (
        not isinstance(background, list)
        or len(background) != 1
        or not isinstance(background[0], dict)
        or background[0].get("uid") != "table"
    ):
        raise ValueError("Baseline preparation requires exactly one background table.")
    if not isinstance(config.get("articulation", []), list) or config.get(
        "articulation", []
    ):
        raise ValueError("Baseline preparation currently supports rigid objects only.")
    entries: list[dict[str, Any]] = []
    for key in ("background", "rigid_object"):
        values = config.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, dict) for v in values):
            raise ValueError(f"Scene export {key} must be a list of objects.")
        entries.extend(values)
    ids = [entry.get("uid") for entry in entries]
    if any(not isinstance(uid, str) or not uid.strip() for uid in ids):
        raise ValueError("Every scene entity must have a non-empty uid.")
    if len(ids) != len(set(ids)):
        raise ValueError("Scene entity UIDs must be unique.")
    for entry in entries:
        if entry.get("is_articulated") is not False:
            raise ValueError(
                "Baseline preparation requires non-articulated rigid entities."
            )
        shape = entry.get("shape")
        if not isinstance(shape, dict) or shape.get("shape_type") != "Mesh":
            raise ValueError("Baseline preparation requires final Mesh assets.")
        for key, default in (
            ("init_pos", [0.0, 0.0, 0.0]),
            ("init_rot", [0.0, 0.0, 0.0]),
            ("body_scale", [1.0, 1.0, 1.0]),
        ):
            values = _vector(entry.get(key, default), 3)
            if key == "body_scale" and np.any(values <= 0.0):
                raise ValueError("Scene body_scale must be positive.")
    return entries


def _support_groups(graph: SceneGraph) -> dict[str, list[tuple[str, str]]]:
    nodes = graph.node_by_id()
    groups: dict[str, list[tuple[str, str]]] = {}
    for uid in sorted(set(nodes) - {"table"}):
        node = nodes[uid]
        if node.parent_relation != "on":
            raise ValueError("Every non-table entity must have an explicit on parent.")
        root = uid
        visited: set[str] = set()
        while nodes[root].parent_id != "table":
            if root in visited or nodes[root].parent_id not in nodes:
                raise ValueError("Scene support hierarchy must reach the table.")
            visited.add(root)
            root = nodes[root].parent_id  # type: ignore[assignment]
        members = groups.setdefault(root, [])
        if uid != root:
            assert node.parent_id is not None
            members.append((uid, node.parent_id))
    return {root: sorted(groups[root]) for root in sorted(groups)}


def _group_data(
    groups: dict[str, list[tuple[str, str]]], geometry: SceneGeometry
) -> list[dict[str, Any]]:
    result = []
    for root, members in groups.items():
        root_pose = geometry.poses_by_uid[root]
        inverse_root = np.linalg.inv(root_pose)
        bounds = np.stack(
            [geometry.aabbs_by_uid[uid] for uid, _ in [(root, "table"), *members]]
        )
        minimum, maximum = bounds[:, 0, :2].min(axis=0), bounds[:, 1, :2].max(axis=0)
        if np.any(maximum <= minimum):
            raise ValueError(f"Group {root!r} must have a positive XY extent.")
        result.append(
            {
                "root_id": root,
                "root_pose": _pose_data(root_pose),
                "members": [
                    {
                        "entity_id": uid,
                        "parent_id": parent,
                        "pose_in_root": _pose_data(
                            inverse_root @ geometry.poses_by_uid[uid]
                        ),
                    }
                    for uid, parent in members
                ],
                "aabb_2d": {
                    "center_xy": _plain_data((minimum + maximum) / 2.0),
                    "size_xy": _plain_data(maximum - minimum),
                },
            }
        )
    return result


def _pose_data(pose: np.ndarray) -> dict[str, list[float]]:
    quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return {
        "position": _plain_data(pose[:3, 3]),
        "quaternion_xyzw": _plain_data(quaternion),
    }


def _plain_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (float, np.floating)):
        result = round(float(value), 12)
        return 0.0 if result == 0.0 else result
    return value


def _source_digest(
    export_root: Path, config_bytes: bytes, graph_bytes: bytes, files: list[Path]
) -> str:
    digest = hashlib.sha256()
    contents = {
        "scene_config.json": config_bytes,
        "scene_graph.json": graph_bytes,
    }
    for name in sorted(contents):
        payload = contents[name]
        digest.update(name.encode() + b"\0" + len(payload).to_bytes(8, "big") + payload)
    for path in sorted(
        files, key=lambda path: path.relative_to(export_root).as_posix()
    ):
        name = path.relative_to(export_root).as_posix()
        digest.update(name.encode() + b"\0" + path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _data_digest(data: dict[str, Any]) -> str:
    payload = {key: value for key, value in data.items() if key != "data_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _cached_base(
    path: Path,
    source: dict[str, str],
    entries: list[dict[str, Any]],
    groups: dict[str, list[tuple[str, str]]],
) -> dict[str, Any] | None:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "coordinate_frame",
            "extraction_version",
            "source_scene",
            "table",
            "groups",
            "data_sha256",
        }:
            return None
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["coordinate_frame"] != _FRAME
            or data["extraction_version"] != _EXTRACTION_VERSION
            or data["source_scene"] != source
            or data["data_sha256"] != _data_digest(data)
        ):
            return None
        table = data["table"]
        if not _keys(table, {"entity_id", "center_xy", "size_xy", "surface_z"}):
            return None
        if table["entity_id"] != "table" or np.any(_vector(table["size_xy"], 2) <= 0):
            return None
        _vector(table["center_xy"], 2)
        _vector([table["surface_z"]], 1)
        poses = {entry["uid"]: _entry_pose(entry) for entry in entries}
        if not isinstance(data["groups"], list) or len(data["groups"]) != len(groups):
            return None
        for group, (root, expected_members) in zip(data["groups"], groups.items()):
            if not _keys(group, {"root_id", "root_pose", "members", "aabb_2d"}):
                return None
            root_pose = _read_pose(group["root_pose"])
            if group["root_id"] != root or not np.allclose(
                root_pose, poses[root], atol=1e-9, rtol=0
            ):
                return None
            aabb = group["aabb_2d"]
            if not _keys(aabb, {"center_xy", "size_xy"}):
                return None
            _vector(aabb["center_xy"], 2)
            if np.any(_vector(aabb["size_xy"], 2) <= 0):
                return None
            if not isinstance(group["members"], list) or len(group["members"]) != len(
                expected_members
            ):
                return None
            for member, (uid, parent) in zip(group["members"], expected_members):
                if not _keys(member, {"entity_id", "parent_id", "pose_in_root"}):
                    return None
                if member["entity_id"] != uid or member["parent_id"] != parent:
                    return None
                if not np.allclose(
                    root_pose @ _read_pose(member["pose_in_root"]),
                    poses[uid],
                    atol=1e-9,
                    rtol=0,
                ):
                    return None
        return data
    except (OSError, ValueError, TypeError, KeyError, OverflowError, yaml.YAMLError):
        return None


def _entry_pose(entry: dict[str, Any]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler(
        "XYZ", entry.get("init_rot", [0, 0, 0]), degrees=True
    ).as_matrix()
    pose[:3, 3] = entry.get("init_pos", [0, 0, 0])
    return pose


def _read_pose(value: Any) -> np.ndarray:
    if not _keys(value, {"position", "quaternion_xyzw"}):
        raise ValueError("Invalid cached pose fields.")
    quaternion = _vector(value["quaternion_xyzw"], 4)
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-9, rtol=0):
        raise ValueError("Cached quaternions must be normalized xyzw values.")
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    pose[:3, 3] = _vector(value["position"], 3)
    return pose


def _keys(value: Any, expected: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == expected


def _vector(value: Any, size: int) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(type(item) not in (int, float) for item in value)
    ):
        raise ValueError(f"Expected {size} finite numeric values.")
    vector = np.asarray(value, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError(f"Expected {size} finite numeric values.")
    return vector


def _check_destination(path: Path, export_root: Path, files: list[Path]) -> None:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Baseline output must be a YAML file.")
    if path.is_symlink():
        raise ValueError("Baseline output must not be a symlink.")
    if path.exists() and not path.is_file():
        raise ValueError("Baseline output must be a file path.")
    resolved = path.resolve()
    if resolved in {p.resolve() for p in files}:
        raise ValueError("Baseline output must not overwrite source assets.")
    for directory in ("mesh_assets", "articulated_assets"):
        if (export_root / directory).resolve() in resolved.parents:
            raise ValueError(
                "Baseline output must stay outside source asset directories."
            )


class _BaseDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
        return super().increase_indent(flow, False)


def _represent_list(
    dumper: yaml.SafeDumper, values: list[Any]
) -> yaml.nodes.SequenceNode:
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        values,
        flow_style=bool(values)
        and all(type(value) in (int, float) for value in values),
    )


_BaseDumper.add_representer(list, _represent_list)


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yaml.dump(
                data, stream, Dumper=_BaseDumper, sort_keys=False, allow_unicode=True
            )
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise ValueError("Baseline output must not be a symlink.")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
