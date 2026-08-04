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

"""Read and normalize an exported Prompt2Scene gym scene.

The source scene remains the authority for object geometry and initial poses.
Generation only makes asset paths absolute, gives runtime objects stable UIDs,
applies one explicit world-frame rotation, and adds conservative physics values
needed by manipulation tasks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any
import warnings

from embodichain.gen_sim.action_engine.config import generation_defaults

from .models import PreparedScene

__all__ = [
    "is_prompt2scene_export",
    "prepare_scene",
    "resolve_gym_config_path",
]

_CONFIG_FILENAMES = ("gym_config_merged.json", "gym_config.json")
_SCENE_SECTIONS = ("background", "rigid_object", "articulation")
_UID_SUFFIX_RE = re.compile(r"_0$")
_UID_INVALID_RE = re.compile(r"[^0-9A-Za-z_.-]+")
_CONTAINER_HINTS = ("basket", "bin", "bowl", "box", "container", "drawer", "tray")

_GENERATION_DEFAULTS = generation_defaults()
_SCENE_DEFAULTS = _GENERATION_DEFAULTS["scene"]
_PHYSICS_DEFAULTS = _GENERATION_DEFAULTS["physics"]
_BACKGROUND_POLICY = _PHYSICS_DEFAULTS["background"]
_RIGID_POLICY = _PHYSICS_DEFAULTS["rigid_object"]
_BACKGROUND_ATTRS = {
    key: value
    for key, value in _BACKGROUND_POLICY.items()
    if key != "max_convex_hull_num"
}
_RIGID_ATTRS = {
    key: value
    for key, value in _RIGID_POLICY.items()
    if key not in {"max_convex_hull_num", "acd_method"}
}
_DEFAULT_BODY_SCALE = tuple(float(value) for value in _SCENE_DEFAULTS["body_scale"])


def resolve_gym_config_path(gym_project: str | Path) -> Path:
    """Resolve one exported gym config without depending on pipeline history."""
    input_path = Path(gym_project).expanduser().resolve()
    if input_path.is_file():
        if input_path.name not in _CONFIG_FILENAMES:
            expected = " or ".join(_CONFIG_FILENAMES)
            raise ValueError(f"Expected {expected}, got: {input_path}")
        return input_path
    if not input_path.is_dir():
        raise FileNotFoundError(f"Gym project does not exist: {input_path}")

    direct = _preferred_config(input_path)
    if direct is not None:
        return direct
    exported = _preferred_config(input_path / "gym_export")
    if exported is not None:
        return exported

    matches = sorted(
        {
            candidate.parent
            for filename in _CONFIG_FILENAMES
            for candidate in input_path.rglob(filename)
        }
    )
    preferred = [
        config
        for directory in matches
        if (config := _preferred_config(directory)) is not None
    ]
    if len(preferred) == 1:
        return preferred[0]
    if not preferred:
        raise FileNotFoundError(
            f"No {' or '.join(_CONFIG_FILENAMES)} found under: {input_path}"
        )
    paths = ", ".join(path.as_posix() for path in preferred)
    raise ValueError(f"Multiple exported gym configs found: {paths}")


def is_prompt2scene_export(gym_project: str | Path) -> bool:
    """Return whether the input has Prompt2Scene export provenance."""
    input_path = Path(gym_project).expanduser().resolve()
    directories = [input_path.parent] if input_path.is_file() else [input_path]
    directories.append(directories[0] / "gym_export")
    return any(
        (directory / "scene_state" / "result.json").is_file()
        for directory in directories
    )


def prepare_scene(
    gym_project: str | Path,
    *,
    z_rotation_degrees: float | None = None,
    body_scale_policy: str = str(_SCENE_DEFAULTS["body_scale_policy"]),
    body_scale: Sequence[float] = _DEFAULT_BODY_SCALE,
) -> PreparedScene:
    """Load a source gym config and return planner/runtime views of one scene."""
    scale_policy = str(body_scale_policy).strip().lower()
    if scale_policy not in {"preserve", "multiply", "absolute"}:
        raise ValueError("body_scale_policy must be preserve, multiply, or absolute.")
    requested_scale = _vector3(body_scale)
    if any(value <= 0.0 for value in requested_scale):
        raise ValueError("body_scale values must be positive.")
    source_path = resolve_gym_config_path(gym_project)
    source = _read_json_object(source_path)
    scene_dir = source_path.parent
    source_entries = _collect_source_entries(source)
    if not source_entries:
        raise ValueError("Gym config has no background, rigid_object, or articulation.")

    table_source_uid = _find_table_source_uid(source_entries)
    uid_map = _make_uid_map(source_entries, table_source_uid=table_source_uid)
    rotation = (
        float(_SCENE_DEFAULTS["prompt2scene_z_rotation_degrees"])
        if z_rotation_degrees is None and is_prompt2scene_export(source_path)
        else float(z_rotation_degrees or 0.0)
    )

    planner_objects: list[dict[str, Any]] = []
    runtime_sections: dict[str, list[dict[str, Any]]] = {
        section: [] for section in _SCENE_SECTIONS
    }
    asset_hashes: dict[str, str] = {}
    for role, source_config in source_entries:
        source_uid = _require_uid(source_config, role=role)
        normalized = deepcopy(source_config)
        normalized["uid"] = uid_map[source_uid]
        _make_asset_paths_absolute(normalized, scene_dir=scene_dir, role=role)
        _normalize_pose_fields(normalized)
        _apply_body_scale_policy(
            normalized,
            policy=scale_policy,
            requested=requested_scale,
        )
        _apply_world_z_rotation(normalized, rotation)
        shape = normalized.get("shape")
        if isinstance(shape, Mapping) and shape.get("fpath"):
            asset_hashes[normalized["uid"]] = _file_hash(Path(str(shape["fpath"])))

        planner_objects.append(
            _planner_object(
                normalized,
                source_uid=source_uid,
                role=role,
            )
        )
        runtime_sections[role].append(_runtime_object(normalized, role=role))

    table = next(
        (obj for obj in runtime_sections["background"] if obj.get("uid") == "table"),
        None,
    )
    table_top_z = _estimate_mesh_top_z(table) if table is not None else None
    return PreparedScene(
        source_config_path=source_path,
        scene_dir=scene_dir,
        planner_objects=tuple(planner_objects),
        background=tuple(runtime_sections["background"]),
        rigid_objects=tuple(runtime_sections["rigid_object"]),
        articulations=tuple(runtime_sections["articulation"]),
        uid_map=uid_map,
        table_top_z=table_top_z,
        z_rotation_degrees=rotation,
        body_scale_policy=scale_policy,
        body_scale=tuple(requested_scale),
        asset_hashes=asset_hashes,
    )


def _preferred_config(directory: Path) -> Path | None:
    for filename in _CONFIG_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in gym config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Gym config must contain a JSON object: {path}")
    return value


def _collect_source_entries(
    source: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for section in _SCENE_SECTIONS:
        value = source.get(section, [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            raise ValueError(f"Gym config section {section!r} must be a list.")
        for config in value:
            if not isinstance(config, Mapping):
                raise ValueError(f"Entries in {section!r} must be JSON objects.")
            entries.append((section, dict(config)))
    return entries


def _find_table_source_uid(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    backgrounds = [(role, config) for role, config in entries if role == "background"]
    if not backgrounds:
        raise ValueError("A tabletop action scene requires a background object.")
    for _, config in backgrounds:
        text = " ".join(
            (
                str(config.get("uid", "")),
                str(config.get("description", "")),
            )
        ).lower()
        if "table" in text:
            return _require_uid(config, role="background")
    return _require_uid(backgrounds[0][1], role="background")


def _make_uid_map(
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    table_source_uid: str,
) -> dict[str, str]:
    uid_map: dict[str, str] = {}
    used: set[str] = set()
    for role, config in entries:
        source_uid = _require_uid(config, role=role)
        if source_uid in uid_map:
            raise ValueError(f"Duplicate scene object UID: {source_uid!r}")
        candidate = (
            "table" if source_uid == table_source_uid else _normalize_uid(source_uid)
        )
        runtime_uid = candidate
        suffix = 2
        while runtime_uid in used:
            runtime_uid = f"{candidate}_{suffix}"
            suffix += 1
        uid_map[source_uid] = runtime_uid
        used.add(runtime_uid)
    return uid_map


def _normalize_uid(source_uid: str) -> str:
    candidate = _UID_SUFFIX_RE.sub("", source_uid.strip())
    candidate = _UID_INVALID_RE.sub("_", candidate).strip("._-")
    if not candidate:
        raise ValueError(f"Cannot derive a runtime UID from {source_uid!r}.")
    if candidate[0].isdigit():
        candidate = f"object_{candidate}"
    return candidate


def _require_uid(config: Mapping[str, Any], *, role: str) -> str:
    uid = str(config.get("uid", "")).strip()
    if not uid:
        raise ValueError(f"Scene object in {role!r} has no UID.")
    return uid


def _make_asset_paths_absolute(
    config: dict[str, Any],
    *,
    scene_dir: Path,
    role: str,
) -> None:
    shape = config.get("shape")
    if isinstance(shape, Mapping):
        normalized_shape = deepcopy(dict(shape))
        fpath = normalized_shape.get("fpath")
        if fpath:
            normalized_shape["fpath"] = _resolve_asset_path(
                scene_dir, str(fpath)
            ).as_posix()
        config["shape"] = normalized_shape
    if role == "articulation" and config.get("fpath"):
        config["fpath"] = _resolve_asset_path(
            scene_dir, str(config["fpath"])
        ).as_posix()


def _resolve_asset_path(scene_dir: Path, fpath: str) -> Path:
    raw = Path(fpath).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (scene_dir / raw).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Scene asset does not exist: {resolved}")
    return resolved


def _normalize_pose_fields(config: dict[str, Any]) -> None:
    config["init_pos"] = _vector3(config.get("init_pos", [0.0, 0.0, 0.0]))
    config["init_rot"] = _vector3(config.get("init_rot", [0.0, 0.0, 0.0]))
    if "body_scale" in config:
        scale = _vector3(config["body_scale"])
        if any(value <= 0.0 for value in scale):
            raise ValueError(
                f"Object {config.get('uid')!r} has non-positive body_scale."
            )
        config["body_scale"] = scale


def _apply_body_scale_policy(
    config: dict[str, Any],
    *,
    policy: str,
    requested: Sequence[float],
) -> None:
    source = _vector3(config.get("body_scale", [1.0, 1.0, 1.0]))
    if policy == "preserve":
        result = source
    elif policy == "multiply":
        result = [left * right for left, right in zip(source, requested)]
    else:
        result = list(requested)
    config["body_scale"] = [_clean_float(value) for value in result]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_world_z_rotation(config: dict[str, Any], degrees: float) -> None:
    if math.isclose(degrees, 0.0, abs_tol=1e-12):
        return
    theta = math.radians(degrees)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    x, y, z = _vector3(config["init_pos"])
    config["init_pos"] = [
        _clean_float(x * cos_theta - y * sin_theta),
        _clean_float(x * sin_theta + y * cos_theta),
        _clean_float(z),
    ]

    # EmbodiChain and Prompt2Scene both interpret these values as intrinsic XYZ.
    from scipy.spatial.transform import Rotation

    original = Rotation.from_euler("XYZ", config["init_rot"], degrees=True)
    world_z = Rotation.from_rotvec([0.0, 0.0, theta])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        rotated = (world_z * original).as_euler("XYZ", degrees=True)
    config["init_rot"] = [_clean_float(value) for value in rotated]
    if "init_local_pose" in config:
        # Keeping two pose representations risks the stale local matrix
        # overriding the rotated Euler pose in ObjectBaseCfg.from_dict.
        del config["init_local_pose"]


def _planner_object(
    config: Mapping[str, Any],
    *,
    source_uid: str,
    role: str,
) -> dict[str, Any]:
    description = str(config.get("description", "")).strip()
    text = f"{config['uid']} {description}".lower()
    shape = deepcopy(dict(config.get("shape", {})))
    return {
        "uid": str(config["uid"]),
        "runtime_uid": str(config["uid"]),
        "source_uid": source_uid,
        "role": role,
        "description": description,
        "shape": shape,
        "init_pos": list(config["init_pos"]),
        "init_rot": list(config["init_rot"]),
        "body_scale": list(config.get("body_scale", [1.0, 1.0, 1.0])),
        "is_container_like": any(hint in text for hint in _CONTAINER_HINTS),
    }


def _runtime_object(config: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if role == "articulation":
        # Articulation schemas vary by asset; preserve their source fields after
        # path and pose normalization instead of guessing a reduced schema.
        result = deepcopy(dict(config))
        result.pop("description", None)
        return result

    result = {
        key: deepcopy(config[key])
        for key in (
            "uid",
            "shape",
            "init_pos",
            "init_rot",
            "body_scale",
        )
        if key in config
    }
    result.setdefault("body_scale", [1.0, 1.0, 1.0])
    source_attrs = dict(config.get("attrs", {}))
    if role == "background":
        result["attrs"] = {**source_attrs, **_BACKGROUND_ATTRS}
        result["body_type"] = "kinematic"
        result["max_convex_hull_num"] = int(_BACKGROUND_POLICY["max_convex_hull_num"])
    else:
        result["attrs"] = {**source_attrs, **_RIGID_ATTRS}
        result["body_type"] = "dynamic"
        hull_limit = int(_RIGID_POLICY["max_convex_hull_num"])
        max_hulls = max(
            1,
            min(int(config.get("max_convex_hull_num", hull_limit)), hull_limit),
        )
        result["max_convex_hull_num"] = max_hulls
        result["acd_method"] = str(_RIGID_POLICY["acd_method"])
        shape = result.get("shape")
        if isinstance(shape, dict):
            shape.setdefault("acd_method", str(_RIGID_POLICY["acd_method"]))
            shape.setdefault("max_convex_hull_num", max_hulls)
    return result


def _estimate_mesh_top_z(config: Mapping[str, Any]) -> float | None:
    shape = config.get("shape", {})
    if not isinstance(shape, Mapping) or not shape.get("fpath"):
        return None
    try:
        import numpy as np
        import trimesh
        from scipy.spatial.transform import Rotation

        loaded = trimesh.load(str(shape["fpath"]), force="scene")
        geometry = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        if vertices.size == 0:
            return None
        # DexSim converts glTF Y-up vertices to its Z-up basis at load time.
        sim_vertices = np.column_stack(
            (vertices[:, 0], -vertices[:, 2], vertices[:, 1])
        )
        sim_vertices *= np.asarray(
            config.get("body_scale", [1.0, 1.0, 1.0]), dtype=np.float64
        )
        rotated = Rotation.from_euler(
            "XYZ", config.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).apply(sim_vertices)
        rotated += np.asarray(config.get("init_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        return float(rotated[:, 2].max())
    except Exception:
        # Mesh bounds improve robot placement but are not needed to preserve the
        # exported scene. The robot builder has a conservative tabletop fallback.
        return None


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Expected a finite xyz vector, got: {value!r}")
    values = [float(item) for item in value]
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"Expected a finite xyz vector, got: {value!r}")
    return values


def _clean_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 1e-12 else rounded
