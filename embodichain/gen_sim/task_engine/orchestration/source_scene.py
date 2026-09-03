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

"""Read and normalize an exported Prompt2Scene source scene.

The source scene remains the authority for object geometry and initial poses.
Generation only makes asset paths absolute, gives runtime objects stable UIDs,
applies one explicit world-frame rotation, and adds conservative physics values
needed by manipulation tasks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any
import warnings

from embodichain.data import get_data_path

__all__ = [
    "PreparedScene",
    "ResolvedSceneSource",
    "is_prompt2scene_export",
    "prepare_scene",
    "resolve_gym_config_path",
    "resolve_source_scene",
]

_LEGACY_CONFIG_FILENAMES = ("gym_config_merged.json", "gym_config.json")
_SCENE_CONFIG_FILENAME = "scene_config.json"
_CONFIG_FILENAMES = (*_LEGACY_CONFIG_FILENAMES, _SCENE_CONFIG_FILENAME)
_EXPORT_DIRECTORY_NAMES = ("gym_export", "scene_export")
_LEGACY_GYM_FORMAT = "legacy_gym_config"
_SCENE_EXPORT_FORMAT = "embodichain.scene-export/v1"
_SCENE_SECTIONS = ("background", "rigid_object", "articulation")
_UID_SUFFIX_RE = re.compile(r"_0$")
_UID_INVALID_RE = re.compile(r"[^0-9A-Za-z_.-]+")

_SCENE_DEFAULTS = {
    "prompt2scene_z_rotation_degrees": -90.0,
    "body_scale_policy": "preserve",
    "body_scale": (1.0, 1.0, 1.0),
}
_BACKGROUND_POLICY = {
    "mass": 10.0,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "restitution": 0.01,
    "max_convex_hull_num": 1,
}
_RIGID_POLICY = {
    "mass": 0.1,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "linear_damping": 0.9,
    "angular_damping": 0.9,
    "contact_offset": 0.003,
    "rest_offset": 0.001,
    "restitution": 0.05,
    "max_depenetration_velocity": 0.8,
    "max_linear_velocity": 5.0,
    "max_angular_velocity": 5.0,
    "min_position_iters": 32,
    "min_velocity_iters": 8,
    "max_convex_hull_num": 16,
    "acd_method": "vhacd",
}
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


@dataclass(frozen=True)
class PreparedScene:
    """A source scene normalized for planning and simulator loading.

    Attributes:
        source_config_path: Absolute source configuration path.
        scene_dir: Directory against which source assets were resolved.
        planner_objects: Semantic object view used during task planning.
        background: Normalized simulator background configurations.
        rigid_objects: Normalized simulator rigid-object configurations.
        articulations: Normalized simulator articulation configurations.
        uid_map: Mapping from source identities to canonical runtime identities.
        table_top_z: Estimated tabletop height when it can be derived.
        z_rotation_degrees: World-frame rotation applied to source poses.
        body_scale_policy: Applied source-scale policy.
        body_scale: Requested scale vector.
        asset_hashes: Runtime-identity to source-asset digest mapping.
        source_scene_xy_translation: World translation applied before rotation.
        asset_provenance: Optional normalized-asset audit entries.
    """

    source_config_path: Path
    scene_dir: Path
    planner_objects: tuple[dict[str, Any], ...]
    background: tuple[dict[str, Any], ...]
    rigid_objects: tuple[dict[str, Any], ...]
    articulations: tuple[dict[str, Any], ...]
    uid_map: dict[str, str]
    table_top_z: float | None
    z_rotation_degrees: float
    body_scale_policy: str
    body_scale: tuple[float, float, float]
    asset_hashes: dict[str, str]
    source_scene_xy_translation: tuple[float, float] = (0.0, 0.0)
    asset_provenance: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ResolvedSceneSource:
    """One validated source-scene config selected from an export layout.

    Attributes:
        path: Absolute path to the selected source configuration.
        source_format: Stable identifier for the detected source schema.
        is_prompt2scene: Whether Prompt2Scene world alignment should be applied.
    """

    path: Path
    source_format: str
    is_prompt2scene: bool


def resolve_source_scene(gym_project: str | Path) -> ResolvedSceneSource:
    """Resolve and classify one supported source-scene configuration.

    Args:
        gym_project: Task root, export directory, or explicit configuration path.

    Returns:
        The selected path together with its source format and provenance.

    Raises:
        FileNotFoundError: If no supported source configuration exists.
        ValueError: If a config is unsupported or recursive discovery is ambiguous.
    """
    input_path = Path(gym_project).expanduser().resolve()
    if input_path.is_file():
        return _classify_source_config(input_path)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Scene project does not exist: {input_path}")

    for directory in (
        input_path,
        *(input_path / name for name in _EXPORT_DIRECTORY_NAMES),
    ):
        preferred = _preferred_config(directory)
        if preferred is not None:
            return _classify_source_config(preferred)

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
        return _classify_source_config(preferred[0])
    if not preferred:
        expected = ", ".join(_CONFIG_FILENAMES)
        raise FileNotFoundError(
            f"No supported scene config ({expected}) found under: {input_path}"
        )
    paths = ", ".join(path.as_posix() for path in preferred)
    raise ValueError(f"Multiple exported scene configs found: {paths}")


def resolve_gym_config_path(gym_project: str | Path) -> Path:
    """Return the selected source config through the compatibility API name.

    Args:
        gym_project: Task root, export directory, or explicit configuration path.

    Returns:
        Absolute path selected by :func:`resolve_source_scene`.

    Raises:
        FileNotFoundError: If no supported source configuration exists.
        ValueError: If source discovery is ambiguous or unsupported.
    """
    return resolve_source_scene(gym_project).path


def is_prompt2scene_export(gym_project: str | Path) -> bool:
    """Return whether the input has Prompt2Scene export provenance.

    Args:
        gym_project: Task root, export directory, or explicit configuration path.

    Returns:
        ``True`` when a supported source carries Prompt2Scene provenance;
        otherwise ``False``, including invalid or missing paths.
    """
    try:
        return resolve_source_scene(gym_project).is_prompt2scene
    except (FileNotFoundError, ValueError):
        return False


def prepare_scene(
    gym_project: str | Path,
    *,
    z_rotation_degrees: float | None = None,
    source_scene_xy_translation: Sequence[float] | None = None,
    body_scale_policy: str = str(_SCENE_DEFAULTS["body_scale_policy"]),
    body_scale: Sequence[float] = _DEFAULT_BODY_SCALE,
) -> PreparedScene:
    """Load a source config and return planner/runtime views of one scene.

    Args:
        gym_project: Task root, export directory, or explicit configuration path.
        z_rotation_degrees: Optional world-frame rotation override. Prompt2Scene
            inputs use the canonical rotation when this value is omitted.
        source_scene_xy_translation: Optional two-value world translation. An
            explicit robot scene otherwise centers itself on its table anchor.
        body_scale_policy: One of ``preserve``, ``multiply``, or ``absolute``.
        body_scale: Positive three-value scale consumed by the selected policy.

    Returns:
        Canonically identified planner and simulator views of the source scene.

    Raises:
        FileNotFoundError: If source discovery or a referenced asset fails.
        ValueError: If the source, transform, scale, identities, or scene
            structure is invalid.
    """
    scale_policy = str(body_scale_policy).strip().lower()
    if scale_policy not in {"preserve", "multiply", "absolute"}:
        raise ValueError("body_scale_policy must be preserve, multiply, or absolute.")
    requested_scale = _vector3(body_scale)
    if any(value <= 0.0 for value in requested_scale):
        raise ValueError("body_scale values must be positive.")
    resolved_source = resolve_source_scene(gym_project)
    source_path = resolved_source.path
    source = _read_json_object(source_path)
    scene_dir = source_path.parent
    source_entries = _collect_source_entries(source)
    if not source_entries:
        raise ValueError(
            "Source scene config has no background, rigid_object, or articulation."
        )

    table_source_uid = _find_table_source_uid(source_entries)
    uid_map = _make_uid_map(source_entries, table_source_uid=table_source_uid)
    source_robot = source.get("robot")
    source_has_robot = isinstance(source_robot, Mapping) and bool(source_robot)
    source_table = next(
        (
            item
            for role, item in source_entries
            if role == "background" and str(item.get("uid", "")) == table_source_uid
        ),
        None,
    )
    if source_scene_xy_translation is not None:
        if len(source_scene_xy_translation) != 2 or any(
            not math.isfinite(float(value)) for value in source_scene_xy_translation
        ):
            raise ValueError(
                "source_scene_xy_translation must contain two finite values."
            )
        resolved_xy_translation = tuple(
            float(value) for value in source_scene_xy_translation
        )
    elif source_has_robot and source_table is not None:
        table_anchor = _vector3(source_table.get("init_pos", (0.0, 0.0, 0.0)))
        resolved_xy_translation = (-table_anchor[0], -table_anchor[1])
    else:
        resolved_xy_translation = (0.0, 0.0)
    rotation = (
        float(_SCENE_DEFAULTS["prompt2scene_z_rotation_degrees"])
        if z_rotation_degrees is None and resolved_source.is_prompt2scene
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
        normalized["init_pos"][0] += resolved_xy_translation[0]
        normalized["init_pos"][1] += resolved_xy_translation[1]
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
        source_scene_xy_translation=resolved_xy_translation,
    )


def _preferred_config(directory: Path) -> Path | None:
    for filename in _CONFIG_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _classify_source_config(path: Path) -> ResolvedSceneSource:
    if path.name not in _CONFIG_FILENAMES:
        source = _read_json_object(path)
        if not any(
            isinstance(source.get(section), Sequence) for section in _SCENE_SECTIONS
        ):
            expected = ", ".join(_CONFIG_FILENAMES)
            raise ValueError(
                f"Expected one of {expected} or an explicit legacy scene JSON, "
                f"got: {path}"
            )
        return ResolvedSceneSource(
            path=path,
            source_format=_LEGACY_GYM_FORMAT,
            is_prompt2scene=False,
        )
    if path.name == _SCENE_CONFIG_FILENAME:
        source = _read_json_object(path)
        source_format = source.get("format")
        if source_format != _SCENE_EXPORT_FORMAT:
            raise ValueError(
                f"Scene config {path} has unsupported format {source_format!r}; "
                f"expected {_SCENE_EXPORT_FORMAT!r}."
            )
        return ResolvedSceneSource(
            path=path,
            source_format=_SCENE_EXPORT_FORMAT,
            is_prompt2scene=True,
        )
    return ResolvedSceneSource(
        path=path,
        source_format=_LEGACY_GYM_FORMAT,
        is_prompt2scene=(
            _has_legacy_prompt2scene_marker(path) or _has_scene_export_companion(path)
        ),
    )


def _has_legacy_prompt2scene_marker(config_path: Path) -> bool:
    config_dir = config_path.parent
    directories = [config_dir, config_dir / "gym_export"]
    return any(
        (directory / "scene_state" / "result.json").is_file()
        for directory in directories
    )


def _has_scene_export_companion(config_path: Path) -> bool:
    config_dir = config_path.parent
    candidates = [config_dir / _SCENE_CONFIG_FILENAME]
    if config_dir.name == "gym_export":
        candidates.append(config_dir.parent / "scene_export" / _SCENE_CONFIG_FILENAME)
    else:
        candidates.append(config_dir / "scene_export" / _SCENE_CONFIG_FILENAME)
    return any(_is_scene_export_v1(candidate) for candidate in candidates)


def _is_scene_export_v1(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return _read_json_object(path).get("format") == _SCENE_EXPORT_FORMAT
    except ValueError:
        return False


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in source scene config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Source scene config must contain a JSON object: {path}")
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
            raise ValueError(f"Source scene section {section!r} must be a list.")
        for config in value:
            if not isinstance(config, Mapping):
                raise ValueError(f"Entries in {section!r} must be JSON objects.")
            entries.append((section, dict(config)))
    return entries


def _find_table_source_uid(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    backgrounds = [config for role, config in entries if role == "background"]
    if len(backgrounds) != 1:
        raise ValueError(
            "A tabletop action scene requires exactly one background object; "
            f"found {len(backgrounds)}."
        )
    return _require_uid(backgrounds[0], role="background")


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
    if not resolved.is_file() and not raw.is_absolute():
        resolved = Path(get_data_path(fpath)).expanduser().resolve()
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
    shape = deepcopy(dict(config.get("shape", {})))
    raw_attributes = config.get("attributes", {})
    if not isinstance(raw_attributes, Mapping):
        raw_attributes = {}
    raw_initial_state = config.get("initial_state", config.get("state", {}))
    if not isinstance(raw_initial_state, Mapping):
        raw_initial_state = {}
    raw_affordances = config.get("affordances", config.get("capabilities", []))
    affordances = (
        [str(value) for value in raw_affordances]
        if isinstance(raw_affordances, Sequence)
        and not isinstance(raw_affordances, (str, bytes))
        else []
    )
    return {
        "uid": str(config["uid"]),
        "runtime_uid": str(config["uid"]),
        "source_uid": source_uid,
        "role": role,
        "name": str(config.get("name", "")).strip(),
        "description": description,
        "shape": shape,
        "init_pos": list(config["init_pos"]),
        "init_rot": list(config["init_rot"]),
        "body_scale": list(config.get("body_scale", [1.0, 1.0, 1.0])),
        "category": config.get("category", config.get("object_category", "")),
        "color": config.get("color", raw_attributes.get("color")),
        "attributes": deepcopy(dict(raw_attributes)),
        "initial_state": deepcopy(dict(raw_initial_state)),
        "affordances": affordances,
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
            shape["acd_method"] = str(_RIGID_POLICY["acd_method"])
            shape["max_convex_hull_num"] = max_hulls
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
