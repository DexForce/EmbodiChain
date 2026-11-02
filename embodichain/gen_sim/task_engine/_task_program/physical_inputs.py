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

"""Explicit, graph-independent physical inputs for GenSim bundle generation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.data import get_data_path
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.utils.utility import load_config

from ..contracts import canonical_hash
from ..robot_profiles import _canonical_robot_profile
from ..orchestration.scene_source import _asset_dependency_files
from ..orchestration.source_scene import (
    PreparedScene,
    _estimate_mesh_top_z,
    _planner_object,
)

__all__ = [
    "FrozenPhysicalInput",
    "load_frozen_physical_input",
    "verify_physical_input_manifest",
]

_COMPONENTS = {
    "scene": "components/scene.yaml",
    "embodiment": "components/embodiment.yaml",
    "execution_policy": "components/execution_policy.yaml",
    "deployment": "task_program_deployment.yaml",
}
_SELECTION_FIELDS = {"id", "scene", "embodiment", "task_program"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _projection(value: FrozenPhysicalInput) -> dict[str, Any]:
    scene = asdict(value.scene)
    scene["source_config_path"] = str(value.scene.source_config_path)
    scene["scene_dir"] = str(value.scene.scene_dir)
    return {
        "scene": scene,
        "scene_simulation": value.scene_simulation,
        "embodiment": value.embodiment,
        "execution_policy": value.execution_policy,
        "deployment": value.deployment,
        "robot_profile": value.robot_profile,
        "input_files": value.input_files,
    }


def _check_files(files: Mapping[str, str]) -> None:
    for name, expected in files.items():
        path = Path(name)
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Frozen physical input file changed: {name}")


def _component_projection(root: Path) -> dict[str, Any]:
    values = {key: load_config(root / name) for key, name in _COMPONENTS.items()}
    values["scene"] = values["scene"]["simulation"]
    values["deployment"] = {
        key: value
        for key, value in values["deployment"].items()
        if key not in _SELECTION_FIELDS
    }
    return values


@dataclass(frozen=True, slots=True)
class FrozenPhysicalInput:
    """A detached physical snapshot, not a runtime physics-switching policy.

    Planner and simulator views are imported together from one explicit model.
    Nested data and referenced files are checked before and after generation.
    """

    scene: PreparedScene
    scene_simulation: dict[str, Any]
    embodiment: dict[str, Any]
    execution_policy: dict[str, Any]
    deployment: dict[str, Any]
    robot_profile: str
    input_files: dict[str, str]
    physical_digest: str

    def assert_unchanged(self) -> None:
        """Reject drift without repairing or resealing the selected input."""
        if canonical_hash(_projection(self)) != self.physical_digest:
            raise ValueError("Frozen physical input data changed.")
        _check_files(self.input_files)

    def manifest(self) -> dict[str, Any]:
        """Return provenance consumed by the ordinary bundle contract reader."""
        self.assert_unchanged()
        components = {
            "scene": self.scene_simulation,
            "embodiment": self.embodiment,
            "execution_policy": self.execution_policy,
            "deployment": self.deployment,
        }
        return {
            "schema_version": "gen_sim.frozen-physical-input/v1",
            "robot_profile": self.robot_profile,
            "physical_digest": self.physical_digest,
            "components_digest": canonical_hash(components),
            "planner_digest": canonical_hash(self.scene.planner_objects),
            "input_files": dict(self.input_files),
        }


def _asset_files(value: Any, *, base_dir: Path) -> set[Path]:
    files: set[Path] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"fpath", "urdf_path", "xml_path", "mjcf_path"} and isinstance(
                item, str
            ):
                path = Path(item).expanduser()
                if not path.is_absolute():
                    local = base_dir / path
                    path = local if local.is_file() else Path(get_data_path(item))
                files.update(_asset_dependency_files(path.resolve()))
                if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                    from pxr import Usd

                    stage = Usd.Stage.Open(str(path))
                    if stage is None:
                        raise ValueError(f"Frozen articulation cannot be read: {path}")
                    files.update(
                        Path(layer.realPath).resolve()
                        for layer in stage.GetUsedLayers()
                        if layer.realPath
                    )
            else:
                files.update(_asset_files(item, base_dir=base_dir))
    elif isinstance(value, (list, tuple)):
        for item in value:
            files.update(_asset_files(item, base_dir=base_dir))
    return files


def _effective_articulation(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    if Path(str(config.get("fpath", ""))).suffix.lower() not in {
        ".usd",
        ".usda",
        ".usdc",
    }:
        return result
    from embodichain.lab.sim.cfg import ArticulationCfg
    from embodichain.lab.sim.spawn.usd import articulation_desc_from_usd

    # Use the same declaration translation as spawn, without creating a World.
    cfg = ArticulationCfg.from_dict(deepcopy(dict(config)))
    descriptor, _ = articulation_desc_from_usd(cfg, per_env=True)
    # The translator selects source scale for preserve mode. Overlay keeps its
    # declared precision; round-tripping the float32 descriptor breaks lineage.
    result["body_scale"] = [float(value) for value in cfg.body_scale]
    result["init_local_pose"] = np.asarray(descriptor.pose).tolist()
    result["init_pos"] = np.asarray(descriptor.pose)[:3, 3].tolist()
    result.setdefault("root_props", {})["fixed_base"] = bool(descriptor.fixed_base)
    return result


def load_frozen_physical_input(
    prepared_scene: PreparedScene,
    bundle: str | Path,
    *,
    robot_profile: str = "dual_franka",
) -> FrozenPhysicalInput:
    """Import an explicitly selected model before semantic planning.

    Args:
        prepared_scene: Fresh source semantic identities, not old observations.
        bundle: Selected reference components; its receipts are never modified.
        robot_profile: Existing Task Engine robot-profile selector.

    Returns:
        One checked snapshot with planner geometry rebuilt from runtime inputs.

    Raises:
        ValueError: If object identities, roles, poses or components disagree.
    """
    root = Path(bundle).expanduser().resolve()
    existing = verify_physical_input_manifest(root)
    if existing is not None and _canonical_robot_profile(
        existing.get("robot_profile", robot_profile)
    ) != _canonical_robot_profile(robot_profile):
        raise ValueError("Frozen reference robot profile differs from the request.")
    input_files = {str(root / name): _sha(root / name) for name in _COMPONENTS.values()}
    if existing is not None:
        input_files[str(root / "physical_input.json")] = _sha(
            root / "physical_input.json"
        )
    scene_manifest = root / "static_scene_manifest.json"
    if scene_manifest.is_file():
        if _canonical_robot_profile(
            load_config(scene_manifest).get("robot_profile")
        ) != _canonical_robot_profile(robot_profile):
            raise ValueError("Frozen reference robot profile differs from the request.")
        input_files[str(scene_manifest)] = _sha(scene_manifest)
    deployment = load_config(root / _COMPONENTS["deployment"])
    for owner in ("scene", "embodiment"):
        if deployment.get(owner) != {"component": _COMPONENTS[owner]}:
            raise ValueError(
                "Frozen import requires canonical component selections without overrides."
            )
    if (
        deployment.get("task_program", {}).get("execution_policy")
        != _COMPONENTS["execution_policy"]
    ):
        raise ValueError(
            "Frozen import requires its selected execution-policy component."
        )
    _resolve_gym_components(deployment, base_dir=root)
    values = _component_projection(root)
    simulation = values["scene"]
    files = _asset_files(simulation, base_dir=root)
    files.update(_asset_files(values["embodiment"], base_dir=root))
    if prepared_scene.source_config_path.is_file():
        files.add(prepared_scene.source_config_path)
    input_files.update({str(path): _sha(path) for path in sorted(files)})
    source = {str(item["runtime_uid"]): item for item in prepared_scene.planner_objects}
    runtime = {
        str(item["uid"]): (role, item)
        for role in ("background", "rigid_object", "articulation")
        for item in simulation.get(role, ())
    }
    if (
        len(source) != len(prepared_scene.planner_objects)
        or len(runtime)
        != sum(
            len(simulation.get(role, ()))
            for role in ("background", "rigid_object", "articulation")
        )
        or source.keys() != runtime.keys()
    ):
        raise ValueError("Frozen scene and source semantic identities disagree.")
    planner = []
    effective_articulations = []
    for uid, (role, config) in runtime.items():
        if source[uid].get("role") != role:
            raise ValueError(f"Frozen scene role changed for {uid!r}.")
        merged = {**deepcopy(source[uid]), **deepcopy(config)}
        # Source semantics may be reused, but omitted runtime fields keep their
        # selected-model defaults rather than inheriting another model's pose.
        for name in (
            "shape",
            "fpath",
            "body_scale",
            "init_pos",
            "init_rot",
            "init_local_pose",
            "attrs",
            "root_props",
            "link_attrs",
            "joint_drive_props",
        ):
            if name not in config:
                merged.pop(name, None)
        merged.setdefault("init_rot", [0.0, 0.0, 0.0])
        if merged.get("init_local_pose") is not None:
            pose = np.asarray(merged["init_local_pose"], dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Frozen scene pose fields disagree for {uid!r}.")
            rotation = pose[:3, :3]
            if not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0
            ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0):
                raise ValueError(f"Frozen scene pose is not rigid for {uid!r}.")
            expected = np.eye(4)
            expected[:3, :3] = (
                Rotation.from_euler("XYZ", config["init_rot"], degrees=True).as_matrix()
                if "init_rot" in config
                else pose[:3, :3]
            )
            expected[:3, 3] = config.get("init_pos", pose[:3, 3])
            if not np.allclose(pose, expected, atol=1e-12, rtol=0):
                raise ValueError(f"Frozen scene pose fields disagree for {uid!r}.")
            merged["init_pos"] = pose[:3, 3].tolist()
            if "init_rot" not in config:
                merged["init_rot"] = (
                    Rotation.from_matrix(pose[:3, :3])
                    .as_euler("XYZ", degrees=True)
                    .tolist()
                )
        if role == "articulation":
            effective = _effective_articulation(config)
            effective_articulations.append(effective)
            merged.update(effective)
        planner.append(
            _planner_object(
                merged, source_uid=str(source[uid].get("source_uid", uid)), role=role
            )
        )
    from .twist_binding import enrich_twist_inventory

    enrich_twist_inventory(planner, effective_articulations)
    scene = replace(
        deepcopy(prepared_scene),
        planner_objects=tuple(planner),
        background=tuple(deepcopy(simulation.get("background", ()))),
        rigid_objects=tuple(deepcopy(simulation.get("rigid_object", ()))),
        articulations=tuple(effective_articulations),
    )
    from ..orchestration.scene_adapter import _discover_part_catalogs

    catalogs = _discover_part_catalogs(scene)
    for item in planner:
        if item["runtime_uid"] in catalogs:
            item["_articulation_part_positions"] = {
                part["part_id"]: list(part["handle_center_world"])
                for part in catalogs[item["runtime_uid"]]
                if "handle_center_world" in part
            }
    table = next((item for item in scene.background if item["uid"] == "table"), None)
    top = _estimate_mesh_top_z(table) if table is not None else None
    scene = replace(
        scene,
        table_top_z=scene.table_top_z if top is None else top,
        asset_hashes={
            uid: _sha(Path(config.get("fpath") or config.get("shape", {}).get("fpath")))
            for uid, (_role, config) in runtime.items()
            if config.get("fpath") or config.get("shape", {}).get("fpath")
        },
    )
    result = FrozenPhysicalInput(
        scene,
        deepcopy(simulation),
        values["embodiment"],
        values["execution_policy"],
        values["deployment"],
        robot_profile,
        input_files,
        "",
    )
    result = replace(result, physical_digest=canonical_hash(_projection(result)))
    result.assert_unchanged()
    return result


def verify_physical_input_manifest(bundle: str | Path) -> dict[str, Any] | None:
    """Verify an explicit input at the existing bundle-reader boundary.

    Args:
        bundle: Bundle directory; legacy bundles have no physical-input marker.

    Returns:
        The checked marker, or None for an unchanged legacy bundle.

    Raises:
        ValueError: If physical declarations or referenced inputs have drifted.
    """
    root = Path(bundle)
    path = root / "physical_input.json"
    if not path.is_file():
        return None
    manifest = load_config(path)
    if manifest.get("schema_version") != "gen_sim.frozen-physical-input/v1":
        raise ValueError("Unsupported frozen physical input manifest.")
    _check_files(manifest["input_files"])
    if canonical_hash(_component_projection(root)) != manifest["components_digest"]:
        raise ValueError("Frozen physical output declarations changed.")
    return manifest
