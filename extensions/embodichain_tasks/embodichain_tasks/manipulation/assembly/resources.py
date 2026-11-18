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

"""Bundle prepared assembly resources with a relocatable run."""

from __future__ import annotations
from pathlib import Path
from typing import Any

from embodichain.data.resources import snapshot_urdf_resources

__all__ = ["export_assembly_resources"]


def export_assembly_resources(
    env: Any, config: dict[str, Any], config_dir: Path
) -> dict[str, Any]:
    """Freeze generated geometry and the prepared robot into the RUN.

    Relative resource references are resolved by the ordinary Gym loader.
    Replay therefore uses this exact model preparation and needs neither
    mjfit nor the source-model cache. The bundle records its engineering
    fidelity and never upgrades the nominal model's qualification.

    Args:
        env: Common environment containing the initialized assembly state.
        config: Owned, component-resolved Gym configuration to freeze.
        config_dir: Destination run configuration directory.

    Returns:
        Configuration using relative references to the bundled resources.
    """
    state = env.assembly_state
    from copy import deepcopy
    import json

    frozen = deepcopy(config)
    for name in ("scene_preparation", "resource_export"):
        operation = getattr(env.cfg, name, None)
        if operation is not None:
            frozen.setdefault("env", {})[name] = operation.to_dict()
    resources = Path(config_dir) / "resources"
    assembly_directory = resources / "assembly"
    manifest_name = f"{state.scene.assembly_asset.content_hash}.json"
    state.scene.assembly_asset.save(assembly_directory / manifest_name)
    (assembly_directory / "fidelity.json").write_text(
        json.dumps(state.scene.assembly_fidelity, indent=2) + "\n"
    )
    robot = env.cfg.robot
    source = Path(robot._pk_urdf_path).resolve()
    robot_directory = resources / "robot"
    provenance = source.parent.parent
    prepared = snapshot_urdf_resources(
        source,
        robot_directory,
        provenance_root=(
            provenance if (provenance / "source_manifest.json").is_file() else None
        ),
    )
    # Fidelity is task metadata; the filesystem helper does not interpret it.
    if (robot_directory / "source_manifest.json").is_file():
        marker = robot_directory / "fidelity.json"
        if marker.is_file():
            fidelity = json.loads(marker.read_text())
            fidelity["source_manifest"] = "./source_manifest.json"
            marker.write_text(json.dumps(fidelity, indent=2) + "\n")
    # Preserve declared robot physics as well as geometry. The task's
    # explicit joint_stiffness/joint_damping fields still own its native
    # impedance override at the normal initialization boundary.
    for name in ("joint_drive_props", "root_props", "attrs"):
        value = getattr(robot, name, None)
        if value is not None:
            frozen["robot"][name] = value.to_dict()
    frozen["robot"]["link_attrs"] = {
        name: value.to_dict()
        for name, value in (getattr(robot, "link_attrs", None) or {}).items()
    }
    config_root = Path(config_dir).resolve()
    frozen["robot"]["fpath"] = f"./{prepared.relative_to(config_root).as_posix()}"
    frozen["robot"].pop("urdf_cfg", None)
    declared_solvers = frozen["robot"].get("solver_cfg") or {}
    # Keep same-directory dependencies together and prevent models from
    # different source trees with the same basename from overwriting each other.
    model_trees = {source.parent: prepared.parent}
    for part, solver in (robot.solver_cfg or {}).items():
        model = Path(solver.urdf_path).resolve() if solver.urdf_path else source
        if part not in declared_solvers:
            if model == source:
                continue
            declared_solvers[part] = solver.to_dict()
        if model.parent not in model_trees:
            model_trees[model.parent] = snapshot_urdf_resources(
                model, resources / "solver_models" / str(len(model_trees))
            ).parent
        bundled = model_trees[model.parent] / model.name
        declared_solvers[part][
            "urdf_path"
        ] = f"./{bundled.relative_to(config_root).as_posix()}"
    if declared_solvers:
        frozen["robot"]["solver_cfg"] = declared_solvers
    frozen.setdefault("env", {}).setdefault("extensions", {})["assembly"][
        "assembly_manifest"
    ] = f"./resources/assembly/{manifest_name}"
    return frozen
