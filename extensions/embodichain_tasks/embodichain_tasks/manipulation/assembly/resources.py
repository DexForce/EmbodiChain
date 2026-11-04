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

__all__ = ["export_assembly_resources"]


def export_assembly_resources(
    env: Any, config: dict[str, Any], config_dir: Path
) -> dict[str, Any]:
    """Freeze generated geometry and the prepared robot into the RUN.

    Relative resource references are resolved by the ordinary Gym loader.
    Replay therefore uses this exact model preparation and needs neither
    mjfit nor the source-model cache. The bundle records its engineering
    fidelity and never upgrades the nominal model's qualification.
    """
    state = env.assembly_state
    from copy import deepcopy
    import json
    import shutil

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
    source = Path(state.env.cfg.robot._pk_urdf_path).resolve()
    robot_directory = resources / "robot"
    if source.parent.resolve() != robot_directory.resolve():
        shutil.copytree(source.parent, robot_directory, dirs_exist_ok=True)
    # Preserve upstream license texts and immutable resource identities,
    # even though generated mesh files live in a separate cache directory.
    licenses = robot_directory / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    for model in ("flexiv_rizon4s", "robotiq_2f85"):
        license_path = source.parent.parent / model / "LICENSE"
        if license_path.is_file():
            shutil.copy2(license_path, licenses / f"{model}.LICENSE")
    source_manifest = source.parent.parent / "source_manifest.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, robot_directory / "source_manifest.json")
        marker = robot_directory / "fidelity.json"
        if marker.is_file():
            fidelity = json.loads(marker.read_text())
            fidelity["source_manifest"] = "./source_manifest.json"
            marker.write_text(json.dumps(fidelity, indent=2) + "\n")
    # Preserve declared robot physics as well as geometry. The task's
    # explicit joint_stiffness/joint_damping fields still own its native
    # impedance override at the normal initialization boundary.
    for name in ("joint_drive_props", "root_props", "attrs"):
        value = getattr(state.env.cfg.robot, name, None)
        if value is not None:
            frozen["robot"][name] = value.to_dict()
    frozen["robot"]["link_attrs"] = {
        name: value.to_dict()
        for name, value in (
            getattr(state.env.cfg.robot, "link_attrs", None) or {}
        ).items()
    }
    prepared = f"./resources/robot/{source.name}"
    frozen["robot"]["fpath"] = prepared
    frozen["robot"].pop("urdf_cfg", None)
    for solver in frozen["robot"].get("solver_cfg", {}).values():
        solver["urdf_path"] = prepared
    frozen.setdefault("env", {}).setdefault("extensions", {})[
        "assembly_manifest"
    ] = f"./resources/assembly/{manifest_name}"
    return frozen
