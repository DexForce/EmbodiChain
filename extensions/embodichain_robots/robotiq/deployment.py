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

"""Explicit source deployment calibration for the qualified Robotiq embodiments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .task_program import robotiq140_task_hand_calibrations

__all__ = [
    "source_robotiq_opening_declaration",
    "source_contact_pad_declaration",
    "source_robot_mount_declaration",
    "source_grip_effort_declaration",
    "source_ur10_idle_posture_declaration",
    "source_arm_stiffness_declaration",
    "source_revolute_grip_calibration",
    "source_robot_lateral_positions",
    "require_source_robot_assets",
]


def source_robotiq_opening_declaration(embodiment: Mapping[str, Any]) -> dict[str, Any]:
    """Cap explicit Robotiq proposal widths at the original 0.128 m free gap.

    A lower authored width and other declared hand models are unchanged. This
    creates an independent deployment declaration; shared components are not edited.
    """
    owned = deepcopy(dict(embodiment))
    generators = owned["skill_profile"]["runtime_services"]["grasp_pose_generators"]
    for generator in generators.values():
        model = generator["model"]
        if model["model_id"] == "robotiq_arg2f_140":
            model["max_opening_width"] = min(float(model["max_opening_width"]), 0.128)
    return owned


def source_contact_pad_declaration(embodiment: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the explicitly selected source pad material overlay to an owned copy.

    The original pad expression and friction values are retained. Mass, inertia,
    collider geometry and shared defaults are untouched; duplicate overlays reject.
    """
    owned = deepcopy(dict(embodiment))
    simulation = owned["simulation"]
    overlays = simulation.setdefault("link_attrs", {})
    if "gen_sim_contact_pads" in overlays:
        raise ValueError("The contact pad material overlay is already declared.")
    overlays["gen_sim_contact_pads"] = {
        "link_names_expr": [".*inner_finger_pad"],
        "attrs": {
            "material_props": {"static_friction": 2.85, "dynamic_friction": 2.70}
        },
    }
    simulation["asset_physics_mode"] = "overlay"
    return owned


def source_robot_mount_declaration(
    embodiment: Mapping[str, Any],
    *,
    table_top_z: float,
    table_bounds: np.ndarray | None = None,
    source_urdf: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a qualified model to explicit source table measurements without IK.

    Args:
        embodiment: Selected robot/sensor/profile component.
        table_top_z: Explicit finite tabletop height.
        table_bounds: Retained optional declaration input for caller compatibility.
        source_urdf: Optional existing assembled URDF for model geometry reads.
            Otherwise the explicit simulation fpath is used. No Robot is built.

    Returns:
        Independent declaration retaining the source bench offsets, floor reserve
        and per-deployment IK tolerance. This is not a collision certificate.
    """
    from embodichain.lab.sim.scene.source_robot_geometry import source_arm_base_position

    if table_top_z is None or not math.isfinite(float(table_top_z)):
        raise ValueError("Robot deployment requires a derived tabletop height.")
    owned = deepcopy(dict(embodiment))
    simulation = owned.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Embodiment component has no simulation mapping.")
    identity = owned["embodiment_id"]
    if identity not in {
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    }:
        raise ValueError("Source mount has no qualified model calibration.")
    for solver in simulation.get("solver_cfg", {}).values():
        if solver.get("class_type") == "PytorchSolver":
            solver.setdefault("pos_eps", 5e-3)
    if identity == "dual_franka_robotiq_arg2f_140":
        position = simulation.get("init_pos")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(
                "Embodiment simulation.init_pos must contain three values."
            )
        simulation["init_pos"] = [
            float(position[0]),
            float(position[1]),
            float(table_top_z) - 0.35,
        ]
    else:
        hands = robotiq140_task_hand_calibrations(owned)
        by_side = {value.side: value for value in hands}
        anchors = np.stack(
            [
                source_arm_base_position(simulation, by_side[side].motion_part)
                for side in ("left", "right")
            ]
        )
        rotation = Rotation.from_euler(
            "XYZ", simulation["init_rot"], degrees=True
        ).as_matrix()
        midpoint = rotation @ anchors.mean(axis=0)
        simulation["init_pos"] = (
            np.array([0.55, 0.0, float(table_top_z) + 0.05]) - midpoint
        ).tolist()
    return owned


def source_grip_effort_declaration(
    embodiment: Mapping[str, Any],
    caps: Mapping[str, float],
) -> dict[str, Any]:
    """Apply explicit resource effort caps without selecting task/recipe eligibility.

    Caps are chosen by the deployment declaration owner. This model adapter maps
    declared resources to actual hand control parts and preserves lower limits.
    """
    owned = deepcopy(dict(embodiment))
    parts = {
        resource["resource_id"]: endpoint["control_part"]
        for resource in owned["skill_profile"]["resources"]
        for endpoint in resource["endpoints"]
        if endpoint["endpoint_id"] == "grasp"
    }
    limits = owned["simulation"]["joint_drive_props"].get("max_effort")
    for resource, cap in caps.items():
        if (
            resource not in parts
            or isinstance(cap, bool)
            or not isinstance(cap, (float, int))
            or not math.isfinite(cap)
            or cap <= 0
        ):
            raise ValueError(
                "Grip effort requires a declared resource and positive finite cap."
            )
        part = parts[resource]
        if isinstance(limits, dict) and part in limits:
            limits[part] = min(float(limits[part]), float(cap))
    return owned


def source_ur10_idle_posture_declaration(
    embodiment: Mapping[str, Any],
    *,
    active_resource_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the source UR10 idle shoulder seed and paired Park declarations.

    Eligibility is explicit in the caller's declaration. No task identifiers or
    semantic call names are inspected here, and actual robot state is not read.
    """
    owned = deepcopy(dict(embodiment))
    if owned["embodiment_id"] != "dual_ur10_robotiq_arg2f_140":
        raise ValueError(
            "Idle shoulder calibration requires the qualified UR10 embodiment."
        )
    hands = {
        value.resource_id: value for value in robotiq140_task_hand_calibrations(owned)
    }
    if active_resource_id not in hands:
        raise ValueError("Active resource is not declared by this embodiment.")
    active = hands[active_resource_id].side
    idle = "right" if active == "left" else "left"
    owned["simulation"]["init_qpos"][3 if idle == "right" else 2] -= 0.5
    endpoints = {
        endpoint["control_part"]: endpoint
        for resource in owned["skill_profile"]["resources"]
        for endpoint in resource["endpoints"]
        if endpoint["endpoint_id"] == "motion"
    }
    presets = {
        value["preset_id"]: value for value in owned["skill_profile"]["command_presets"]
    }
    for hand in hands.values():
        preset = presets[endpoints[hand.motion_part]["command_preset"]]
        preset["commands"]["park"][1] -= 0.5
    return owned, {
        "schema_version": "gen_sim.ur10-initial-articulation-posture/v1",
        "reason": "initial_prismatic_operation_idle_arm_clearance",
        "active_resource": active_resource_id,
        "tucked_initial_resource": next(
            value.resource_id for value in hands.values() if value.side == idle
        ),
        "shoulder_offset_rad": -0.5,
        "physical_clearance_certified": False,
    }


def source_arm_stiffness_declaration(
    embodiment: Mapping[str, Any],
    *,
    stiffness: float,
) -> dict[str, Any]:
    """Apply an explicit motion-endpoint gain override without recipe inference."""
    if (
        isinstance(stiffness, bool)
        or not isinstance(stiffness, (int, float))
        or not math.isfinite(stiffness)
        or stiffness < 0
    ):
        raise ValueError("Arm stiffness must be a finite nonnegative value.")
    owned = deepcopy(dict(embodiment))
    values = owned["simulation"]["joint_drive_props"]["stiffness"]
    for resource in owned["skill_profile"]["resources"]:
        for endpoint in resource.get("endpoints", []):
            if endpoint["endpoint_id"] == "motion":
                values[endpoint["control_part"]] = float(stiffness)
    return owned


def source_revolute_grip_calibration() -> dict[str, Any]:
    """Return original Robotiq source-control depth/opening calibration.

    This explicit model-owned metadata retains the source 7.5 mm minimum axial
    engagement and measured pad dimensions. A host selects this calibration;
    no task identifier or source asset hash changes its values.
    """
    return dict(
        maximum_opening_width=0.128,
        minimum_knob_grip_depth=0.0075,
        pad_collision_extents=[0.027, 0.065, 0.0075],
        basis="canonical_pad_mesh_and_7p5mm_grip_depth_physical_qualification",
    )


def source_robot_lateral_positions(
    model_id: str,
    embodiment: Mapping[str, Any] | None = None,
) -> dict[str, float] | None:
    """Retain original semantic lateral hints through model-owned source frames.

    Args:
        model_id: Explicit selected source embodiment identity.
        embodiment: Optional complete physical component with its original pose
            and URDF/assembly declaration. No robot runtime is needed.

    Returns:
        Original side-to-world-Y hints for UR10 when a physical component
        is supplied. Franka and UR10 without one retain None.
        These semantic hints do not certify collision clearance or kinematics.
    """
    from embodichain.lab.sim.scene.source_robot_geometry import source_arm_base_position

    models = {
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    }
    if model_id not in models:
        raise ValueError("No source lateral calibration for this model.")
    if model_id == "dual_franka_robotiq_arg2f_140":
        return None
    if embodiment is not None:
        if embodiment.get("embodiment_id") != model_id:
            raise ValueError("Lateral hint model differs from the selected embodiment.")
        simulation = embodiment["simulation"]
        if simulation.get("init_local_pose") is not None:
            pose = np.asarray(simulation["init_local_pose"], dtype=float)
        else:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler(
                "XYZ", simulation.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
            ).as_matrix()
            pose[:3, 3] = simulation.get("init_pos", [0.0, 0.0, 0.0])
        motion_parts = {
            value["resource_id"]: next(
                endpoint["control_part"]
                for endpoint in value["endpoints"]
                if endpoint["endpoint_id"] == "motion"
            )
            for value in embodiment["skill_profile"]["resources"]
        }
        return {
            side: float(
                (
                    pose[:3, :3]
                    @ source_arm_base_position(simulation, motion_parts[side])
                    + pose[:3, 3]
                )[1]
            )
            for side in ("left", "right")
        }
    return None


def require_source_robot_assets(model_id: str) -> None:
    """Preflight the selected model's existing registered assets and configuration.

    Args:
        model_id: Explicit selected source embodiment identity.

    Raises:
        ValueError: The model, source asset or basic model configuration cannot
            be resolved. No Robot, IK solver, simulator or provider is constructed.

    Task-package/component selection belongs to the calling application; this
    owner checks model files and kinematic configuration rather than task imports.
    """
    from embodichain.data import get_data_path
    from embodichain_robots import URRobotCfg

    assets = {
        "dual_franka_robotiq_arg2f_140": "Franka/Panda/Panda.urdf",
        "dual_ur10_robotiq_arg2f_140": "UniversalRobots/UR10/UR10.urdf",
    }
    if model_id not in assets:
        raise ValueError("No qualified source robot asset model was selected.")
    for asset in (assets[model_id], "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"):
        try:
            path = Path(get_data_path(asset))
            if not path.is_file():
                raise FileNotFoundError(path)
        except (RuntimeError, OSError) as error:
            raise ValueError(
                f"Source model {model_id!r} requires registered asset {asset!r}."
            ) from error
    try:
        if model_id == "dual_ur10_robotiq_arg2f_140":
            cfg = URRobotCfg.from_dict({"robot_type": "ur10"})
        else:
            return
        cfg.validate()
    except (ImportError, TypeError, ValueError, RuntimeError, OSError) as error:
        raise ValueError(
            f"Source model {model_id!r} configuration preflight failed: {error}"
        ) from error
