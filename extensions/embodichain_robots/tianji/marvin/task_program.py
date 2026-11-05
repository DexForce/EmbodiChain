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

"""Native Marvin hand declarations for explicit source Task Program assembly."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any
import xml.etree.ElementTree as ET

__all__ = [
    "TianjiMarvinTaskHandCalibration",
    "declare_tianji_task_program_embodiment",
    "tianji_marvin_task_hand_calibrations",
]


@dataclass(frozen=True, slots=True)
class TianjiMarvinTaskHandCalibration:
    """Native topology and nominal authored reserves, without runtime evidence."""

    resource_id: str
    side: str
    grasp_target_id: str
    motion_part: str
    hand_part: str
    command_preset: str
    hand_root_link: str
    pad_link_names: tuple[str, str]
    finger_link_names: tuple[str, ...]
    geometry: Mapping[str, Any]
    nominal_envelope: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "geometry", MappingProxyType(deepcopy(dict(self.geometry)))
        )
        object.__setattr__(
            self, "nominal_envelope", MappingProxyType(dict(self.nominal_envelope))
        )

    def opening_driver_indices(self, joint_names: tuple[str, ...]) -> tuple[int, ...]:
        """Select finger 1; finger 2 follows its authored one-to-one mimic."""
        driver = f"{self.side.upper()}_HAND_FINGER_1"
        if joint_names.count(driver) != 1:
            raise ValueError(
                "Native Marvin opening requires its explicit independent finger joint."
            )
        return (joint_names.index(driver),)

    def geometry_declaration(self) -> dict[str, Any]:
        """Return complete native sampling inputs without performing capture."""
        return deepcopy(dict(self.geometry))

    def nominal_envelope_declaration(self) -> dict[str, Any]:
        """Return authored collider reserves, independently of cooked geometry."""
        return dict(
            grasp_target_id=self.grasp_target_id,
            model_id="tianji_marvin_gripper",
            envelope=dict(self.nominal_envelope),
        )

    def nominal_pour_radius(self, max_opening_width: float) -> float:
        """Apply the existing nominal reserve formula to this native hand."""
        import numpy as np

        if not math.isfinite(max_opening_width) or max_opening_width <= 0:
            raise ValueError("The nominal Pour reserve requires a positive opening.")
        envelope = self.nominal_envelope
        return float(
            np.linalg.norm(
                [
                    (max_opening_width + envelope["finger_thickness"]) * 0.5,
                    envelope["finger_width"] * 0.5,
                    envelope["finger_length"] * 0.5,
                ]
            )
        )

    def revolute_grip_calibration(self) -> Mapping[str, Any] | None:
        """Use the existing conservative generic screen, without Robotiq proof."""
        return None


def _nominal_envelope(
    simulation: Mapping[str, Any], side: str, root_from_tcp: Any
) -> dict[str, float]:
    """Read deterministic authored collider bounds; acquire no native geometry."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.utility.urdf_collision import load_urdf_collision_inputs
    from embodichain.toolkits.graspkit.hand_envelope import NominalHandEnvelope

    root_link = f"{side}_hand_base_link"
    pads = (f"{side}_hand_finger_1_link", f"{side}_hand_finger_2_link")
    path = Path(simulation["fpath"])
    body_scale = simulation.get("body_scale", (1.0, 1.0, 1.0))
    geometry = load_urdf_collision_inputs(path, (root_link, *pads), vertices_only=True)
    document = ET.parse(path).getroot()
    tcp_from_root = np.linalg.inv(np.asarray(root_from_tcp, dtype=float))
    extents = []
    for index, pad in enumerate(pads, 1):
        joint = document.find(f"joint[@name='{side.upper()}_HAND_FINGER_{index}']")
        if joint is None or joint.find("parent").get("link") != root_link:
            raise ValueError(
                "Native Marvin fingers must retain their declared hand-root attachment."
            )
        origin = joint.find("origin")
        rotation = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        translation = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        vertices = (
            np.concatenate([part.vertices for part in geometry[pad]]) @ rotation.T
            + translation
        ) * np.asarray(body_scale)
        vertices = vertices @ tcp_from_root[:3, :3].T + tcp_from_root[:3, 3]
        extents.append(np.ptp(vertices, axis=0))
    palm = np.concatenate([part.vertices for part in geometry[root_link]]) * np.asarray(
        body_scale
    )
    palm = palm @ tcp_from_root[:3, :3].T + tcp_from_root[:3, 3]
    sizes = np.max(np.stack(extents), axis=0)
    result = dict(
        finger_length=float(sizes[2]),
        finger_width=float(sizes[1]),
        finger_thickness=float(sizes[0]),
        palm_depth=float(-palm[:, 2].min()),
        opening_margin=0.003,
    )
    NominalHandEnvelope(**result)
    return result


def declare_tianji_task_program_embodiment(
    embodiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand the existing native component for an explicit source integration.

    The original component and ordinary Task Program defaults are not changed.
    RobotCfg owns the complete native joint order, solver/TCP, inertials and
    mounted cameras. Only source integration services and named park commands
    are added to an independently owned declaration; no simulator is created.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from embodichain.compute.geometry.pose_checks import as_rigid_matrix
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.task_program.integrations.hand_calibration import (
        physical_hand_resources,
    )
    from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model

    owned = deepcopy(dict(embodiment))
    if (
        owned.get("embodiment_id") != "tianji_marvin"
        or owned["simulation"].get("class_type") != "TianjiMarvin"
        or owned["simulation"].get("with_gripper", True) is not True
    ):
        raise ValueError(
            "Native Marvin task declarations require the existing gripper-equipped TianjiMarvin component."
        )
    simulation = owned["simulation"]
    cfg = resolve_robot_cfg(
        "TianjiMarvin",
        {key: value for key, value in simulation.items() if key != "class_type"},
    )
    cfg.validate()
    owned["simulation"] = {**cfg.to_dict(), "class_type": "TianjiMarvin"}
    simulation = owned["simulation"]
    document = ET.parse(simulation["fpath"]).getroot()
    joints = [
        joint.get("name")
        for joint in document.findall("joint")
        if joint.get("type") != "fixed"
    ]
    if len(joints) != len(simulation["init_qpos"]) or len(set(joints)) != len(joints):
        raise ValueError(
            "Native Marvin ready pose must retain its complete authored joint order."
        )
    ready = dict(zip(joints, simulation["init_qpos"], strict=True))
    profile = owned["skill_profile"]
    leaves = physical_hand_resources(profile)
    services = profile["runtime_services"]
    generators = services["grasp_pose_generators"]
    geometries = services.setdefault("hand_geometries", {})
    presets = {preset["preset_id"]: preset for preset in profile["command_presets"]}
    motion_presets = {}
    for resource in leaves:
        endpoints = {
            endpoint["endpoint_id"]: endpoint for endpoint in resource["endpoints"]
        }
        grasp, motion = endpoints["grasp"], endpoints["motion"]
        part, motion_part = grasp["control_part"], motion["control_part"]
        generator = generators[part]
        model = resolve_parallel_jaw_gripper_model(generator["model"])
        side = generator.get("side")
        if model.model_id != "tianji_marvin_gripper" or side not in {"left", "right"}:
            raise ValueError(
                "Native Marvin hands require the declared native model and explicit side."
            )
        solver = simulation["solver_cfg"][motion_part]
        if solver["end_link_name"] != f"{side}_hand_tool_link":
            raise ValueError(
                "Native Marvin source hand calibration requires its actual fixed tool frame."
            )
        tool = document.find(f"joint[@name='{side.upper()}_HAND_TOOL']/origin")
        if tool is None:
            raise ValueError(
                "Native Marvin is missing its authored fixed tool transform."
            )
        root_from_tool = np.eye(4)
        root_from_tool[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(tool.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        root_from_tool[:3, 3] = np.fromstring(
            tool.get("xyz", "0 0 0"), sep=" "
        ) * np.asarray(simulation.get("body_scale", (1.0, 1.0, 1.0)))
        actual_tcp = root_from_tool @ as_rigid_matrix(solver["tcp"])
        declared_tcp = as_rigid_matrix(generator["root_from_tcp"])
        if not np.allclose(actual_tcp, declared_tcp, rtol=0, atol=1e-6):
            raise ValueError(
                "Native Marvin grasp TCP differs from its configured native tool/TCP."
            )
        expected = [f"{side.upper()}_HAND_FINGER_{index}" for index in (1, 2)]
        if simulation["control_parts"].get(part) != expected:
            raise ValueError(
                "Native Marvin requires its two authored finger joints, including the mimic follower."
            )
        preset = presets[grasp["command_preset"]]
        if (
            preset["control_part"] != part
            or preset["commands"].get("open") != [0.0, 0.0]
            or preset["commands"].get("grasp") != [-0.05, -0.05]
        ):
            raise ValueError(
                "Native Marvin hand commands cannot borrow another gripper's units or linkage."
            )
        geometries.setdefault(
            part,
            dict(
                kind="sampled_parallel_jaw",
                motion_part=motion_part,
                hand_part=part,
                hand_root_link=f"{side}_hand_base_link",
                pad_link_names=[
                    f"{side}_hand_finger_2_link",
                    f"{side}_hand_finger_1_link",
                ],
                command_preset=preset["preset_id"],
                sample_count=65,
                body_max_edge=0.01,
                pad_max_edge=0.004,
                max_subdivisions=10,
            ),
        )
        park_id = motion.get("command_preset", f"{motion_part}_postures")
        if park_id not in presets:
            park = dict(
                preset_id=park_id,
                control_part=motion_part,
                commands={
                    "park": [
                        ready[name] for name in simulation["control_parts"][motion_part]
                    ]
                },
            )
            profile["command_presets"].append(park)
            presets[park_id] = park
        elif (
            presets[park_id]["control_part"] != motion_part
            or "park" not in presets[park_id]["commands"]
        ):
            raise ValueError(
                "Native Marvin park requires an explicit complete arm command."
            )
        motion_presets[motion_part] = park_id
    for resource in profile["resources"]:
        for endpoint in resource["endpoints"]:
            if (
                endpoint["endpoint_id"] == "motion"
                and endpoint["control_part"] in motion_presets
            ):
                endpoint["command_preset"] = motion_presets[endpoint["control_part"]]
                if "motion.joint_position" not in endpoint["capabilities"]:
                    endpoint["capabilities"].append("motion.joint_position")
    physical_hand_resources(profile)
    return owned


def tianji_marvin_task_hand_calibrations(
    embodiment: Mapping[str, Any],
) -> tuple[TianjiMarvinTaskHandCalibration, ...]:
    """Bind explicit native hand declarations to the public task-hand interface."""
    from embodichain.lab.task_program.integrations.hand_calibration import (
        physical_hand_resources,
    )
    from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model

    if (
        embodiment.get("embodiment_id") != "tianji_marvin"
        or embodiment["simulation"].get("class_type") != "TianjiMarvin"
    ):
        raise ValueError(
            "Native Marvin calibration requires its own existing component."
        )
    embodiment = declare_tianji_task_program_embodiment(embodiment)
    profile = embodiment["skill_profile"]
    services = profile["runtime_services"]
    geometries = services["hand_geometries"]
    selected = []
    for resource in physical_hand_resources(profile):
        endpoints = {
            endpoint["endpoint_id"]: endpoint for endpoint in resource["endpoints"]
        }
        grasp, motion = endpoints["grasp"], endpoints["motion"]
        part = grasp["control_part"]
        generator = services["grasp_pose_generators"][part]
        model = resolve_parallel_jaw_gripper_model(generator["model"])
        side = generator.get("side")
        if model.model_id != "tianji_marvin_gripper" or side not in {"left", "right"}:
            raise ValueError(
                "Native Marvin calibration cannot replace its physical hand model."
            )
        geometry = geometries[part]
        expected_pads = {f"{side}_hand_finger_{index}_link" for index in (1, 2)}
        if (
            geometry.get("kind") != "sampled_parallel_jaw"
            or geometry.get("motion_part") != motion["control_part"]
            or geometry.get("hand_part") != part
            or geometry.get("command_preset") != grasp["command_preset"]
            or geometry.get("hand_root_link") != f"{side}_hand_base_link"
            or set(geometry.get("pad_link_names", ())) != expected_pads
            or len(geometry["pad_link_names"]) != 2
        ):
            raise ValueError(
                "Native Marvin hand geometry differs from its actual model/endpoints."
            )
        selected.append(
            TianjiMarvinTaskHandCalibration(
                resource["resource_id"],
                side,
                part,
                motion["control_part"],
                part,
                grasp["command_preset"],
                geometry["hand_root_link"],
                tuple(geometry["pad_link_names"]),
                (f"{side}_hand_finger_1_link", f"{side}_hand_finger_2_link"),
                geometry,
                _nominal_envelope(
                    embodiment["simulation"], side, generator["root_from_tcp"]
                ),
            )
        )
    if (
        len(selected) != 2
        or {hand.side for hand in selected} != {"left", "right"}
        or set(geometries) != {hand.hand_part for hand in selected}
    ):
        raise ValueError(
            "Native Marvin source assembly requires exactly its two physical hands."
        )
    return tuple(selected)
