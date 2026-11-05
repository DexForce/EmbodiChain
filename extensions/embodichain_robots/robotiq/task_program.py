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

"""Explicit Robotiq 2F-140 bindings for the three qualified dual-arm embodiments."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy
from types import MappingProxyType
from typing import Any

__all__ = ["Robotiq140TaskHandCalibration", "robotiq140_task_hand_calibrations"]

_EMBODIMENTS = frozenset(
    {
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
        "tianji_marvin_robotiq_arg2f_140",
    }
)
_FINGERS_BY_ROOT = {
    "left_robotiq_arg2f_base_link": (
        "left",
        (
            "left_outer_finger",
            "left_inner_finger",
            "left_inner_finger_pad",
            "left_right_outer_finger",
            "left_right_inner_finger",
            "left_right_inner_finger_pad",
        ),
    ),
    "right_robotiq_arg2f_base_link": (
        "right",
        (
            "right_left_outer_finger",
            "right_left_inner_finger",
            "right_left_inner_finger_pad",
            "right_outer_finger",
            "right_inner_finger",
            "right_inner_finger_pad",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class Robotiq140TaskHandCalibration:
    """Model-owned link/command identities, not a runtime hand observation."""

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

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "geometry", MappingProxyType(deepcopy(dict(self.geometry)))
        )

    def nominal_pour_radius(self, max_opening_width: float) -> float:
        """Return the source nominal swept-hand reserve for an explicit opening.

        Args:
            max_opening_width: The selected grasp model's declared maximum width.

        Returns:
            Radius from the source nominal finger dimensions; not native geometry.
        """
        import math

        if not math.isfinite(max_opening_width) or max_opening_width <= 0:
            raise ValueError("The nominal Pour reserve requires a positive opening.")
        envelope = self.nominal_envelope_declaration()["envelope"]
        # Keep the source NumPy reduction order, not an alternative norm formula.
        import numpy as np

        return float(
            np.linalg.norm(
                [
                    (max_opening_width + envelope["finger_thickness"]) * 0.5,
                    envelope["finger_width"] * 0.5,
                    envelope["finger_length"] * 0.5,
                ]
            )
        )

    def opening_driver_indices(self, joint_names: tuple[str, ...]) -> tuple[int, ...]:
        """Resolve the model's independent opening driver in declared joint order.

        Args:
            joint_names: Complete configured hand joint roster.

        Returns:
            The single non-mimic driver index for this explicit model hand.
        """
        driver = {"left": "left_finger_joint", "right": "right_finger_joint"}[self.side]
        if joint_names.count(driver) != 1:
            raise ValueError(
                "Qualified Robotiq hand requires its explicit opening driver."
            )
        return (joint_names.index(driver),)

    def geometry_declaration(self) -> dict[str, Any]:
        """Return the explicit source-compatible shared hand geometry service.

        Returns:
            Closed ``sampled_parallel_jaw`` declaration. The complete hand subtree
            remains selected by root; pad order is qualified at live capture.
            This selects sampling but performs no FK, mesh reads or simulation.
        """
        return deepcopy(dict(self.geometry))

    def nominal_envelope_declaration(self) -> dict[str, Any]:
        """Return the original explicit task-reserve calibration, not collision facts.

        Returns:
            Existing shared alias envelope schema keyed by exact grasp target.
            The source reserve dimensions remain distinct from actual URDF
            surfaces and do not replace physical geometry qualification.
        """
        return dict(
            grasp_target_id=self.grasp_target_id,
            model_id="robotiq_arg2f_140",
            envelope=dict(
                finger_length=0.13,
                finger_width=0.03,
                finger_thickness=0.01,
                palm_depth=0.08,
                opening_margin=0.003,
            ),
        )


def robotiq140_task_hand_calibrations(
    embodiment: Mapping[str, Any],
    *,
    hand_geometries: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Robotiq140TaskHandCalibration, ...]:
    """Bind existing embodiment geometry declarations to explicit model contact roles.

    Args:
        embodiment: Qualified source physical/skill-profile declaration with
            expanded control parts. Command presets remain its authority.
        hand_geometries: Explicit geometry service mappings from the canonical
            embodiment owner when the frozen historical snapshot predates them.
            Omission consumes the selected embodiment's own existing declarations.
            This function never invents or defaults root/pad/sampling calibration.

    Returns:
        Exactly two hand descriptors retaining caller semantic resource IDs.
        Existing geometry mappings are copied, not replaced by a parallel model
        default. Complete finger rosters and nominal source task reserves are
        model-owned additions, still not native observations or certificates.

    Raises:
        ValueError: Source embodiment, model, command or explicit geometry binding
            is missing, ambiguous or inconsistent.
    """
    if embodiment.get("embodiment_id") not in _EMBODIMENTS:
        raise ValueError("No source Robotiq task-hand calibration for this embodiment.")
    profile = embodiment["skill_profile"]
    services = profile["runtime_services"]
    geometries = (
        services.get("hand_geometries") if hand_geometries is None else hand_geometries
    )
    if not isinstance(geometries, Mapping):
        raise ValueError(
            "Source hand calibration requires explicit embodiment geometry declarations."
        )
    generators = services["grasp_pose_generators"]
    commands = {value["preset_id"]: value for value in profile["command_presets"]}
    parts = embodiment["simulation"]["control_parts"]
    selected = {}
    for resource in profile["resources"]:
        endpoints = {value["endpoint_id"]: value for value in resource["endpoints"]}
        grasp, motion = endpoints.get("grasp"), endpoints.get("motion")
        if grasp is None:
            continue
        part = grasp["control_part"]
        geometry = geometries.get(part)
        if not isinstance(geometry, Mapping) or motion is None or part in selected:
            raise ValueError(
                "Source hand resources require unique explicit geometry bindings."
            )
        root = geometry.get("hand_root_link")
        if (
            root not in _FINGERS_BY_ROOT
            or geometry.get("kind") != "sampled_parallel_jaw"
            or geometry.get("motion_part") != motion["control_part"]
            or geometry.get("hand_part") != part
            or geometry.get("command_preset") != grasp.get("command_preset")
        ):
            raise ValueError(
                "Source hand geometry differs from its explicit motion/grasp endpoints."
            )
        side, fingers = _FINGERS_BY_ROOT[root]
        pads = tuple(geometry.get("pad_link_names", ()))
        if len(pads) != 2 or len(set(pads)) != 2 or not set(pads) <= set(fingers):
            raise ValueError(
                "Source hand pad roles must belong to its explicit finger roster."
            )
        model = generators.get(part, {})
        if (
            model.get("kind") != "antipodal_parallel_jaw"
            or model.get("model", {}).get("model_id") != "robotiq_arg2f_140"
        ):
            raise ValueError(
                "Source hand calibration requires its explicitly declared Robotiq model."
            )
        command = commands.get(grasp.get("command_preset"))
        joints = parts.get(part, ())
        if (
            not joints
            or command is None
            or command.get("control_part") != part
            or any(
                len(command.get("commands", {}).get(name, ())) != len(joints)
                for name in ("open", "grasp")
            )
        ):
            raise ValueError(
                "Source hand commands must retain their complete declared joint order."
            )
        selected[part] = Robotiq140TaskHandCalibration(
            resource_id=resource["resource_id"],
            side=side,
            grasp_target_id=part,
            motion_part=motion["control_part"],
            hand_part=part,
            command_preset=command["preset_id"],
            hand_root_link=root,
            pad_link_names=pads,
            finger_link_names=fingers,
            geometry=geometry,
        )
    if len(selected) != 2 or {
        value.hand_root_link for value in selected.values()
    } != set(_FINGERS_BY_ROOT):
        raise ValueError(
            "Source task-hand calibration requires both explicitly bound hands."
        )
    if set(geometries) != set(selected):
        raise ValueError("Source hand geometry declares unmatched grasp targets.")
    return tuple(selected.values())
