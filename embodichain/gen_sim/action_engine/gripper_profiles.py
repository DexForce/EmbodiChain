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

"""Validated GenSim gripper assets, controls, TCPs, and grasp geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "GraspModelSpec",
    "GripperModel",
    "GripperProfile",
    "get_gripper_profile",
]

_Side = str
_Transform = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class GripperModel(str, Enum):
    """Gripper models supported by the GenSim composition root."""

    PGI = "pgi"
    ROBOTIQ = "robotiq"


@dataclass(frozen=True, slots=True)
class GraspModelSpec:
    """Collision geometry used to interpret sampled poses as gripper TCP poses."""

    model_id: str
    min_opening_width: float
    max_opening_width: float
    finger_length: float
    finger_width: float
    finger_thickness: float
    palm_depth: float
    opening_margin: float

    def as_mapping(self) -> dict[str, float | str]:
        """Return a detached JSON-compatible geometry description."""
        return {
            "model_id": self.model_id,
            "min_opening_width": self.min_opening_width,
            "max_opening_width": self.max_opening_width,
            "finger_length": self.finger_length,
            "finger_width": self.finger_width,
            "finger_thickness": self.finger_thickness,
            "palm_depth": self.palm_depth,
            "opening_margin": self.opening_margin,
        }


@dataclass(frozen=True, slots=True)
class GripperProfile:
    """One indivisible simulator, controller, kinematics, and grasp contract.

    ``tcp_transform`` is the row-major homogeneous transform from each solver's
    configured ``end_link_name`` frame to the tool center point. No quaternion
    conversion is involved in this contract.
    """

    model: GripperModel
    asset_path: str
    assembly_name: str
    tcp_transform: _Transform
    left_control_joints: tuple[str, ...]
    right_control_joints: tuple[str, ...]
    left_mimic_joints: tuple[str, ...]
    right_mimic_joints: tuple[str, ...]
    mimic_multipliers: tuple[float, ...]
    mimic_offsets: tuple[float, ...]
    simulated_joint_initial_positions: tuple[float, ...]
    open_positions: tuple[float, ...]
    close_positions: tuple[float, ...]
    control_limits: tuple[tuple[float, float], ...]
    drive_stiffness: float
    drive_damping: float
    drive_max_effort: float
    grasp_model: GraspModelSpec

    def __post_init__(self) -> None:
        control_count = len(self.left_control_joints)
        if not control_count or len(self.right_control_joints) != control_count:
            raise ValueError(
                "Gripper profiles require matching non-empty hand controls."
            )
        if not (
            len(self.open_positions)
            == len(self.close_positions)
            == len(self.control_limits)
            == control_count
        ):
            raise ValueError(
                "Gripper control states and limits must match control joints."
            )
        mimic_count = len(self.left_mimic_joints)
        if not (
            len(self.right_mimic_joints)
            == len(self.mimic_multipliers)
            == len(self.mimic_offsets)
            == mimic_count
        ):
            raise ValueError("Gripper mimic metadata must have matching lengths.")
        if len(self.simulated_joint_initial_positions) != len(
            self.simulated_joint_names("left")
        ):
            raise ValueError(
                "Gripper simulated initial positions must match physical joints."
            )

    def control_joint_names(self, side: _Side) -> tuple[str, ...]:
        """Return the exact assembled control-joint names for one hand."""
        self._validate_side(side)
        return self.left_control_joints if side == "left" else self.right_control_joints

    def mimic_joint_names(self, side: _Side) -> tuple[str, ...]:
        """Return exact assembled mimic-joint names for one hand."""
        self._validate_side(side)
        return self.left_mimic_joints if side == "left" else self.right_mimic_joints

    def simulated_joint_names(self, side: _Side) -> tuple[str, ...]:
        """Return physical movable joints in assembled qpos order for one hand."""
        return tuple(
            dict.fromkeys(
                (*self.control_joint_names(side), *self.mimic_joint_names(side))
            )
        )

    def runtime_manifest(
        self,
        *,
        tcp_parent_frames: dict[str, str],
    ) -> dict[str, Any]:
        """Describe the selected physical and planning contract for diagnostics."""
        if set(tcp_parent_frames) != {"left", "right"} or not all(
            isinstance(value, str) and value for value in tcp_parent_frames.values()
        ):
            raise ValueError(
                "TCP parent frames require non-empty left and right links."
            )
        return {
            "model": self.model.value,
            "asset_path": self.asset_path,
            "control_joints": {
                side: list(self.control_joint_names(side)) for side in ("left", "right")
            },
            "mimic_joints": {
                side: [
                    {
                        "name": name,
                        "source": self.control_joint_names(side)[0],
                        "multiplier": self.mimic_multipliers[index],
                        "offset": self.mimic_offsets[index],
                    }
                    for index, name in enumerate(self.mimic_joint_names(side))
                ]
                for side in ("left", "right")
            },
            "open_positions": list(self.open_positions),
            "close_positions": list(self.close_positions),
            "control_limits": [list(limit) for limit in self.control_limits],
            "tcp": {
                "parent_frames": dict(tcp_parent_frames),
                "transform_direction": "parent_link_to_tcp",
                "matrix_layout": "row_major_homogeneous_4x4",
                "quaternion_order": "not_applicable",
                "transform": [list(row) for row in self.tcp_transform],
            },
            "grasp_model": self.grasp_model.as_mapping(),
        }

    @staticmethod
    def _validate_side(side: _Side) -> None:
        if side not in {"left", "right"}:
            raise ValueError("Gripper side must be 'left' or 'right'.")


_PGI_PROFILE = GripperProfile(
    model=GripperModel.PGI,
    asset_path="DH_PGI_140_80/DH_PGI_140_80.urdf",
    assembly_name="dh_pgi_140_80",
    tcp_transform=(
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.121),
        (0.0, 0.0, 0.0, 1.0),
    ),
    left_control_joints=("left_gripper_finger1_joint_1",),
    right_control_joints=("right_gripper_finger1_joint_1",),
    left_mimic_joints=("left_gripper_finger2_joint_1",),
    right_mimic_joints=("right_gripper_finger2_joint_1",),
    mimic_multipliers=(1.0,),
    mimic_offsets=(0.0,),
    simulated_joint_initial_positions=(0.0, 0.0),
    open_positions=(0.0,),
    close_positions=(0.04,),
    control_limits=((0.0, 0.04),),
    drive_stiffness=1.0e3,
    drive_damping=1.0e2,
    drive_max_effort=1.0e4,
    grasp_model=GraspModelSpec(
        model_id="dh_pgi_140_80",
        min_opening_width=0.003,
        max_opening_width=0.100,
        finger_length=0.10,
        finger_width=0.040,
        finger_thickness=0.01,
        palm_depth=0.096,
        opening_margin=0.03,
    ),
)

_ROBOTIQ_MIMIC_MULTIPLIERS = (-1.0, 1.0, -1.0, -1.0, 1.0)
_ROBOTIQ_PROFILE = GripperProfile(
    model=GripperModel.ROBOTIQ,
    asset_path="Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf",
    assembly_name="robotiq_arg2f_140",
    tcp_transform=(
        (0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.2),
        (0.0, 0.0, 0.0, 1.0),
    ),
    left_control_joints=(
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "left_right_outer_knuckle_joint",
        "left_right_inner_knuckle_joint",
        "left_right_inner_finger_joint",
    ),
    right_control_joints=(
        "right_finger_joint",
        "right_left_inner_knuckle_joint",
        "right_left_inner_finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
    ),
    left_mimic_joints=(
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "left_right_outer_knuckle_joint",
        "left_right_inner_knuckle_joint",
        "left_right_inner_finger_joint",
    ),
    right_mimic_joints=(
        "right_left_inner_knuckle_joint",
        "right_left_inner_finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
    ),
    mimic_multipliers=_ROBOTIQ_MIMIC_MULTIPLIERS,
    mimic_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),
    simulated_joint_initial_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    open_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    close_positions=(0.7, -0.7, 0.7, -0.7, -0.7, 0.7),
    control_limits=(
        (0.0, 0.7),
        (-0.8757, 0.8757),
        (-0.8757, 0.8757),
        (-0.725, 0.725),
        (-0.8757, 0.8757),
        (-0.8757, 0.8757),
    ),
    drive_stiffness=50.0,
    drive_damping=5.0,
    drive_max_effort=500.0,
    grasp_model=GraspModelSpec(
        model_id="robotiq_arg2f_140",
        min_opening_width=0.01,
        max_opening_width=0.15,
        finger_length=0.13,
        finger_width=0.03,
        finger_thickness=0.01,
        palm_depth=0.08,
        opening_margin=0.01,
    ),
)

_GRIPPER_PROFILES = {
    GripperModel.PGI: _PGI_PROFILE,
    GripperModel.ROBOTIQ: _ROBOTIQ_PROFILE,
}


def get_gripper_profile(model: GripperModel | str) -> GripperProfile:
    """Return one strictly selected GenSim gripper profile."""
    if isinstance(model, GripperModel):
        selected = model
    elif isinstance(model, str):
        try:
            selected = GripperModel(model)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported gripper model {model!r}; expected one of: pgi, robotiq."
            ) from exc
    else:
        raise TypeError(
            f"Gripper model must be a string; expected one of: pgi, robotiq, got "
            f"{type(model).__name__}."
        )
    return _GRIPPER_PROFILES[selected]
