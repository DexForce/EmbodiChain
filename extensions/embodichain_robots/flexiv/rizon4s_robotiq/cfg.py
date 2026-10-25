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
"""Nominal public Flexiv Rizon 4S with Robotiq 2F-85 development preset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
    LinkPhysicsOverrideCfg,
    RigidBodyPhysicsCfg,
    NewtonCollisionPropertiesCfg,
    NewtonRigidBodyMaterialCfg,
)
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["FlexivRizon4SRobotiqCfg"]


@configclass
class FlexivRizon4SRobotiqCfg(RobotCfg):
    """Public Rizon arm with an explicitly approximate parallel mimic gripper.

    The default assets preserve the pinned Menagerie arm model. The prepared
    Robotiq model cannot preserve its closed loops through the current URDF
    boundary; it is intended for development and is not the author's identified
    model. ``require_exact_dynamics`` fails before simulation setup unless a
    separately qualified URDF and matching fidelity record are supplied.
    """

    gripper_model: str = "four_bar_mimic"
    """Explicit development model: four_bar_mimic or parallel_jaw."""
    source_urdf: str | None = None
    """Optional prepared replacement URDF with the same joint/link contract."""
    require_exact_dynamics: bool = False
    """Require a neighboring fidelity.json attesting exact source dynamics."""

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> FlexivRizon4SRobotiqCfg:
        """Build preset defaults and apply ordinary robot overrides."""
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        init_dict = init_dict or {}
        self.gripper_model = init_dict.get("gripper_model", self.gripper_model)
        if self.gripper_model not in {"four_bar_mimic", "parallel_jaw"}:
            raise ValueError("gripper_model must be four_bar_mimic or parallel_jaw.")
        self.source_urdf = init_dict.get("source_urdf", self.source_urdf)
        self.require_exact_dynamics = init_dict.get(
            "require_exact_dynamics", self.require_exact_dynamics
        )
        urdf = self._pk_urdf_path
        if self.require_exact_dynamics:
            import json

            marker = Path(urdf).with_name("fidelity.json")
            fidelity = json.loads(marker.read_text()) if marker.is_file() else {}
            if not fidelity.get("exact_source_dynamics", False):
                raise ValueError(
                    "Exact assembly training requires a qualified Rizon4S/2F85 model: "
                    "the public URDF preset approximates Robotiq closed loops. "
                    "Supply source_urdf with neighboring fidelity.json containing "
                    "exact_source_dynamics=true after physical validation, or select "
                    "the explicitly approximate development deployment."
                )
        self.uid = "FlexivRizon4SRobotiq"
        self.urdf_cfg = URDFCfg(
            components=[{"component_type": "arm", "urdf_path": urdf}]
        )
        self.control_parts = {
            "arm": [f"joint{i}" for i in range(1, 8)],
            "eef": (
                ["gripper_right_jaw_joint", "gripper_left_jaw_joint"]
                if self.gripper_model == "parallel_jaw"
                else ["gripper_right_driver_joint"]
            ),
        }
        self.solver_cfg = {
            "arm": PytorchSolverCfg(
                end_link_name="gripper_tcp", root_link_name="base", num_samples=1
            )
        }
        stiffness = [289.0, 673.0, 224.0, 373.0, 237.0, 232.0, 186.0]
        damping = [61.0, 143.0, 36.0, 59.0, 13.0, 12.0, 9.9]
        armature = [3.17, 3.17, 1.38, 1.38, 0.13, 0.13, 0.13]
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            target_mode="position_velocity",
            stiffness={
                **{f"joint{i+1}": value for i, value in enumerate(stiffness)},
                "gripper_.*": (
                    5000.0 if self.gripper_model == "parallel_jaw" else 100.0
                ),
            },
            damping={
                **{f"joint{i+1}": value for i, value in enumerate(damping)},
                "gripper_.*": (10.0 if self.gripper_model == "parallel_jaw" else 10.0),
            },
            armature={f"joint{i+1}": value for i, value in enumerate(armature)},
            max_effort={
                **{
                    f"joint{i+1}": value
                    for i, value in enumerate([123, 123, 64, 64, 39, 39, 39])
                },
                "gripper_.*": (30.0 if self.gripper_model == "parallel_jaw" else 5.0),
            },
            max_velocity={
                "joint[1-7]": 2.0,
                "gripper_.*": (0.2 if self.gripper_model == "parallel_jaw" else 2.0),
            },
        )
        if self.gripper_model == "parallel_jaw":
            self.link_attrs = {
                "nominal_pad_contacts": LinkPhysicsOverrideCfg(
                    link_names_expr=["gripper_(left|right)_pad"],
                    attrs=RigidBodyPhysicsCfg(
                        collision_props=NewtonCollisionPropertiesCfg(
                            contact_offset=0.0,
                            rest_offset=0.0,
                            condim=4,
                        ),
                        material_props=NewtonRigidBodyMaterialCfg(
                            static_friction=1.0,
                            dynamic_friction=1.0,
                            ke=40000.0,
                            kd=40.0,
                            torsional_friction=0.02,
                            rolling_friction=0.001,
                        ),
                    ),
                )
            }
        # Source order: 7 arm joints then right/left four-bar tree joints.
        self.init_qpos = [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0] + [0.0] * (
            2 if self.gripper_model == "parallel_jaw" else 8
        )

    @property
    def _pk_urdf_path(self) -> str:
        if self.source_urdf is not None:
            return get_data_path(self.source_urdf)
        from ._menagerie import prepare_rizon4s_urdf

        return prepare_rizon4s_urdf(
            get_data_path("FlexivRizon4SMenagerie"), gripper_model=self.gripper_model
        )

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Return the same 7-DOF arm/TCP kinematics used by simulation."""
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        return {
            "arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                device=device,
                end_link_name="gripper_tcp",
                root_link_name="base",
            )
        }
