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
from copy import deepcopy
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

    ``asset_source="menagerie"`` preserves legacy checkpoint construction.
    ``asset_source="default"`` composes the registered FlexivRizon4S and
    Robotiq2F85 URDF assets, retaining their visuals, collisions and mimic tree.
    Neither deployment is qualified as the author's identified closed-loop
    dynamics. ``require_exact_dynamics`` fails before simulation setup unless a
    separately qualified URDF and matching fidelity record are supplied.
    """

    gripper_model: str = "four_bar_mimic"
    """Explicit development model: four_bar_mimic or parallel_jaw."""
    source_urdf: str | None = None
    """Optional prepared replacement URDF with the same joint/link contract."""
    require_exact_dynamics: bool = False
    """Require a neighboring fidelity.json attesting exact source dynamics."""
    asset_source: str = "menagerie"
    """Asset selection: legacy Menagerie or registered default arm/hand assets."""
    grasp_link_names: tuple[str, str] = ("gripper_right_pad", "gripper_left_pad")
    """Opposing contact links, independent of their source-defined spelling."""
    grasp_aperture_coefficients: tuple[float, float, float] = (0.0057, 0.037, -0.044)
    """Half aperture in metres: constant + cosine*cos(q) + sine*sin(q)."""
    grasp_joint_upper: float = 0.8
    """Upper joint position used to initialize a grasp."""

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> FlexivRizon4SRobotiqCfg:
        """Build preset defaults and apply ordinary robot overrides."""
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        init_dict = init_dict or {}
        self.asset_source = init_dict.get("asset_source", self.asset_source)
        if self.asset_source not in {"menagerie", "default"}:
            raise ValueError("asset_source must be menagerie or default.")
        self.gripper_model = init_dict.get("gripper_model", self.gripper_model)
        if self.gripper_model not in {"four_bar_mimic", "parallel_jaw"}:
            raise ValueError("gripper_model must be four_bar_mimic or parallel_jaw.")
        if self.asset_source == "default" and self.gripper_model != "four_bar_mimic":
            raise ValueError("Default 2F85 assets retain their source mimic tree.")
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
        if self.asset_source == "default":
            self._build_default_assets(urdf)
            return
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

    def _build_default_assets(self, urdf: str) -> None:
        """Reuse the default arm configuration and the unconverted hand URDF."""
        from embodichain_robots.flexiv.rizon4s import FlexivRizon4SCfg

        arm = FlexivRizon4SCfg.from_dict({})
        self.uid = "FlexivRizon4SRobotiq"
        self.urdf_cfg = URDFCfg(
            components=[{"component_type": "arm", "urdf_path": urdf}]
        )
        self.control_parts = {
            "arm": list(arm.control_parts["arm"]),
            "eef": ["finger_joint"],
        }
        self.solver_cfg = {
            "arm": PytorchSolverCfg(
                end_link_name="gripper_tcp", root_link_name="base_link", num_samples=1
            )
        }
        self.root_props = deepcopy(arm.root_props)
        self.attrs = deepcopy(arm.attrs)
        self.joint_drive_props = deepcopy(arm.joint_drive_props)
        for field, value in (
            ("stiffness", 100.0),
            ("damping", 10.0),
            ("max_effort", 5.0),
            ("max_velocity", 2.0),
        ):
            getattr(self.joint_drive_props, field)["finger_joint"] = value
        self.grasp_link_names = ("right_inner_finger", "left_inner_finger")
        # Conservative inward bounds of the registered coarse finger meshes.
        self.grasp_aperture_coefficients = (
            0.004898087275028228,
            0.034345679203517304,
            -0.04567720361798315,
        )
        self.grasp_joint_upper = 0.725
        self.link_attrs = {
            "default_pad_contacts": LinkPhysicsOverrideCfg(
                link_names_expr=list(self.grasp_link_names),
                attrs=RigidBodyPhysicsCfg(
                    collision_props=NewtonCollisionPropertiesCfg(
                        contact_offset=0.0, rest_offset=0.0, condim=4
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
        self.init_qpos = list(arm.init_qpos) + [0.0] * 6

    @property
    def _pk_urdf_path(self) -> str:
        if self.source_urdf is not None:
            return get_data_path(self.source_urdf)
        if self.asset_source == "default":
            from ._default_assets import prepare_default_assets

            return prepare_default_assets()
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
                end_link_name=self.solver_cfg["arm"].end_link_name,
                root_link_name=self.solver_cfg["arm"].root_link_name,
            )
        }
