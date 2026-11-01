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

"""One Franka family configuration with explicit arm and tool identity."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Literal
import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    ArticulationRootPropertiesCfg,
    LinkPhysicsOverrideCfg,
    RigidBodyPhysicsCfg,
    NewtonCollisionPropertiesCfg,
    NewtonRigidBodyMaterialCfg,
)
from embodichain.lab.sim.motion.solvers import FEPSolverCfg, PinkSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass
from ._specs import combination

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["FrankaCfg"]


@configclass
class FrankaCfg(RobotCfg):
    """Configure a supported Franka arm and parallel-gripper combination.

    The default ``fr3/panda_hand`` preserves the historical FrankaPanda preset.
    ``panda/robotiq_2f85`` uses the original Panda chain and a coupled tool.
    Other cross-combinations require separate qualification and are rejected.
    Tool calibration belongs to the selected embodiment; it is applied once
    to both the assembled tool and solver-owned TCP. The integrated FR3 hand
    uses its authored TCP and accepts no external-tool calibration fields.
    """

    arm_model: Literal["panda", "fr3"] = "fr3"
    gripper: Literal["panda_hand", "robotiq_2f85"] = "panda_hand"
    gripper_mount_z_m: float | None = None
    gripper_mount_yaw_rad: float | None = None
    tcp_offset_m: float | None = None

    @classmethod
    def from_dict(cls, init_dict: dict) -> FrankaCfg:
        """Decode model selections and ordinary RobotCfg overrides.

        Args:
            init_dict: Explicit arm/tool selection, calibration and overrides.

        Returns:
            An independently owned complete configuration.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        values = init_dict or {}
        if "robot_type" in values:
            raise ValueError(
                "robot_type is legacy-only; FrankaCfg selects arm_model and gripper"
            )
        self.arm_model = values.get("arm_model", self.arm_model)
        self.gripper = values.get("gripper", self.gripper)
        if not isinstance(self.arm_model, str) or not isinstance(self.gripper, str):
            raise ValueError("Franka arm_model and gripper must be supported strings")
        spec = combination(self.arm_model, self.gripper)
        self.uid = spec["uid"]
        self.control_parts = {
            "arm": list(spec["arm_joints"]),
            "eef": list(spec["eef_joints"]),
        }
        self.init_qpos = spec["init_qpos"]
        self.joint_drive_props = JointDrivePropertiesCfg(**spec["drive"])
        calibration = ("gripper_mount_z_m", "gripper_mount_yaw_rad", "tcp_offset_m")
        if spec["hand_asset"] is None and any(
            values.get(name, getattr(self, name)) is not None for name in calibration
        ):
            raise ValueError(
                "integrated Panda Hand uses its authored mount/TCP; override solver_cfg explicitly"
            )
        components = [
            dict(
                component_type="arm",
                urdf_path=get_data_path(spec["arm_asset"]),
                transform=np.eye(4),
            )
        ]
        options = {}
        tcp = np.eye(4)
        if spec["hand_asset"] is not None:
            for name, default in zip(
                calibration, (0.0, 0.0, spec["tool"]["tcp_offset_m"])
            ):
                value = values.get(name, getattr(self, name))
                value = default if value is None else value
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite calibration parameter")
                setattr(self, name, float(value))
            if self.gripper_mount_z_m < 0 or self.tcp_offset_m <= 0:
                raise ValueError("gripper mounting/TCP distances are invalid")
            mount = np.eye(4)
            angle = self.gripper_mount_yaw_rad
            mount[:3, :3] = [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
            mount[2, 3] = self.gripper_mount_z_m
            components.append(
                dict(
                    component_type="hand",
                    urdf_path=get_data_path(spec["hand_asset"]),
                    transform=mount,
                )
            )
            options["fname"] = (
                "panda_robotiq85_"
                + hashlib.sha256(
                    mount.astype("<f8").tobytes()
                    + np.asarray([self.tcp_offset_m], dtype="<f8").tobytes()
                ).hexdigest()[:12]
            )
            tcp = mount.copy()
            tcp[:3, 3] += tcp[:3, :3] @ np.array([0.0, 0.0, self.tcp_offset_m])
            self.qpos_limits = spec["tool"]["qpos_limits"]
            self.root_props = ArticulationRootPropertiesCfg(
                fixed_base=True, min_position_iters=64, min_velocity_iters=16
            )
            from ._physics import make_inertial_overlays as arm_inertials
            from embodichain_robots.robotiq._physics import (
                make_inertial_overlays as tool_inertials,
            )

            self.link_attrs = {**arm_inertials(), **tool_inertials()}
        else:
            self.link_attrs = {
                "newton_gripper_contacts": LinkPhysicsOverrideCfg(
                    link_names_expr=["fr3_leftfinger|fr3_rightfinger"],
                    attrs=RigidBodyPhysicsCfg(
                        collision_props=NewtonCollisionPropertiesCfg(condim=4),
                        material_props=NewtonRigidBodyMaterialCfg(
                            ke=40000.0,
                            kd=400.0,
                            torsional_friction=0.1,
                            rolling_friction=0.01,
                        ),
                    ),
                )
            }
        self.urdf_cfg = URDFCfg(components=components, **options)
        solver_options = dict(
            root_link_name=spec["root_link"],
            end_link_name=spec["end_link"],
            tcp=tcp.tolist(),
        )
        self.solver_cfg = {
            "arm": (
                FEPSolverCfg(redundancy_search=True, **solver_options)
                if spec["solver"] == "fep"
                else PinkSolverCfg(
                    pos_eps=1e-5,
                    rot_eps=1e-4,
                    max_iterations=100,
                    show_ik_warnings=False,
                    **solver_options,
                )
            )
        }

    @property
    def _pk_urdf_path(self) -> str:
        return get_data_path(combination(self.arm_model, self.gripper)["arm_asset"])

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: object
    ) -> dict[str, pk.SerialChain]:
        """Build the selected arm chain; the solver applies the tool TCP.

        Args:
            device: Torch device for kinematics.
            **kwargs: Compatibility arguments for the robot configuration port.

        Returns:
            The named seven-joint arm chain with source-defined joint names.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        spec = combination(self.arm_model, self.gripper)
        return {
            "arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                device=device,
                end_link_name=spec["end_link"],
                root_link_name=spec["root_link"],
            )
        }
