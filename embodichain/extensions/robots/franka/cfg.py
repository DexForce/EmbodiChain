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

"""Franka arm configuration using the same external-tool assembly as UR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RobotCfg, URDFCfg, JointDrivePropertiesCfg
from embodichain.lab.sim.motion.solvers import FEPSolverCfg, PinkSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass
from ._specs import arm_spec

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["FrankaCfg"]


@configclass
class FrankaCfg(RobotCfg):
    """Configure a bare seven-DOF Panda or FR3 ending at its flange.

    End effectors use ordinary ``urdf_cfg.components`` overrides, exactly as
    for UR arms. Embodiments own the added control parts, tool drives, mount
    transform and ``solver_cfg.arm.tcp``. No tool is attached implicitly.
    """

    arm_model: Literal["panda", "fr3"] = "fr3"

    @classmethod
    def from_dict(cls, init_dict: dict) -> FrankaCfg:
        """Build arm defaults and apply ordinary assembly overrides.

        Args:
            init_dict: Arm selection and standard RobotCfg overrides.

        Returns:
            An independently owned arm or explicitly assembled configuration.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        values = init_dict or {}
        removed = {
            "gripper",
            "gripper_mount_z_m",
            "gripper_mount_yaw_rad",
            "tcp_offset_m",
        } & values.keys()
        if removed:
            raise ValueError(
                "FrankaCfg is arm-only; configure end effectors with urdf_cfg.components, control_parts and solver_cfg.arm.tcp instead of "
                + ", ".join(sorted(removed))
            )
        if "robot_type" in values:
            raise ValueError("robot_type is legacy-only; FrankaCfg selects arm_model")
        self.arm_model = values.get("arm_model", self.arm_model)
        spec = arm_spec(self.arm_model)
        self.uid = "Franka"
        self.control_parts = {"arm": list(spec["joints"])}
        self.init_qpos = spec["init_qpos"]
        self.joint_drive_props = JointDrivePropertiesCfg(**spec["drive"])
        self.urdf_cfg = URDFCfg(
            components=[
                dict(
                    component_type="arm",
                    urdf_path=get_data_path(spec["asset"]),
                    transform=np.eye(4),
                )
            ]
        )
        if self.arm_model == "panda":
            from ._physics import make_inertial_overlays

            self.link_attrs = make_inertial_overlays()
        options = dict(root_link_name=spec["root_link"], end_link_name=spec["end_link"])
        self.solver_cfg = {
            "arm": (
                FEPSolverCfg(redundancy_search=True, **options)
                if spec["solver"] == "fep"
                else PinkSolverCfg(
                    pos_eps=1e-5,
                    rot_eps=1e-4,
                    max_iterations=100,
                    show_ik_warnings=False,
                    **options,
                )
            )
        }

    @property
    def _pk_urdf_path(self) -> str:
        return get_data_path(arm_spec(self.arm_model)["asset"])

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: object
    ) -> dict[str, pk.SerialChain]:
        """Build the arm-local flange chain independently of mounted tools.

        Args:
            device: Torch device for kinematics.
            **kwargs: Compatibility arguments for the robot configuration port.

        Returns:
            The named seven-joint source chain. Solvers apply the configured TCP.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        spec = arm_spec(self.arm_model)
        return {
            "arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                device=device,
                end_link_name=spec["end_link"],
                root_link_name=spec["root_link"],
            )
        }
