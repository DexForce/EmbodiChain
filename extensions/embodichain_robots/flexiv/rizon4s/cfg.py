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

"""Flexiv Rizon 4S with published dynamics, tuned damping and a bare flange."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["FlexivRizon4SCfg"]


@configclass
class FlexivRizon4SCfg(RobotCfg):
    """Configure the seven-axis Rizon 4S without an attached gripper.

    Source kinematics, mass, COM, inertia and joint limits come from Flexiv's
    published robot description. Drive stiffness, reflected rotor inertia and
    root solver iterations come from Flexiv's Isaac Sim workspace. Damping is
    three times its published simulation reference to suppress rebound during
    native IK dragging on the Default backend. Public inertial parameters are
    blurred by the manufacturer; these simulation defaults are not a calibrated
    model of the real robot or its controller.

    ``arm`` contains ``joint1`` through ``joint7``. FK and Warp RR IK use
    ``base_link`` to ``flange`` with an identity TCP. An explicit URDF
    ``fpath`` override is shared by simulation and serial-chain construction.

    Example:
        >>> cfg = FlexivRizon4SCfg.from_dict({})
        >>> robot = sim.add_robot(cfg=cfg)
    """

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> FlexivRizon4SCfg:
        """Build the Rizon 4S defaults and merge caller overrides.

        Args:
            init_dict: Robot fields overriding the preset, including an
                optional local URDF ``fpath``.

        Returns:
            A Rizon 4S robot configuration.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        cfg = merge_robot_cfg(cfg, init_dict)
        # The application may attach tools through ordinary URDF components.
        # Simulation and kinematics must use that same assembled source.
        cfg.fpath = cfg.urdf_cfg.assemble_urdf()
        for solver in cfg.solver_cfg.values():
            if hasattr(solver, "urdf_path"):
                solver.urdf_path = cfg.fpath
        return cfg

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        """Populate source dynamics, tuned simulation drives and the ready pose."""
        init_dict = init_dict or {}
        self.uid = "FlexivRizon4S"
        self.fpath = get_data_path(
            init_dict.get("fpath") or "FlexivRizon4S/rizon4s.urdf"
        )
        self.urdf_cfg = URDFCfg(
            components=[{"component_type": "arm", "urdf_path": self.fpath}],
            sensors=[],
        )
        self.control_parts = {"arm": [f"joint{i}" for i in range(1, 8)]}
        self.solver_cfg = {
            "arm": SSIKSolverCfg(
                geometry_mode="rr",
                rr_search="seed_local",
                urdf_path=self._pk_urdf_path,
                root_link_name="base_link",
                end_link_name="flange",
            )
        }
        # Flexiv isaac_sim_ws 9378b06f, assets/rizon_4s/payloads/Physics/.
        # physics.usda angular gains multiply by 180/pi for per-radian units.
        # Its damping reference (80/70/15) is scaled by three for the Default
        # backend: the source gains cause repeated physical rebound during IK
        # dragging. Armature from physx.usda remains in kg*m^2. Speeds follow
        # the published URDF in rad/s for native drive enforcement.
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={
                "joint[1-2]": 4000.0,
                "joint[3-4]": 5000.0,
                "joint[5-7]": 1500.0,
            },
            damping={"joint[1-2]": 240.0, "joint[3-4]": 210.0, "joint[5-7]": 45.0},
            max_effort={"joint[1-2]": 123.0, "joint[3-4]": 64.0, "joint[5-7]": 39.0},
            max_velocity={
                "joint[1-2]": 2.0944,
                "joint[3-4]": 2.4435,
                "joint[5-7]": 4.8869,
            },
            friction=0.0,
            armature={
                "joint[1-2]": 3.169792,
                "joint[3-4]": 1.375488,
                "joint[5-7]": 0.134325,
            },
        )
        self.attrs = RigidBodyPhysicsCfg()
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True,
            self_collision_enabled=False,
            min_position_iters=32,
            min_velocity_iters=16,
        )
        # Official flexiv_description/config/Rizon4s/initial_positions.yaml.
        self.init_qpos = [0.0, -0.69813, 0.0, 1.570796, 0.0, 0.69813, 0.0]

    @property
    def _pk_urdf_path(self) -> str:
        """URDF shared by the simulator and the differentiable serial chain."""
        return self.fpath

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build the seven-joint flange chain from the simulation's source URDF.

        Args:
            device: Device for the chain's tensors.
            **kwargs: Reserved for the common robot configuration protocol.

        Returns:
            The ``arm`` chain with the configured solver root and end links.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        solver = self.solver_cfg["arm"]
        chain = create_pk_serial_chain(
            urdf_path=self._pk_urdf_path,
            device=device,
            root_link_name=solver.root_link_name,
            end_link_name=solver.end_link_name,
        )
        return {"arm": chain}
