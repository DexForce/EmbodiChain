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

"""CobotMagic robot definition using the RobotDef protocol."""

from __future__ import annotations

import torch
import numpy as np

from typing import Dict, Union

from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.solvers import SolverCfg, OPWSolverCfg
from embodichain.lab.sim.robots.protocol import RobotDef
from embodichain.lab.sim.robots.registry import register_robot
from embodichain.data import get_data_path

__all__ = ["CobotMagicDef", "CobotMagicCfg"]


@register_robot("CobotMagic")
class CobotMagicDef:
    """Robot definition for the CobotMagic dual-arm robot.

    This class satisfies the :class:`~embodichain.lab.sim.robots.protocol.RobotDef`
    protocol and is registered in the global robot registry under the name
    ``"CobotMagic"``.

    Attributes:
        name: Unique identifier string for this robot.
        urdf_cfg: URDF assembly configuration for both arms.
        control_parts: Mapping from part name to joint name lists.
        solver_cfg: OPW IK solver configuration for each arm.
        drive_pros: Joint drive properties (stiffness, damping, max_effort).
        attrs: Rigid-body physics attributes.
        min_position_iters: Minimum position iterations for the solver.
        min_velocity_iters: Minimum velocity iterations for the solver.
    """

    name: str = "CobotMagic"

    # -- URDF configuration ----------------------------------------------------

    @property
    def urdf_cfg(self) -> URDFCfg:
        """URDF assembly configuration for both arms."""
        arm_urdf = get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf")
        left_arm_xpos = np.array(
            [
                [1.0, 0.0, 0.0, 0.233],
                [0.0, 1.0, 0.0, 0.300],
                [0.0, 0.0, 1.0, 0.000],
                [0.0, 0.0, 0.0, 1.000],
            ]
        )
        right_arm_xpos = np.array(
            [
                [1.0, 0.0, 0.0, 0.233],
                [0.0, 1.0, 0.0, -0.300],
                [0.0, 0.0, 1.0, 0.000],
                [0.0, 0.0, 0.0, 1.000],
            ]
        )
        return URDFCfg(
            components=[
                {
                    "component_type": "left_arm",
                    "urdf_path": arm_urdf,
                    "transform": left_arm_xpos,
                },
                {
                    "component_type": "right_arm",
                    "urdf_path": arm_urdf,
                    "transform": right_arm_xpos,
                },
            ]
        )

    # -- Control parts ---------------------------------------------------------

    control_parts: dict[str, list[str]] = {
        "left_arm": [
            "LEFT_JOINT1",
            "LEFT_JOINT2",
            "LEFT_JOINT3",
            "LEFT_JOINT4",
            "LEFT_JOINT5",
            "LEFT_JOINT6",
        ],
        "left_eef": ["LEFT_JOINT7", "LEFT_JOINT8"],
        "right_arm": [
            "RIGHT_JOINT1",
            "RIGHT_JOINT2",
            "RIGHT_JOINT3",
            "RIGHT_JOINT4",
            "RIGHT_JOINT5",
            "RIGHT_JOINT6",
        ],
        "right_eef": ["RIGHT_JOINT7", "RIGHT_JOINT8"],
    }

    # -- Solver configuration --------------------------------------------------

    solver_cfg: dict[str, SolverCfg] = {
        "left_arm": OPWSolverCfg(
            end_link_name="left_link6",
            root_link_name="left_arm_base",
            tcp=np.array(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]]
            ),
        ),
        "right_arm": OPWSolverCfg(
            end_link_name="right_link6",
            root_link_name="right_arm_base",
            tcp=np.array(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]]
            ),
        ),
    }

    # -- Drive properties ------------------------------------------------------

    drive_pros: JointDrivePropertiesCfg = JointDrivePropertiesCfg(
        stiffness={
            "LEFT_JOINT[1-6]": 7e4,
            "RIGHT_JOINT[1-6]": 7e4,
            "LEFT_JOINT[7-8]": 3e2,
            "RIGHT_JOINT[7-8]": 3e2,
        },
        damping={
            "LEFT_JOINT[1-6]": 1e3,
            "RIGHT_JOINT[1-6]": 1e3,
            "LEFT_JOINT[7-8]": 3e1,
            "RIGHT_JOINT[7-8]": 3e1,
        },
        max_effort={
            "LEFT_JOINT[1-6]": 3e6,
            "RIGHT_JOINT[1-6]": 3e6,
            "LEFT_JOINT[7-8]": 3e3,
            "RIGHT_JOINT[7-8]": 3e3,
        },
    )

    # -- Rigid-body attributes -------------------------------------------------

    attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg(
        mass=0.1,
        static_friction=0.95,
        dynamic_friction=0.9,
        linear_damping=0.7,
        angular_damping=0.7,
        contact_offset=0.001,
        rest_offset=0.001,
        restitution=0.01,
        max_depenetration_velocity=1e1,
    )

    # -- Extra fields ----------------------------------------------------------

    min_position_iters: int = 8
    min_velocity_iters: int = 2

    # -- Methods ---------------------------------------------------------------

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        """Build pytorch-kinematics serial chains for each arm.

        Args:
            device: Torch device to place chains on.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary mapping ``"left_arm"`` and ``"right_arm"`` to their
            respective serial chain objects.
        """
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_chain,
            create_pk_serial_chain,
        )

        urdf_path = get_data_path("CobotMagicArm/CobotMagicNoGripper.urdf")
        chain = create_pk_chain(urdf_path, device)

        left_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="link6", root_link_name="base_link"
        ).to(device=device)
        right_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="link6", root_link_name="base_link"
        ).to(device=device)
        return {"left_arm": left_arm_chain, "right_arm": right_arm_chain}

    def build_cfg(self, **overrides: object) -> RobotCfg:
        """Build a :class:`RobotCfg` from this robot definition.

        Delegates to :meth:`RobotDef.build_cfg` for the default
        implementation.

        Args:
            **overrides: Optional overrides applied on top of the defaults.

        Returns:
            A fully-populated :class:`RobotCfg`.
        """
        return RobotDef.build_cfg(self, **overrides)


class CobotMagicCfg(RobotCfg):
    """Backward-compatible wrapper around :class:`CobotMagicDef`.

    Existing code that calls ``CobotMagicCfg.from_dict(...)`` continues to
    work via this thin shim that delegates to the registered
    :class:`CobotMagicDef`.
    """

    urdf_cfg: URDFCfg = None
    control_parts: Dict[str, list[str]] | None = None
    solver_cfg: Dict[str, "SolverCfg"] | None = None

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Union[str, float, int]]) -> CobotMagicCfg:
        """Create a :class:`RobotCfg` via :class:`CobotMagicDef`.

        Args:
            init_dict: Dictionary of configuration overrides.

        Returns:
            A fully-populated :class:`RobotCfg` with overrides applied.
        """
        return CobotMagicDef().build_cfg(**init_dict)


if __name__ == "__main__":
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(
        headless=False,
        sim_device="cpu",
        num_envs=2,
        render_cfg=RenderCfg(renderer="fast-rt"),
    )
    sim = SimulationManager(config)

    overrides = {
        "init_pos": [0.0, 0.0, 1.0],
    }

    cfg = CobotMagicDef().build_cfg(**overrides)
    robot = sim.add_robot(cfg=cfg)

    print("CobotMagic added to the simulation.")

    from IPython import embed

    embed()
