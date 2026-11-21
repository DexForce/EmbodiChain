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

"""Standalone fixed-base BrainCo Revo1 hand with six joint commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

import embodichain.data as data
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
)
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

from ._specs import get_side_spec

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["BrainCoRevo1Cfg"]


@configclass
class BrainCoRevo1Cfg(RobotCfg):
    """Configure a left or right hand without imposing a grasping interface.

    ``eef`` commands the six independent source joints. Four additional PIP
    joints retain their source mimic relations. The source position limits,
    mass and inertia are preserved; drives are initial simulation defaults,
    not manufacturer motor specifications or hardware calibration.

    Build the asset defaults with :meth:`from_dict`. No FK/IK solver or single
    tool TCP is declared for this branched hand. Reset starts active joints at
    zero; the backend's existing mimic constraints govern passive followers.
    The fixed palm starts 0.3 m above the ground so the source finger geometry
    clears the default floor; callers can override the root pose.
    """

    side: Literal["left", "right"] = "right"
    """Native hand side; independent of any mounting robot's arm side."""

    def __post_init__(self) -> None:
        """Reject unknown sides before resolving or downloading an asset."""
        get_side_spec(self.side)

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> BrainCoRevo1Cfg:
        """Build hand defaults and apply isolated caller overrides.

        Args:
            init_dict: Configuration overrides, including ``side`` and optional
                local ``fpath`` to bypass the bundled asset download.

        Returns:
            The configured standalone hand.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        init_dict = init_dict or {}
        self.side = init_dict.get("side", self.side)
        spec = get_side_spec(self.side)
        self.uid = f"BrainCoRevo1{self.side.title()}"
        self.fpath = init_dict.get("fpath") or data.get_data_path(spec.urdf_path)
        self.init_pos = [0.0, 0.0, 0.3]
        self.control_parts = {"eef": list(spec.joint_names)}
        self.init_qpos = None
        self.solver_cfg = {}
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"eef": 100.0},
            damping={"eef": 10.0},
            max_effort={"eef": 1000.0},
            max_velocity={"eef": 1.0},
        )
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True,
            min_position_iters=64,
            min_velocity_iters=16,
        )
        self.attrs = RigidBodyPhysicsCfg()

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Return no serial chains for the branched joint-controlled hand.

        Args:
            device: Reserved for the RobotCfg protocol.
            **kwargs: Reserved for the RobotCfg protocol.

        Returns:
            An empty mapping; no finger FK/IK capability is configured.
        """
        return {}
