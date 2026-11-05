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

"""Standalone fixed-base SharpaWave hand with 22 independent joint commands."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
)
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["SharpaWaveCfg"]


@configclass
class SharpaWaveCfg(RobotCfg):
    """Configure a native left/right hand using joint position commands.

    ``eef`` contains the 22 independent source joints in URDF order.
    Native names and physical values remain source-owned. Source effort/velocity
    limits and link inertials are explicitly reapplied through portable fields
    to preserve them across native import. Force-drive gains (stiffness 3,
    damping 0.1) are simulation defaults, not manufacturer calibration. All 22 joints start at zero, within the source position limits.

    Build defaults with :meth:`from_dict`. The fixed palm starts 0.3 m above
    the floor. No single TCP, finger IK, grasp or retargeting service is declared.
    """

    side: Literal["left", "right"] = "right"
    """Native hand side, independent of a mounting robot's arm side."""

    def __post_init__(self) -> None:
        """Reject an unsupported side before asset lookup."""
        if self.side not in ("left", "right"):
            raise ValueError(
                f"Unsupported hand side {self.side!r}; expected 'left' or 'right'."
            )

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> SharpaWaveCfg:
        """Build native defaults and apply isolated caller overrides.

        Args:
            init_dict: Overrides including ``side``, an optional local ``fpath``
                and caller-defined control groups or physical settings.

        Returns:
            The configured standalone hand.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        init_dict = init_dict or {}
        self.side = init_dict.get("side", self.side)
        self.__post_init__()
        self.uid = f"SharpaWave{self.side.title()}"
        self.fpath = init_dict.get("fpath") or get_data_path(
            f"SharpaWave/{self.side}_sharpa_wave/{self.side}_sharpa_wave.urdf"
        )
        self.init_pos = [0.0, 0.0, 0.3]
        source = ET.parse(self.fpath)
        joints = [
            joint for joint in source.findall("joint") if joint.get("type") != "fixed"
        ]
        self.control_parts = {"eef": [joint.get("name") for joint in joints]}
        self.init_qpos = [
            max(
                float(joint.find("limit").get("lower")),
                min(0.0, float(joint.find("limit").get("upper"))),
            )
            for joint in joints
        ]
        self.solver_cfg = {}
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=3.0,
            damping=0.1,
            max_effort={
                joint.get("name"): float(joint.find("limit").get("effort"))
                for joint in joints
            },
            max_velocity={
                joint.get("name"): float(joint.find("limit").get("velocity"))
                for joint in joints
            },
        )
        self.root_props = ArticulationRootPropertiesCfg(fixed_base=True)
        self.attrs = RigidBodyPhysicsCfg()
        # Explicit source overlays also preserve inertials on small/visual-only
        # links whose native import defaults can differ from the authored URDF.
        self.link_attrs = {}
        for link in source.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            origin = inertial.find("origin")
            xyz = [
                float(value)
                for value in (
                    origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
                ).split()
            ]
            rpy = [
                float(value)
                for value in (
                    origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
                ).split()
            ]
            tensor = inertial.find("inertia")
            matrix = np.array(
                [
                    [
                        float(tensor.get("i" + a + b) or tensor.get("i" + b + a))
                        for b in "xyz"
                    ]
                    for a in "xyz"
                ]
            )
            rotation = Rotation.from_euler("xyz", rpy).as_matrix()
            name = link.get("name")
            self.link_attrs[name] = LinkPhysicsOverrideCfg(
                link_names_expr=[re.escape(name)],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(
                        mass=float(inertial.find("mass").get("value")),
                        inertia=(rotation @ matrix @ rotation.T).tolist(),
                        com_position=xyz,
                    )
                ),
            )

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Return no serial chains for this branched joint-controlled hand.

        Args:
            device: Reserved for the RobotCfg protocol.
            **kwargs: Reserved for the RobotCfg protocol.

        Returns:
            An empty mapping; no finger FK/IK solver is configured.
        """
        return {}
