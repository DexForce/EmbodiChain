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

"""Joint-control preset for the original AgiBot G1_120s USD."""

from __future__ import annotations

from typing import Any

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    RobotCfg,
    UsdSourceCfg,
)
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

from .assets import _ROBOT_USD

__all__ = ["AgibotG1Cfg"]


@configclass
class AgibotG1Cfg(RobotCfg):
    """Configure the original G1_120s USD for Default joint control.

    The preset preserves the authored mass, limits, drives, mimic joints and
    closed-loop grippers. Each end-effector group commands its outer driver
    joint. Passive and mimic joints follow the source mechanism. Build the
    defaults with :meth:`from_dict` or resolve the ``AgibotG1`` catalog ID.
    """

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> AgibotG1Cfg:
        """Build G1 defaults and apply caller overrides.

        Args:
            init_dict: Robot fields overriding the preset. A local USD
                ``source`` can point to an existing complete G1_120s bundle.

        Returns:
            A configured USD-backed robot.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        overrides = init_dict or {}
        self.uid = "AgibotG1"
        if "source" not in overrides and not overrides.get("fpath"):
            self.source = UsdSourceCfg(
                file_path=get_data_path(f"AgibotG1/{_ROBOT_USD}"),
                source_prim_path="/G1",
            )
        self.asset_physics_mode = "preserve"
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=None, self_collision_enabled=None
        )
        self.joint_drive_props = None
        self.build_pk_chain = False
        self.control_parts = {
            "torso": ["idx01_body_joint1", "idx02_body_joint2"],
            "head": ["idx11_head_joint1", "idx12_head_joint2"],
            "left_arm": [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)],
            "right_arm": [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)],
            "left_eef": ["idx41_gripper_l_outer_joint1"],
            "right_eef": ["idx81_gripper_r_outer_joint1"],
        }
