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

"""Dual-arm robot provider for coordinated Atomic Skill benchmarks."""

from __future__ import annotations

from embodichain.lab.sim.cfg import RobotCfg
from scripts.tutorials.atomic_action.scenario_utils import (
    DUAL_UR5_INIT_POS,
    DUAL_UR5_INIT_ROT,
    create_dual_tutorial_robot_cfg,
)

from ..registry import register_robot_provider
from .base import RobotProvider

__all__ = ["DualArmProvider"]


def _optional_floats(value: object, name: str) -> list[float] | None:
    """Return an optional list of floats from a YAML value."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"robot.config.{name} must be a list of numbers.")
    return [float(item) for item in value]


class DualArmProvider(RobotProvider):
    """Build the tutorial dual-arm robot: two arms with one gripper each.

    The control parts are ``left_arm``/``right_arm`` and
    ``left_eef``/``right_eef``.  ``robot.config`` selects the arm family and
    the tutorial construction parameters, so each coordinated skill suite can
    reuse its tutorial's validated robot.
    """

    control_part = "left_arm"

    def build_cfg(self) -> RobotCfg:
        """Build the configured dual-arm robot."""
        config = dict(self.spec.config)
        robot_type = str(config.get("robot_type", "ur5"))
        solver = str(config.get("solver", "ur"))
        if solver not in {"ur", "pytorch"}:
            raise ValueError("robot.config.solver must be 'ur' or 'pytorch'.")
        return create_dual_tutorial_robot_cfg(
            robot_type=robot_type,
            uid=str(config.get("uid", f"benchmark_dual_{robot_type}")),
            urdf_name=str(config.get("urdf_name", f"benchmark_dual_{robot_type}")),
            tcp_z=float(config.get("tcp_z", 0.15)),
            solver=solver,
            ur_ik_nearest_weight=_optional_floats(
                config.get("ur_ik_nearest_weight"), "ur_ik_nearest_weight"
            ),
            pytorch_num_samples=int(config.get("pytorch_num_samples", 30)),
            init_pos=tuple(config.get("init_pos", DUAL_UR5_INIT_POS)),
            init_rot=tuple(config.get("init_rot", DUAL_UR5_INIT_ROT)),
            left_arm_home=_optional_floats(
                config.get("left_arm_home"), "left_arm_home"
            ),
            right_arm_home=_optional_floats(
                config.get("right_arm_home"), "right_arm_home"
            ),
            hand_stiffness=float(config.get("hand_stiffness", 1.0e3)),
            hand_damping=float(config.get("hand_damping", 1.0e2)),
            hand_max_effort=float(config.get("hand_max_effort", 1.0e4)),
        )


register_robot_provider("dual_arm", DualArmProvider)
