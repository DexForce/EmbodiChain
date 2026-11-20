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

"""Configuration for the 12-joint Unitree Go2 quadruped."""

from __future__ import annotations

from typing import Any

from embodichain.utils import configclass

from ..._locomotion import _LocomotionRobotCfg, _apply_profile

__all__ = ["UnitreeGo2Cfg"]


@configclass
class UnitreeGo2Cfg(_LocomotionRobotCfg):
    """Configure the bundled 12-joint Unitree Go2 quadruped with its default PD drives.

    The preset preserves the existing locomotion simulation asset and standing
    pose. It provides the locomotion control group without an IK solver.
    Construct it with :meth:`from_dict` to assemble defaults and caller overrides.
    """

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        """Populate the model's asset, control group, pose and drive defaults."""
        _apply_profile(self, "UnitreeGo2", init_dict)
