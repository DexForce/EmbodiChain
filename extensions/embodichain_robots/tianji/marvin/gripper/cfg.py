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

"""Standalone TianjiMarvinGripper and its shared assembly/grasp defaults."""

from __future__ import annotations

from embodichain.lab.sim.cfg.gripper import ParallelJawGripperCfg
from embodichain.utils import configclass
from ._gripper_defaults import get_gripper_defaults

__all__ = ["TianjiMarvinGripperCfg"]


@configclass
class TianjiMarvinGripperCfg(ParallelJawGripperCfg):
    """Configure the exact source hand independently of a mounting robot."""

    @classmethod
    def get_gripper_defaults(cls, variant: str | None = None) -> dict[str, object]:
        """Return model-owned data without resolving the source asset."""
        return get_gripper_defaults(variant)
