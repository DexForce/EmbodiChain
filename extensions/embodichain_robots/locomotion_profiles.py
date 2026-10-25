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

"""Backend-independent descriptions of the bundled locomotion robot models."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any

__all__ = ["load_locomotion_robot_profile"]


@lru_cache(maxsize=1)
def _profiles() -> dict[str, dict[str, Any]]:
    resource = files("embodichain_robots").joinpath("_locomotion_profiles.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_locomotion_robot_profile(robot_id: str) -> dict[str, Any]:
    """Load an isolated model description without importing simulation code.

    Args:
        robot_id: Exact bundled locomotion robot catalog ID.

    Returns:
        A fresh mapping containing ``asset`` construction defaults, ``robot``
        joint order, standing pose and drive parameters, and ``frames`` model
        reference points. Joint positions in ``asset.init_qpos`` follow source
        asset order; ``robot.default_joint_position`` follows ``robot.joint_names``.
        Profiles describe the existing simulation models, not hardware calibration.

    Raises:
        ValueError: The ID has no bundled locomotion profile.
    """
    try:
        return deepcopy(_profiles()[robot_id])
    except KeyError as error:
        raise ValueError(f"Unknown locomotion robot profile {robot_id!r}.") from error
