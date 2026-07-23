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

"""Deprecation shim — task environments have moved to ``embodichain_tasks``.

This module re-exports all task classes from the new ``embodichain_tasks``
package for backward compatibility. It will be removed in a future version.

.. deprecated::
    Import from ``embodichain_tasks`` directly instead of
    ``embodichain.lab.gym.envs.tasks``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "embodichain.lab.gym.envs.tasks is deprecated. "
    "Import from embodichain_tasks instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv  # noqa: F401
    from embodichain_tasks.tableware.pour_water.pour_water import (  # noqa: F401
        PourWaterEnv,
        PourWaterAgentEnv,
    )
    from embodichain_tasks.tableware.rearrangement import (  # noqa: F401
        RearrangementEnv,
        RearrangementAgentEnv,
    )
    from embodichain_tasks.tableware.stack_blocks_two import (
        StackBlocksTwoEnv,
    )  # noqa: F401
    from embodichain_tasks.tableware.stack_cups import StackCupsEnv  # noqa: F401
    from embodichain_tasks.tableware.scoop_ice import ScoopIce  # noqa: F401
    from embodichain_tasks.tableware.blocks_ranking_rgb import (  # noqa: F401
        BlocksRankingRGBEnv,
    )
    from embodichain_tasks.tableware.blocks_ranking_size import (  # noqa: F401
        BlocksRankingSizeEnv,
    )
    from embodichain_tasks.tableware.match_object_container import (  # noqa: F401
        MatchObjectContainerEnv,
    )
    from embodichain_tasks.tableware.place_object_drawer import (  # noqa: F401
        PlaceObjectDrawerEnv,
    )
    from embodichain_tasks.rl.push_cube import PushCubeEnv  # noqa: F401
    from embodichain_tasks.rl.basic.cart_pole import CartPoleEnv  # noqa: F401
    from embodichain_tasks.special.simple_task import SimpleTaskEnv  # noqa: F401

    __all__ = [
        "BaseAgentEnv",
        "PourWaterEnv",
        "PourWaterAgentEnv",
        "RearrangementEnv",
        "RearrangementAgentEnv",
        "StackBlocksTwoEnv",
        "StackCupsEnv",
        "ScoopIce",
        "BlocksRankingRGBEnv",
        "BlocksRankingSizeEnv",
        "MatchObjectContainerEnv",
        "PlaceObjectDrawerEnv",
        "PushCubeEnv",
        "CartPoleEnv",
        "SimpleTaskEnv",
    ]
except ImportError:
    # embodichain_tasks is not installed — tasks will be discovered via
    # entry_points instead.
    __all__: list[str] = []
