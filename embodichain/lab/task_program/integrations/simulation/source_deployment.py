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

"""Provider-free physical environment declarations for source Task Program deployments."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any

__all__ = ["source_task_program_environment_declaration"]


def source_task_program_environment_declaration(
    rigid_uids: Sequence[str],
    *,
    max_episodes: int = 1,
    max_episode_steps: int = 10000,
) -> dict[str, Any]:
    """Declare the source physical reset/cadence contract, without task semantics.

    Args:
        rigid_uids: All dynamic scene rigid identities in source order.
        max_episodes: Explicit positive episode limit.
        max_episode_steps: Explicit positive control-step limit.

    Returns:
        Independent Default environment with the source CCD, control cadence,
        arena and complete-scene settling declaration. No simulator is started.
    """
    if any(
        type(value) is not int or value <= 0
        for value in (max_episodes, max_episode_steps)
    ):
        raise ValueError("Source episode limits must be positive integers.")
    if len(set(rigid_uids)) != len(rigid_uids) or any(
        type(uid) is not str or not uid or uid != uid.strip() for uid in rigid_uids
    ):
        raise ValueError("Source reset requires unique exact rigid identities.")
    return dict(
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        num_envs=1,
        arena_space=2.5,
        physics="default",
        physics_config=dict(enable_ccd=True),
        env=dict(
            sim_steps_per_control=4,
            events=dict(
                settle_objects_on_reset=dict(
                    func="wait_for_dynamic_objects_to_settle",
                    mode="reset",
                    params=dict(
                        entity_cfgs=[dict(uid=uid) for uid in rigid_uids],
                        min_steps=10,
                        max_steps=600,
                        check_interval_steps=2,
                        required_stable_checks=3,
                        timeout_behavior="raise",
                    ),
                )
            ),
            extensions={},
        ),
    )
