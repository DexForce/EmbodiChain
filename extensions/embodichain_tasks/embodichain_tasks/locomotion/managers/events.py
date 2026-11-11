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

"""Reset-time physical context for G1 dynamics learning."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from embodichain_tasks.locomotion.tracking.g1_motion import UnitreeG1MotionEnv

__all__ = ["randomize_motion_context"]


def randomize_motion_context(env: UnitreeG1MotionEnv, env_ids: torch.Tensor) -> None:
    """Set ground friction, joint encoder bias and torso COM for reset rows.

    The sampled context is retained by the task and stored with each measured
    replay transition. The physics scene applies pending property writes before
    the reset state is installed.
    """
    ids = env.episode_context._ids(env_ids)
    if not len(ids):
        return
    generator = env.get_generator("motion_context")
    draw = lambda width: torch.rand(
        (len(ids), width), device=env.device, generator=generator
    )
    env._friction[ids] = 0.3 + 0.9 * draw(1)
    env.position_action.position_bias[ids] = (2 * draw(env.task.dof) - 1) * 0.01
    env._com_offset[ids] = (2 * draw(3) - 1) * env._com_offset.new_tensor(
        [0.025, 0.05, 0.05]
    )
    env.sim.get_rigid_object("floor").set_friction(env._friction[ids, 0], env_ids=ids)
    com = env._default_torso_com[ids].clone()
    com[:, :, :3] += env._com_offset[ids, None]
    env.robot.set_com_pose(com, link_names=["torso_link"], env_ids=ids)
    env.sim.prepare()
