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

"""ANYbotics ANYmal-C flat-ground velocity task."""

from __future__ import annotations

from embodichain.lab.gym.utils.registration import register_env
from embodichain.learning.rl.policy_evaluation.camera import PolicyViewerCameraCfg

from ._embodichain import EmbodiChainVelocityEnv
from .contracts.anymal_c.config import load_config
from .contracts.anymal_c.mdp import (
    ANYmalCState,
    build_observations,
    compute_rewards,
    compute_termination,
)

__all__ = ["ANYmalCFlatEnv"]

_CONFIG = load_config()


@register_env(
    "ANYmalCFlatRL-v1",
    max_episode_steps=_CONFIG.max_episode_steps,
    override=True,
    supports_rl=True,
)
class ANYmalCFlatEnv(EmbodiChainVelocityEnv):
    """Track planar velocity commands with the 12-DOF ANYmal-C model."""

    policy_viewer_camera_cfg = PolicyViewerCameraCfg(
        eye_offset=(-1.65, -1.3, 0.9), target_height=0.43
    )
    velocity_task_config = _CONFIG
    state_type = ANYmalCState
    build_observations_fn = staticmethod(build_observations)
    compute_rewards_fn = staticmethod(compute_rewards)
    compute_termination_fn = staticmethod(compute_termination)
    foot_link_names = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")
    foot_offsets = ((0.0, 0.0, 0.0),) * 4
    illegal_contact_link_names = (
        "base",
        "LF_THIGH",
        "LH_THIGH",
        "RF_THIGH",
        "RH_THIGH",
    )
    orientation_link_name = "base"
