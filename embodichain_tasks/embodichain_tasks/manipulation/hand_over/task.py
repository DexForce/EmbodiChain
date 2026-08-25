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

"""Dual-UR5 hand-over task and Gym registration."""

from __future__ import annotations

import json
from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramEnvironmentAdapter,
    create_simulation_expert_program_adapter,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import register_env

from .._expert import _task_config_path, create_parallel_jaw_grasp_pose_generator
from .expert.binding import (
    HAND_OVER_EXPERT_PROGRAM_REGISTRATION,
    HAND_OVER_GRASP_SAMPLE_COUNT,
)

__all__ = ["HandOverEnv"]

ENV_ID = "HandOver-v1"


def _create_default_env_cfg() -> EmbodiedEnvCfg:
    """Load the directly-instantiable task configuration from package data."""
    path = _task_config_path("manipulation", "hand_over", "env.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return config_to_cfg(payload, source_path=path)


@register_env(
    ENV_ID,
    max_episode_steps=1200,
    expert_program_registration=HAND_OVER_EXPERT_PROGRAM_REGISTRATION,
)
class HandOverEnv(EmbodiedEnv):
    """Transfer a can between two UR5 arms through a semantic program."""

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the configured scene and Expert Program adapter.

        Args:
            cfg: Environment configuration. Package data is loaded when omitted.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.
        """
        if cfg is None:
            cfg = _create_default_env_cfg()
        super().__init__(cfg, **kwargs)
        grasp_pose_generators = {
            f"{side}_hand": create_parallel_jaw_grasp_pose_generator(
                sample_count=HAND_OVER_GRASP_SAMPLE_COUNT,
                opening_margin=0.002,
            )
            for side in ("left", "right")
        }
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            registration=HAND_OVER_EXPERT_PROGRAM_REGISTRATION,
            grasp_pose_generators=grasp_pose_generators,
        )

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the production adapter used by ``EmbodiedEnv``.

        Returns:
            Adapter that compiles and executes the configured Expert Program.
        """
        return self._expert_program_adapter
