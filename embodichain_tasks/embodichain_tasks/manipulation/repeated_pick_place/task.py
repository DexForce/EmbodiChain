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

"""Repeated pick/place task and Gym registration."""

from __future__ import annotations

from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramEnvironmentAdapter,
    create_simulation_expert_program_adapter,
)
from embodichain.lab.gym.utils.registration import register_env

from .._expert import (
    HAND_CONTROL_PART,
    create_parallel_jaw_grasp_pose_generator,
    load_bundled_expert_program,
)
from .expert.binding import REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION

__all__ = ["ExpertProgramRepeatedPickPlaceEnv"]

ENV_ID = "ExpertProgramRepeatedPickPlace-v1"
DEFAULT_GRASP_SAMPLES = 10_000


@register_env(
    ENV_ID,
    max_episode_steps=1200,
    expert_program_registration=REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION,
)
class ExpertProgramRepeatedPickPlaceEnv(EmbodiedEnv):
    """Run three declarative Pick/Place cycles through the semantic runtime."""

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the configured task and its Expert Program adapter.

        Args:
            cfg: Environment configuration. The bundled program is installed
                when ``cfg.expert_program`` is unset.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.
        """
        if cfg.expert_program is None:
            cfg.expert_program = load_bundled_expert_program(
                "manipulation", "repeated_pick_place"
            )

        super().__init__(cfg, **kwargs)

        grasp_pose_generator = create_parallel_jaw_grasp_pose_generator(
            sample_count=DEFAULT_GRASP_SAMPLES,
            opening_margin=0.002,
        )
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            registration=REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION,
            grasp_pose_generators={HAND_CONTROL_PART: grasp_pose_generator},
        )

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the production adapter used by ``EmbodiedEnv``.

        Returns:
            Adapter that compiles and executes the configured Expert Program.
        """
        return self._expert_program_adapter
