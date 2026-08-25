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

"""Open Drawer task and Gym registration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import ExpertProgramEnvironmentAdapter
from embodichain.lab.gym.utils.registration import register_env

from .._expert import load_bundled_expert_program
from .expert.binding import (
    DEFAULT_TRANSLATION_AXIS,
    create_open_drawer_expert_program_adapter,
)

__all__ = ["ExpertProgramOpenDrawerEnv"]

ENV_ID = "ExpertProgramOpenDrawer-v1"


@register_env(ENV_ID, max_episode_steps=600)
class ExpertProgramOpenDrawerEnv(EmbodiedEnv):
    """Open a passive drawer through Expert Program and the atomic Slide skill."""

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the configured task and its Expert Program adapter.

        Args:
            cfg: Environment configuration. The bundled program is installed
                when ``cfg.expert_program`` is unset.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.

        Raises:
            TypeError: If task extensions are not a mapping.
        """
        if cfg.expert_program is None:
            cfg.expert_program = load_bundled_expert_program(
                "manipulation", "open_drawer"
            )
        extensions = self._extensions(cfg.extensions)
        translation_axis = extensions.get("translation_axis", DEFAULT_TRANSLATION_AXIS)

        super().__init__(cfg, **kwargs)
        self._expert_program_adapter = create_open_drawer_expert_program_adapter(
            self,
            translation_axis=translation_axis,
        )

    @staticmethod
    def _extensions(value: object) -> Mapping[str, object]:
        """Return task extensions as a strict mapping."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("cfg.extensions must be a mapping.")
        return value

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the production adapter used by ``EmbodiedEnv``.

        Returns:
            Adapter that compiles and executes the configured Expert Program.
        """
        return self._expert_program_adapter
