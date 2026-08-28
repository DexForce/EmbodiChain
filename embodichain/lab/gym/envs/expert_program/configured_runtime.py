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

"""Register configuration-defined Expert Program environments."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

from ._configured_runtime_decoder import (
    _ConfiguredExpertProgramRuntime,
    _identifier,
    _integer,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.utils.registration import EnvSpec

__all__: list[str] = []

_REGISTER_LOCK = Lock()


def _register_configured_expert_program_runtime(
    env_id: object,
    runtime: _ConfiguredExpertProgramRuntime,
    *,
    max_episode_steps: object,
) -> EnvSpec:
    """Register ``EmbodiedEnv`` under a config-owned ID, idempotently."""
    selected_env_id = _identifier(env_id, path="id")
    selected_max_episode_steps = _integer(
        max_episode_steps,
        path="max_episode_steps",
        minimum=1,
    )
    if type(runtime) is not _ConfiguredExpertProgramRuntime:
        raise TypeError("runtime must be exactly _ConfiguredExpertProgramRuntime.")

    from gymnasium.envs.registration import registry as gym_registry

    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        get_env_spec,
        register_env_function,
    )

    with _REGISTER_LOCK:
        existing = REGISTERED_ENVS.get(selected_env_id)
        if existing is None:
            if selected_env_id in gym_registry:
                raise ValueError(
                    f"Cannot register configured Expert Program environment "
                    f"{selected_env_id!r}: Gymnasium already owns that ID."
                )
            register_env_function(
                EmbodiedEnv,
                selected_env_id,
                max_episode_steps=selected_max_episode_steps,
                expert_program_adapter_factory=runtime.adapter_factory,
            )
            return get_env_spec(selected_env_id)

        existing_factory = existing.expert_program_adapter_factory
        existing_fingerprint = getattr(
            existing_factory,
            "configuration_fingerprint",
            None,
        )
        if (
            existing.cls is not EmbodiedEnv
            or existing.max_episode_steps != selected_max_episode_steps
            or existing_fingerprint != runtime.configuration_fingerprint
        ):
            raise ValueError(
                f"Configured Expert Program environment ID {selected_env_id!r} "
                "is already registered with a different environment or runtime "
                "configuration. Choose another id instead of overriding it."
            )
        if selected_env_id not in gym_registry:
            raise RuntimeError(
                f"Configured environment {selected_env_id!r} is missing from the "
                "Gymnasium registry."
            )
        existing_registration = existing.expert_program_registration
        if existing_registration is None:
            raise RuntimeError(
                f"Configured environment {selected_env_id!r} lost its Expert "
                "Program registration."
            )
        existing_registration.assert_unchanged()
        if existing_registration.fingerprint != runtime.registration.fingerprint:
            raise ValueError(
                f"Configured Expert Program environment ID {selected_env_id!r} "
                "is already registered with a different integration."
            )
        return existing
