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

"""Strict decoding and process-local registration of configuration-owned RL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from threading import Lock
from typing import Any

from embodichain.lab.gym.envs.managers.cfg import (
    ConditionCfg,
    RLTaskCfg,
    SceneEntityCfg,
)
from embodichain.utils.module_utils import find_function_from_modules

__all__: list[str] = []

_LOCK = Lock()


def _decode_rl_task(value: object, modules: Sequence[str]) -> RLTaskCfg:
    if not isinstance(value, Mapping) or set(value) - {
        "success",
        "failure",
        "truncation",
    }:
        raise ValueError(
            "rl_task accepts only success, failure and truncation mappings."
        )
    groups = {}
    for group in ("success", "failure", "truncation"):
        entries = value.get(group, {})
        if not isinstance(entries, Mapping):
            raise TypeError(f"rl_task.{group} must be a mapping.")
        groups[group] = {}
        for name, declaration in entries.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("Condition names must be nonempty strings.")
            if (
                not isinstance(declaration, Mapping)
                or set(declaration) - {"func", "params"}
                or "func" not in declaration
            ):
                raise ValueError(
                    f"rl_task.{group}.{name} requires func and optional params."
                )
            func_name = declaration["func"]
            if not isinstance(func_name, str):
                raise TypeError(
                    "Condition func must name a registered function or class."
                )
            params = deepcopy(declaration.get("params", {}))
            if not isinstance(params, dict):
                raise TypeError("Condition params must be a mapping.")
            for key in tuple(params):
                if key.endswith("entity_cfg") and isinstance(params[key], Mapping):
                    params[key] = SceneEntityCfg(**params[key])
            groups[group][name] = ConditionCfg(
                func=find_function_from_modules(
                    func_name, list(modules), raise_if_not_found=True
                ),
                params=params,
            )
    return RLTaskCfg(**groups)


def _register_configured_rl(config: Mapping[str, Any], max_episode_steps: int) -> None:
    from gymnasium.envs.registration import registry as gym_registry
    from embodichain.lab.gym.envs.configured_rl_env import ConfiguredRLEnv
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        register_env_function,
    )

    env_id = config["id"]
    if not isinstance(env_id, str) or not env_id or env_id != env_id.strip():
        raise ValueError("Configured RL id must be a nonempty string.")
    if type(max_episode_steps) is not int or max_episode_steps <= 0:
        raise ValueError("Configured RL max_episode_steps must be a positive integer.")
    signature = {
        "rl_task": config["rl_task"],
        "env": {
            key: value
            for key, value in config["env"].items()
            if key
            in {
                "events",
                "observations",
                "rewards",
                "actions",
                "extensions",
                "control_parts",
                "active_joint_ids",
                "sim_steps_per_control",
                "target_control_frequency",
            }
        },
        "physics": config["physics"],
        "max_episode_steps": max_episode_steps,
    }
    signature["manager_order"] = {
        name: list(config["env"].get(name, {}))
        for name in ("events", "observations", "rewards", "actions")
    }
    signature["condition_order"] = {
        name: list(config["rl_task"].get(name, {}))
        for name in ("success", "failure", "truncation")
    }
    signature["scene"] = {
        key: config[key]
        for key in (
            "robot",
            "sensor",
            "light",
            "background",
            "visual_object",
            "rigid_object",
            "rigid_object_group",
            "articulation",
            "physics_config",
        )
        if key in config
    }
    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    with _LOCK:
        existing = REGISTERED_ENVS.get(env_id)
        if existing is not None:
            if (
                existing.cls is not ConfiguredRLEnv
                or getattr(existing, "_configured_rl_fingerprint", None) != fingerprint
            ):
                raise ValueError(
                    f"Configured RL environment ID {env_id!r} already has a different declaration."
                )
            if env_id not in gym_registry:
                raise RuntimeError(
                    f"Configured RL environment {env_id!r} lost its Gym registration."
                )
            return
        if env_id in gym_registry:
            raise ValueError(f"Gymnasium already owns configured RL ID {env_id!r}.")
        register_env_function(
            ConfiguredRLEnv,
            env_id,
            max_episode_steps=max_episode_steps,
            supports_rl=True,
        )
        REGISTERED_ENVS[env_id]._configured_rl_fingerprint = fingerprint
