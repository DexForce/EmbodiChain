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

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
import tqdm

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (  # noqa: F401
    AgenticGenSimEnv,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.runner import (
    add_vectorized_reset_randomization as _add_vectorized_reset_randomization,
    generate_action_agent_trajectory,
    log_task_success as _log_task_success,
    normalize_legacy_dataset_functor_config as _normalize_legacy_dataset_functor_config,
    prepare_gym_config_for_run_agent as _modify_gym_config_for_run_agent,
    run_action_agent,
)
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.utils.logger import log_info, log_warning
from embodichain.utils.utility import load_config

__all__ = ["build_parser", "cli"]

_RUN_AGENT_DEFAULTS = load_config(Path(__file__).with_name("run_agent_defaults.yaml"))
_PHYSICAL_COLLISION_CONFIG = _RUN_AGENT_DEFAULTS["physical_collision"]
_WINDOW_LOOK_AT_CONFIG = _RUN_AGENT_DEFAULTS["window_look_at"]

_SHOW_PHYSICAL_COLLISION_ENV = _PHYSICAL_COLLISION_CONFIG["environment_variable"]
_PHYSICAL_COLLISION_RGBA = tuple(_PHYSICAL_COLLISION_CONFIG["rgba"])
_FALSE_ENV_VALUES = frozenset(_PHYSICAL_COLLISION_CONFIG["false_env_values"])


def cli() -> None:
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = build_parser()
    args = parser.parse_args()

    env_cfg, gym_config, _ = build_env_cfg_from_args(
        args,
        gym_config_modifier=_modify_gym_config_for_run_agent,
    )
    agent_config = load_config(args.agent_config)

    env = gymnasium.make(
        id=gym_config["id"],
        cfg=env_cfg,
        agent_config=agent_config,
        agent_config_path=args.agent_config,
        task_name=args.task_name,
    )
    _show_physical_collision_if_requested(env)
    _set_default_window_look_at(env, gym_config.get("num_envs", 1))
    _run_action_agent(args, env, gym_config)


def build_parser() -> argparse.ArgumentParser:
    """Build the action-agent runner argument parser."""
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the task.",
        required=True,
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        help="Path to the agent configuration file.",
        required=True,
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force Seed Graph v3 to be reparsed and revalidated before execution.",
        default=False,
    )
    return parser


def _run_action_agent(args: argparse.Namespace, env: gymnasium.Env, gym_config: dict):
    """Compatibility wrapper around the runtime-owned episode runner."""
    if getattr(args, "preview", False):
        log_warning("Preview mode is handled by the shared runner and is skipped here.")
    run_action_agent(
        env=env,
        gym_config=gym_config,
        task_name=args.task_name,
        regenerate=getattr(args, "regenerate", False),
        save_path=getattr(args, "save_path", ""),
        save_video=getattr(args, "save_video", False),
        debug_mode=getattr(args, "debug_mode", False),
        reset=lambda *reset_args, **reset_kwargs: _reset_env_with_physical_collision(
            env, *reset_args, **reset_kwargs
        ),
        action_iterator=lambda actions, episode_index: tqdm.tqdm(
            actions,
            desc=f"Executing action list #{episode_index}",
            unit="step",
        ),
        final_reset=bool(getattr(args, "headless", False)),
    )


def _generate_action_agent_trajectory(
    args: argparse.Namespace,
    env: gymnasium.Env,
    trajectory_idx: int,
    *,
    runtime_run_id: str,
) -> bool:
    return generate_action_agent_trajectory(
        env=env,
        episode_index=trajectory_idx,
        runtime_run_id=runtime_run_id,
        regenerate=getattr(args, "regenerate", False),
        save_path=getattr(args, "save_path", ""),
        save_video=getattr(args, "save_video", False),
        debug_mode=getattr(args, "debug_mode", False),
        reset=lambda *reset_args, **reset_kwargs: _reset_env_with_physical_collision(
            env, *reset_args, **reset_kwargs
        ),
        action_iterator=lambda actions, episode_index: tqdm.tqdm(
            actions,
            desc=f"Executing action list #{trajectory_idx}",
            unit="step",
        ),
    )


def _reset_env_with_physical_collision(
    env: gymnasium.Env,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    result = env.reset(*args, **kwargs)
    _show_physical_collision_if_requested(env)
    return result


def _show_physical_collision_if_requested(env: gymnasium.Env) -> None:
    if not _physical_collision_debug_enabled():
        return

    sim = _get_wrapped_attr(env, "sim")
    if sim is None:
        log_warning("Physical collision visualization skipped: env.sim is unavailable.")
        return

    asset_uids: list[str] = []
    for getter_name in (
        "get_rigid_object_uid_list",
        "get_rigid_object_group_uid_list",
        "get_articulation_uid_list",
    ):
        getter = getattr(sim, getter_name, None)
        if getter is not None:
            asset_uids.extend(getter())

    visible_count = 0
    for uid in asset_uids:
        asset = sim.get_asset(uid)
        if asset is None or not hasattr(asset, "set_physical_visible"):
            continue
        try:
            asset.set_physical_visible(
                visible=True,
                rgba=_PHYSICAL_COLLISION_RGBA,
            )
        except Exception as exc:
            log_warning(f"Failed to show physical collision for asset '{uid}': {exc}")
            continue
        visible_count += 1

    if not getattr(env, "_physical_collision_debug_logged", False):
        log_info(
            "Physical collision visualization enabled "
            f"for {visible_count} scene assets via {_SHOW_PHYSICAL_COLLISION_ENV}.",
            color="green",
        )
        setattr(env, "_physical_collision_debug_logged", True)


def _set_default_window_look_at(env: gymnasium.Env, num_envs: int) -> None:
    """Set the action-agent runner's default simulator-window viewpoint."""
    sim = _get_wrapped_attr(env, "sim")
    window = getattr(sim, "_window", None)
    if window is None:
        return

    look_at_config = _WINDOW_LOOK_AT_CONFIG[
        "single_env" if num_envs == 1 else "multiple_envs"
    ]
    eye = np.array(look_at_config["eye"], dtype=np.float32)
    look_at = np.array(look_at_config["look_at"], dtype=np.float32)
    up = np.array(look_at_config["up"], dtype=np.float32)

    window.set_look_at(eye=eye, look_at=look_at, up=up)


def _physical_collision_debug_enabled() -> bool:
    value = os.environ.get(_SHOW_PHYSICAL_COLLISION_ENV, "")
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _get_wrapped_attr(env: gymnasium.Env, name: str) -> Any:
    if hasattr(env, "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr(name)
        except AttributeError:
            pass
    return getattr(env, name, None)


if __name__ == "__main__":
    cli()
