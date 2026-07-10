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
from typing import Any

import gymnasium
import numpy as np
import torch
import tqdm

from embodichain.gen_sim.action_agent_pipeline.utils.timing import timing_scope
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (  # noqa: F401
    AgenticGenSimEnv,
)
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.utils.logger import log_info, log_warning
from embodichain.utils.utility import load_config

__all__ = ["cli"]

_SHOW_PHYSICAL_COLLISION_ENV = "EMBODICHAIN_SHOW_PHYSICAL_COLLISION"
_PHYSICAL_COLLISION_RGBA = (0.0, 1.0, 0.0, 0.85)
_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}
# hard-coded param for waic demo.
_RIGID_OBJECT_POSITION_RANGE = [[-0.04, -0.04, 0.0], [0.04, 0.04, 0.0]]
_RIGID_OBJECT_ROTATION_RANGE = [[0.0, 0.0, -45.0], [0.0, 0.0, 45.0]]
_TABLE_HEIGHT_DELTA_RANGE = [[-0.05], [0.05]]


def cli() -> None:
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

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
        help="Whether to regenerate code if already existed.",
        default=False,
    )

    args = parser.parse_args()

    env_cfg, gym_config, _ = build_env_cfg_from_args(
        args,
        gym_config_modifier=_add_vectorized_reset_randomization,
    )
    agent_config = load_config(args.agent_config)

    with timing_scope(
        "run_agent.make_env",
        metadata={"task_name": args.task_name, "gym_id": gym_config["id"]},
    ):
        env = gymnasium.make(
            id=gym_config["id"],
            cfg=env_cfg,
            agent_config=agent_config,
            agent_config_path=args.agent_config,
            task_name=args.task_name,
        )
    _show_physical_collision_if_requested(env)

    with timing_scope("run_agent.total", metadata={"task_name": args.task_name}):
        _run_action_agent(args, env, gym_config)

    if args.headless:
        with timing_scope("run_agent.final_reset"):
            _reset_env_with_physical_collision(env, options={"final": True})


def _add_vectorized_reset_randomization(gym_config: dict[str, Any]) -> None:
    """Add default reset randomization for parallel action-agent environments.

    Dataset functors are removed because dataset recorders are not supported for
    vectorized action-agent execution. Plain dataset configuration is retained
    for consumers that use it as metadata.

    A pose randomizer is added for every configured rigid object. The table-height
    randomizer runs after those pose randomizers so all randomized objects are
    shifted together with the table.

    Args:
        gym_config: Merged gym configuration that will be parsed into the
            environment configuration.
    """
    if gym_config.get("num_envs", 1) <= 1:
        return

    env_config = gym_config.setdefault("env", {})
    dataset_config = env_config.get("dataset")
    if isinstance(dataset_config, dict):
        dataset_functor_names = [
            dataset_name
            for dataset_name, dataset_params in dataset_config.items()
            if isinstance(dataset_params, dict) and "func" in dataset_params
        ]
        for dataset_name in dataset_functor_names:
            del dataset_config[dataset_name]

    events = env_config.setdefault("events", {})
    for rigid_object in gym_config.get("rigid_object", []):
        uid = rigid_object.get("uid")
        if not isinstance(uid, str) or not uid:
            log_warning(
                "Skipping reset pose randomization for a rigid object without a UID."
            )
            continue

        events.setdefault(
            f"init_{uid}_pose",
            {
                "func": "randomize_rigid_object_pose",
                "mode": "reset",
                "params": {
                    "entity_cfg": {"uid": uid},
                    "position_range": [
                        list(_RIGID_OBJECT_POSITION_RANGE[0]),
                        list(_RIGID_OBJECT_POSITION_RANGE[1]),
                    ],
                    "rotation_range": [
                        list(_RIGID_OBJECT_ROTATION_RANGE[0]),
                        list(_RIGID_OBJECT_ROTATION_RANGE[1]),
                    ],
                    "relative_position": True,
                },
            },
        )

    events.setdefault(
        "random_table_height",
        {
            "func": "randomize_anchor_height",
            "mode": "reset",
            "params": {
                "anchor_uid": "table",
                "height_delta_range": [
                    list(_TABLE_HEIGHT_DELTA_RANGE[0]),
                    list(_TABLE_HEIGHT_DELTA_RANGE[1]),
                ],
            },
        },
    )


def _run_action_agent(args: argparse.Namespace, env: gymnasium.Env, gym_config: dict):
    """Run action-agent graphs without relying on the shared run_env runner."""
    if getattr(args, "preview", False):
        log_warning("Preview mode is handled by the shared runner and is skipped here.")

    log_info("Start action-agent data generation.", color="green")
    for trajectory_idx in range(gym_config.get("max_episodes", 1)):
        _generate_action_agent_trajectory(
            args,
            env,
            trajectory_idx,
        )
    _, _ = _reset_env_with_physical_collision(env)


def _generate_action_agent_trajectory(
    args: argparse.Namespace,
    env: gymnasium.Env,
    trajectory_idx: int,
) -> bool:
    with timing_scope(
        "run_agent.trajectory_reset",
        metadata={"trajectory_idx": trajectory_idx},
    ):
        _, _ = _reset_env_with_physical_collision(env)
    with timing_scope(
        "run_agent.create_demo_action_list",
        metadata={"trajectory_idx": trajectory_idx},
    ):
        action_list = env.get_wrapper_attr("create_demo_action_list")(
            action_sentence=str(trajectory_idx),
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            regenerate=getattr(args, "regenerate", False),
        )
    if action_list is None or len(action_list) == 0:
        log_warning("Action is invalid. Skip to next generation.")
        return False

    if getattr(action_list, "already_executed", False):
        log_info("Action list was already executed by the action-agent runtime.")
        with timing_scope(
            "run_agent.evaluate_success",
            metadata={"trajectory_idx": trajectory_idx},
        ):
            _log_task_success(env)
        return True

    with timing_scope(
        "run_agent.execute_action_list",
        metadata={"trajectory_idx": trajectory_idx, "actions": len(action_list)},
    ):
        for action in tqdm.tqdm(
            action_list,
            desc=f"Executing action list #{trajectory_idx}",
            unit="step",
        ):
            env.step(action)
    with timing_scope(
        "run_agent.evaluate_success",
        metadata={"trajectory_idx": trajectory_idx},
    ):
        _log_task_success(env)
    return True


def _log_task_success(env: gymnasium.Env) -> bool | None:
    try:
        success_fn = (
            env.get_wrapper_attr("is_task_success")
            if hasattr(env, "get_wrapper_attr")
            else env.is_task_success
        )
        success = success_fn()
    except Exception as exc:
        log_warning(f"Failed to evaluate task success after execution: {exc}")
        return None

    if isinstance(success, torch.Tensor):
        success_bool = success.detach().cpu().flatten().bool()
        n_success = int(success_bool.sum().item())
        n_total = int(success_bool.numel())
        log_info(
            f"Task success after execution: {n_success}/{n_total} environments succeeded.",
            color="green",
        )
        success_value = bool(success_bool.all().item())
    else:
        success_value = bool(np.asarray(success).flatten().all())
        log_info(f"Task success after execution: {success_value}", color="green")
    return success_value


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
