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

"""Prepare and run action-agent episodes without a command-line dependency."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.utils.logger import log_info, log_warning

__all__ = [
    "add_vectorized_reset_randomization",
    "generate_action_agent_trajectory",
    "log_task_success",
    "normalize_legacy_dataset_functor_config",
    "prepare_gym_config_for_run_agent",
    "run_action_agent",
]

_RUNNER_DEFAULTS = defaults_section("runner")


def prepare_gym_config_for_run_agent(gym_config: dict[str, Any]) -> None:
    """Apply runtime-required compatibility and vectorization transforms."""
    normalize_legacy_dataset_functor_config(gym_config)
    add_vectorized_reset_randomization(gym_config)


def normalize_legacy_dataset_functor_config(gym_config: dict[str, Any]) -> None:
    """Move legacy manager-only dataset options out of functor parameters."""
    env_config = gym_config.get("env")
    if not isinstance(env_config, dict):
        return
    dataset_config = env_config.get("dataset")
    if not isinstance(dataset_config, dict):
        return
    for functor_config in dataset_config.values():
        if not isinstance(functor_config, dict) or "func" not in functor_config:
            continue
        params = functor_config.get("params")
        if not isinstance(params, dict) or "save_failed_episodes" not in params:
            continue
        legacy_value = params.pop("save_failed_episodes")
        functor_config.setdefault("save_failed_episodes", legacy_value)


def add_vectorized_reset_randomization(gym_config: dict[str, Any]) -> None:
    """Add deterministic default reset randomizers for vectorized execution."""
    if gym_config.get("num_envs", 1) <= 1:
        return
    env_config = gym_config.setdefault("env", {})
    dataset_config = env_config.get("dataset")
    if isinstance(dataset_config, dict):
        for name in [
            name
            for name, params in dataset_config.items()
            if isinstance(params, dict) and "func" in params
        ]:
            del dataset_config[name]

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
                        list(_RUNNER_DEFAULTS["rigid_object_position_range"][0]),
                        list(_RUNNER_DEFAULTS["rigid_object_position_range"][1]),
                    ],
                    "rotation_range": [
                        list(_RUNNER_DEFAULTS["rigid_object_rotation_range"][0]),
                        list(_RUNNER_DEFAULTS["rigid_object_rotation_range"][1]),
                    ],
                    "relative_position": True,
                    "relative_rotation": True,
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
                    list(_RUNNER_DEFAULTS["table_height_delta_range"][0]),
                    list(_RUNNER_DEFAULTS["table_height_delta_range"][1]),
                ],
            },
        },
    )


def run_action_agent(
    *,
    env: Any,
    gym_config: dict[str, Any],
    task_name: str,
    regenerate: bool = False,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    reset: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
    action_iterator: Callable[[Iterable[Any], int], Iterable[Any]] | None = None,
    runtime_graph_renderer: Callable[[Mapping[str, Any]], bytes] | None = None,
    final_reset: bool = False,
) -> None:
    """Execute every configured episode and flush its final recorder state."""
    reset_fn = reset or env.reset
    log_info("Start action-agent data generation.", color="green")
    runtime_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_info(
        f"Runtime task graph run_id: {runtime_run_id}; output root: "
        f"outputs/graph/{task_name}/runs/{runtime_run_id}",
        color="green",
    )
    for episode_index in range(gym_config.get("max_episodes", 1)):
        generate_action_agent_trajectory(
            env=env,
            episode_index=episode_index,
            runtime_run_id=runtime_run_id,
            regenerate=regenerate,
            save_path=save_path,
            save_video=save_video,
            debug_mode=debug_mode,
            reset=reset_fn,
            action_iterator=action_iterator,
            runtime_graph_renderer=runtime_graph_renderer,
        )
    reset_fn()
    if final_reset:
        reset_fn(options={"final": True})


def generate_action_agent_trajectory(
    *,
    env: Any,
    episode_index: int,
    runtime_run_id: str,
    regenerate: bool = False,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    reset: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
    action_iterator: Callable[[Iterable[Any], int], Iterable[Any]] | None = None,
    runtime_graph_renderer: Callable[[Mapping[str, Any]], bytes] | None = None,
) -> bool:
    """Execute one online or precomputed action-agent trajectory."""
    (reset or env.reset)()
    action_list = env.get_wrapper_attr("create_demo_action_list")(
        action_sentence=str(episode_index),
        save_path=save_path,
        save_video=save_video,
        debug_mode=debug_mode,
        regenerate=regenerate,
        runtime_run_id=runtime_run_id,
        episode_index=episode_index,
        runtime_graph_renderer=runtime_graph_renderer,
    )
    if action_list is None or len(action_list) == 0:
        log_warning("Action is invalid. Skip to next generation.")
        return False
    if getattr(action_list, "already_executed", False):
        log_info("Action list was already executed by the action-agent runtime.")
        runtime_graph_dir = getattr(action_list, "runtime_graph_output_dir", None)
        if runtime_graph_dir:
            log_info(f"Runtime task graphs saved to: {runtime_graph_dir}")
        log_task_success(
            env,
            semantic_success=getattr(action_list, "runtime_success", None),
        )
        return True

    actions = (
        action_iterator(action_list, episode_index)
        if action_iterator is not None
        else action_list
    )
    for action in actions:
        env.step(action)
    log_task_success(env)
    return True


def log_task_success(
    env: Any,
    *,
    semantic_success: torch.Tensor | None = None,
) -> bool | None:
    """Evaluate and report the final task mask without changing environment state."""
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
        if semantic_success is not None:
            success_bool &= semantic_success.detach().cpu().flatten().bool()
        n_success = int(success_bool.sum().item())
        n_total = int(success_bool.numel())
        log_info(
            f"Task success after execution: {n_success}/{n_total} environments succeeded.",
            color="green",
        )
        return bool(success_bool.all().item())

    success_value = bool(np.asarray(success).flatten().all())
    log_info(f"Task success after execution: {success_value}", color="green")
    return success_value
