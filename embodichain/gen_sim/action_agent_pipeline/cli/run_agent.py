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

__all__ = ["build_parser", "cli"]

_RUN_AGENT_DEFAULTS = load_config(Path(__file__).with_name("run_agent_defaults.yaml"))
_PHYSICAL_COLLISION_CONFIG = _RUN_AGENT_DEFAULTS["physical_collision"]
_VECTORIZED_RESET_CONFIG = _RUN_AGENT_DEFAULTS["vectorized_reset_randomization"]
_WINDOW_LOOK_AT_CONFIG = _RUN_AGENT_DEFAULTS["window_look_at"]
_DEFAULT_TABLE_TEXTURE_PATH = str(
    Path(__file__).resolve().parents[4] / "gym_project" / "background_texture"
)

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
        gym_config_modifier=lambda config: _modify_gym_config_for_run_agent(
            config, table_texture_path=args.table_texture_path
        ),
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
    _set_default_window_look_at(env, gym_config.get("num_envs", 1))

    with timing_scope("run_agent.total", metadata={"task_name": args.task_name}):
        _run_action_agent(args, env, gym_config)

    if args.headless:
        with timing_scope("run_agent.final_reset"):
            _reset_env_with_physical_collision(env, options={"final": True})


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
        help="Whether to regenerate code if already existed.",
        default=False,
    )
    parser.add_argument(
        "--table_texture_path",
        "--table-texture-path",
        dest="table_texture_path",
        type=str,
        default=_DEFAULT_TABLE_TEXTURE_PATH,
        help=(
            "Directory of table textures used for reset randomization. "
            "Defaults to gym_project/background_texture in the repository."
        ),
    )
    return parser


def _modify_gym_config_for_run_agent(
    gym_config: dict[str, Any], table_texture_path: str | None = None
) -> None:
    """Apply action-agent demo defaults to a merged gym configuration."""
    _set_rect_light_intensity_for_parallel_envs(gym_config)
    _add_vectorized_reset_randomization(
        gym_config,
        table_texture_path=table_texture_path or _DEFAULT_TABLE_TEXTURE_PATH,
    )


def _set_rect_light_intensity_for_parallel_envs(gym_config: dict[str, Any]) -> None:
    """Reduce direct rect-light intensity for vectorized environment runs."""
    if gym_config.get("num_envs", 1) <= 1:
        return

    light_config = gym_config.get("light")
    if not isinstance(light_config, dict):
        return
    direct_lights = light_config.get("direct")
    if not isinstance(direct_lights, list):
        return

    for light in direct_lights:
        if isinstance(light, dict) and light.get("light_type") == "rect":
            light["intensity"] = 10.0


def _add_vectorized_reset_randomization(
    gym_config: dict[str, Any], *, table_texture_path: str | None = None
) -> None:
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
        table_texture_path: Directory containing texture images for the table.
            Uses the repository demo texture directory when omitted.
    """
    if gym_config.get("num_envs", 1) <= 1:
        return

    table_texture_path = table_texture_path or _DEFAULT_TABLE_TEXTURE_PATH

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
                        list(
                            _VECTORIZED_RESET_CONFIG["rigid_object_position_range"][0]
                        ),
                        list(
                            _VECTORIZED_RESET_CONFIG["rigid_object_position_range"][1]
                        ),
                    ],
                    "rotation_range": [
                        list(
                            _VECTORIZED_RESET_CONFIG["rigid_object_rotation_range"][0]
                        ),
                        list(
                            _VECTORIZED_RESET_CONFIG["rigid_object_rotation_range"][1]
                        ),
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
                    list(_VECTORIZED_RESET_CONFIG["table_height_delta_range"][0]),
                    list(_VECTORIZED_RESET_CONFIG["table_height_delta_range"][1]),
                ],
            },
        },
    )
    events.setdefault(
        "random_table_visual_material",
        {
            "func": "randomize_visual_material",
            "mode": "reset",
            "params": {
                "entity_cfg": {"uid": "table"},
                "random_texture_prob": 1.0,
                "texture_path": table_texture_path,
                "texture_sampling": "without_replacement",
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
