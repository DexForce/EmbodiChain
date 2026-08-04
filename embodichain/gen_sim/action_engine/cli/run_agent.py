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

"""Run a generated Action Engine configuration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import gymnasium
import numpy as np
import torch

from embodichain.gen_sim.action_engine.config import generation_defaults
from embodichain.gen_sim.action_engine.environment import (  # noqa: F401
    ACTION_ENGINE_ENV_ID,
)
from embodichain.gen_sim.action_engine.runtime import load_agent_execution_program
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.utils import set_seed
from embodichain.utils.logger import log_info, log_warning
from embodichain.utils.utility import load_config

__all__ = ["build_parser", "cli"]

_DEFAULT_MAX_EPISODES = int(generation_defaults()["task"]["max_episodes"])


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by generated demo commands."""
    parser = argparse.ArgumentParser(description="Execute an Action Engine task agent.")
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--task_name", required=True, help="Generated task name.")
    parser.add_argument(
        "--agent_config",
        required=True,
        help="Path to action_engine_config_v1 JSON.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Recompile task_agent in memory before execution.",
    )
    parser.add_argument(
        "--show-physical-collision",
        action="store_true",
        help="Show physical collision geometry after every reset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed; episode N uses seed + N.",
    )
    parser.add_argument(
        "--runtime-backend",
        choices=("independent",),
        default="independent",
        help="Execution backend. Action Engine owns the production runtime.",
    )
    return parser


def _validate_gym_id(config: dict[str, Any]) -> None:
    if config.get("id") != ACTION_ENGINE_ENV_ID:
        raise ValueError(
            f"Gym config id must be {ACTION_ENGINE_ENV_ID!r}, "
            f"got {config.get('id')!r}."
        )


def _validate_run_contract(
    gym_config: dict[str, Any],
    agent_config: dict[str, Any],
    task_name: str,
) -> None:
    """Validate the small cross-artifact contract before simulator startup."""
    configured_task = agent_config.get("task_name")
    if configured_task != task_name:
        raise ValueError(
            f"--task_name {task_name!r} does not match agent_config task "
            f"{configured_task!r}."
        )
    extension = gym_config.get("env", {}).get("extensions", {}).get("action_engine", {})
    if extension.get("task_name") != task_name:
        raise ValueError("Gym and agent configs describe different tasks.")
    gym_hash = extension.get("execution_program_hash")
    agent_hash = agent_config.get("execution_program_hash")
    if not isinstance(agent_hash, str) or not agent_hash or gym_hash != agent_hash:
        raise ValueError("Gym and agent configs have different program hashes.")


def cli() -> None:
    """Launch the environment and execute all configured episodes."""
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    args = build_parser().parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    env_cfg, gym_config, _ = build_env_cfg_from_args(args)
    if args.seed is not None:
        env_cfg.seed = args.seed
    _validate_gym_id(gym_config)
    agent_config = load_config(args.agent_config)
    if not isinstance(agent_config, dict):
        raise ValueError("agent_config must contain a JSON object.")
    _validate_run_contract(gym_config, agent_config, args.task_name)
    load_agent_execution_program(
        agent_config,
        agent_config_path=args.agent_config,
        regenerate=bool(args.regenerate),
    )

    env = gymnasium.make(
        id=gym_config["id"],
        cfg=env_cfg,
        agent_config=agent_config,
        agent_config_path=args.agent_config,
        task_name=args.task_name,
        runtime_backend=args.runtime_backend,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    episodes = int(gym_config.get("max_episodes", _DEFAULT_MAX_EPISODES))
    try:
        for episode_index in range(episodes):
            episode_seed = None if args.seed is None else int(args.seed) + episode_index
            env.reset(seed=episode_seed)
            if args.show_physical_collision:
                _show_physical_collision(env)
            execute = env.get_wrapper_attr("create_demo_action_list")
            result = execute(
                regenerate=bool(args.regenerate),
                runtime_run_id=run_id,
                episode_index=episode_index,
            )
            if not getattr(result, "already_executed", False):
                raise RuntimeError(
                    "Action Engine env returned an offline action sequence."
                )
            success = torch.as_tensor(
                getattr(result, "runtime_success"),
                dtype=torch.bool,
            )
            log_info(
                "Action Engine episode "
                f"{episode_index}: {int(success.sum())}/{success.numel()} "
                "environments succeeded.",
                color="green",
            )
            record_dir = getattr(result, "runtime_graph_output_dir", None)
            if record_dir:
                log_info(f"Runtime records: {record_dir}", color="green")
        # EmbodiedEnv publishes the just-finished rollout during reset. Flush
        # the final episode as well; otherwise only episodes followed by a next
        # iteration reach the configured dataset recorder.
        env.reset(options={"final": True})
    except KeyboardInterrupt:
        log_warning("Action Engine run interrupted by user.")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _show_physical_collision(env: gymnasium.Env) -> None:
    """Enable physical-shape visualization for all supported scene assets."""
    sim = env.get_wrapper_attr("sim")
    uids: list[str] = []
    for getter_name in (
        "get_rigid_object_uid_list",
        "get_rigid_object_group_uid_list",
        "get_articulation_uid_list",
    ):
        getter = getattr(sim, getter_name, None)
        if callable(getter):
            uids.extend(getter())
    visible = 0
    for uid in uids:
        asset = sim.get_asset(uid)
        if asset is None or not hasattr(asset, "set_physical_visible"):
            continue
        try:
            asset.set_physical_visible(
                visible=True,
                rgba=[1.0, 0.15, 0.1, 0.35],
            )
            visible += 1
        except Exception as exc:
            log_warning(f"Unable to show collision geometry for {uid!r}: {exc}")
    log_info(f"Physical collision geometry visible for {visible} assets.")


if __name__ == "__main__":
    cli()
