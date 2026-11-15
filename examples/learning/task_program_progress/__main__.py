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

"""Run a seeded policy rollout with measured stages and separate outcomes.

Run: python -m examples.learning.task_program_progress --steps 16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


def _json(value: Any) -> Any:
    """Convert owned reports to JSON without changing runtime state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def main() -> None:
    """Run an untrained random policy until a measured terminal or deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/task_program_progress.json")
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.utils.utility import load_config

    path = (
        Path(__file__).resolve().parents[3]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.policy.cobotmagic.yaml"
    )
    config = load_config(path)
    config.update(
        seed=args.seed,
        device=args.device,
        num_envs=1,
        headless=True,
        enable_sensor=False,
        renderer="no-render",
    )
    cfg = config_to_cfg(config, source_path=path)
    cfg.max_episode_steps = args.steps
    cfg.init_rollout_buffer = False
    env = gym.make(config["id"], cfg=cfg).unwrapped
    trace = []
    try:
        env.reset(seed=args.seed)
        generator = torch.Generator(device=env.device).manual_seed(args.seed)
        for step in range(args.steps):
            # A policy produces the complete ActionManager vector; loading the
            # declarative pick/place steps never invokes their expert planners.
            action = (
                2
                * torch.rand(
                    env.action_space.shape, generator=generator, device=env.device
                )
                - 1
            )
            _, reward, terminated, truncated, info = env.step(action)
            trace.append(
                _json(
                    {
                        "step": step,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "progress": info["task_program_progress"],
                        "outcome": info["task_outcome"],
                        "terminal_outcomes": info.get("terminal_outcomes", {}),
                        "episode_context": info["episode_context"],
                    }
                )
            )
        report = {
            "seed": args.seed,
            "deployment": str(path),
            "policy": "seeded_uniform_untrained",
            "trace": trace,
            "post_rollout_progress": _json(env.task_program_progress.snapshot()),
            "post_rollout_context": _json(env.episode_context.snapshot()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output.resolve()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
