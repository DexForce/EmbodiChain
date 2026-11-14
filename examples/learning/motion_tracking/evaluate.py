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

"""Evaluate a tracking policy once from the start of every motion clip."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.learning.rl.evaluation import infer_policy_action
from embodichain.learning.rl.policy_evaluation.manifest import RunManifest
from embodichain.learning.rl.runtime import build_gym_policy_runtime
from embodichain.utils.utility import load_config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = RunManifest.load(args.run)
    _, checkpoint = manifest.select_checkpoint("latest")
    config = load_config(manifest.configs["train"])
    config["trainer"]["gym_config"] = str(manifest.configs["gym"])
    discover_task_packages()
    execute_init_hooks()
    runtime = build_gym_policy_runtime(
        config,
        device=torch.device(args.device),
        headless=True,
        renderer="no-render",
        num_envs=1,
        gpu_id=0,
    )
    try:
        env = runtime.env.unwrapped
        runtime.policy.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["policy"]
        )
        runtime.policy.eval()
        env.motion_cfg.observation_noise = False
        env.motion_cfg.random_start = False
        env.max_episode_steps = 2**31 - 1
        results = []
        for clip, name in enumerate(env.motion.names):
            obs, _ = env.reset(
                seed=0, options={"motion_frames": env.motion.starts[clip : clip + 1]}
            )
            total = 0.0
            failed = False
            steps = 0
            length = int(env.motion.lengths[clip]) - 1
            for _ in range(length):
                action = infer_policy_action(
                    runtime.policy, obs, device=runtime.device, num_envs=1
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward[0])
                steps += 1
                if bool(terminated[0] | truncated[0]):
                    failed = bool(terminated[0])
                    break
            results.append(
                {
                    "clip": name,
                    "steps": steps,
                    "reference_steps": length,
                    "failed": failed,
                    "return": total,
                    "normalized_return": total / (length * env.task.max_step_reward),
                }
            )
        result = {
            "checkpoint": str(checkpoint),
            "clips": results,
            "failures": sum(x["failed"] for x in results),
            "macro_normalized_return": sum(x["normalized_return"] for x in results)
            / len(results),
        }
        output = manifest.root / "evaluations/motion-clips.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Evaluation report: {output}")
        print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
