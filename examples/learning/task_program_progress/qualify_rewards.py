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

"""Reproducible native reward acceptance and checkpoint video qualification.

Run under xvfb-run when recording without a display. Physical cases use a
fixed scene and zero actions; they qualify the evaluator, not a learned task.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
import yaml


def main() -> None:
    """Qualify a physical positive/negative fixture or preview a saved policy."""
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("supported", "floating", "empty_grip"), default="supported"
    )
    parser.add_argument(
        "--run", type=Path, help="Saved PPO run to load instead of a physical fixture"
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help="Qualify the calibrated planning expert with independent measured objectives",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reward_qualification")
    )
    args = parser.parse_args()
    if args.expert and args.run:
        parser.error("expert and run are mutually exclusive")
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num-envs must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from embodichain.utils.utility import load_config
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    import gymnasium as gym

    root = Path(__file__).resolve().parents[3]
    renderer = "hybrid" if args.video else "no-render"
    policy = None
    runtime = None
    label = args.case
    started = time.perf_counter()
    if args.expert:
        from embodichain.lab.gym.envs.managers.skill_objective import _bind_objective

        source = (
            root
            / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.cobotmagic.yaml"
        )
        config = load_config(source)
        config.update(
            id="ExpertRewardEvidence-v1",
            device="cpu",
            num_envs=1,
            renderer=renderer,
            headless=True,
            enable_sensor=False,
        )
        cfg = config_to_cfg(config, source_path=source)
        cfg.events = None
        cfg.init_rollout_buffer = False
        cfg.dataset = None
        cfg.record_trajectory = False
        binding = {
            "robot": "CobotMagic",
            "motion_part": "right_arm",
            "object": "block_2",
            "support": "table",
            "pad_links": ["right_link7", "right_link8"],
            "hand_links": ["right_gripper_base", "right_link7", "right_link8"],
        }
        _bind_objective(
            cfg,
            "expert_pick",
            {
                "skill_id": "pick_up",
                "binding": binding,
                "goal": {"position": [0.75, 0.1, 0.98]},
            },
            stage="diagnostic_pick",
        )
        _bind_objective(
            cfg,
            "expert_place",
            {
                "skill_id": "place",
                "binding": {**binding, "support": "block_1"},
                "goal": {
                    "position": [0.0, 0.0, 0.05],
                    "reference": "block_1",
                    "frame": "world_axes",
                },
            },
            stage="diagnostic_place",
        )
        env = gym.make(config["id"], cfg=cfg).unwrapped
        label = "expert_grasp_place"
    elif args.run:
        from embodichain.learning.rl.policy_evaluation.manifest import RunManifest
        from embodichain.learning.rl.policy_evaluation.cli import (
            _load_policy_state_dict,
        )
        from embodichain.learning.rl.runtime import build_gym_policy_runtime

        manifest = RunManifest.load(args.run)
        config = load_config(manifest.configs["train"])
        config["trainer"]["gym_config"] = str(manifest.configs["gym"])
        runtime = build_gym_policy_runtime(
            config,
            device=torch.device("cuda:0"),
            num_envs=args.num_envs,
            headless=True,
            renderer=renderer,
            gpu_id=0,
            seed=args.seed,
            config_dir=manifest.configs["train"].parent,
        )
        _, checkpoint = manifest.select_checkpoint("latest")
        policy = runtime.policy
        policy.load_state_dict(_load_policy_state_dict(checkpoint))
        policy.eval()
        env = runtime.env.unwrapped
        label = "trained_policy"
    else:
        source = Path(__file__).with_name("joint_environment.yaml")
        component = load_config(source)
        component["env"]["events"] = {}
        component["env"]["rewards"] = {}
        component["max_episode_steps"] = max(100, args.steps + 1)
        upper = component["simulation"]["rigid_object"][1]
        upper["init_pos"] = [0.75, -0.1, 0.95 if args.case != "floating" else 1.1]
        if args.case == "floating":
            upper["body_type"] = "kinematic"
            component["simulation"]["rigid_object"].remove(upper)
            component["simulation"]["background"].append(upper)
        physical = args.output / f"{label}_environment.yaml"
        physical.write_text(yaml.safe_dump(component, sort_keys=False))
        objective = {
            "skill_id": "pick_up" if args.case == "empty_grip" else "place",
            "binding": {
                "robot": "CobotMagic",
                "motion_part": "right_arm",
                "object": "block_2",
                "support": "block_1",
                "pad_links": ["right_link7", "right_link8"],
                "hand_links": ["right_gripper_base", "right_link7", "right_link8"],
            },
            "goal": {
                "position": [0.0, 0.0, 0.2 if args.case == "floating" else 0.05],
                "reference": "block_1",
            },
        }
        config = {
            "id": f"RewardQualification_{label}-v1",
            "environment": {"component": str(physical)},
            "embodiment": {
                "component": str(
                    root
                    / "extensions/embodichain_tasks/configs/components/embodiments/cobotmagic.yaml"
                )
            },
            "device": "cuda:0",
            "num_envs": args.num_envs,
            "renderer": renderer,
            "headless": True,
            "enable_sensor": False,
            "rl_task": {"skill": objective, "reward_composition": {"discount": 0.99}},
        }
        cfg = config_to_cfg(config, source_path=source)
        cfg.init_rollout_buffer = False
        env = gym.make(config["id"], cfg=cfg).unwrapped
    env._demo_no_auto_reset = True
    recorder = None
    trace = []
    try:
        if args.video:
            from embodichain.lab.gym.envs.managers.cfg import FunctorCfg
            from embodichain.lab.gym.envs.managers.record import record_camera_data

            recorder = record_camera_data(
                FunctorCfg(
                    params={
                        "name": label,
                        "save_path": str(args.output / "videos"),
                        "resolution": (640, 480),
                        "eye": (1.8, -1.5, 1.8),
                        "target": (0.6, 0, 0.9),
                        "max_env_num": 1,
                        "fps": 1 / (env.step_dt * (2 if args.expert else 1)),
                    }
                ),
                env,
            )
        observation, _ = env.reset(seed=args.seed)
        expert_result = None
        if args.expert:
            from embodichain.lab.gym.envs.demo import execute_demo_episode

            def observe(active, observation, info):
                measured = {}
                for name in ("expert_pick", "expert_place"):
                    term = env.observation_manager.get_functor_cfg(
                        "skill_objective_" + name
                    ).func
                    measured[name] = {
                        "sample": term._last_sample.values.tolist(),
                        "valid": term._last_sample.valid.tolist(),
                        "held_seconds": term._held_time.tolist(),
                        "stable_seconds": term._stable_time.tolist(),
                    }
                trace.append(
                    {
                        "step": len(trace),
                        "reward": [0.0],
                        "success": [False],
                        "failure": [False],
                        "measured": measured,
                    }
                )
                if recorder and len(trace) % 2 == 0:
                    recorder(env, env_ids=[0], name=label)

            expert_result = execute_demo_episode(
                env, step_observer=observe, should_stop=lambda: len(trace) >= args.steps
            ).to_metadata()
        else:
            for step in range(args.steps):
                if policy is None:
                    action = torch.zeros(env.action_space.shape, device=env.device)
                else:
                    from embodichain.learning.rl.evaluation import (
                        infer_policy_action,
                        convert_policy_action_for_env,
                    )

                    with torch.inference_mode():
                        action = convert_policy_action_for_env(
                            env,
                            infer_policy_action(
                                policy,
                                observation,
                                device=env.device,
                                num_envs=env.num_envs,
                            ),
                        )
                observation, reward, terminated, truncated, info = env.step(action)
                if recorder:
                    recorder(env, env_ids=[0], name=label)
                trace.append(
                    {
                        "step": step,
                        "reward": reward.tolist(),
                        "success": info["success"].tolist(),
                        "failure": info["fail"].tolist(),
                        "terminated": terminated.tolist(),
                        "truncated": truncated.tolist(),
                        "terms": {
                            key: value.tolist()
                            for key, value in info["rewards"].items()
                        },
                    }
                )
                if policy is not None and bool((terminated | truncated).any()):
                    break
        if recorder:
            recorder.save_and_clear([0], purpose="diagnostic")
        successes = [
            any(row["success"][i] for row in trace) for i in range(env.num_envs)
        ]
        report = {
            "case": label,
            "source": (
                "checkpoint_policy"
                if policy is not None
                else "fixed_physical_fixture_zero_actions"
            ),
            "run": str(args.run.resolve()) if args.run else None,
            "seed": args.seed,
            "steps": len(trace),
            "num_envs": env.num_envs,
            "success_rows": sum(successes),
            "success_rate": sum(successes) / env.num_envs,
            "finite_rewards": all(
                torch.isfinite(torch.tensor(row["reward"])).all().item()
                for row in trace
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "trace": trace,
        }
        if args.expert:
            report["source"] = "planning_expert_not_learned_policy"
            report["expert_result"] = expert_result
            report["measured_pick_success"] = any(
                row["measured"]["expert_pick"]["sample"][0][1] == 1 for row in trace
            )
            report["measured_place_success"] = any(
                row["measured"]["expert_place"]["sample"][0][1] == 1 for row in trace
            )
            report["success_rows"] = int(
                all(expert_result["success"])
                and report["measured_pick_success"]
                and report["measured_place_success"]
            )
            report["success_rate"] = float(report["success_rows"])
        (args.output / f"{label}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("trace", "expert_result")}
            ),
            flush=True,
        )
    finally:
        if recorder:
            recorder.close()
        if runtime:
            runtime.close()
        else:
            env.close()


if __name__ == "__main__":
    main()
