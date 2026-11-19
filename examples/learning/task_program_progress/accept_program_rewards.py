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

"""Accept Task Program rewards with scripted native actions and independent arithmetic.

No policy model or optimizer is constructed. First collect native targets from
an existing planning Task Program, then drive a policy-owned program through its
ordinary ActionManager. Every env.step reward is audited against measured stage
state and an independently evaluated shaping equation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import os

import torch
import yaml


def _json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _source(root):
    return (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.cobotmagic.yaml"
    )


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_inputs(root):
    task = _source(root)
    paths = [
        task,
        task.with_name("env.yaml"),
        *[
            (task.parent / "task_program" / name)
            for name in ("program.yaml", "integration.yaml", "evaluation.yaml")
        ],
        root
        / "extensions/embodichain_tasks/configs/components/embodiments/cobotmagic.yaml",
        root
        / "extensions/embodichain_tasks/configs/components/execution_policies/trajectory_open_loop.yaml",
    ]
    return {str(path.relative_to(root)): _hash(path) for path in paths}


def _finite(step, *values):
    for value in values:
        tensor = torch.as_tensor(value)
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(
                f"Nonfinite reward/potential in acceptance step {step}"
            )


def _collect(args, root):
    import gymnasium as gym
    from embodichain.utils.utility import load_config
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    path = _source(root)
    config = load_config(path)
    config.update(
        id="ProgramRewardActionSource-v1",
        device="cpu",
        num_envs=1,
        renderer="no-render",
        headless=True,
        enable_sensor=False,
    )
    cfg = config_to_cfg(config, source_path=path)
    cfg.events = None
    cfg.dataset = None
    cfg.record_trajectory = False
    cfg.init_rollout_buffer = False
    env = gym.make(config["id"], cfg=cfg).unwrapped
    targets = []
    try:
        env.reset(seed=args.seed)
        initial_qpos = env.robot.get_qpos().clone()

        def observe(active, observation, info):
            targets.append(env.robot.get_qpos(target=True)[0].detach().cpu().tolist())

        result = execute_demo_episode(
            env, step_observer=observe, should_stop=lambda: len(targets) >= 1200
        )
        if not all(result.success):
            raise RuntimeError(
                f"Expert trajectory did not complete: {result.to_metadata()}"
            )
        artifact = {
            "source": "planning_task_program_controller_targets",
            "seed": args.seed,
            "control_dt": env.step_dt,
            "joint_names": list(env.robot.joint_names),
            "initial_qpos": initial_qpos[0].tolist(),
            "targets": targets,
            "result": result.to_metadata(),
            "source_config": str(path),
            "source_config_sha256": _hash(path),
            "input_hashes": _source_inputs(root),
        }
        (args.output / "controller_source.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "collected_steps": len(targets),
                    "expert_success": list(result.success),
                    "control_dt": env.step_dt,
                }
            ),
            flush=True,
        )
    finally:
        env.close()


def _configuration(args, root):
    from embodichain.utils.utility import load_config

    source = _source(root)
    expert = load_config(source)
    physical = load_config(source.with_name("env.yaml"))
    physical["environment_id"] = "program_reward_acceptance"
    physical["env"]["events"] = {}
    physical["env"]["rewards"] = {}
    physical["env"]["dataset"] = {}
    physical["env"]["actions"] = load_config(
        Path(__file__).with_name("joint_environment.yaml")
    )["env"]["actions"]
    for term in physical["env"]["actions"].values():
        term["func"] = "JointPositionAction"
        term["params"]["scale"] = 1.0
    physical["max_episode_steps"] = 600 if args.case == "truncated" else 2400
    if args.case == "failure":
        physical["simulation"]["rigid_object"][1]["init_pos"] = [0.75, 0.1, 0.02]
    physical_path = args.output / f"environment_{args.case}.yaml"
    physical_path.write_text(yaml.safe_dump(physical, sort_keys=False))
    program = load_config(Path(__file__).with_name("program_rewards.yaml"))
    program["rewards"]["skills"][0]["objective"]["goal"]["position"] = [0.75, 0.1, 0.98]
    program_path = args.output / f"program_{args.case}.yaml"
    program_path.write_text(yaml.safe_dump(program, sort_keys=False))
    config = load_config(Path(__file__).with_name("program_task.yaml"))
    config["id"] = f"ProgramRewardAcceptance_{args.case}-v1"
    config["environment"]["component"] = str(physical_path)
    config["evaluation"]["component"] = str(
        source.parent / "task_program/evaluation.measured.yaml"
    )
    integration = deepcopy(expert["task_program"]["integration"])
    integration["component"] = str(source.parent / "task_program/integration.yaml")
    config["task_program"]["integration"] = integration
    config["task_program"]["program"] = str(program_path)
    config["task_program"]["execution_policy"] = {
        "component": str(
            root
            / "extensions/embodichain_tasks/configs/components/execution_policies/trajectory_open_loop.yaml"
        )
    }
    config["embodiment"] = deepcopy(expert["embodiment"])
    config["embodiment"]["component"] = str(
        root
        / "extensions/embodichain_tasks/configs/components/embodiments/cobotmagic.yaml"
    )
    config.update(
        device="cpu",
        num_envs=2,
        renderer="hybrid" if args.video else "no-render",
        headless=True,
        enable_sensor=False,
    )
    return config, source


def _potential(env):
    # Independent scalar oracle. Do not call ComposedPotentialReward._potential.
    progress = env.task_program_progress.snapshot()
    achieved = progress["achieved"].tolist()
    local = [
        env.observation_manager.get_functor_cfg(f"skill_objective_{i}")
        .func._last_sample.values[:, 0]
        .tolist()
        for i in range(2)
    ]
    phi = []
    for row, bits in enumerate(achieved):
        completed = sum(bits)
        phi.append(
            completed / 2 + (local[completed][row] / 2 if completed < 2 else 0.0)
        )
    return phi, achieved


def _replay(args, root):
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    artifact = json.loads((args.output / "controller_source.json").read_text())
    source_sha256 = _hash(args.output / "controller_source.json")
    if artifact.get("input_hashes") != _source_inputs(root) or artifact[
        "source_config_sha256"
    ] != _hash(_source(root)):
        raise ValueError(
            "Planning source configurations changed or provenance is missing; recollect controller_source.json"
        )
    config, source = _configuration(args, root)
    cfg = config_to_cfg(config, source_path=source)
    cfg.init_rollout_buffer = False
    cfg.dataset = None
    cfg.record_trajectory = False
    from embodichain.lab.gym.envs.managers.skill_objective import (
        SkillObjectiveObservation,
    )

    effective_contract = {
        "configuration": config,
        "source_inputs": _source_inputs(root),
        "physical_component_sha256": _hash(
            args.output / f"environment_{args.case}.yaml"
        ),
        "evaluation_component_sha256": _hash(config["evaluation"]["component"]),
        "program_sha256": _hash(args.output / f"program_{args.case}.yaml"),
        "rewards": {
            name: {
                "weight": term.weight,
                "params": _json(term.params),
                "extra": _json(term.extra),
            }
            for name, term in vars(cfg.rewards).items()
        },
        "objectives": {
            name: _json(term.params)
            for name, term in vars(cfg.observations).items()
            if term.func is SkillObjectiveObservation
        },
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(effective_contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    env = gym.make(config["id"], cfg=cfg).unwrapped
    env._demo_no_auto_reset = True
    recorder = None
    trace = []
    captured_steps = []
    video_capture = None
    checks = []
    previous_paid = [False, False]
    start = time.perf_counter()
    try:
        if args.video:
            from embodichain.lab.gym.envs.managers.cfg import FunctorCfg
            from embodichain.lab.gym.envs.managers.record import record_camera_data

            recorder = record_camera_data(
                FunctorCfg(
                    params={
                        "name": f"program_rewards_{args.case}",
                        "save_path": str(args.output / "videos"),
                        "resolution": (640, 480),
                        "eye": (1.8, -1.5, 1.8),
                        "target": (0.6, 0, 0.9),
                        "max_env_num": 1,
                        "fps": 1 / (2 * env.step_dt),
                    }
                ),
                env,
            )
        env.reset(seed=args.seed)
        if (
            list(env.robot.joint_names) != artifact["joint_names"]
            or abs(env.step_dt - artifact["control_dt"]) > 1e-12
        ):
            raise ValueError("Controller source joint layout/control cadence mismatch")
        expected_initial = torch.tensor(artifact["initial_qpos"], device=env.device)
        torch.testing.assert_close(
            env.robot.get_qpos(), expected_initial.expand(2, -1), atol=1e-5, rtol=0
        )
        descriptors = env.action_manager.descriptors
        by_name = {name: i for i, name in enumerate(artifact["joint_names"])}
        joint_indices = [
            torch.tensor(
                [by_name[name] for name in d.term.joint_names], device=env.device
            )
            for d in descriptors
        ]
        hold = env.robot.get_qpos(target=True).clone()
        previous_phi, previous_achieved = _potential(env)
        initial_phi = list(previous_phi)
        _finite("initial", initial_phi)
        count = 64 if args.case == "hold" else len(artifact["targets"])
        if args.case == "truncated":
            count = 600
        if args.case == "failure":
            count = 1
        max_error = 0.0
        max_repeat_error = 0.0
        for step in range(count):
            requested = hold.clone()
            if args.case != "hold":
                requested[0] = torch.tensor(
                    artifact["targets"][step], device=env.device
                )
            action = torch.empty(env.action_space.shape, device=env.device)
            for descriptor, indices in zip(descriptors, joint_indices, strict=True):
                action[:, descriptor.start : descriptor.stop] = requested.index_select(
                    1, indices
                )
            _, reward, terminated, truncated, info = env.step(action)
            phi, achieved = _potential(env)
            expected = {
                "composed_potential": [],
                "composed_outcome": [],
                "composed_stage_0": [],
                "composed_stage_1": [],
            }
            for row in range(2):
                expected["composed_potential"].append(
                    (0.0 if terminated[row] else 0.99 * phi[row]) - previous_phi[row]
                )
                accepted = bool(info["task_outcome"]["accepted"][row])
                failed = bool(info["task_outcome"]["failure"][row])
                expected["composed_outcome"].append(
                    0.0
                    if previous_paid[row]
                    else 10.0 if accepted else -10.0 if failed else 0.0
                )
                previous_paid[row] |= accepted or failed
                for stage in range(2):
                    expected[f"composed_stage_{stage}"].append(
                        float(
                            achieved[row][stage] and not previous_achieved[row][stage]
                        )
                    )
            weights = {
                "composed_potential": 1.0,
                "composed_outcome": 1.0,
                "composed_stage_0": 0.5,
                "composed_stage_1": 0.5,
            }
            expected_total = [
                sum(weights[name] * values[row] for name, values in expected.items())
                for row in range(2)
            ]
            _finite(
                step,
                reward,
                previous_phi,
                phi,
                expected_total,
                *info["rewards"].values(),
                *expected.values(),
            )
            errors = {
                name: max(
                    abs(a - b)
                    for a, b in zip(info["rewards"][name].tolist(), values, strict=True)
                )
                for name, values in expected.items()
            }
            total_error = max(
                abs(a - b) for a, b in zip(reward.tolist(), expected_total, strict=True)
            )
            max_error = max(max_error, total_error, *errors.values())
            # Re-reading the same transition must not consume/re-pay event state.
            repeated, _ = env.reward_manager.compute({}, action, {})
            _finite(step, repeated)
            max_repeat_error = max(
                max_repeat_error, float((repeated - reward).abs().max())
            )
            trace.append(
                {
                    "step": step,
                    "reward": reward.tolist(),
                    "expected_reward": expected_total,
                    "terms": _json(info["rewards"]),
                    "expected_terms": expected,
                    "phi_before": previous_phi,
                    "phi_after": phi,
                    "achieved_before": previous_achieved,
                    "achieved_after": achieved,
                    "terminated": terminated.tolist(),
                    "truncated": truncated.tolist(),
                    "success": info["success"].tolist(),
                    "failure": info["fail"].tolist(),
                    "invalid_measurement": info["task_outcome"][
                        "invalid_measurement"
                    ].tolist(),
                    "episode_ids": list(env.episode_context.episode_ids),
                    "max_error": max(total_error, *errors.values()),
                }
            )
            previous_phi, previous_achieved = phi, achieved
            if recorder and (
                step % 2 == 1 or bool(terminated[0]) or bool(truncated[0])
            ):
                recorder(env, env_ids=[0], name=f"program_rewards_{args.case}")
                captured_steps.append(step)
            if bool(terminated[0]) or (args.case == "truncated" and bool(truncated[0])):
                break
        stage_sums = [
            [
                sum(item["terms"][f"composed_stage_{i}"][env_id] for item in trace)
                for i in range(2)
            ]
            for env_id in range(2)
        ]
        outcome_sums = [
            sum(row["terms"]["composed_outcome"][i] for row in trace) for i in range(2)
        ]
        # The still-unstarted peer must never get milestone or outcome bonuses.
        checks.append(
            {
                "name": "peer_independent_event_rewards",
                "passed": stage_sums[1] == [0.0, 0.0]
                and outcome_sums[1] == (-10.0 if args.case == "failure" else 0.0),
            }
        )
        if args.case == "success":
            checks.append(
                {"name": "each_milestone_once", "passed": stage_sums[0] == [1.0, 1.0]}
            )
            checks.append(
                {
                    "name": "final_success_once",
                    "passed": outcome_sums[0] == 10.0 and trace[-1]["success"][0],
                }
            )
        elif args.case == "truncated":
            checks.append(
                {
                    "name": "partial_progress_no_final_bonus",
                    "passed": stage_sums[0] == [1.0, 0.0]
                    and outcome_sums[0] == 0.0
                    and trace[-1]["truncated"][0]
                    and not trace[-1]["terminated"][0],
                }
            )
        elif args.case == "failure":
            checks.append(
                {
                    "name": "failure_penalty_without_success",
                    "passed": outcome_sums[0] == -10.0
                    and stage_sums[0] == [0.0, 0.0]
                    and trace[-1]["failure"][0],
                }
            )
        else:
            checks.append(
                {
                    "name": "no_progress_no_event_reward",
                    "passed": stage_sums[0] == [0.0, 0.0] and outcome_sums[0] == 0.0,
                }
            )
        # Check PBRS telescoping independently over the actual terminal boundary.
        shape_returns = [
            sum(
                0.99**t * row["terms"]["composed_potential"][i]
                for t, row in enumerate(trace)
            )
            for i in range(2)
        ]
        endpoint = [
            (
                0.0
                if trace[-1]["terminated"][i]
                else 0.99 ** len(trace) * previous_phi[i]
            )
            - initial_phi[i]
            for i in range(2)
        ]
        telescope_error = max(
            abs(a - b) for a, b in zip(shape_returns, endpoint, strict=True)
        )
        checks.append(
            {
                "name": "finite_valid_measurements",
                "passed": not any(any(row["invalid_measurement"]) for row in trace),
            }
        )
        checks.append({"name": "numeric_oracle", "passed": max_error <= 2e-6})
        checks.append(
            {"name": "repeat_reads_idempotent", "passed": max_repeat_error == 0.0}
        )
        checks.append(
            {"name": "discounted_shaping_identity", "passed": telescope_error <= 2e-6}
        )
        if recorder:
            episode = env.episode_context.episode_ids[0]
            recorder.save_and_clear([0], purpose="diagnostic")
            path = (
                args.output
                / "videos"
                / "diagnostics"
                / "env_000"
                / f"episode_{episode:06d}_program_rewards_{args.case}.mp4"
            )
            video_capture = {
                "path": str(path.relative_to(args.output)),
                "sha256": _hash(path),
                "step_indices": captured_steps,
                "trace_sha256": hashlib.sha256(
                    json.dumps(trace, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
            }
        post_reset = None
        if args.case == "success":
            peer_phi = previous_phi[1]
            peer_episode = env.episode_context.episode_ids[1]
            env.reset(options={"reset_ids": torch.tensor([0], device=env.device)})
            after_reset, _ = _potential(env)
            checks.append(
                {
                    "name": "partial_reset_peer_unchanged",
                    "passed": after_reset[1] == peer_phi
                    and env.episode_context.episode_ids[1] == peer_episode,
                }
            )
            requested = env.robot.get_qpos(target=True).clone()
            action = torch.empty(env.action_space.shape, device=env.device)
            for descriptor, indices in zip(descriptors, joint_indices, strict=True):
                action[:, descriptor.start : descriptor.stop] = requested.index_select(
                    1, indices
                )
            _, next_reward, next_terminal, next_truncated, next_info = env.step(action)
            next_phi, _ = _potential(env)
            expected_next = [0.99 * next_phi[i] - after_reset[i] for i in range(2)]
            _finite(
                "post_reset",
                next_reward,
                next_phi,
                expected_next,
                *next_info["rewards"].values(),
            )
            peer_error = abs(float(next_reward[1]) - expected_next[1])
            reset_error = max(
                abs(a - b)
                for a, b in zip(next_reward.tolist(), expected_next, strict=True)
            )
            no_events = all(
                float(next_info["rewards"][key][i]) == 0.0
                for key in ("composed_outcome", "composed_stage_0", "composed_stage_1")
                for i in range(2)
            )
            checks.append(
                {
                    "name": "post_reset_reward_history_preserved",
                    "passed": reset_error <= 2e-6
                    and no_events
                    and not bool((next_terminal | next_truncated).any()),
                }
            )
            post_reset = {
                "reward": next_reward.tolist(),
                "expected_reward": expected_next,
                "terms": _json(next_info["rewards"]),
                "peer_reward_error": peer_error,
                "max_reward_error": reset_error,
                "episode_ids": list(env.episode_context.episode_ids),
            }
        report = {
            "case": args.case,
            "source": "planning_targets_through_ActionManager_no_model",
            "policy_training": False,
            "seed": args.seed,
            "controller_source_sha256": source_sha256,
            "effective_contract_sha256": contract_sha256,
            "effective_contract": effective_contract,
            "post_reset": post_reset,
            "video_capture": video_capture,
            "program_identity": env.task_program_progress.program_identity,
            "evaluation_identity": env.task_program_progress.compiled.identity,
            "execution_mode": cfg.task_program_execution,
            "program_id": cfg.task_program.program_id,
            "num_envs": 2,
            "steps": len(trace),
            "control_dt": env.step_dt,
            "max_reward_error": max_error,
            "max_repeat_read_error": max_repeat_error,
            "shaping_identity_error": telescope_error,
            "unweighted_stage_event_sums": stage_sums,
            "weighted_milestone_sums": [[v * 0.5 for v in row] for row in stage_sums],
            "outcome_sums": outcome_sums,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
            "elapsed_seconds": time.perf_counter() - start,
            "trace": trace,
        }
        (args.output / f"reward_{args.case}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("trace", "effective_contract", "post_reset")
                }
            ),
            flush=True,
        )
        if not report["passed"]:
            raise RuntimeError(
                "Task Program reward acceptance failed; inspect reward trace"
            )
    finally:
        if recorder:
            recorder.close()
        env.close()


def _report(args, root):
    import subprocess

    reports = []
    for name in ("success", "hold", "truncated", "failure"):
        result = json.loads((args.output / f"reward_{name}.json").read_text())
        if result.get("case") != name:
            raise ValueError(
                f"Acceptance result case must match filename: expected {name!r}, "
                f"got {result.get('case')!r}"
            )
        reports.append(result)
    source_hash = _hash(args.output / "controller_source.json")
    for result in reports:
        if (
            not result["passed"]
            or result.get("controller_source_sha256") != source_hash
        ):
            raise ValueError(
                "Every acceptance case must pass and reference the current sealed action source"
            )
        actual_contract = hashlib.sha256(
            json.dumps(
                result["effective_contract"], sort_keys=True, allow_nan=False
            ).encode()
        ).hexdigest()
        if actual_contract != result["effective_contract_sha256"]:
            raise ValueError("Effective reward contract hash does not match the report")
        contract = result["effective_contract"]
        configuration = contract["configuration"]
        paths = {
            "physical_component_sha256": configuration["environment"]["component"],
            "program_sha256": configuration["task_program"]["program"],
            "evaluation_component_sha256": configuration["evaluation"]["component"],
        }
        if contract["source_inputs"] != _source_inputs(root) or any(
            contract.get(key) != _hash(path) for key, path in paths.items()
        ):
            raise ValueError(
                "Acceptance configuration contents changed or an input hash is missing"
            )
    success = reports[0]
    gamma = 0.99
    events = []
    for row in success["trace"]:
        for index, label in enumerate(("lift", "place")):
            if row["terms"][f"composed_stage_{index}"][0]:
                events.append(
                    {
                        "event": label,
                        "step_zero_based": row["step"],
                        "milestone_bonus": 0.5,
                        "shape": row["terms"]["composed_potential"][0],
                        "program_progress_shape": (
                            0.0
                            if row["terminated"][0]
                            else 0.99 * sum(row["achieved_after"][0]) / 2
                        )
                        - sum(row["achieved_before"][0]) / 2,
                        "skill_goal_shape": (
                            0.0
                            if row["terminated"][0]
                            else 0.99
                            * (row["phi_after"][0] - sum(row["achieved_after"][0]) / 2)
                        )
                        - (row["phi_before"][0] - sum(row["achieved_before"][0]) / 2),
                        "outcome": row["terms"]["composed_outcome"][0],
                        "total": row["reward"][0],
                    }
                )
    metrics = {
        "scope": "Task Program reward composition, without a trained policy or optimizer",
        "program_id": success["program_id"],
        "seed": success["seed"],
        "controller_source_sha256": source_hash,
        "cases": {
            r["case"]: {
                k: r[k]
                for k in (
                    "steps",
                    "num_envs",
                    "passed",
                    "max_reward_error",
                    "max_repeat_read_error",
                    "shaping_identity_error",
                    "weighted_milestone_sums",
                    "outcome_sums",
                    "effective_contract_sha256",
                )
            }
            for r in reports
        },
        "env_step_calls": sum(
            r["steps"] + int(r["post_reset"] is not None) for r in reports
        ),
        "reward_samples": sum(
            (r["steps"] + int(r["post_reset"] is not None)) * r["num_envs"]
            for r in reports
        ),
        "max_reward_error": max(r["max_reward_error"] for r in reports),
        "max_repeat_read_error": max(r["max_repeat_read_error"] for r in reports),
        "max_shaping_identity_error": max(r["shaping_identity_error"] for r in reports),
        "post_reset": success["post_reset"],
        "events": events,
        "oracle": "Independent scalar reward-composition arithmetic using canonical measured skill potentials; not an independent geometry/skill-formula oracle",
    }
    preview = None
    capture = success.get("video_capture")
    if capture is not None:
        raw_video = (args.output / capture["path"]).resolve()
        if (
            not raw_video.is_relative_to(args.output)
            or _hash(raw_video) != capture["sha256"]
        ):
            raise ValueError(
                "Recorded preview video content does not match this acceptance run"
            )
        trace_hash = hashlib.sha256(
            json.dumps(success["trace"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if trace_hash != capture["trace_sha256"]:
            raise ValueError(
                "Recorded preview video is bound to a different reward trace"
            )
        dt = 2 * success["control_dt"]
        captured = [
            row
            for row in success["trace"]
            if row["step"] % 2 == 1 or row["terminated"][0] or row["truncated"][0]
        ]

        if [row["step"] for row in captured] != capture["step_indices"]:
            raise ValueError(
                "Recorded preview frame indices do not match the reward trace"
            )

        def stamp(seconds):
            total = int(round(seconds * 100))
            return f"{total//360000}:{total//6000%60:02d}:{total//100%60:02d}.{total%100:02d}"

        subtitle = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 640",
            "PlayResY: 480",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Audit,DejaVu Sans,16,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,7,10,10,10,1",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        last_event = "none"
        for index, row in enumerate(captured):
            stage = 0.5 * (
                row["terms"]["composed_stage_0"][0]
                + row["terms"]["composed_stage_1"][0]
            )
            outcome = row["terms"]["composed_outcome"][0]
            if row["terms"]["composed_stage_0"][0]:
                last_event = "lift milestone +0.5"
            if row["terms"]["composed_stage_1"][0]:
                last_event = "place milestone +0.5; task success +10"
            text = (
                "Task Program reward audit | scripted planning actions"
                + r"\N"
                + f"step={row['step']+1}  total={row['reward'][0]:+.6f}  shape={row['terms']['composed_potential'][0]:+.6f}"
                + r"\N"
                + f"milestone={stage:+.1f}  outcome={outcome:+.1f}  abs_error={row['max_error']:.2e}"
                + r"\N"
                + "Last event: "
                + last_event
            )
            end = (index + 1) * dt + (1 if index == len(captured) - 1 else 0)
            subtitle.append(
                f"Dialogue: 0,{stamp(index*dt)},{stamp(end)},Audit,,0,0,0,,{text}"
            )
        ass = args.output / "reward_overlay.ass"
        ass.write_text("\n".join(subtitle) + "\n")
        preview = args.output / "task_program_rewards.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(raw_video),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration=1,drawbox=x=0:y=0:w=iw:h=90:color=black@0.65:t=fill,subtitles={ass}",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(preview),
            ],
            check=True,
        )
        metrics["preview_sha256"] = _hash(preview)
        metrics["preview_source"] = {
            "video_sha256": capture["sha256"],
            "trace_sha256": capture["trace_sha256"],
            "frame_count": len(capture["step_indices"]),
        }
    (args.output / "SUMMARY.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = [
        "# Task Program reward 验收",
        "",
        "本轮不构建策略模型或优化器。先由现有规划 Task Program 采集控制目标，再通过普通 ActionManager 驱动配置化 Task Program，核对 env.step() 返回的实际 reward。execution: policy 表示控制动作由外部输入，与是否进行策略训练无关。",
        "",
        "## 验收结果",
        "",
        "| 场景 | 控制步 | 阶段奖励（环境 0） | 结果奖励（环境 0） | 最大逐步误差 |",
        "|---|---:|---|---:|---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r['case']} | {r['steps']} | {r['weighted_milestone_sums'][0]} | {r['outcome_sums'][0]:g} | {r['max_reward_error']:.3e} |"
        )
    lines += [
        "",
        f"四个场景全部通过，共 {metrics['env_step_calls']} 次原生 env.step、{metrics['reward_samples']} 个环境奖励样本（包含一次部分 reset 后的额外转移）。重复读取最大误差为 {metrics['max_repeat_read_error']:g}；折扣塑形累计恒等式最大误差为 {metrics['max_shaping_identity_error']:.3e}。所有实际值、期望值和重复读取值均显式检查为有限数。",
        "",
        "完整轨迹中，每个阶段奖励只触发一次，成功奖励只触发一次。第二个环境始终保持原姿态，没有获得阶段或成功奖励；失败场景两行均按各自的失败状态得到 -10。部分 reset 后继续执行一步，未 reset 环境的下一步奖励也经过独立复算。",
        "",
        "## 数值口径",
        "",
        "原子技能提供当前物理测量得到的局部势函数 u；程序势函数为已完成阶段比例加当前阶段局部势函数的相应份额。本例两个阶段等权：Phi = completed / 2 + active_u / 2。",
        "",
        "每步塑形：r_shape = 0.99 * Phi_next - Phi_before；真实任务终止时 Phi_next 取 0。总奖励：r = r_shape + 0.5 * lift_first + 0.5 * place_first + 10 * success_first - 10 * failure_first。时间截断保留实际 Phi_next，不当作失败或真实终止。",
        "",
        "独立复算器使用 Python 标量公式，不调用奖励实现的势函数组合方法；局部技能势函数取自规范物理观测。本轮验证奖励组合、权重和生命周期，不将它描述成对几何或技能势函数公式的独立证明。",
        "",
        "### 完成事件",
        "",
        "| 事件 | 从 0 开始的 step | 阶段 bonus | 程序进度 shape | 技能目标 shape | 结果 bonus | 总 reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for event in events:
        lines.append(
            f"| {event['event']} | {event['step_zero_based']} | {event['milestone_bonus']:.1f} | {event['program_progress_shape']:+.8f} | {event['skill_goal_shape']:+.8f} | {event['outcome']:+.1f} | {event['total']:+.8f} |"
        )
    lines += [
        "",
        "最终步的总奖励包含负的终止势函数修正，因此不必等于 10 或 10.5；上表按完整公式复算。",
        "",
        "## 文件与复现",
        "",
        f"控制源 SHA-256：`{source_hash}`。每个场景保存有效奖励参数、物理目标参数、控制源依赖文件和有效配置的哈希；回放前验证源配置未改变。",
        "",
        "- [机器可读汇总](" + str(args.output / "SUMMARY.json") + ")",
        *[
            "- [reward_"
            + r["case"]
            + ".json]("
            + str(args.output / f"reward_{r['case']}.json")
            + ")"
            for r in reports
        ],
        "",
        "```bash",
        "python -m examples.learning.task_program_progress.accept_program_rewards --phase collect",
        "xvfb-run -a python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case success --video",
        "python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case hold",
        "python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case truncated",
        "python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case failure",
        "python -m examples.learning.task_program_progress.accept_program_rewards --phase report",
        "```",
    ]
    if preview:
        lines += [
            "",
            "视频中的数字直接来自该次成功回放的 reward 流水，末帧额外停留 1 秒便于查看终止奖励。",
            "",
            f"![Task Program reward]({preview})",
        ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(args.output / "REPORT.md", flush=True)


def main() -> None:
    """Collect planning commands or audit Task Program rewards on native physics."""
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("collect", "replay", "report"), required=True
    )
    parser.add_argument(
        "--case", choices=("success", "hold", "truncated", "failure"), default="success"
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/task_program_reward_acceptance")
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    {"collect": _collect, "replay": _replay, "report": _report}[args.phase](args, root)


if __name__ == "__main__":
    main()
