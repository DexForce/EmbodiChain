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

"""Run one configured expert and dynamic replay with independent physical outcomes.

Invoke with ``python -m embodichain.lab.scripts.evaluate_task_objective``.
This intentionally supports one environment and one concrete deployment per run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any

import yaml

__all__: list[str] = []


def _json_value(value: Any) -> Any:
    """Snapshot resolved configuration without silently stringifying objects."""
    if isinstance(value, Enum):
        return _json_value(value.value)
    if hasattr(type(value), "__members__") and hasattr(value, "name"):
        return {
            "enum": f"{type(value).__module__}:{type(value).__qualname__}",
            "name": value.name,
            "value": _json_value(value.value),
        }
    if value is MISSING:
        return {"unresolved": True}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, type) or callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    # Manager component configs acquire named terms after construction. Walking
    # their live attributes also preserves typed children until this serializer
    # handles them (class_to_dict can erase enum and device identities).
    if hasattr(value, "to_dict") and hasattr(value, "__dict__"):
        return {
            key: _json_value(item)
            for key, item in vars(value).items()
            if not key.startswith("__")
        }
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if type(value).__module__ == "torch" and type(value).__name__ in {
        "device",
        "dtype",
    }:
        return str(value)
    raise TypeError(
        f"Cannot snapshot configuration value of type {type(value).__name__}"
    )


def _component_snapshot(source: Path) -> dict[str, dict[str, str]]:
    """Capture authored deployment dependencies and their content hashes."""
    source = source.resolve()
    result: dict[str, dict[str, str]] = {}
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        content = path.read_bytes()
        result[os.path.relpath(path, source.parent)] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "content": content.decode("utf-8"),
        }
        data = yaml.safe_load(content)
        if not isinstance(data, Mapping):
            return
        for name in ("environment", "embodiment", "scene", "objective"):
            selection = data.get(name)
            if isinstance(selection, Mapping) and isinstance(
                selection.get("component"), str
            ):
                visit(path.parent / selection["component"])
        program = data.get("task_program")
        if isinstance(program, Mapping):
            for name in ("program", "integration", "execution_policy"):
                if isinstance(program.get(name), str):
                    visit(path.parent / program[name])

    visit(source)
    return result


def _outcomes(
    episode: dict[str, Any] | None,
    physical: dict[str, Any] | None,
    *,
    action_source: str,
    program_completed: bool = False,
) -> dict[str, Any]:
    """Keep execution evidence, physical truth and acceptance independent."""
    execution: dict[str, Any] = {"status": "not_available"}
    acceptance: dict[str, Any] = {"status": "not_applicable"}
    if episode is not None:
        segments = episode.get("segments", [])
        statuses = [
            segment.get("metadata", {}).get("runtime", {}).get("status")
            for segment in segments
        ]
        execution = {
            "status": "completed" if program_completed else "incomplete",
            "segment_statuses": statuses,
        }
        acceptance = {
            "status": "evaluated",
            "success": bool(episode["completed"]),
            "reason": episode["terminal_reason"],
        }
    return {
        "action_source": action_source,
        "execution_outcome": execution,
        "physical_outcome": (
            deepcopy(physical) if physical is not None else {"status": "not_evaluated"}
        ),
        "demo_acceptance": acceptance,
        "persistence_status": {
            "dataset": "not_requested",
            "diagnostic_trajectory": "not_written",
        },
    }


def _revision(path: Path) -> dict[str, Any]:
    """Record code provenance where a Git checkout is available."""
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": bool(status)}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _error_snapshot(error: BaseException) -> dict[str, str]:
    """Preserve diagnostics without retaining native resources in frames."""
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def _release_error_tracebacks(error: BaseException) -> None:
    """Release constructor locals throughout chained exceptions before teardown."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
            current.__traceback__ = None
        for chained in (current.__cause__, current.__context__):
            if chained is not None:
                pending.append(chained)


def _run(args: argparse.Namespace) -> Path:
    """Execute the single-deployment sample using existing Gym lifecycle owners."""
    import gymnasium as gym
    import torch
    from tensordict import TensorDict

    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.envs.types import ControllerAction
    from embodichain.lab.gym.envs.wrapper import ReplayWrapper
    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.config_paths import resolve_config_path

    if args.num_envs not in (None, 1):
        raise ValueError("This expert/replay sample requires --num_envs 1.")
    if (
        not math.isfinite(args.initial_position_jitter)
        or not 0 <= args.initial_position_jitter <= 0.02
    ):
        raise ValueError("initial-position-jitter must be between 0 and 0.02 meters.")
    if not 0 <= args.settle_steps <= 100:
        raise ValueError("settle-steps must be between 0 and 100.")
    source = resolve_config_path(args.gym_config)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "deployment": str(source),
        "code": _revision(Path(__file__).resolve().parent),
        "components": _component_snapshot(source),
        "status": "initializing",
        "variation": {
            "scope": "per_episode",
            "kind": "initial_xy_uniform",
            "half_range_m": args.initial_position_jitter,
        },
        "results": [],
    }
    _write_report(report_path, report)
    env = None
    failure: BaseException | None = None
    try:
        discover_task_packages()
        execute_init_hooks()
        args.num_envs = 1
        args.record_trajectory = True
        args.filter_dataset_saving = True

        def configure(config: dict[str, Any]) -> None:
            if "objective" not in config or "task_program" not in config:
                raise ValueError(
                    "Select a deployment with objective and task_program components."
                )
            if args.initial_position_jitter:
                objective_path = source.parent / config["objective"]["component"]
                objective_data = yaml.safe_load(objective_path.read_text())
                radius = args.initial_position_jitter
                config["env"].setdefault("events", {})["objective_initial_pose"] = {
                    "func": "randomize_rigid_object_pose",
                    "mode": "reset",
                    "params": {
                        "entity_cfg": {"uid": objective_data["object_uid"]},
                        "position_range": [
                            [-radius, -radius, 0.0],
                            [radius, radius, 0.0],
                        ],
                        "relative_position": True,
                    },
                }

        cfg, merged, action_config = build_env_cfg_from_args(
            args, gym_config_modifier=configure
        )
        if action_config:
            raise ValueError("This sample does not support --action_config.")
        cfg.trajectory_auto_save = False
        cfg.init_rollout_buffer = True
        report["resolved_config"] = _json_value(cfg)
        report["merged_deployment"] = merged
        report["seed"] = cfg.seed
        _write_report(report_path, report)
        env = gym.make(id=merged["id"], cfg=cfg).unwrapped
        env.reset(seed=cfg.seed, options={"save_data": False})
        report["resolved_config"] = _json_value(cfg)
        report["seed"] = cfg.seed
        objective = env.physical_objective
        obj = env.sim.get_rigid_object(objective.cfg.object_uid)
        report["variation"]["actual_initial_pose"] = (
            obj.get_local_pose().detach().cpu().tolist()
        )
        episode = execute_demo_episode(env)
        episode_metadata = episode.to_metadata()
        report["expert_episode"] = episode_metadata

        # The explicitly reported settling tail is recorded and replayed too.
        available = max(0, env.max_episode_steps - int(env._elapsed_steps[0]))
        tail_steps = min(args.settle_steps, available)
        prior_guard = getattr(env, "_demo_no_auto_reset", False)
        prior_active = env._demo_active_mask.clone()
        env._demo_no_auto_reset = True
        # Keep accepted rollout and segment boundaries frozen. The diagnostic
        # trajectory alone includes the additional physical settling interval.
        env._demo_active_mask.fill_(False)
        try:
            qpos = env.robot.get_qpos().clone()
            command = {"qpos": qpos}
            if env.expert_action_spec.joint_command_mode == "position_velocity":
                command["qvel"] = torch.zeros_like(qpos)
            for _ in range(tail_steps):
                env.step(ControllerAction(TensorDict(command, batch_size=[1])))
                env._demo_active_mask.fill_(True)
                env._write_trajectory_step()
                env._demo_active_mask.fill_(False)
        finally:
            env._demo_no_auto_reset = prior_guard
            env._demo_active_mask.copy_(prior_active)
        report["settle_steps"] = tail_steps
        expert = _outcomes(
            episode_metadata,
            objective.snapshot_row(0),
            action_source="task_program",
            program_completed=env._active_task_program_bridge.program_completed,
        )
        report["results"].append(expert)
        trajectory_path = output / "expert.pt"
        env.save_trajectory(str(trajectory_path))
        trajectory = torch.load(trajectory_path, weights_only=False)
        trajectory["meta"]["action_kind"] = "expert_controller"
        torch.save(trajectory, trajectory_path)
        expert["persistence_status"]["diagnostic_trajectory"] = "written"
        report["trajectory"] = trajectory_path.name
        report["status"] = "expert_finished"
        _write_report(report_path, report)

        if trajectory["meta"]["lengths"][0] > 0:
            replay = ReplayWrapper(env, trajectory, mode="dynamic")
            replay.reset(seed=cfg.seed, options={"save_data": False})
            for _ in range(trajectory["meta"]["lengths"][0]):
                replay.step(None)
            replay_result = _outcomes(
                None, objective.snapshot_row(0), action_source="dynamic_replay"
            )
            replay_result["execution_outcome"] = {
                "status": "completed",
                "steps": trajectory["meta"]["lengths"][0],
            }
            report["results"].append(replay_result)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "error"
        report["error"] = _error_snapshot(error)
        failure = error
        _release_error_tracebacks(error)

    # Drain only after the failed constructor and its exception handler have
    # unwound. A traceback may otherwise keep unregistered native wrappers alive.
    try:
        if env is not None:
            env.close(exit_process=False)
    except BaseException as error:
        report.setdefault("cleanup_errors", []).append(_error_snapshot(error))
        if failure is None:
            failure = error
        _release_error_tracebacks(error)
    try:
        SimulationManager.flush_cleanup_queue()
    except BaseException as error:
        report.setdefault("cleanup_errors", []).append(_error_snapshot(error))
        if failure is None:
            failure = error
        _release_error_tracebacks(error)
    if failure is not None:
        report["status"] = "error"
    _write_report(report_path, report)
    if failure is not None:
        raise failure from None
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sample and print its local report path."""
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

    parser = argparse.ArgumentParser(description=__doc__)
    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-position-jitter", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=20)
    args = parser.parse_args(argv)
    print(_run(args))


if __name__ == "__main__":
    main()
