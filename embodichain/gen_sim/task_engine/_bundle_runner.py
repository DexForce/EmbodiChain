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

"""Private subprocess boundary for canonical Task Program execution."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, NoReturn, Sequence

import numpy as np
import torch

from embodichain.utils.utility import load_config

from .orchestration.artifacts import STATIC_SCENE_MANIFEST_FILENAME
from .orchestration.scene_source import verify_scene_source_fingerprint
from .reporting import write_execution_report
from .semantic_graph import validate_semantic_task_graph

__all__ = ["execute_bundle", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the private runner protocol and execute one bundle."""
    parser = argparse.ArgumentParser(
        prog="embodichain.gen_sim.task_engine._bundle_runner",
        add_help=False,
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--execution-output", required=True)
    protocol, forwarded = parser.parse_known_args(argv)
    return execute_bundle(
        protocol.bundle,
        forwarded,
        execution_output=protocol.execution_output,
    )


def execute_bundle(
    bundle: str | Path,
    forwarded: Sequence[str] = (),
    *,
    execution_output: str | Path | None = None,
) -> int:
    """Execute one semantic bundle through the ordinary Gym Task Program bridge."""
    root = Path(bundle).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {root}")
    output = (
        root / "execution"
        if execution_output is None
        else Path(execution_output).expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    deployment_path = root / "task_program_deployment.yaml"
    program_path = root / "task_program/program.yaml"
    graph_path = root / "semantic_task_graph.json"
    fingerprint_path = root / "integration_fingerprint.json"
    for path in (deployment_path, program_path, graph_path, fingerprint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Bundle is missing required artifact: {path}")
    _verify_source(root)
    graph = validate_semantic_task_graph(_read_json(graph_path))
    fingerprint = _read_json(fingerprint_path)
    deployment = _verify_integration_fingerprint(
        root, deployment_path, graph, fingerprint
    )
    _verify_program_projection(program_path, graph)

    args = _runner_parser().parse_args(list(forwarded))
    args.gym_config = deployment_path.as_posix()
    args.record_trajectory = True
    trajectory_root = output / "trajectory"
    args.trajectory_save_dir = trajectory_root.as_posix()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    result_metadata: dict[str, Any] | None = None
    cargo_report: dict[str, Any] | None = None
    row_success = [False] * int(args.num_envs)
    terminal_reasons = ["runtime_not_started"] * int(args.num_envs)
    failure: dict[str, Any] | None = None
    env: Any = None
    probe: dict[str, Any] | None = None
    try:
        import gymnasium

        from embodichain.lab.gym.envs.demo import execute_demo_episode
        from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
        from embodichain.lab.gym.utils.registration import (
            discover_task_packages,
            execute_init_hooks,
        )

        discover_task_packages()
        execute_init_hooks()
        from ._task_program.assembly import register_deployment
        from embodichain.lab.task_program.language import load_task_program

        deployment_cfg = load_config(deployment_path)
        register_deployment(
            deployment,
            environment_id=deployment_cfg["id"],
            max_episode_steps=int(deployment_cfg["max_episode_steps"]),
        )

        def configure_environment(value: dict[str, Any]) -> None:
            _configure_recording(value, output)
            value.pop("task_program", None)

        env_cfg, gym_config, action_config = build_env_cfg_from_args(
            args,
            gym_config_modifier=configure_environment,
        )
        env_cfg.task_program = load_task_program(
            program_path,
            integration=deployment.selection,
            validation_context=deployment.integration.registration.catalog,
        )
        deployment.integration.registration.catalog.preflight(env_cfg.task_program)
        env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)
        env.reset(seed=args.seed, options={"save_data": False})
        from ._task_program.cargo import capture_cargo, check_cargo

        cargo = capture_cargo(env, load_config(root / "components/scene.yaml"), graph)
        probe = _isolated_initial_probe(env, deployment, env_cfg.task_program)
        probe["integration_fingerprint"] = graph["integration_fingerprint"]
        (output / "planning_probe.json").write_text(
            json.dumps(probe, indent=2), encoding="utf-8"
        )
        if args.plan_probe_only or not any(probe["plan_success"]):
            terminal_reasons = ["initial_plan_rejected"] * int(args.num_envs)
            _write_terminal_robot_state(env, output)
        else:
            result = execute_demo_episode(env, episode_index=0, attempt_id=0)
            result_metadata = result.to_metadata()
            _write_terminal_robot_state(env, output)
            row_success = [bool(value) for value in result.success]
            terminal_reasons = list(result.terminal_reasons) or [
                str(result.terminal_reason)
            ] * len(row_success)
            target = getattr(env, "unwrapped", env)
            buffer = getattr(target, "_traj_buffer", None)
            if cargo and buffer is None:
                raise ValueError("Cargo acceptance requires a recorded trajectory.")
            cargo_report = check_cargo(
                cargo,
                len(row_success),
                trajectory=None if buffer is None else buffer["states"],
                step_counts=getattr(target, "_traj_steps", None),
            )
            (output / "cargo_envelope.json").write_text(
                json.dumps(cargo_report, indent=2), encoding="utf-8"
            )
            for index, accepted in enumerate(cargo_report["accepted_mask"]):
                if not accepted and row_success[index]:
                    row_success[index] = False
                    terminal_reasons[index] = "cargo_envelope_failed"
            if result.completed and all(row_success):
                env.reset()
            else:
                _preserve_failed_execution_recording(
                    env, output, num_envs=len(row_success)
                )
                env.reset(options={"save_data": False})
    except Exception as exc:
        failure = _exception_metadata(exc)
        if env is not None:
            _write_terminal_robot_state(env, output)
            try:
                _preserve_failed_execution_recording(
                    env,
                    output,
                    num_envs=int(args.num_envs),
                )
            except Exception as recording_error:
                failure["recording_error"] = _exception_metadata(recording_error)
            try:
                env.reset(options={"save_data": False})
            except Exception as abort_error:
                failure["abort_error"] = _exception_metadata(abort_error)
    finally:
        if env is not None:
            try:
                getattr(env, "unwrapped", env).close(exit_process=False)
            except Exception as cleanup_error:
                cleanup = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
                if failure is None:
                    failure = cleanup
                else:
                    failure["cleanup_error"] = cleanup
        try:
            from embodichain.lab.sim.sim_manager import SimulationManager

            SimulationManager.flush_cleanup_queue()
        except Exception as cleanup_error:
            cleanup = {
                "type": type(cleanup_error).__name__,
                "message": str(cleanup_error),
            }
            if failure is None:
                failure = cleanup
            else:
                failure["simulation_cleanup_error"] = cleanup

    if args.plan_probe_only:
        probe = probe or {"scope": "initial_call", "plan_success": []}
        probe["failure"] = failure
        probe["integration_fingerprint"] = graph["integration_fingerprint"]
        (output / "planning_probe.json").write_text(
            json.dumps(probe, indent=2), encoding="utf-8"
        )
        _print_json(probe)
        return (
            0
            if failure is None and probe["plan_success"] and all(probe["plan_success"])
            else 2
        )
    if len(row_success) != int(args.num_envs):
        row_success = (row_success + [False] * int(args.num_envs))[: int(args.num_envs)]
    if len(terminal_reasons) != len(row_success):
        terminal_reasons = [
            (
                str(result_metadata.get("terminal_reason", "runtime_failed"))
                if result_metadata is not None
                else "runtime_failed"
            )
        ] * len(row_success)
    report = _build_execution_report(
        graph,
        result_metadata,
        row_success=row_success,
        terminal_reasons=terminal_reasons,
        failure=failure,
        trajectory_root=trajectory_root,
        cargo_report=cargo_report,
    )
    if probe is not None and not any(probe["plan_success"]) and failure is None:
        report["status"] = "rejected"
    write_execution_report(output, report)
    _print_json(report)
    return 0 if report["status"] == "succeeded" else 2


from ._task_program.assembly import probe_initial_plan as _probe_initial_plan


def _isolated_initial_probe(env: Any, deployment: Any, program: Any) -> dict[str, Any]:
    """Keep diagnostic sampling from perturbing the subsequent execution stream."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return _probe_initial_plan(env, deployment, program)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _write_terminal_robot_state(env: Any, output: Path) -> None:
    """Persist measured joint state before reset; this is not success evidence."""
    path = output / "terminal_robot_state.json"
    if path.exists():
        return
    payload: dict[str, Any] = {"schema_version": "gen_sim.terminal-robot-state/v1"}
    try:
        robot = getattr(env, "unwrapped", env).robot
        payload["control_parts"] = {
            name: {
                "joint_ids": list(robot.get_joint_ids(name=name)),
                "joint_names": [
                    robot.joint_names[index] for index in robot.get_joint_ids(name=name)
                ],
                "measured_qpos": robot.get_qpos(name=name).detach().cpu().tolist(),
            }
            for name in robot.cfg.control_parts
        }
    except Exception as exc:
        payload["observation_error"] = _exception_metadata(exc)
    try:
        path.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(
            f"[Task Engine] Terminal robot snapshot unavailable: {exc}", file=sys.stderr
        )


def _build_execution_report(
    graph: dict[str, Any],
    runtime_result: dict[str, Any] | None,
    *,
    row_success: list[bool],
    terminal_reasons: list[str],
    failure: dict[str, Any] | None,
    trajectory_root: Path,
    cargo_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report without turning row-local failure into global failure."""
    semantic_success = _semantic_success_by_env(
        graph,
        runtime_result,
        num_envs=len(row_success),
    )
    if cargo_report is not None:
        row_success = list(row_success)
        terminal_reasons = list(terminal_reasons)
        for env_id, accepted in enumerate(cargo_report["accepted_mask"]):
            if accepted is not True and row_success[env_id]:
                row_success[env_id] = False
                terminal_reasons[env_id] = "cargo_envelope_failed"
        nodes = {node["id"]: node for node in graph["nodes"]}
        for group in graph["task_groups"]:
            carriers = {
                nodes[node_id]["call"].get("arguments", {}).get("object")
                for node_id in group["node_ids"]
                if nodes[node_id].get("task_type") == "E5"
            }
            for content in cargo_report["contents"]:
                if content["carrier"] not in carriers:
                    continue
                for env_id, accepted in enumerate(content["accepted_mask"]):
                    if accepted is not True:
                        semantic_success[env_id][str(group["id"])] = False
    return {
        "schema_version": "task_program_execution_report/v1",
        "status": "succeeded" if failure is None and all(row_success) else "failed",
        "task_id": str(graph["task_id"]),
        "semantic_call_count": len(graph["nodes"]),
        "integration_fingerprint": str(graph["integration_fingerprint"]),
        "record_dir": trajectory_root.as_posix(),
        "environments": [
            {
                "env_id": env_id,
                "success": success and failure is None,
                "terminal_reason": str(terminal_reasons[env_id]),
                "semantic_success": semantic_success[env_id],
            }
            for env_id, success in enumerate(row_success)
        ],
        "runtime_result": deepcopy(runtime_result),
        "failure": failure,
    }


def _exception_metadata(exc: BaseException) -> dict[str, Any]:
    """Preserve one exception and its explicit causal chain as JSON evidence."""
    if not isinstance(exc, BaseException):
        raise TypeError("exc must be a BaseException.")
    result = {"type": type(exc).__name__, "message": str(exc)}
    causes: list[dict[str, str]] = []
    seen = {id(exc)}
    current = exc
    while len(causes) < 8:
        next_error = current.__cause__
        if next_error is None and not current.__suppress_context__:
            next_error = current.__context__
        if next_error is None or id(next_error) in seen:
            break
        seen.add(id(next_error))
        causes.append(
            {
                "type": type(next_error).__name__,
                "message": str(next_error),
            }
        )
        current = next_error
    if causes:
        result["causes"] = causes
    return result


def _preserve_failed_execution_recording(
    env: Any,
    output: Path,
    *,
    num_envs: int,
) -> None:
    """Commit an audit copy of a failed attempt before the reset discards it.

    The common demo executor intentionally asks callers to discard invalid
    episodes.  Task Engine still needs a causal trajectory and camera artifact
    for diagnosing a failed physical boundary, so this helper commits only to
    the isolated execution directory and never to the training dataset.
    """
    target = getattr(env, "unwrapped", env)
    trajectory = getattr(target, "_traj_buffer", None)
    trajectory_steps = getattr(target, "_traj_steps", None)
    if trajectory is not None and trajectory_steps is not None:
        active_env_ids = [
            env_id
            for env_id in range(num_envs)
            if int(trajectory_steps[env_id].item()) > 0
        ]
        if active_env_ids:
            trajectory_dir = output / "trajectory"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            target.save_trajectory(
                trajectory_dir / "failed_attempt.pt",
                env_ids=active_env_ids,
            )

    event_manager = getattr(target, "event_manager", None)
    mode_cfgs = getattr(event_manager, "_mode_functor_cfgs", {})
    try:
        from embodichain.lab.gym.envs.managers.record import record_camera_data

        for configured_functors in mode_cfgs.values():
            for functor_cfg in configured_functors:
                if isinstance(functor_cfg.func, record_camera_data):
                    try:
                        # A first-step failure can precede the recording interval.
                        # Fetch the last rendered state without advancing physics.
                        functor_cfg.func(target, env_ids=None, **functor_cfg.params)
                    finally:
                        functor_cfg.func.save_and_clear()
    except (AttributeError, TypeError, RuntimeError, OSError):
        # The trajectory is the required audit artifact.  Camera persistence is
        # best effort because custom environments may not expose this manager.
        return


def _verify_integration_fingerprint(
    bundle: Path,
    deployment_path: Path,
    graph: dict[str, Any],
    fingerprint: dict[str, Any],
) -> Any:
    """Recompose provider-free integration identity before simulation starts."""
    from ._task_program.assembly import ADAPTER_CONTRACT, load_deployment

    if (
        fingerprint.get("schema_version") != "semantic_integration_fingerprint/v2"
        or fingerprint.get("adapter_contract") != ADAPTER_CONTRACT
    ):
        raise ValueError(
            "Incompatible GenSim bundle contract; regenerate the bundle against "
            "the public 2620929c baseline."
        )

    deployment_cfg = load_config(deployment_path)
    embodiment_cfg = load_config(bundle / "components/embodiment.yaml")
    if type(deployment_cfg) is not dict or type(embodiment_cfg) is not dict:
        raise ValueError("Configured deployment components must be exact mappings.")
    task_program = deployment_cfg.get("task_program")
    skill_profile = embodiment_cfg.get("skill_profile")
    if type(task_program) is not dict or type(skill_profile) is not dict:
        raise ValueError(
            "Configured deployment must declare task_program and skill_profile."
        )
    composed = load_deployment(
        task_program=task_program,
        skill_profile=skill_profile,
        base_dir=bundle,
    )
    if fingerprint.get("integration_id") != composed.integration_id:
        raise ValueError(
            "GenSim bundle integration identity does not match its declaration."
        )
    expected = graph["integration_fingerprint"]
    actual = composed.integration.integration_fingerprint
    recorded = fingerprint.get("integration_fingerprint")
    if expected != recorded or expected != actual:
        raise ValueError(
            "Semantic integration fingerprint drifted before execution: "
            f"graph={expected!r}, recorded={recorded!r}, actual={actual!r}. "
            "Regenerate the GenSim bundle against the current baseline contract."
        )
    recorded_registration = fingerprint.get("registration_fingerprint")
    actual_registration = composed.integration.registration.fingerprint
    if recorded_registration != actual_registration:
        raise ValueError(
            "Semantic registration fingerprint drifted before execution: "
            f"recorded={recorded_registration!r}, actual={actual_registration!r}."
        )
    return composed


def _verify_program_projection(
    program_path: Path,
    graph: dict[str, Any],
) -> None:
    """Require the executable program to be an exact projection of the graph."""
    program = load_config(program_path)
    if type(program) is not dict:
        raise ValueError("Task Program artifact must contain an exact mapping.")
    if program.get("targets") != graph["targets"]:
        raise ValueError("Task Program targets do not match SemanticTaskGraph targets.")
    body = program.get("program")
    if type(body) is not dict or body.get("kind") != "sequence":
        raise ValueError("Generated Task Program must contain one sequence body.")
    items = body.get("items")
    if type(items) is not list or len(items) != len(graph["nodes"]):
        raise ValueError(
            "Task Program segment count does not match SemanticTaskGraph nodes."
        )
    for index, (item, node) in enumerate(zip(items, graph["nodes"], strict=True)):
        if type(item) is not dict or item.get("kind") != "segment":
            raise ValueError(f"Task Program item {index} must be one segment.")
        if item.get("name") != node["id"]:
            raise ValueError(
                f"Task Program item {index} does not match graph node ID "
                f"{node['id']!r}."
            )
        steps = item.get("steps")
        if (
            type(steps) is not dict
            or steps.get("kind") != "invoke"
            or steps.get("call") != node["call"]
        ):
            raise ValueError(
                f"Task Program segment {node['id']!r} is not an exact "
                "SemanticTaskGraph call projection."
            )


def _semantic_success_by_env(
    graph: dict[str, Any],
    runtime_result: dict[str, Any] | None,
    *,
    num_envs: int,
) -> list[dict[str, bool]]:
    """Project verified runtime segment outcomes onto immutable TaskGroups."""
    node_success = {str(node["id"]): [False] * num_envs for node in graph["nodes"]}
    segments = (
        runtime_result.get("segments", []) if type(runtime_result) is dict else []
    )
    if type(segments) is list:
        for segment in segments:
            if type(segment) is not dict:
                continue
            name = segment.get("name")
            if name not in node_success:
                continue
            successes = segment.get("successes")
            active = segment.get("active")
            if type(successes) is not list or len(successes) != num_envs:
                continue
            if type(active) is not list or len(active) != num_envs:
                active = [False] * num_envs
            node_success[str(name)] = [
                is_active is True and success is True
                for is_active, success in zip(active, successes, strict=True)
            ]
    result: list[dict[str, bool]] = []
    for env_id in range(num_envs):
        result.append(
            {
                str(group["id"]): all(
                    node_success[str(node_id)][env_id] for node_id in group["node_ids"]
                )
                for group in graph["task_groups"]
            }
        )
    return result


def _runner_parser() -> argparse.ArgumentParser:
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

    parser = argparse.ArgumentParser(add_help=True)
    add_env_launcher_args_to_parser(parser, require_gym_config=False)
    parser.set_defaults(seed=0)
    parser.add_argument("--plan-probe-only", action="store_true")
    parser.add_argument(
        "--failure-policy", choices=("stop", "continue"), default="stop"
    )
    return parser


def _configure_recording(config: dict[str, Any], output: Path) -> None:
    env_config = config.setdefault("env", {})
    events = env_config.setdefault("events", {})
    events["record_camera"] = {
        "func": "record_camera_data",
        "mode": "interval",
        "interval_step": 5,
        "params": {
            "name": "task_program_audience_view",
            "resolution": [640, 360],
            "intrinsics": [280.0, 280.0, 320.0, 180.0],
            "eye": [0.6, 0.0, 1.8],
            "target": [0.0, 0.0, 0.75],
            "up": [-1.0, 0.0, 0.0],
            "save_path": (output / "videos").as_posix(),
        },
    }


def _verify_source(bundle: Path) -> None:
    static_manifest_path = bundle / STATIC_SCENE_MANIFEST_FILENAME
    if not static_manifest_path.is_file():
        return
    static_manifest = _read_json(static_manifest_path)
    source = static_manifest.get("source", {})
    if isinstance(source, dict) and isinstance(source.get("source_fingerprint"), dict):
        verify_scene_source_fingerprint(source["source_fingerprint"])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON artifact {path}: {exc}") from exc
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _module_entrypoint() -> NoReturn:
    """Run the private protocol and bypass unsafe native interpreter teardown."""
    exit_code = main()
    # DexSim owns native CUDA/Vulkan state whose interpreter-order teardown is
    # unsafe after the explicit environment cleanup above.  This private runner
    # is already an isolated subprocess, so flush the published protocol output
    # and use the same fast-exit boundary as the canonical simulation CLI.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    _module_entrypoint()
