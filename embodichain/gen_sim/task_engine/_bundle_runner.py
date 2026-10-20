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
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import sys
import tempfile
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
    normative_spec = None
    if (root / "task_spec.json").is_file():
        from embodichain.gen_sim.task_spec import TaskSpec, validate_task_spec

        normative_spec = TaskSpec.from_dict(_read_json(root / "task_spec.json"))
        validate_task_spec(normative_spec)
    frozen_scene = (
        _read_json(root / "scene_variant.json")
        if (root / "scene_variant.json").is_file()
        else None
    )
    if frozen_scene is not None and normative_spec is None:
        raise ValueError(
            "A frozen scene-layout bundle requires its authoritative TaskSpec."
        )
    if normative_spec is not None:
        from embodichain.gen_sim.task_spec import scene_layout_requirement

        if (
            scene_layout_requirement(normative_spec) is not None
            and frozen_scene is None
        ):
            raise ValueError(
                "TaskSpec scene-layout execution requires a frozen SceneVariant."
            )
    task_roles = (
        _verify_task_spec_bundle(root, normative_spec, frozen_scene)
        if normative_spec is not None
        else {}
    )

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
    initial_observations: tuple[dict, ...] = ()
    recording_checks = ExitStack()
    finish_task = None
    prepare_callback = None

    def reset_episode(**kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        if prepare_callback is not None:
            options["episode_prepare"] = prepare_callback
        return env.reset(options=options, **kwargs)

    runtime_assets = tempfile.TemporaryDirectory(prefix="ec-task-assets-")
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
            routes = deployment.integration.adapter_factory.press_routes
            if routes:
                source = load_config(root / "components/scene.yaml")["simulation"]
                art = next(
                    a
                    for a in source["articulation"]
                    if a["uid"] == routes[0].binding.object_id
                )
                position = np.asarray(art["init_local_pose"], dtype=float)[:3, 3]
                target = position + np.array([0.0, 0.0, 0.05])
                camera = deepcopy(value["env"]["events"]["record_camera"])
                camera["params"].update(
                    name="e9_button_view",
                    resolution=[960, 540],
                    intrinsics=[800.0, 800.0, 480.0, 270.0],
                    eye=(target + np.array([0.30, -0.25, 0.20])).tolist(),
                    target=target.tolist(),
                    up=[0.0, 0.0, 1.0],
                )
                value["env"]["events"]["record_press_camera"] = camera
            value.pop("task_program", None)

        env_cfg, gym_config, action_config = build_env_cfg_from_args(
            args,
            gym_config_modifier=configure_environment,
        )
        _alias_runtime_meshes(env_cfg, Path(runtime_assets.name))
        env_cfg.task_program = load_task_program(
            program_path,
            integration=deployment.selection,
            validation_context=deployment.integration.registration.catalog,
        )
        compiled_program = deployment.integration.registration.catalog.preflight(
            env_cfg.task_program
        )
        env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)
        target = getattr(env, "unwrapped", env)
        if normative_spec is not None or frozen_scene is not None:
            rows = list(range(target.num_envs))
            physical_scene = load_config(root / "components/scene.yaml")["simulation"]
            frozen_poses = {
                entry["uid"]: entry["init_local_pose"]
                for section in ("background", "rigid_object")
                for entry in physical_scene.get(section, [])
                if "init_local_pose" in entry
            }
            finish_task = recording_checks.enter_context(
                target.episode_context.check("gensim/task_spec", rows)
            )

            def prepare_episode() -> None:
                nonlocal initial_observations, result_metadata, terminal_reasons
                with target.episode_context.check(
                    "gensim/frozen_layout", rows
                ) as finish_initial:
                    if frozen_scene is not None:
                        from embodichain.lab.sim.scene.expansion import (
                            ScenePoseChange,
                            SceneVariant,
                            apply_scene_variants,
                        )

                        # Reset events may settle or reposition authored rigids.
                        # Restore the complete frozen component before first observations.
                        restore = SceneVariant(
                            frozen_scene["layout"]["parent_scene_id"],
                            frozen_scene["seed"],
                            frozen_scene["layout"]["ordinal"],
                            tuple(
                                ScenePoseChange(uid, pose)
                                for uid, pose in frozen_poses.items()
                            ),
                        )
                        apply_scene_variants(
                            target.sim,
                            [restore] * target.num_envs,
                            env_ids=rows,
                            movable_entity_ids=tuple(frozen_poses),
                        )
                        for uid, expected in frozen_poses.items():
                            actual = target.sim.get_rigid_object(uid).get_local_pose(
                                to_matrix=True
                            )
                            if not np.allclose(
                                actual.detach().cpu().numpy(),
                                np.asarray(expected)[None],
                                atol=2e-6,
                                rtol=0,
                            ):
                                raise ValueError(
                                    f"Runtime initial pose differs from frozen layout: {uid!r}."
                                )
                    from .scene_expansion import (
                        _observe_initial_conditions,
                        evaluate_scene_task_spec,
                    )
                    from embodichain.gen_sim.task_spec import evaluate_predicate

                    parents = (frozen_scene or {}).get("parent_by_uid", {})
                    initial_observations = _observe_initial_conditions(
                        normative_spec, target.sim, task_roles, parents
                    )
                    checks = [
                        [
                            evaluate_predicate(predicate, observed).to_dict()
                            for predicate in normative_spec.init
                        ]
                        for observed in initial_observations
                    ]
                    (output / "task_spec_initial_conditions.json").write_text(
                        json.dumps(
                            {"observations": initial_observations, "checks": checks},
                            indent=2,
                        )
                    )
                    if any(
                        check["status"] != "satisfied"
                        for row in checks
                        for check in row
                    ):
                        result_metadata = {"completed": False, "segments": []}
                        evaluations = evaluate_scene_task_spec(
                            normative_spec,
                            graph,
                            result_metadata,
                            num_envs=len(row_success),
                            initial_observations=initial_observations,
                        )
                        result_metadata["task_spec_evaluation"] = {
                            "semantic_hash": normative_spec.semantic_hash,
                            "rows": list(evaluations),
                        }
                        terminal_reasons = [
                            "task_spec_" + value["acceptance_status"]
                            for value in evaluations
                        ]
                        (output / "task_spec_evaluation.json").write_text(
                            json.dumps(
                                result_metadata["task_spec_evaluation"], indent=2
                            )
                        )
                        raise ValueError(
                            "TaskSpec initial conditions are failed or unverified; execution and persistence are blocked."
                        )
                    finish_initial(
                        accepted=True,
                        metadata=frozen_scene
                        or {"task_spec_semantic_hash": normative_spec.semantic_hash},
                    )

            prepare_callback = prepare_episode
        reset_episode(seed=args.seed, options={"save_data": False})
        from ._task_program.cargo import capture_cargo, check_cargo

        cargo = capture_cargo(env, load_config(root / "components/scene.yaml"), graph)

        def publish_initial_plan(evidence: dict[str, Any]) -> None:
            nonlocal probe
            probe = {
                **evidence,
                "integration_fingerprint": graph["integration_fingerprint"],
            }
            (output / "planning_probe.json").write_text(
                json.dumps(probe, indent=2), encoding="utf-8"
            )

        if args.plan_probe_only:
            publish_initial_plan(
                _isolated_initial_probe(env, deployment, env_cfg.task_program)
            )
            _write_terminal_robot_state(env, output)
            if finish_task is not None:
                finish_task(accepted=False, reasons="plan_probe_only")
                recording_checks.close()
        else:
            from ._task_program.planning_probe import capture_initial_plan

            with capture_initial_plan(
                compiled_program, publish_initial_plan, _exception_metadata
            ):
                result = execute_demo_episode(env, episode_index=0, attempt_id=0)
            result_metadata = result.to_metadata()
            _write_terminal_robot_state(env, output)
            row_success = [bool(value) for value in result.success]
            terminal_reasons = list(result.terminal_reasons) or [
                str(result.terminal_reason)
            ] * len(row_success)
            initial_rejected = (
                probe is not None
                and bool(probe["plan_success"])
                and not any(probe["plan_success"])
            )
            if not initial_rejected:
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
            if normative_spec is not None:
                from .scene_expansion import evaluate_scene_task_spec

                evaluations = evaluate_scene_task_spec(
                    normative_spec,
                    graph,
                    result_metadata,
                    num_envs=len(row_success),
                    initial_observations=initial_observations,
                )
                result_metadata["task_spec_evaluation"] = {
                    "semantic_hash": normative_spec.semantic_hash,
                    "rows": list(evaluations),
                }
                for row, evaluation in enumerate(evaluations):
                    if evaluation["certificate_ready"] is not True:
                        row_success[row] = False
                        terminal_reasons[row] = (
                            "task_spec_" + evaluation["acceptance_status"]
                        )
                (output / "task_spec_evaluation.json").write_text(
                    json.dumps(result_metadata["task_spec_evaluation"], indent=2)
                )
                target = getattr(env, "unwrapped", env)
                finish_task(
                    accepted=row_success,
                    metadata={
                        row: {
                            "task_spec_evaluation": evaluation,
                            "execution_success": row_success[row],
                        }
                        for row, evaluation in enumerate(evaluations)
                    },
                    reasons={
                        row: terminal_reasons[row]
                        for row, passed in enumerate(row_success)
                        if not passed
                    },
                )
                recording_checks.close()
                accepted_rows = [
                    row for row, accepted in enumerate(row_success) if accepted
                ]
                rejected_rows = [
                    row for row, accepted in enumerate(row_success) if not accepted
                ]
                if rejected_rows:
                    _preserve_failed_execution_recording(
                        env, output, num_envs=len(row_success), env_ids=rejected_rows
                    )
                reset_episode(
                    options={
                        "save_data": bool(accepted_rows),
                        "commit_env_ids": accepted_rows,
                    }
                )
            elif result.completed and all(row_success):
                reset_episode()
            else:
                _preserve_failed_execution_recording(
                    env, output, num_envs=len(row_success)
                )
                reset_episode(options={"save_data": False})
    except Exception as exc:
        recording_checks.__exit__(type(exc), exc, exc.__traceback__)
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
                reset_episode(options={"save_data": False})
            except Exception as abort_error:
                failure["abort_error"] = _exception_metadata(abort_error)
    finally:
        recording_checks.__exit__(*sys.exc_info())
        if env is not None:
            try:
                from ._task_program.press_runtime import SENSOR_UID, PressContactSensor

                sensor = (
                    getattr(env, "unwrapped", env).sim.get_sensor(SENSOR_UID)
                    if getattr(
                        deployment.integration.adapter_factory, "press_routes", ()
                    )
                    else None
                )
                if isinstance(sensor, PressContactSensor):
                    (output / "press_evidence.json").write_text(
                        json.dumps(
                            {
                                "route": sensor.route.payload(),
                                "acceptance": sensor.acceptance,
                                "target_actor_id": sensor.target_actor,
                                "parent_actor_id": sensor.parent_actor,
                                "contact_envelopes": sensor.contact_envelopes,
                                "finger_actor_ids": sorted(sensor.finger_actors),
                                "runtime_mass": sensor.art.get_mass(
                                    sensor.route.binding.link
                                ).tolist(),
                                "runtime_limits": sensor.art.get_qpos_limits()[
                                    0, sensor.joint_index
                                ].tolist(),
                                "runtime_drive": [
                                    v.tolist()
                                    for v in sensor.art.get_joint_drive(
                                        joint_ids=[sensor.joint_index]
                                    )
                                ],
                                "trace": sensor.trace,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            except Exception as evidence_error:
                evidence_failure = {
                    "type": type(evidence_error).__name__,
                    "message": str(evidence_error),
                }
                if failure is None:
                    failure = evidence_failure
                else:
                    failure["press_evidence_error"] = evidence_failure
                row_success = [False] * int(args.num_envs)
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
        runtime_assets.cleanup()

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
    if probe is None:
        probe = {
            "scope": "initial_call",
            "planning_source": "execution",
            "plan_success": [],
            "task_success": None,
            "failure": failure,
            "integration_fingerprint": graph["integration_fingerprint"],
            "diagnostic": "Execution did not reach initial atomic planning.",
        }
        (output / "planning_probe.json").write_text(
            json.dumps(probe, indent=2), encoding="utf-8"
        )
    if probe["plan_success"] and not any(probe["plan_success"]) and failure is None:
        report["status"] = "rejected"
        for row in report["environments"]:
            row["terminal_reason"] = "initial_plan_rejected"
    write_execution_report(output, report)
    _print_json(report)
    return 0 if report["status"] == "succeeded" else 2


from ._task_program.planning_probe import probe_initial_plan as _probe_initial_plan


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
    env_ids: Sequence[int] | None = None,
) -> None:
    """Commit an audit copy of a failed attempt before the reset discards it.

    The common demo executor intentionally asks callers to discard invalid
    episodes.  Task Engine still needs a causal trajectory and camera artifact
    for diagnosing a failed physical boundary, so this helper commits only to
    the isolated execution directory and never to the training dataset.
    """
    target = getattr(env, "unwrapped", env)
    selected = list(range(num_envs)) if env_ids is None else list(env_ids)
    trajectory = getattr(target, "_traj_buffer", None)
    trajectory_steps = getattr(target, "_traj_steps", None)
    if trajectory is not None and trajectory_steps is not None:
        active_env_ids = [
            env_id for env_id in selected if int(trajectory_steps[env_id].item()) > 0
        ]
        if active_env_ids:
            trajectory_dir = output / "diagnostics"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            target.save_trajectory(
                trajectory_dir / "failed_attempt.pt",
                env_ids=active_env_ids,
                purpose="diagnostic",
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
                        functor_cfg.func(target, env_ids=selected, **functor_cfg.params)
                    finally:
                        functor_cfg.func.save_and_clear(
                            env_ids=selected, purpose="diagnostic"
                        )
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
        projected_call = steps.get("call") if type(steps) is dict else None
        # Long-horizon relative-placement routes carry a runtime-only stage
        # selector.  It is deliberately absent from the semantic graph so
        # graph identity and old callers remain stable; strip it before
        # checking the immutable graph-to-program projection.
        if type(projected_call) is dict:
            projected_call = deepcopy(projected_call)
            arguments = projected_call.get("arguments")
            if type(arguments) is dict and "stage_id" in arguments:
                arguments = dict(arguments)
                arguments.pop("stage_id", None)
                projected_call["arguments"] = arguments
        if (
            type(steps) is not dict
            or steps.get("kind") != "invoke"
            or projected_call != node["call"]
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
    parser.add_argument("--failure-policy", choices=("stop",), default="stop")
    return parser


def _alias_runtime_meshes(config: Any, directory: Path) -> None:
    """Keep native material names below their path-derived 255-byte limit.

    Nested transactional bundle paths can exceed that limit before publication.
    Alias asset directories only in the in-memory runtime configuration; source
    files, fingerprints, persisted bundles and relative texture references retain
    their original meaning. The runner owns these aliases until native cleanup.
    """
    from embodichain.lab.sim.shapes import MeshCfg

    aliases: dict[Path, Path] = {}
    for entity in (*config.background, *config.rigid_object):
        shape = entity.shape
        if not isinstance(shape, MeshCfg) or not isinstance(shape.fpath, str):
            continue
        original = Path(shape.fpath).resolve(strict=True)
        parent = original.parent
        if parent not in aliases:
            alias = directory / str(len(aliases))
            alias.symlink_to(parent, target_is_directory=True)
            aliases[parent] = alias
        shape.fpath = str(aliases[parent] / original.name)


def _verify_task_spec_bundle(root: Path, spec: Any, frozen: dict | None) -> dict:
    """Bind normative semantics and saved poses before creating a simulator."""
    from .task_spec import (
        task_spec_from_candidate,
        task_spec_role_bindings,
        _preserves_task_spec,
    )
    from .contracts import validate_task_candidate_set

    candidates = validate_task_candidate_set(
        _read_json(root / "task_candidate_set.json")
    )
    bindings = _read_json(root / "role_bindings.json")
    selected = next(
        (
            value
            for value in candidates["candidates"]
            if value["candidate_id"] == bindings["candidate_id"]
        ),
        None,
    )
    if selected is None or not _preserves_task_spec(
        spec, task_spec_from_candidate(selected)
    ):
        raise ValueError("Bundle semantics contradict the authoritative TaskSpec.")
    roles = task_spec_role_bindings(selected, bindings)
    if frozen is not None:
        from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant

        data = frozen["layout"]
        variant = SceneVariant(
            data["parent_scene_id"],
            data["seed"],
            data["ordinal"],
            tuple(ScenePoseChange(uid, pose) for uid, pose in data["poses"].items()),
            tuple(tuple(pair) for pair in data["metadata"]),
        )
        provenance = dict(variant.metadata)
        if (
            frozen["geometry_status"] != "passed"
            or frozen["source_sha256"] != provenance.get("source_sha256")
            or frozen["geometry_sha256"] != provenance.get("geometry_sha256")
        ):
            raise ValueError("Frozen scene provenance differs from its SceneVariant.")
        if (
            frozen["task_spec_semantic_hash"] != spec.semantic_hash
            or frozen["variant_id"] != variant.variant_id
            or frozen["seed"] != variant.seed
            or frozen["proposal_ordinal"] != variant.ordinal
            or {
                role: tuple(uids) for role, uids in frozen["task_role_bindings"].items()
            }
            != roles
            or frozen["role_bindings"] != bindings["reference_bindings"]
        ):
            raise ValueError(
                "Frozen scene identity or task-role bindings differ from the bundle."
            )
        source = load_config(root / "components/scene.yaml")["simulation"]
        entries = {
            item["uid"]: item
            for section in ("background", "rigid_object")
            for item in source.get(section, [])
        }
        for change in variant.pose_changes:
            entry = entries.get(change.entity_uid, {})
            if "init_local_pose" not in entry or not np.allclose(
                np.asarray(entry["init_local_pose"]),
                change.arena_pose,
                atol=1e-9,
                rtol=0,
            ):
                raise ValueError(
                    "Bundle initial poses differ from its frozen SceneVariant."
                )
    return roles


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
