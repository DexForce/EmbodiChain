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
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterator, NoReturn, Sequence

import numpy as np
import torch

from embodichain.utils.utility import load_config

from .orchestration.artifacts import STATIC_SCENE_MANIFEST_FILENAME
from .orchestration.scene_source import verify_scene_source_fingerprint
from .reporting import write_execution_report
from .semantic_graph import validate_semantic_task_graph

__all__ = ["execute_bundle", "main"]


@contextmanager
def _cpu_runtime_threads(env: Any) -> Iterator[None]:
    """Avoid thread-pool overhead for one CPU episode in the isolated worker."""
    runtime = env.unwrapped
    if runtime.num_envs != 1 or torch.device(runtime.device).type != "cpu":
        yield
        return
    from ._task_program.motion import _CPU_BATCH_THREADS

    previous = torch.get_num_threads()
    token = _CPU_BATCH_THREADS.set(min(previous, 8))
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous)
        _CPU_BATCH_THREADS.reset(token)


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
    from .semantic_graph_validation import verify_graph_contract

    task_spec_contract = verify_graph_contract(root, graph)

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
    feedback_evidence: dict[str, Any] = {}
    control_evidence: list[dict[str, Any]] = []
    grip_evidence: dict[str, Any] | None = None
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
            routes = deployment.integration.adapter_factory.press_routes
            twists = getattr(deployment.integration.adapter_factory, "twist_routes", ())
            routes = twists or routes
            if routes:
                source = load_config(root / "components/scene.yaml")["simulation"]
                art = next(
                    a
                    for a in source["articulation"]
                    if a["uid"] == routes[0].binding.object_id
                )
                position = np.asarray(art["init_local_pose"], dtype=float)[:3, 3]
                target = position + np.array([0.0, 0.0, 0.05])
                if twists:
                    pose = np.asarray(art["init_local_pose"], dtype=float)
                    target = (
                        pose[:3, :3] @ np.asarray(twists[0].binding.outer_point)
                        + position
                    )
                camera = deepcopy(value["env"]["events"]["record_camera"])
                camera["params"].update(
                    name="e8_knob_view" if twists else "e9_button_view",
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
        _configure_twist_recorders(
            env_cfg,
            bool(getattr(deployment.integration.adapter_factory, "twist_routes", ())),
        )
        env_cfg.task_program = load_task_program(
            program_path,
            integration=deployment.selection,
            validation_context=deployment.integration.registration.catalog,
        )
        compiled_program = deployment.integration.registration.catalog.preflight(
            env_cfg.task_program
        )
        env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)
        env.reset(seed=args.seed, options={"save_data": False})
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
            with _cpu_runtime_threads(env):
                publish_initial_plan(
                    _isolated_initial_probe(env, deployment, env_cfg.task_program)
                )
            _write_terminal_robot_state(env, output)
            grip_evidence = _snapshot_grip_contact_evidence(env, deployment)
        else:
            from ._task_program.planning_probe import capture_initial_plan

            with (
                _cpu_runtime_threads(env),
                capture_initial_plan(
                    compiled_program, publish_initial_plan, _exception_metadata
                ),
            ):
                result = execute_demo_episode(env, episode_index=0, attempt_id=0)
            result_metadata = result.to_metadata()
            if task_spec_contract is not None:
                from embodichain.gen_sim.task_spec import TaskSpec
                from ._task_program.task_spec_runtime import (
                    capture_final_goals,
                    capture_placement_milestones,
                    evaluate_runtime_evidence,
                )
                from ._task_program.control_evidence import capture_control_operations

                spec = TaskSpec.from_dict(_read_json(root / "task_spec.json"))
                try:
                    final_goals = capture_final_goals(
                        env,
                        deployment,
                        root,
                        graph,
                        task_spec_contract,
                        len(result.success),
                        runtime=result_metadata,
                    )
                except (AttributeError, TypeError, ValueError, KeyError) as error:
                    final_goals = {p.id: {"unavailable": str(error)} for p in spec.goal}
                placement_milestones = capture_placement_milestones(
                    env,
                    deployment,
                    graph,
                    task_spec_contract,
                    result_metadata,
                    len(result.success),
                    spec=spec,
                )
                evaluation = evaluate_runtime_evidence(
                    spec,
                    task_spec_contract,
                    graph,
                    result_metadata,
                    final_goals,
                    num_envs=len(result.success),
                    run_id=output.as_posix(),
                    program=load_config(program_path),
                    constraints=load_config(root / "task_program/constraints.json"),
                    placement_milestones=placement_milestones,
                    control_milestones=capture_control_operations(
                        deployment,
                        graph,
                        result_metadata,
                        load_config(program_path),
                        len(result.success),
                    )
                    or None,
                )
                result_metadata["task_spec_evaluation"] = evaluation
                (output / "task_spec_evaluation.json").write_text(
                    json.dumps(
                        evaluation, ensure_ascii=False, indent=2, allow_nan=False
                    )
                    + "\n",
                    encoding="utf-8",
                )
            _write_terminal_robot_state(env, output)
            row_success = [bool(value) for value in result.success]
            terminal_reasons = list(result.terminal_reasons) or [
                str(result.terminal_reason)
            ] * len(row_success)
            if task_spec_contract is not None:
                result_metadata["program_success"] = list(row_success)
                row_success, terminal_reasons = _apply_task_spec_verdict(
                    result_metadata, row_success, terminal_reasons
                )
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
            feedback_evidence = _snapshot_twist_feedback_audit(env, deployment)
            control_evidence = _snapshot_control_evidence(env, deployment)
            grip_evidence = _snapshot_grip_contact_evidence(env, deployment)
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
            if grip_evidence is None:
                grip_evidence = _snapshot_grip_contact_evidence(env, deployment)
            if not feedback_evidence:
                feedback_evidence = _snapshot_twist_feedback_audit(env, deployment)
            try:
                if not control_evidence:
                    control_evidence = _snapshot_control_evidence(env, deployment)
            except Exception as error:
                failure["control_evidence_error"] = _exception_metadata(error)
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
            if grip_evidence is not None:
                try:
                    (output / "grip_contact_evidence.json").write_text(
                        json.dumps(
                            {
                                **grip_evidence,
                                "integration_fingerprint": graph[
                                    "integration_fingerprint"
                                ],
                            },
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                except (OSError, TypeError, ValueError) as evidence_error:
                    print(
                        f"[Task Engine] Grip trace sidecar unavailable: {evidence_error}",
                        file=sys.stderr,
                    )
            try:
                if control_evidence:
                    (output / "control_operations.json").write_text(
                        json.dumps(control_evidence, indent=2), encoding="utf-8"
                    )
                    for family in ("twist", "press"):
                        records = [r for r in control_evidence if r["family"] == family]
                        if records:
                            payload = (
                                records[0]
                                if len(records) == 1
                                else {"operations": records}
                            )
                            (output / f"{family}_evidence.json").write_text(
                                json.dumps(payload, indent=2), encoding="utf-8"
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
    try:
        from ._task_group_results import task_group_results

        groups = task_group_results(graph, result_metadata, num_envs=len(row_success))
        for env_id, values in enumerate(groups):
            for group_id, group in values.items():
                if (
                    group["execution_status"] == "passed"
                    and report["environments"][env_id]["semantic_success"].get(group_id)
                    is False
                ):
                    group.update(status="failed", reason="group_postcondition_failed")
        (output / "task_group_results.json").write_text(
            json.dumps(
                {
                    "schema_version": "gen_sim_task_group_results/v1",
                    "source": "execution_report.json",
                    "environments": groups,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as evidence_error:
        print(
            f"[Task Engine] E-group summary unavailable: {evidence_error}",
            file=sys.stderr,
        )
    try:
        (output / "recovery_summary.json").write_text(
            json.dumps(_recovery_summary(report), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as evidence_error:
        print(
            f"[Task Engine] Recovery summary unavailable: {evidence_error}",
            file=sys.stderr,
        )
    if task_spec_contract is not None:
        from .semantic_graph_html import write_semantic_task_graph_html

        write_semantic_task_graph_html(
            graph,
            output / "semantic_graph.html",
            report,
            task_spec=_read_json(root / "task_spec.json"),
            validation=task_spec_contract,
        )
    # The parent CLI prints the run summary; the full report stays on disk.
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
    row_success, terminal_reasons = _apply_task_spec_verdict(
        runtime_result, row_success, terminal_reasons
    )
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


def _apply_task_spec_verdict(
    runtime_result: dict[str, Any] | None,
    row_success: list[bool],
    terminal_reasons: list[str],
) -> tuple[list[bool], list[str]]:
    """Keep legacy success unchanged; new contracts require a verified verdict."""
    if not runtime_result or "task_spec_evaluation" not in runtime_result:
        return row_success, terminal_reasons
    rows = runtime_result["task_spec_evaluation"]["environments"]
    if [r.get("env_id") for r in rows] != list(range(len(row_success))):
        raise ValueError("TaskSpec evaluation environment identities differ.")
    success, reasons = list(row_success), list(terminal_reasons)
    for row in rows:
        index = row["env_id"]
        if success[index] and not (
            row.get("certificate_ready") is True and row.get("status") == "satisfied"
        ):
            success[index] = False
            reasons[index] = (
                "task_spec_failed"
                if row.get("status") == "failed"
                else "task_spec_unverified"
            )
    return success, reasons


def _snapshot_grip_contact_evidence(env: Any, deployment: Any) -> dict[str, Any] | None:
    """Copy the registered native observer's trace without creating or sampling it."""
    from ._task_program.grip_evidence import (
        NativeGripEvidenceFactory,
        NativeGripObserver,
        SENSOR_UID,
    )

    registration = getattr(
        getattr(deployment, "integration", None), "registration", None
    )
    if not isinstance(
        getattr(registration, "control_part_evidence_factory", None),
        NativeGripEvidenceFactory,
    ):
        return None
    try:
        sensor = getattr(env, "unwrapped", env).sim.get_sensor(SENSOR_UID)
        observer = getattr(sensor, "_gen_sim_grip_observer", None)
        if not isinstance(observer, NativeGripObserver):
            raise ValueError("Registered native grip observer is unavailable.")
        return observer.snapshot_trace()
    except Exception as error:
        return {
            "schema_version": "gen_sim.grip-contact-evidence/v1",
            "source": "native_contact_sensor",
            "observed": False,
            "status": "unavailable",
            "snapshot_error": _exception_metadata(error),
        }


def _recovery_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Classify real workflow recovery independently of the unchanged verdict."""
    rows = report["environments"]
    count = len(rows)
    result = report.get("runtime_result") or {}
    if not isinstance(result, dict):
        result = {}
    successes = result.get("program_success", result.get("success"))
    if (
        not isinstance(successes, list)
        or len(successes) != count
        or any(type(value) is not bool for value in successes)
    ):
        successes = [None] * count
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    known = [bool(segments)] * count
    covered = [False] * count
    attempts, recovered, reacquired = [0] * count, [0] * count, [0] * count
    traces = []
    roles = {"retry_retained", "reacquire", "retry_reacquired"}
    for segment in segments:
        if not isinstance(segment, dict):
            known = [False] * count
            continue
        active = segment.get("active")
        if (
            not isinstance(active, list)
            or len(active) != count
            or any(type(value) is not bool for value in active)
        ):
            known = [False] * count
            continue
        active_ids = [index for index, value in enumerate(active) if value]
        metadata = segment.get("metadata")
        runtime = metadata.get("runtime") if isinstance(metadata, dict) else None
        if not isinstance(runtime, dict):
            for env_id in active_ids:
                known[env_id] = False
            continue
        env_ids = runtime.get("env_ids", [])
        ledger = runtime.get("workflow_recoveries")
        if (
            runtime.get("kind") != "skill_result"
            or runtime.get("schema_version") != 2
            or not isinstance(env_ids, list)
            or any(
                type(value) is not int or value < 0 or value >= count
                for value in env_ids
            )
            or len(set(env_ids)) != len(env_ids)
            or not isinstance(ledger, list)
        ):
            for env_id in active_ids:
                known[env_id] = False
            continue
        for env_id in active_ids:
            covered[env_id] |= env_id in env_ids
            known[env_id] &= env_id in env_ids
        # Core recovery is not silently reclassified from an empty workflow ledger.
        events, failures = runtime.get("events"), runtime.get("failures")
        if (
            not isinstance(events, list)
            or not isinstance(failures, list)
            or failures
            or any(
                not isinstance(event, dict)
                or type(event.get("kind")) is not str
                or event.get("kind")
                in {"replanned", "recovery_required", "recovery_exhausted"}
                for event in events or []
            )
        ):
            for env_id in active_ids:
                known[env_id] = False
        for trace in ledger:
            traces.append(
                {
                    "segment": segment.get("name"),
                    "env_ids": list(env_ids),
                    "trace": deepcopy(trace),
                }
            )
            if not isinstance(trace, dict):
                for env_id in active_ids:
                    known[env_id] = False
                continue
            masks = trace.get("masks", {})
            valid = (
                type(trace.get("role")) is str
                and trace["role"] in roles
                and isinstance(masks, dict)
                and all(
                    isinstance(masks.get(name), list)
                    and len(masks[name]) == len(env_ids)
                    and all(type(value) is bool for value in masks[name])
                    for name in ("entered", "completed", "failed")
                )
            )
            if not valid:
                for env_id in active_ids:
                    known[env_id] = False
                continue
            for index, env_id in enumerate(env_ids):
                entered, completed, failed = (
                    masks[name][index] for name in ("entered", "completed", "failed")
                )
                if (completed or failed) and not entered or completed and failed:
                    known[env_id] = False
                    continue
                if not entered:
                    continue
                attempts[env_id] += 1
                call = trace.get("call")
                commands = call.get("command_count") if isinstance(call, dict) else None
                if completed and type(commands) is int and commands > 0:
                    recovered[env_id] += 1
                    reacquired[env_id] += int(trace["role"] == "reacquire")
                else:
                    known[env_id] = False
    return {
        "schema_version": "gen_sim.recovery-summary/v1",
        "source": "runtime.workflow_recoveries",
        "scope": "program_completion_not_taskspec_verdict",
        "original_failure_source": "execution_report.json:runtime_result.segments[].metadata.runtime",
        "environments": [
            {
                "env_id": row["env_id"],
                "program_success": successes[index],
                "certified_success": row["success"],
                "completion_kind": (
                    "not_completed"
                    if successes[index] is False
                    else (
                        "recovered_completion"
                        if successes[index] is True and recovered[index]
                        else (
                            "first_pass_completion"
                            if successes[index] is True
                            and known[index]
                            and covered[index]
                            and attempts[index] == 0
                            else "unknown"
                        )
                    )
                ),
                "workflow_recovery_attempt_count": attempts[index],
                "completed_recovery_call_count": recovered[index],
                "completed_reacquisition_call_count": reacquired[index],
            }
            for index, row in enumerate(rows)
        ],
        "workflow_recoveries": traces,
    }


def _snapshot_control_evidence(env: Any, deployment: Any) -> list[dict[str, Any]]:
    """Own all operation evidence before episode reset changes native state."""
    from ._task_program.press_runtime import PressContactSensor

    records = []
    adapter = deployment.integration.adapter_factory
    for family in ("twist", "press"):
        for route in getattr(adapter, family + "_routes", ()):
            sensor = getattr(env, "unwrapped", env).sim.get_sensor(route.sensor_uid)
            if not isinstance(sensor, PressContactSensor):
                continue
            record = {
                "family": family,
                "operation_id": route.operation_id,
                "route": route.payload(),
                "acceptance": deepcopy(sensor.acceptance),
                "target_actor_id": sensor.target_actor,
                "parent_actor_id": sensor.parent_actor,
                "contact_envelopes": deepcopy(sensor.contact_envelopes),
                "finger_actor_ids": sorted(sensor.finger_actors),
                "runtime_mass": sensor.art.get_mass(route.binding.link).tolist(),
                "runtime_limits": sensor.art.get_qpos_limits()[
                    0, sensor.joint_index
                ].tolist(),
                "runtime_drive": [
                    v.tolist()
                    for v in sensor.art.get_joint_drive(joint_ids=[sensor.joint_index])
                ],
                "trace": deepcopy(sensor.trace),
            }
            if family == "twist":
                record.update(
                    table_actor_id=sensor.table_actor,
                    startup_evidence=sensor.startup_evidence(),
                    geometry_evidence=sensor.geometry_evidence(),
                )
                feedback = getattr(sensor, "_feedback_state", None)
                if feedback is not None:
                    try:
                        record["feedback_audit"] = deepcopy(list(feedback.audit))
                    except Exception as error:
                        record["feedback_audit_error"] = {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
            records.append(record)
    return records


def _snapshot_twist_feedback_audit(env: Any, deployment: Any) -> dict[str, Any]:
    """Own optional diagnostics before reset clears the episode ledger."""
    try:
        routes = getattr(deployment.integration.adapter_factory, "twist_routes", ())
        audits = {}
        for route in routes:
            if getattr(route, "feedback_enabled", False) is not True:
                continue
            sensor = getattr(env, "unwrapped", env).sim.get_sensor(route.sensor_uid)
            feedback = getattr(sensor, "_feedback_state", None)
            if feedback is not None:
                audits[route.operation_id] = deepcopy(list(feedback.audit))
        return (
            {"feedback_audit": next(iter(audits.values()))}
            if len(audits) == 1
            else {"feedback_audits": audits} if audits else {}
        )
    except Exception as error:
        return {
            "feedback_audit_error": {
                "type": type(error).__name__,
                "message": str(error),
            }
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
    from ._task_program.identity import bundle_registration_fingerprint

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
    actual_registration = bundle_registration_fingerprint(
        composed.integration.registration
    )
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
        # A runtime-only placement selector must name this graph stage before
        # it can be omitted from the immutable call projection.
        if type(projected_call) is dict:
            projected_call = deepcopy(projected_call)
            arguments = projected_call.get("arguments")
            graph_arguments = node["call"].get("arguments")
            declared_stage = (
                type(graph_arguments) is dict and "stage_id" in graph_arguments
            )
            if (
                type(arguments) is dict
                and "stage_id" in arguments
                and not declared_stage
            ):
                if (
                    projected_call.get("kind") != "registered"
                    or projected_call.get("call_id")
                    not in {
                        "simulation.place_relative",
                        "gen_sim.place_upright",
                        "gen_sim.stack_place",
                    }
                    or type(arguments["stage_id"]) is not str
                    or arguments["stage_id"] != node["task_instance_id"]
                ):
                    raise ValueError(
                        f"Task Program segment {node['id']!r} runtime stage_id "
                        "must select its exact relative-placement task instance."
                    )
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


def _configure_twist_recorders(env_cfg: Any, enabled: bool) -> None:
    """Select the private E8 recorder after the legacy short-name parser.

    The Gym YAML parser resolves only names in manager modules. Keep its normal
    recorder declaration and replace the already decoded callable locally,
    without modifying the shared manager registry or other task recorders.
    """
    if not enabled:
        return
    from ._task_program.twist_recording import E8CameraRecorder

    for name in ("record_camera", "record_press_camera"):
        getattr(env_cfg.events, name).func = E8CameraRecorder


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
