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

"""Run an immutable Task Engine bundle through ordinary shared Task Program APIs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from .bundle_loading import load_current_task_program_bundle
from .orchestration.workflow_artifacts import (
    validate_run_output_directory,
    write_run_json,
)
from .reporting import (
    TASK_PROGRAM_EXECUTION_REPORT_SCHEMA,
    build_bundle_execution_report,
    build_recovery_summary,
    verify_bundle_execution_report,
    write_execution_report,
)

__all__ = ["execute_bundle", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the existing worker protocol and preserve its process exit status.

    Args:
        argv: Original worker arguments; remaining launcher arguments are forwarded.

    Returns:
        Zero for accepted execution/probe and nonzero for rejection or failure.
    """
    parser = argparse.ArgumentParser(prog=__name__, add_help=False)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--execution-output", required=True)
    parser.add_argument(
        "--acceptance-mode", choices=("strict", "experiment"), default="strict"
    )
    selected, forwarded = parser.parse_known_args(argv)
    return execute_bundle(
        selected.bundle,
        forwarded,
        execution_output=selected.execution_output,
        acceptance_mode=selected.acceptance_mode,
    )


def _runner_parser() -> argparse.ArgumentParser:
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

    parser = argparse.ArgumentParser(add_help=True)
    add_env_launcher_args_to_parser(parser, require_gym_config=False)
    parser.set_defaults(seed=0)
    parser.add_argument("--plan-probe-only", action="store_true")
    parser.add_argument("--failure-policy", choices=("stop",), default="stop")
    return parser


def _configured_environment(bundle, args, output):
    bundle.assert_current_inputs()
    import yaml
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.gym.utils.gym_utils import (
        config_to_cfg,
        merge_args_with_gym_config,
    )

    if args.num_envs != 1:
        raise ValueError("Source Task Engine execution supports num_envs=1 only.")
    if args.action_config is not None:
        raise ValueError(
            "A sealed Task Program owns its action declaration; rebuild the bundle for a different action configuration."
        )
    declarations = {
        name: yaml.safe_load(data)
        for name, data in bundle.snapshot.files.items()
        if name.endswith((".yaml", ".yml"))
    }
    selected = deepcopy(declarations["task_program_deployment.yaml"])
    physical = resolve_gym_components(
        selected, base_dir=bundle.root, component_sources=declarations
    )
    runtime = merge_args_with_gym_config(args, physical.config)
    # Preserve component selections for the public parser's ownership checks.
    # The resolved view supplies launcher defaults and backend validation only.
    config = deepcopy(selected)
    for key in (
        "num_envs",
        "seed",
        "device",
        "headless",
        "renderer",
        "gpu_id",
        "arena_space",
        "max_episodes",
        "enable_sensor",
        "visualization",
    ):
        if key in runtime:
            config[key] = deepcopy(runtime[key])
    config["task_program"] = {
        key: {"inline": declarations[path]}
        for key, path in selected["task_program"].items()
    }
    events = config.setdefault("env", {}).setdefault("events", {})
    events["record_camera"] = dict(
        func="record_camera_data",
        mode="interval",
        interval_step=5,
        params=dict(
            name="task_program_audience_view",
            resolution=[640, 360],
            intrinsics=[280.0, 280.0, 320.0, 180.0],
            eye=[0.6, 0.0, 1.8],
            target=[0.0, 0.0, 0.75],
            up=[-1.0, 0.0, 0.0],
            save_path=str(output / "videos"),
        ),
    )
    cfg = config_to_cfg(
        config,
        source_path=bundle.root / "task_program_deployment.yaml",
        component_sources=declarations,
    )
    cfg.filter_visual_rand = args.filter_visual_rand
    cfg.filter_dataset_saving = args.filter_dataset_saving
    cfg.record_trajectory = True
    cfg.trajectory_save_dir = str(output / "trajectory")
    cfg.validate()
    bundle.assert_current_inputs()
    return config["id"], cfg


def _validate_native_trial_mode(bundle, acceptance_mode):
    from embodichain.lab.task_program.semantics.integration import (
        SemanticIntegrationManifest,
    )

    registration = bundle.deployment.integration.registration
    catalog = registration.catalog
    manifest = SemanticIntegrationManifest(
        scene=catalog.scene,
        robot_profile=catalog.robot_profile,
        call_catalog=catalog.call_catalog,
        runtime_preset=bundle.deployment.selection.runtime_preset,
    )
    presets = {
        value.invocation_id: value.preset_id
        for value in registration.invocation_policy_bindings
    }
    selected = []
    for segment in bundle.program.iter_segments():
        for call in segment.calls:
            occurrence = f"{bundle.program.program_id}/{segment.segment_id}:{call.segment_call_index}"
            linked = manifest.link_call(call.call, preset_id=presets.get(occurrence))
            options = catalog.robot_profile.presets[
                linked.preset_id
            ].action_option_template(call.call.semantic_id)
            if getattr(options, "native_scene_trial", False):
                selected.append(occurrence)
    if selected and acceptance_mode != "experiment":
        raise ValueError("Native Pick exploration requires experiment acceptance mode.")
    return selected


def _persist_terminal(environment, output, *, diagnostic_camera):
    environment.save_trajectory(str(output / "trajectory.pt"), purpose="diagnostic")
    if diagnostic_camera:
        environment.event_manager.get_functor_cfg("record_camera").func.save_and_clear(
            purpose="diagnostic"
        )


def _mark_run_failure(bundle, output, report, error, acceptance_mode):
    if report is None:
        return _failure_report(bundle, output, error, acceptance_mode)
    failed = deepcopy(report)
    failed.update(status="failed", failure=_exception_metadata(error))
    for row in failed["environments"]:
        row["success"] = False
    return failed


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


def _failure_report(bundle, output, error, acceptance_mode):
    graph = json.loads(bundle.snapshot.files["semantic_task_graph.json"])
    identity = bundle.snapshot.identity.to_dict()
    return dict(
        schema_version=TASK_PROGRAM_EXECUTION_REPORT_SCHEMA,
        status="failed",
        task_id=graph["task_id"],
        semantic_call_count=len(graph["nodes"]),
        integration_fingerprint=graph["integration_fingerprint"],
        record_dir=str(output),
        environments=[
            dict(
                env_id=0,
                success=False,
                terminal_reason="runner_error",
                semantic_success={},
            )
        ],
        runtime_result=dict(
            bundle_identity=bundle.snapshot.identity.digest,
            program_fingerprint=identity["program_fingerprint"],
            acceptance_mode=acceptance_mode,
        ),
        failure=_exception_metadata(error),
    )


def execute_bundle(
    bundle: str | Path,
    forwarded: Sequence[str] = (),
    *,
    execution_output: str | Path | None = None,
    acceptance_mode: str = "strict",
) -> int:
    """Execute a new run using sealed declarations and the standard demo executor.

    Args:
        bundle: Existing current-contract bundle to verify before simulation.
        forwarded: Existing environment launcher and probe-only arguments.
        execution_output: Fresh attempt output outside every sealed bundle. The
            default creates a unique sibling directory beside the bundle.
        acceptance_mode: Original strict or explicitly selected experiment policy.

    Returns:
        Zero only when the selected acceptance mask succeeds, or a probe reports
        a feasible initial plan. Program, physical effects, strict TaskSpec and
        unavailable observations remain separate recorded facts.

    Old contracts or current input/revision drift reject before simulation. This
    entry never replaces registration, instantiates a custom skill or converts a
    historical execution into a new certificate.
    """
    if acceptance_mode not in {"strict", "experiment"}:
        raise ValueError("Unknown execution acceptance mode.")
    loaded = load_current_task_program_bundle(bundle)
    native_trials = _validate_native_trial_mode(loaded, acceptance_mode)
    args = _runner_parser().parse_args(forwarded)
    output = validate_run_output_directory(
        Path(loaded.root).parent
        / (Path(loaded.root).name + ".execution-" + uuid4().hex)
        if execution_output is None
        else execution_output
    )
    for name in (
        "execution_attempt.json",
        "execution_report.json",
        "planning_probe.json",
        "physical_run_report.json",
        "recorded_measurements.json",
        "raw_evidence.json",
    ):
        if (output / name).exists():
            raise FileExistsError(
                "Historical execution artifacts already exist; choose a fresh attempt directory."
            )
    environment_id, cfg = _configured_environment(loaded, args, output)
    attempt = dict(
        schema_version="task_engine.execution_attempt/v2",
        attempt_id=str(uuid4()),
        bundle_identity=loaded.snapshot.identity.digest,
        status="starting",
        seed=args.seed,
        num_envs=args.num_envs,
        acceptance_mode=acceptance_mode,
        plan_probe_only=args.plan_probe_only,
        record_dir=str(output),
        launcher_arguments=list(forwarded),
    )
    write_run_json(output / "execution_attempt.json", attempt, overwrite=False)
    if native_trials:
        write_run_json(
            output / "native_pick_exploration.json",
            dict(
                schema_version="gen_sim.native-pick-exploration/v1",
                configured=True,
                occurrence_ids=native_trials,
                scope="ordinary_pick_only; actual use is recorded in per-call planner diagnostics",
            ),
            overwrite=False,
        )
    environment = None
    report = None
    alternate = None
    probe = None
    status = 2
    terminal_saved = False
    finalized_episode = False
    try:
        import gymnasium
        from embodichain.lab.sim.motion.cpu_threads import cpu_planning_threads

        loaded.assert_current_inputs()
        environment = gymnasium.make(id=environment_id, cfg=cfg).unwrapped
        loaded.assert_current_inputs()
        environment.reset(seed=args.seed, options={"save_data": False})
        loaded.assert_current_inputs()
        with cpu_planning_threads(
            device=environment.device, num_envs=environment.num_envs
        ):
            if args.plan_probe_only:
                from embodichain.lab.task_program.integrations.simulation.planning_probe import (
                    probe_configured_task_program_initial_plan,
                )

                probe = probe_configured_task_program_initial_plan(
                    environment, loaded.deployment
                )
                loaded.assert_current_inputs()
                probe.update(
                    failure=None,
                    bundle_identity=loaded.snapshot.identity.digest,
                    integration_fingerprint=loaded.snapshot.identity.to_dict()[
                        "integration_fingerprint"
                    ],
                )
                write_run_json(output / "planning_probe.json", probe, overwrite=False)
                status = (
                    0 if probe["plan_success"] and all(probe["plan_success"]) else 2
                )
            else:
                from .execution import execute_task_program_bundle_episode
                from .runtime_bindings import BundleEvidenceProjector

                projector = BundleEvidenceProjector(
                    loaded, step_dt=environment.step_dt, robot_uid=environment.robot.uid
                )

                def observe_initial_plan(value: dict[str, Any]) -> None:
                    nonlocal probe
                    probe = deepcopy(value)
                    probe.update(
                        integration_fingerprint=loaded.snapshot.identity.to_dict()[
                            "integration_fingerprint"
                        ],
                        bundle_identity=loaded.snapshot.identity.digest,
                    )
                    write_run_json(
                        output / "planning_probe.json", probe, overwrite=False
                    )

                episode = execute_task_program_bundle_episode(
                    loaded,
                    environment,
                    prepare_evidence=projector.prepare,
                    final_evidence=projector.finish,
                    control_operation_ids=projector.control_operation_ids,
                    initial_plan_observer=observe_initial_plan,
                )
                write_run_json(
                    output / "raw_evidence.json",
                    episode.to_raw_evidence(),
                    overwrite=False,
                )
                write_run_json(
                    output / "physical_run_report.json",
                    episode.evidence.to_dict(),
                    overwrite=False,
                )
                write_run_json(
                    output / "recorded_measurements.json",
                    episode.measurements,
                    overwrite=False,
                )
                report = build_bundle_execution_report(
                    loaded, episode, record_dir=output, acceptance_mode=acceptance_mode
                )
                alternate_mode = (
                    "experiment" if acceptance_mode == "strict" else "strict"
                )
                alternate = build_bundle_execution_report(
                    loaded, episode, record_dir=output, acceptance_mode=alternate_mode
                )
                status = (
                    0 if all(row["success"] for row in report["environments"]) else 2
                )
                attempt.update(
                    run_id=episode.evidence.run_id,
                    projection_unavailable=projector.unavailable,
                )
                _persist_terminal(environment, output, diagnostic_camera=status != 0)
                terminal_saved = True
                environment.reset(options={"save_data": status == 0})
                finalized_episode = True
    except (Exception, KeyboardInterrupt) as error:
        status = 2
        attempt["error"] = _exception_metadata(error)
        if not args.plan_probe_only:
            report = _mark_run_failure(loaded, output, report, error, acceptance_mode)
        print(str(error), file=sys.stderr)
    finally:
        if environment is not None:
            try:
                if not terminal_saved and not args.plan_probe_only:
                    _persist_terminal(environment, output, diagnostic_camera=True)
            except Exception as error:
                attempt["persistence_error"] = _exception_metadata(error)
                report = _mark_run_failure(
                    loaded, output, report, error, acceptance_mode
                )
                status = 2
            finally:
                if not args.plan_probe_only and not finalized_episode:
                    try:
                        environment.reset(options={"save_data": False})
                    except Exception as error:
                        attempt["abort_error"] = _exception_metadata(error)
                        report = _mark_run_failure(
                            loaded, output, report, error, acceptance_mode
                        )
                        status = 2
                try:
                    environment.close(exit_process=False)
                except Exception as error:
                    attempt["cleanup_error"] = _exception_metadata(error)
                    report = _mark_run_failure(
                        loaded, output, report, error, acceptance_mode
                    )
                    status = 2
        try:
            from embodichain.lab.sim import SimulationManager

            SimulationManager.flush_cleanup_queue()
        except Exception as error:
            attempt["simulation_cleanup_error"] = _exception_metadata(error)
            report = _mark_run_failure(loaded, output, report, error, acceptance_mode)
            status = 2
        if args.plan_probe_only:
            probe = probe or dict(
                scope="initial_call", plan_success=[], task_success=None
            )
            probe["failure"] = next(
                (
                    attempt[key]
                    for key in ("error", "cleanup_error", "simulation_cleanup_error")
                    if key in attempt
                ),
                None,
            )
            probe["integration_fingerprint"] = loaded.snapshot.identity.to_dict()[
                "integration_fingerprint"
            ]
            write_run_json(output / "planning_probe.json", probe)
            print(json.dumps(probe, ensure_ascii=False))
            status = (
                0
                if probe["failure"] is None
                and probe["plan_success"]
                and all(probe["plan_success"])
                else 2
            )
            report = None
        if report is not None:
            from .experiment_policy import program_completion_summary

            if probe is None:
                write_run_json(
                    output / "planning_probe.json",
                    dict(
                        scope="initial_call",
                        planning_source="execution",
                        plan_success=[],
                        task_success=None,
                        failure=report["failure"],
                        integration_fingerprint=loaded.snapshot.identity.to_dict()[
                            "integration_fingerprint"
                        ],
                        diagnostic="Execution did not reach initial atomic planning.",
                    ),
                    overwrite=False,
                )

            if alternate is None:
                alternate = deepcopy(report)
                alternate["runtime_result"]["acceptance_mode"] = (
                    "experiment" if acceptance_mode == "strict" else "strict"
                )
            if report["failure"] is not None:
                alternate.update(status="failed", failure=deepcopy(report["failure"]))
                for row in alternate["environments"]:
                    row["success"] = False
            strict = report if acceptance_mode == "strict" else alternate
            experimental = report if acceptance_mode == "experiment" else alternate
            verify_bundle_execution_report(
                report,
                loaded,
                acceptance_mode=acceptance_mode,
                expected_record_dir=output,
            )
            write_execution_report(output, report)
            if acceptance_mode == "experiment":
                write_run_json(
                    output / "strict_execution_report.json", strict, overwrite=False
                )
            write_run_json(
                output / "program_completion_summary.json",
                program_completion_summary(
                    strict["runtime_result"],
                    strict,
                    experimental,
                    acceptance_mode=acceptance_mode,
                ),
                overwrite=False,
            )
            write_run_json(
                output / "recovery_summary.json",
                build_recovery_summary(report),
                overwrite=False,
            )
            try:
                from .semantic_graph_html import write_semantic_task_graph_html

                coverage = loaded.snapshot.files.get("semantic_graph_validation.json")
                write_semantic_task_graph_html(
                    json.loads(loaded.snapshot.files["semantic_task_graph.json"]),
                    output / "semantic_graph.html",
                    report,
                    task_spec=loaded.task_spec,
                    validation=None if coverage is None else json.loads(coverage),
                    bundle_root=loaded.root,
                )
            except (OSError, TypeError, ValueError) as error:
                attempt["display_error"] = _exception_metadata(error)
        attempt.update(
            status="succeeded" if status == 0 else "failed", exit_code=status
        )
        write_run_json(output / "execution_attempt.json", attempt)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
