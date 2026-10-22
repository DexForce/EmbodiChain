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

"""Unified CLI for complete Task Engine workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Final, Sequence

from .config import load_task_engine_config
from .orchestration.scene_adapter import SceneAdapter
from .run_directory import reserve_run_directory
from .robot_profiles import _CLI_PROFILES, _require_robot_assets
from .workflow import (
    SubprocessActionExecutor,
    TaskEngineWorkflow,
    _bundle_success_terms,
    _environment_successes,
)
from .workflow_contracts import (
    TASK_RUN_REQUEST_SCHEMA,
    validate_scene_history_root,
    validate_scene_output_separation,
)

__all__ = ["build_parser", "main"]


_ROBOT_PROFILES = _CLI_PROFILES
_MODES: Final = ("scene",)


def build_parser() -> argparse.ArgumentParser:
    """Build the Task Engine parser."""
    parser = argparse.ArgumentParser(
        prog="embodichain task-engine",
        description="Prepare, run, or complete one Scene and Action workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare a bundle without simulator execution."
    )
    _add_workflow_arguments(prepare_parser)
    run_all_parser = subparsers.add_parser(
        "run-all", help="Prepare and execute one complete workflow."
    )
    _add_workflow_arguments(run_all_parser)
    _add_open_window_argument(run_all_parser)
    run_parser = subparsers.add_parser(
        "run", help="Execute an already prepared Task Engine bundle."
    )
    run_parser.add_argument("--bundle", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--config", default=None)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--num-envs", type=int, default=None)
    run_parser.add_argument("--dataset-saving", action="store_true")
    _add_open_window_argument(run_parser)
    _add_failure_policy_argument(run_parser)
    visualize_parser = subparsers.add_parser(
        "visualize", help="Render an existing semantic graph as offline HTML."
    )
    for name in ("graph", "output"):
        visualize_parser.add_argument(f"--{name}", required=True)
    for name in ("report", "task-spec", "validation"):
        visualize_parser.add_argument(f"--{name}", default=None)
    validate_parser = subparsers.add_parser(
        "validate-graph", help="Strictly audit an existing bundle without simulation."
    )
    validate_parser.add_argument("--bundle", required=True)
    return parser


def _add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--task-id", "--task_id", required=True)
    instruction = parser.add_mutually_exclusive_group(required=True)
    instruction.add_argument("--instruction")
    instruction.add_argument("--task-file", "--task_file")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--dataset_saving",
        action="store_true",
        help="Opt in to the Gym project's dataset recorder during execution.",
    )
    parser.add_argument(
        "--robot-profile",
        choices=_ROBOT_PROFILES,
        default="dual_franka",
        help="Robot deployment selector; tianji_marvin is a legacy alias of dual_tianji.",
    )
    _add_failure_policy_argument(parser)


def _add_failure_policy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--failure-policy",
        choices=("stop",),
        default="stop",
        help=(
            "Stop affected execution after a dependency failure. "
            "Diagnostic continuation is currently unsupported."
        ),
    )


def _add_open_window_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--open-window",
        action="store_true",
        help="Open the native DexSim window during simulator execution.",
    )


def main(argv: Sequence[str] | None = None, *, started_at: float | None = None) -> int:
    """Dispatch one Task Engine workflow command.

    Args:
        argv: Arguments, or the current process arguments when omitted.
        started_at: Optional monotonic bootstrap timestamp supplied by the module entry point.

    Returns:
        Command exit status.
    """
    started_at = perf_counter() if started_at is None else started_at
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in {
        "prepare",
        "run",
        "run-all",
        "visualize",
        "validate-graph",
        "-h",
        "--help",
    }:
        arguments.insert(0, "run-all")
    args = parser.parse_args(arguments)
    if args.command == "visualize":
        return _run_visualize(args, parser)
    if args.command == "validate-graph":
        return _run_validate_graph(args, parser)
    if args.command == "run":
        return _run_prepared_bundle(args, started_at=started_at)
    return _run_workflow(
        args,
        execute=args.command == "run-all",
        parser=parser,
        started_at=started_at,
    )


def _run_visualize(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Render recorded evidence without invoking a model or simulator."""
    from .semantic_graph_html import write_semantic_task_graph_html

    try:
        root = Path(args.graph).expanduser().resolve().parent
        if Path(args.output).suffix.lower() != ".html":
            raise ValueError("Visualization output must end in .html.")
        companions = {}
        for key, explicit, default in (
            ("task_spec", args.task_spec, "task_spec.json"),
            ("validation", args.validation, "semantic_graph_validation.json"),
        ):
            path = Path(explicit).expanduser() if explicit else root / default
            if explicit or path.is_file():
                companions[key] = json.loads(path.read_text(encoding="utf-8"))
        output = write_semantic_task_graph_html(
            json.loads(Path(args.graph).expanduser().read_text(encoding="utf-8")),
            args.output,
            (
                None
                if args.report is None
                else json.loads(
                    Path(args.report).expanduser().read_text(encoding="utf-8")
                )
            ),
            bundle_root=root,
            **companions,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    _print_json({"output": output.as_posix()})
    return 0


def _run_validate_graph(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    from ._bundle_runner import (
        _verify_source,
        _verify_program_projection,
        _verify_integration_fingerprint,
    )
    from .semantic_graph import validate_semantic_task_graph
    from .semantic_graph_validation import verify_graph_contract

    try:
        root = Path(args.bundle).expanduser().resolve()
        graph = validate_semantic_task_graph(
            json.loads((root / "semantic_task_graph.json").read_text(encoding="utf-8"))
        )
        _verify_source(root)
        _verify_integration_fingerprint(
            root,
            root / "task_program_deployment.yaml",
            graph,
            json.loads(
                (root / "integration_fingerprint.json").read_text(encoding="utf-8")
            ),
        )
        _verify_program_projection(root / "task_program/program.yaml", graph)
        contract = verify_graph_contract(root, graph)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    _print_json(
        {
            "bundle": str(root),
            "structure_and_integration": "validated",
            "task_spec_coverage": (
                "unavailable" if contract is None else contract["coverage"]["status"]
            ),
            "runtime_evidence": "not_run",
        }
    )
    return 0


def _run_workflow(
    args: argparse.Namespace,
    *,
    execute: bool,
    parser: argparse.ArgumentParser,
    started_at: float,
) -> int:
    try:
        image, scene, edit = _mode_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))
    if scene is not None:
        validate_scene_history_root(scene, args.output_root)
    instruction = _instruction(args)
    workflow_cfg, planning_cfg, execution_cfg = load_task_engine_config(args.config)
    if planning_cfg.physical_reference is None:
        try:
            _require_robot_assets(args.robot_profile)
        except ValueError as exc:
            parser.error(str(exc))
    adapter = SceneAdapter(model=args.model, robot_profile=args.robot_profile)
    workflow = TaskEngineWorkflow(scene_adapter=adapter)
    with reserve_run_directory(args.output_root) as allocation:
        if scene is not None:
            validate_scene_output_separation(scene, allocation.path)
        result = workflow.run(
            {
                "schema_version": TASK_RUN_REQUEST_SCHEMA,
                "task_id": args.task_id,
                "task_instruction": instruction,
                "image_path": image,
                "gym_project": scene,
                "scene_edit_prompt": edit,
                "output_dir": allocation.path.as_posix(),
            },
            config_path=args.config,
            workflow_cfg=workflow_cfg,
            planning_cfg=planning_cfg,
            execution_cfg=execution_cfg,
            model=args.model,
            base_seed=args.base_seed,
            dataset_saving=args.dataset_saving,
            failure_policy=args.failure_policy,
            open_window=bool(getattr(args, "open_window", False)),
            run_id=allocation.run_id,
            created_at=allocation.created_at,
            execute=execute,
        )
    payload = {
        "run_id": allocation.run_id,
        "status": result.status,
        "failure_class": result.failure_class,
        "elapsed_seconds": None,
        "output_dir": result.output_dir.as_posix(),
        "manifest": result.manifest_path.as_posix(),
        "final_bundle": (
            None if result.final_bundle is None else result.final_bundle.as_posix()
        ),
        "video_paths": _report_saved_videos(result.output_dir) if execute else [],
    }
    payload["elapsed_seconds"] = round(perf_counter() - started_at, 3)
    payload["semantic_graph_html"] = _finish_html(
        result.output_dir,
        result.final_bundle,
        payload,
        task_id=args.task_id,
        created_at=allocation.created_at.isoformat(),
    )
    _print_json(payload)
    accepted = result.succeeded if execute else result.status == "prepared"
    return 0 if accepted else 2


def _run_prepared_bundle(args: argparse.Namespace, *, started_at: float) -> int:
    _, _, execution_cfg = load_task_engine_config(args.config)
    num_envs = execution_cfg.num_envs if args.num_envs is None else int(args.num_envs)
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    execution_cfg = replace(
        execution_cfg,
        num_envs=num_envs,
        min_successful_envs=(
            num_envs
            if execution_cfg.success_policy == "all"
            else execution_cfg.min_successful_envs
        ),
    )
    with reserve_run_directory(args.output_root) as allocation:
        report = SubprocessActionExecutor()(
            args.bundle,
            allocation.path,
            seed=int(args.seed),
            num_envs=num_envs,
            dataset_saving=bool(args.dataset_saving),
            failure_policy=args.failure_policy,
            open_window=bool(args.open_window),
            **(
                {"acceptance_mode": execution_cfg.acceptance_mode}
                if execution_cfg.acceptance_mode != "strict"
                else {}
            ),
        )
    successes = _environment_successes(
        report,
        required_semantic_steps=_bundle_success_terms(Path(args.bundle)),
        bundle=Path(args.bundle),
        acceptance_mode=execution_cfg.acceptance_mode,
    )
    accepted = (
        str(report.get("status")) not in {"rejected", "aborted"}
        and len(successes) == num_envs
        and sum(successes) >= execution_cfg.required_successes
    )
    payload = {
        "run_id": allocation.run_id,
        "status": "succeeded" if accepted else "failed",
        "elapsed_seconds": None,
        "output_dir": allocation.path.as_posix(),
        "execution_report": report,
        "video_paths": _report_saved_videos(allocation.path),
    }
    payload["elapsed_seconds"] = round(perf_counter() - started_at, 3)
    payload["semantic_graph_html"] = _finish_html(
        allocation.path,
        Path(args.bundle).expanduser(),
        payload,
        task_id=report.get("task_id"),
        created_at=allocation.created_at.isoformat(),
        html_path=allocation.path / "semantic_graph.html",
    )
    _print_json(payload)
    return 0 if accepted else 2


def _finish_html(
    output_dir: Path,
    bundle: Path | None,
    payload: dict[str, Any],
    *,
    task_id: str,
    created_at: str,
    html_path: Path | None = None,
) -> str | None:
    from .semantic_graph_html import _finalize_run_html

    path = _finalize_run_html(
        output_dir,
        bundle=bundle,
        html_path=html_path,
        summary={
            "task_id": task_id,
            "run_id": payload["run_id"],
            "created_at": created_at,
            "status": payload["status"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "elapsed_scope": "CLI wall time through publication, excluding HTML refresh",
        },
    )
    if path is not None:
        print(f"[Task Engine] Semantic graph HTML: {path}", file=sys.stderr)
    return None if path is None else path.as_posix()


def _report_saved_videos(output_dir: Path) -> list[str]:
    """List recordings only after the current run has finished publishing."""
    try:
        paths = sorted(
            {
                path.resolve().as_posix()
                for pattern in (
                    "videos/*.mp4",
                    "videos/env_*/*.mp4",
                    "attempts/scene_*/action_attempts/action_*/videos/*.mp4",
                    "attempts/scene_*/action_attempts/action_*/videos/env_*/*.mp4",
                )
                for path in output_dir.glob(pattern)
                if path.is_file()
                and not path.name.endswith(".partial.mp4")
                and path.resolve().is_relative_to(output_dir.resolve())
            }
        )
    except OSError as exc:
        print(f"[Task Engine] Unable to list saved videos: {exc}", file=sys.stderr)
        return []
    for path in paths:
        print(f"[Task Engine] Video saved: {path}", file=sys.stderr)
    if not paths:
        print("[Task Engine] No video files generated for this run.", file=sys.stderr)
    return paths


def _instruction(args: argparse.Namespace) -> str:
    instruction = (
        str(args.instruction).strip()
        if args.instruction is not None
        else Path(args.task_file).expanduser().read_text(encoding="utf-8").strip()
    )
    if not instruction:
        raise ValueError("Task instruction must not be empty.")
    return instruction


def _mode_inputs(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    if args.mode != "scene" or not str(args.scene).strip():
        raise ValueError("Task Engine requires --mode scene and an existing --scene.")
    return None, str(args.scene).strip(), None


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
