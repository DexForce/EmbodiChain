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

"""CLI for collaboration preparation and execution."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Iterator, Sequence

from embodichain.gen_sim.action_engine.protocol import (
    AGENT_CONFIG_FILENAME,
    EXECUTION_PROGRAM_FILENAME,
    FAST_GYM_CONFIG_FILENAME,
)
from embodichain.gen_sim.action_engine.runtime import ExecutionReport
from embodichain.gen_sim.action_engine.agent import ActionAgent

from .artifacts import GROUNDED_TASK_PLAN_FILENAME, write_execution_report
from .contracts import validate_grounded_task_plan
from .coordinator import CollaborationCoordinator
from .scene_adapter import SceneAdapter
from .scene_store import ScenePackageRef, ScenePackageStore, SceneSourceRef

__all__ = ["build_parser", "main"]


_ROBOT_PROFILES = (
    "ur5",
    "ur10",
    "dual_ur5",
    "dual_ur10",
    "franka",
    "dual_franka",
)
_PREPARED_RUN_ARGS = ("--filter_dataset_saving", "--headless")


def build_parser() -> argparse.ArgumentParser:
    """Build the Gen Sim collaboration parser."""
    parser = argparse.ArgumentParser(
        prog="python -m embodichain.gen_sim.collaboration",
        description="Prepare and run a three-agent collaboration task.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    import_parser = subparsers.add_parser(
        "import-scene",
        help="Import an exact scene and its assets into the local Data Bank.",
    )
    import_parser.add_argument("--scene", required=True)
    import_parser.add_argument("--data-bank", default=None)
    _add_scene_policy_arguments(import_parser)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Generate, bind, compile, and publish a task bundle.",
    )
    prepare_parser.add_argument("--task-id", "--task_id", required=True)
    instruction = prepare_parser.add_mutually_exclusive_group(required=True)
    instruction.add_argument("--instruction")
    instruction.add_argument("--task-file", "--task_file")
    source = prepare_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene")
    source.add_argument("--scene-package", "--scene_package")
    prepare_parser.add_argument("--output", "--output-dir", required=True)
    prepare_parser.add_argument("--data-bank", default=None)
    prepare_parser.add_argument("--model", default=None)
    prepare_parser.add_argument("--vlm-model", default=None)
    prepare_parser.add_argument("--candidate-count", type=int, default=3)
    prepare_parser.add_argument(
        "--planning-mode", choices=("offline", "ab"), default="offline"
    )
    prepare_parser.add_argument("--max-episodes", type=int, default=None)
    prepare_parser.add_argument("--max-episode-steps", type=int, default=None)
    prepare_parser.add_argument("--randomize-scene", action="store_true")
    prepare_parser.add_argument("--randomize-table-material", action="store_true")
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument(
        "--run-after-prepare",
        "--run_after_prepare",
        action="store_true",
        help="Run the bound bundle immediately after preparation succeeds.",
    )
    _add_scene_policy_arguments(prepare_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a published bundle with the existing simulator launcher.",
    )
    run_parser.add_argument("--bundle", required=True)
    run_parser.set_defaults(run_args=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one collaboration command without retaining global argv state."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if arguments and arguments[0] == "run":
        args, forwarded = parser.parse_known_args(arguments)
        args.run_args.extend(forwarded)
    else:
        args = parser.parse_args(arguments)
    if args.subcommand == "import-scene":
        return _import_scene(args)
    if args.subcommand == "prepare":
        return _prepare(args)
    if args.subcommand == "run":
        return _run(args)
    raise AssertionError(f"Unknown gen-sim-task command: {args.subcommand}")


def _import_scene(args: argparse.Namespace) -> int:
    store = ScenePackageStore(args.data_bank)
    package = store.import_scene(
        SceneSourceRef(
            args.scene,
            robot_profile=args.robot_profile,
            z_rotation_degrees=args.source_scene_z_rotation_degrees,
            body_scale_policy=args.body_scale_policy,
            body_scale=tuple(args.body_scale),
        )
    )
    _print_json(
        {
            "status": "imported",
            "package_id": package.package_id,
            "package_path": str(package.package_path),
            "config_path": str(package.config_path),
        }
    )
    return 0


def _prepare(args: argparse.Namespace) -> int:
    instruction = (
        str(args.instruction).strip()
        if args.instruction is not None
        else Path(args.task_file).expanduser().read_text(encoding="utf-8").strip()
    )
    if not instruction:
        raise ValueError("Task instruction must not be empty.")
    store = ScenePackageStore(args.data_bank)
    adapter = SceneAdapter(
        store=store,
        model=args.model,
        robot_profile=args.robot_profile,
    )
    coordinator = CollaborationCoordinator(scene_adapter=adapter)
    if args.scene_package:
        source: SceneSourceRef | ScenePackageRef = ScenePackageRef(
            args.scene_package,
            robot_profile=args.robot_profile,
        )
    else:
        source = SceneSourceRef(
            args.scene,
            robot_profile=args.robot_profile,
            z_rotation_degrees=args.source_scene_z_rotation_degrees,
            body_scale_policy=args.body_scale_policy,
            body_scale=tuple(args.body_scale),
        )
    result = coordinator.prepare(
        args.task_id,
        instruction,
        source,
        args.output,
        model=args.model,
        candidate_count=args.candidate_count,
        overwrite=args.overwrite,
        planning_mode=args.planning_mode,
        vlm_model=args.vlm_model,
        max_episodes=args.max_episodes,
        max_episode_steps=args.max_episode_steps,
        randomize_scene=args.randomize_scene,
        randomize_table_material=args.randomize_table_material,
    )
    _print_json(
        {
            "status": result.status,
            "task_id": args.task_id,
            "selected_candidate_id": result.selected_candidate_id,
            "output_dir": str(result.output_dir),
            "grounded_task_plan": (
                str(result.collaboration_artifacts.grounded_task_plan)
                if result.bound
                else None
            ),
            "preparation_failure": (
                str(result.collaboration_artifacts.preparation_failure)
                if result.collaboration_artifacts.preparation_failure.is_file()
                else None
            ),
            "run_command": (
                _bundle_run_command(result.output_dir) if result.bound else None
            ),
        }
    )
    if not result.bound:
        return 2
    if args.run_after_prepare:
        return _run(
            argparse.Namespace(
                bundle=result.output_dir,
                run_args=list(_PREPARED_RUN_ARGS),
            )
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle}")
    agent_config = bundle / AGENT_CONFIG_FILENAME
    gym_config = bundle / FAST_GYM_CONFIG_FILENAME
    for path in (agent_config, gym_config):
        if not path.is_file():
            raise FileNotFoundError(f"Bundle is missing required artifact: {path}")
    task_id = _bundle_task_id(bundle, agent_config)
    forwarded = list(args.run_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    rejection = _preflight_bundle(
        bundle,
        agent_config=agent_config,
        gym_config=gym_config,
        forwarded=forwarded,
    )
    if rejection is not None:
        write_execution_report(bundle, rejection)
        _print_json(rejection.as_mapping())
        return 2
    legacy_argv = [
        "--task_name",
        task_id,
        "--gym_config",
        str(gym_config),
        "--agent_config",
        str(agent_config),
        "--collaboration-report",
        *forwarded,
    ]
    from embodichain.gen_sim.action_engine.cli import run_agent

    with _temporary_argv(["run_agent", *legacy_argv]):
        return int(run_agent.cli() or 0)


def _preflight_bundle(
    bundle: Path,
    *,
    agent_config: Path,
    gym_config: Path,
    forwarded: Sequence[str],
) -> ExecutionReport | None:
    """Return a rejected report, or ``None`` when the graph is executable."""
    grounded_path = bundle / GROUNDED_TASK_PLAN_FILENAME
    if not grounded_path.is_file():
        return None
    grounded = validate_grounded_task_plan(_read_json(grounded_path))
    agent = _read_json(agent_config)
    graph_value = agent.get("seed_task_graph", EXECUTION_PROGRAM_FILENAME)
    if not isinstance(graph_value, str) or not graph_value:
        raise ValueError("Bundle agent_config.seed_task_graph must be a path string.")
    graph_path = Path(graph_value).expanduser()
    if not graph_path.is_absolute():
        graph_path = (bundle / graph_path).resolve()
    else:
        graph_path = graph_path.resolve()
    if graph_path != bundle and bundle not in graph_path.parents:
        raise ValueError("Bundle SeedGraph path escapes the bundle directory.")
    if not graph_path.is_file():
        raise FileNotFoundError(f"Bundle is missing SeedGraph: {graph_path}")
    action_agent = ActionAgent()
    try:
        action_agent.preflight(
            graph_path,
            scene_manifest=grounded["scene_manifest"],
        )
    except (TypeError, ValueError, OSError) as exc:
        return action_agent.rejection_report(
            graph_path,
            exc,
            grounded_plan=grounded,
            environment_count=_environment_count(gym_config, forwarded),
        )
    return None


def _environment_count(gym_config: Path, forwarded: Sequence[str]) -> int:
    value: Any = _read_json(gym_config).get("num_envs", 1)
    for index, argument in enumerate(forwarded):
        if argument == "--num_envs" and index + 1 < len(forwarded):
            value = forwarded[index + 1]
        elif argument.startswith("--num_envs="):
            value = argument.partition("=")[2]
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _bundle_task_id(bundle: Path, agent_config: Path) -> str:
    grounded_path = bundle / GROUNDED_TASK_PLAN_FILENAME
    if grounded_path.is_file():
        grounded = _read_json(grounded_path)
        task_id = grounded.get("task_id")
    else:
        task_id = _read_json(agent_config).get("task_name")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Bundle does not declare a non-empty task ID.")
    return task_id.strip()


def _bundle_run_command(bundle: str | Path) -> str:
    """Return a shell-safe command for the next collaboration stage."""
    return shlex.join(
        [
            "python",
            "-m",
            "embodichain.gen_sim.collaboration",
            "run",
            "--bundle",
            str(Path(bundle).expanduser().resolve()),
            *_PREPARED_RUN_ARGS,
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


@contextmanager
def _temporary_argv(arguments: list[str]) -> Iterator[None]:
    original = sys.argv
    sys.argv = arguments
    try:
        yield
    finally:
        sys.argv = original


def _add_scene_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robot-profile",
        choices=_ROBOT_PROFILES,
        default="franka",
    )
    parser.add_argument(
        "--source-scene-z-rotation-degrees",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--body-scale-policy",
        choices=("preserve", "multiply", "absolute"),
        default="preserve",
    )
    parser.add_argument(
        "--body-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
