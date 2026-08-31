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

"""Private subprocess boundary for executing one prepared Task Engine bundle."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Sequence

from embodichain.gen_sim.action_engine.agent import ActionAgent
from embodichain.gen_sim.action_engine.generation.config_builder import (
    _runtime_articulation_config,
)
from embodichain.gen_sim.action_engine.protocol import (
    AGENT_CONFIG_FILENAME,
    EXECUTION_PROGRAM_FILENAME,
    FAST_GYM_CONFIG_FILENAME,
)
from embodichain.gen_sim.action_engine.runtime import ExecutionReport

from .orchestration.artifacts import (
    GROUNDED_TASK_PLAN_FILENAME,
    STATIC_SCENE_MANIFEST_FILENAME,
    write_execution_report,
)
from .orchestration.contracts import validate_grounded_task_plan
from .orchestration.scene_source import verify_scene_source_fingerprint

__all__ = ["execute_bundle", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the private runner protocol and execute one bundle."""
    parser = argparse.ArgumentParser(
        prog="embodichain.gen_sim.task_engine._bundle_runner"
    )
    parser.add_argument("--bundle", required=True)
    args, forwarded = parser.parse_known_args(argv)
    return execute_bundle(args.bundle, forwarded)


def execute_bundle(
    bundle: str | Path,
    forwarded: Sequence[str] = (),
) -> int:
    """Execute one prepared bundle through the existing Action Engine launcher."""
    root = Path(bundle).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {root}")
    agent_config = root / AGENT_CONFIG_FILENAME
    gym_config = root / FAST_GYM_CONFIG_FILENAME
    for path in (agent_config, gym_config):
        if not path.is_file():
            raise FileNotFoundError(f"Bundle is missing required artifact: {path}")
    task_id = _bundle_task_id(root, agent_config)
    run_args = list(forwarded)
    if run_args and run_args[0] == "--":
        run_args.pop(0)
    rejection = _preflight_bundle(
        root,
        agent_config=agent_config,
        gym_config=gym_config,
        forwarded=run_args,
    )
    if rejection is not None:
        write_execution_report(root, rejection)
        _print_json(rejection.as_mapping())
        return 2
    from embodichain.gen_sim.action_engine.cli import run_agent

    with _runtime_gym_config(gym_config) as runtime_gym_config:
        legacy_argv = [
            "--task_name",
            task_id,
            "--gym_config",
            str(runtime_gym_config),
            "--agent_config",
            str(agent_config),
            "--task-engine-report",
            *run_args,
        ]
        with _temporary_argv(["run_agent", *legacy_argv]):
            return int(run_agent.cli() or 0)


def _preflight_bundle(
    bundle: Path,
    *,
    agent_config: Path,
    gym_config: Path,
    forwarded: Sequence[str],
) -> ExecutionReport | None:
    static_manifest_path = bundle / STATIC_SCENE_MANIFEST_FILENAME
    if static_manifest_path.is_file():
        static_manifest = _read_json(static_manifest_path)
        source = static_manifest.get("source", {})
        if isinstance(source, dict) and isinstance(
            source.get("source_fingerprint"), dict
        ):
            verify_scene_source_fingerprint(source["source_fingerprint"])
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
        task_id = _read_json(grounded_path).get("task_id")
    else:
        task_id = _read_json(agent_config).get("task_name")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Bundle does not declare a non-empty task ID.")
    return task_id.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


@contextmanager
def _runtime_gym_config(source: Path) -> Iterator[Path]:
    """Yield a simulator-only config without modifying a prepared bundle."""
    original = _read_json(source)
    normalized = deepcopy(original)
    articulations = normalized.get("articulation")
    if isinstance(articulations, list):
        normalized["articulation"] = [
            _runtime_articulation_config(item) if isinstance(item, dict) else item
            for item in articulations
        ]
    if normalized == original:
        yield source
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".task-engine-runtime-",
            suffix=".json",
            dir=source.parent,
            delete=False,
        ) as stream:
            json.dump(normalized, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            temporary_path = Path(stream.name)
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def _temporary_argv(arguments: list[str]) -> Iterator[None]:
    original = sys.argv
    sys.argv = arguments
    try:
        yield
    finally:
        sys.argv = original


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
