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

"""Inspect configured Task Program deployments without constructing a simulator."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys

from embodichain.lab.gym.utils._component_composition import (
    _resolve_gym_components,
    _validate_scene_binding_targets,
)
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.utils.utility import load_config

__all__ = ["inspect_deployment", "main"]


def inspect_deployment(path: str | Path) -> dict[str, object]:
    """Decode, compose, and preflight one configured Task Program deployment.

    Args:
        path: Runnable Gym deployment containing a `task_program` selection.

    Returns:
        JSON-compatible identifiers and compiled program statistics.

    Raises:
        FileNotFoundError: If the deployment or one selected component is absent.
        TypeError: If a decoded configuration value has the wrong shape.
        ValueError: If component, contract, scene, or program validation fails.
    """
    deployment_path = Path(path).expanduser().resolve()
    if not deployment_path.is_file():
        raise FileNotFoundError(
            f"Task Program deployment is not a file: {deployment_path}."
        )
    payload = load_config(deployment_path)
    if not isinstance(payload, Mapping):
        raise TypeError("Runnable Task Program deployment must decode to a mapping.")

    physical = _resolve_gym_components(
        payload,
        base_dir=deployment_path.parent,
    )
    task_program = physical.config.get("task_program")
    if task_program is None:
        raise ValueError("Runnable deployment does not declare task_program.")
    if physical.embodiment_skill_profile is None:
        raise ValueError(
            "Configured Task Program deployment must select an embodiment "
            "component with skill_profile."
        )
    if physical.scene_config is None:
        raise ValueError(
            "Configured Task Program deployment must select or declare a "
            "physical scene."
        )

    deployment = _load_configured_task_program_deployment(
        task_program=task_program,
        skill_profile=physical.embodiment_skill_profile,
        base_dir=deployment_path.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding,
        simulation=physical.scene_config,
    )

    registration = deployment.integration.registration
    registration.assert_unchanged()
    catalog = registration.catalog
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=catalog,
    )
    if program.program_id != deployment.program_id:
        raise ValueError(
            "Task integration expects program_id "
            f"{deployment.program_id!r}, got {program.program_id!r}."
        )
    compiled = catalog.preflight(program)
    calls = tuple(
        compiled_call.call
        for segment in compiled.iter_segments()
        for compiled_call in segment.calls
    )
    semantic_calls = sorted({call.semantic_id for call in calls})

    environment_id = physical.config.get("id")
    if type(environment_id) is not str or not environment_id:
        raise ValueError("Runnable deployment must declare a non-empty string id.")
    return {
        "deployment": str(deployment_path),
        "environment_id": environment_id,
        "program_id": program.program_id,
        "integration_id": deployment.integration_id,
        "scene_registry": deployment.selection.scene_registry,
        "robot_profile": deployment.selection.robot_profile,
        "runtime_preset": deployment.selection.runtime_preset,
        "segment_count": compiled.segment_count,
        "call_count": len(calls),
        "semantic_calls": semantic_calls,
    }


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Statically resolve and preflight configured Task Program deployments."
        )
    )
    parser.add_argument(
        "deployments",
        nargs="+",
        type=Path,
        help="Runnable task.<embodiment>.yaml deployment paths.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable result object.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run static inspection for one or more deployments.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Zero when every deployment passes, otherwise one.
    """
    args = _parser().parse_args(argv)
    summaries: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in args.deployments:
        try:
            summaries.append(inspect_deployment(path))
        except Exception as error:
            errors.append(
                {
                    "deployment": str(path.expanduser().resolve()),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    if args.json:
        print(
            json.dumps(
                {"deployments": summaries, "errors": errors},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for summary in summaries:
            print(
                "[PASS] "
                f"{summary['environment_id']}: "
                f"{summary['segment_count']} segments, "
                f"{summary['call_count']} calls "
                f"({', '.join(summary['semantic_calls'])})"
            )
        for error in errors:
            print(
                "[FAIL] "
                f"{error['deployment']}: "
                f"{error['error_type']}: {error['message']}",
                file=sys.stderr,
            )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
