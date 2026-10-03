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

"""Bounded E8 support candidates through canonical generation and execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any, TYPE_CHECKING

from embodichain.utils.utility import load_config

from .twist_support import TwistSupportPenetrationError

if TYPE_CHECKING:
    from ..config import TaskEnginePlanningCfg
    from ..orchestration.coordinator import PreparationResult

__all__: list[str] = []

SUPPORT_HEIGHTS = (0.001, 0.002, 0.003, 0.004)


def is_twist_bundle(bundle: Path) -> bool:
    """Identify E8 by its semantic calls, independently of the renderer."""
    path = bundle / "task_program/program.yaml"
    if not path.is_file():
        return False
    program = load_config(path)
    return any(
        item.get("steps", {}).get("call", {}).get("call_id") == "gen_sim.twist"
        for item in program.get("program", {}).get("items", ())
    )


def _support_height(bundle: Path) -> float | None:
    path = bundle / "twist_adaptation.json"
    if not path.is_file():
        return None
    adaptation = load_config(path)
    if (
        adaptation.get("policy")
        != "gripper_measured_scale_bounded_height_constrained_relayout"
    ):
        return None
    records = adaptation.get("records", ())
    if len(records) != 1:
        return None
    return records[0]["support_clearance"].get("requested_total_lift_m")


def support_failure(output: Path) -> bool:
    """Retry only support-related startup failures, never a missed target."""
    path = output / "twist_evidence.json"
    if not path.is_file():
        return False
    accepted = load_config(path).get("acceptance", {})
    if (
        accepted.get("phase") != "initial_stable"
        or accepted.get("accepted") is not False
    ):
        return False
    startup = accepted.get("startup_check", {})
    return startup.get("reason") in {"startup_table_contact", "startup_motion"}


def execute_support_candidates(
    preparation: PreparationResult,
    output: Path,
    executor: Callable[..., Mapping[str, Any]],
    options: dict[str, Any],
    planning: TaskEnginePlanningCfg,
) -> tuple[Mapping[str, Any], PreparationResult, Path, list[dict[str, Any]]]:
    """Rebuild only the deployment height after observed unsafe startup.

    Every simulator call remains the supplied ordinary executor. Fresh bundles
    are generated from the original PreparedScene, never from a previously
    lifted copy, so neither support height nor scale can accumulate on retries.
    """
    from ..task_program_bundle import generate_task_program_bundle

    height = _support_height(preparation.output_dir)
    if not is_twist_bundle(preparation.output_dir) or height not in SUPPORT_HEIGHTS:
        report = executor(preparation.output_dir, output, **options)
        return report, preparation, output, []
    current, execution_root, attempts = preparation, output, []
    first = SUPPORT_HEIGHTS.index(height)
    for index in range(first, len(SUPPORT_HEIGHTS)):
        candidate_height = SUPPORT_HEIGHTS[index]
        if index > first:
            bundle = output.parent / f"{output.name}_height_{index + 1:02d}_bundle"
            next_execution = output.parent / f"{output.name}_height_{index + 1:02d}"
            try:
                graph, generated = generate_task_program_bundle(
                    preparation.semantic_task_graph,
                    preparation.adaptation.prepared_scene,
                    bundle,
                    robot_profile=preparation.adaptation.scene_manifest[
                        "robot_profile"
                    ],
                    max_episodes=planning.max_episodes,
                    max_episode_steps=planning.max_episode_steps,
                    fit_grasp_assets=planning.fit_grasp_assets,
                    twist_support_lift_m=candidate_height,
                )
            except TwistSupportPenetrationError as error:
                attempts.append(
                    {
                        "height_m": candidate_height,
                        "status": "geometric_penetration",
                        "error": str(error),
                    }
                )
                continue
            # Preserve provider/source receipts alongside the new canonical bundle.
            for path in preparation.output_dir.iterdir():
                if path.is_file() and not (bundle / path.name).exists():
                    if path.name == "planner_report.json":
                        from ..semantic_graph import semantic_task_graph_hash

                        receipt = load_config(path)
                        receipt.update(
                            integration_fingerprint=graph["integration_fingerprint"],
                            semantic_task_graph_hash=semantic_task_graph_hash(graph),
                            source_preparation=str(preparation.output_dir),
                            support_lift_m=candidate_height,
                        )
                        (bundle / path.name).write_text(
                            json.dumps(receipt, indent=2), encoding="utf-8"
                        )
                    else:
                        shutil.copy2(path, bundle / path.name)
            current = replace(
                preparation,
                output_dir=bundle,
                semantic_task_graph=graph,
                generated_paths=generated,
            )
            execution_root = next_execution
        report = executor(current.output_dir, execution_root, **options)
        attempts.append(
            {
                "height_m": candidate_height,
                "bundle": str(current.output_dir),
                "execution_root": str(execution_root),
                "status": str(report.get("status")),
                "support_startup_rejected": support_failure(execution_root),
            }
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "support_attempts.json").write_text(
            json.dumps(attempts, indent=2), encoding="utf-8"
        )
        if not attempts[-1]["support_startup_rejected"]:
            break
    return report, current, execution_root, attempts
