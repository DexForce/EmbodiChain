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
import hashlib
import math
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
GRIP_DEPTHS = (0.010, 0.0125, 0.015, 0.020)


def _save_scale_attempts(output: Path, attempts: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "scale_attempts.json").write_text(
        json.dumps(attempts, indent=2), encoding="utf-8"
    )


def _candidate_bundle_path(output: Path, kind: str, index: int) -> Path:
    # Native material identifiers can include the asset path; avoid nested retry names.
    identity = hashlib.sha256(output.name.encode("utf-8")).hexdigest()[:10]
    return output.parent.parent / f"b{identity}{kind}{index}"


def _dictionaries(value: Any) -> Any:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _dictionaries(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _dictionaries(child)


def size_failure(output: Path, report: Mapping[str, Any] | None = None) -> bool:
    """Admit dimension retries only from qualified terminal physical receipts."""
    try:
        evidence = load_config(output / "twist_evidence.json")
        report = (
            load_config(output / "execution_report.json") if report is None else report
        )
        if report.get("status") != "failed":
            return False
        runtime = report.get("runtime_result", {})
        if (
            runtime.get("completed") is not False
            or runtime.get("terminal_reason") != "segment_validation_failed"
        ):
            return False
        geometry = evidence["geometry_evidence"]
        mass = geometry["mass_properties"]
        sha = evidence["route"]["binding"]["source_sha256"]
        if (
            mass.get("verified") is not True
            or mass.get("source_edited") is not False
            or geometry.get("source_edited") is True
            or mass.get("source_sha256") != sha
            or geometry.get("source_sha256") != sha
            or not isinstance(sha, str)
            or len(sha) != 64
        ):
            return False
        startup = evidence["startup_evidence"]
        if (
            startup.get("complete") is not True
            or startup.get("overflow") is not False
            or startup.get("summary", {}).get("accepted") is not True
        ):
            return False
        rows = startup["trace"] + evidence["trace"]
        if (
            not startup["trace"]
            or not evidence["trace"]
            or any(
                row.get("valid") is not True
                or row.get("dropped_contacts") != 0
                or not math.isfinite(row["timestamp"])
                or not math.isfinite(row["qpos"])
                for row in rows
            )
        ):
            return False
        # Earlier diagnostics cannot turn an unrelated terminal failure into a retry.
        allowed = {"e8_table_clearance_rejected"}
        for item in _dictionaries(report):
            code = item.get("failure_code")
            failure = item.get("failure")
            if isinstance(failure, Mapping):
                code = failure.get("code", code)
            if code and code not in allowed:
                return False
        for segment in runtime.get("segments", ()):
            for call in segment.get("metadata", {}).get("runtime", {}).get("calls", ()):
                if (
                    call.get("semantic_id") != "gen_sim.twist"
                    or call.get("status") != "failed"
                ):
                    continue
                for attempt in call.get("plan_attempts", ()):
                    diagnostic = attempt.get("planner_diagnostics", {})
                    screen = diagnostic.get("metadata", {}).get(
                        "gen_sim_twist_table_clearance", {}
                    )
                    if (
                        diagnostic.get("failure", {}).get("code")
                        == "e8_table_clearance_rejected"
                        and screen.get("supported") is True
                        and screen.get("accepted") is False
                    ):
                        return True
        accepted = evidence.get("acceptance", {})
        order = accepted.get("candidate_order", ())
        count, limit = accepted.get("candidate_fallback_count"), accepted.get(
            "candidate_fallback_limit"
        )
        if not (
            accepted.get("accepted") is False
            and accepted.get("phase") == "bounded_chunk"
            and accepted.get("candidate_exhausted") is True
            and len(order) == 3
            and type(count) is int
            and type(limit) is int
            and limit > 0
            and count == min(limit, len(order) - 1)
            and accepted.get("candidate_attempt_count") == count + 1
            and accepted.get("candidate_selected") == order[count]
            and accepted.get("target_contact", accepted.get("target_contact_seen"))
            is False
            and accepted.get("already_reached", False) is False
            and accepted.get("reason", "target_contact_not_observed")
            == "target_contact_not_observed"
        ):
            return False
        target = evidence["target_actor_id"]
        fingers = set(evidence["finger_actor_ids"])
        for row in evidence["trace"]:
            if row.get("target_contact") is not False:
                return False
            pairs, impulses = row["contact_pairs"], row["impulses"]
            if len(pairs) != len(impulses):
                return False
            for pair, impulse in zip(pairs, impulses, strict=True):
                if not math.isfinite(impulse) or impulse < 0:
                    return False
                if impulse > 0 and target in pair and fingers.intersection(pair):
                    return False
        return True
    except (ValueError, OSError, KeyError, TypeError, IndexError, AttributeError):
        return False


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


def _execute_support_heights(
    preparation: PreparationResult,
    output: Path,
    executor: Callable[..., Mapping[str, Any]],
    options: dict[str, Any],
    planning: TaskEnginePlanningCfg,
    *,
    grip_depth_m: float | None = None,
    contact_feedback: bool = False,
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
            bundle = _candidate_bundle_path(output, "h", index + 1)
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
                    **(
                        {"twist_grip_depth_m": grip_depth_m}
                        if grip_depth_m is not None
                        else {}
                    ),
                    **({"twist_contact_feedback": True} if contact_feedback else {}),
                    **(
                        {"twist_mass_source": True}
                        if planning.twist_mass_source
                        else {}
                    ),
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


def execute_support_candidates(
    preparation: PreparationResult,
    output: Path,
    executor: Callable[..., Mapping[str, Any]],
    options: dict[str, Any],
    planning: TaskEnginePlanningCfg,
) -> tuple[Mapping[str, Any], PreparationResult, Path, list[dict[str, Any]]]:
    """Keep the original deployment first; search finite sizes after typed mismatch.

    Each candidate is rebuilt from the original scene, then uses the same
    executor, seed and physics. A resource/source/IK/velocity failure never
    advances the dimensional search. Height retries retain their original
    support-only trigger instead of forming a height-by-scale grid.
    """
    from .twist_adaptation import TwistGripDimensionError
    from ..task_program_bundle import generate_task_program_bundle

    report, selected, execution_root, attempts = _execute_support_heights(
        preparation, output, executor, options, planning
    )
    if not is_twist_bundle(selected.output_dir) or not size_failure(
        execution_root, report
    ):
        return report, selected, execution_root, attempts
    adaptation = load_config(selected.output_dir / "twist_adaptation.json")
    depth = adaptation.get("scale_candidate", {}).get("candidate_grip_depth")
    height = _support_height(selected.output_dir)
    acceptance = load_config(execution_root / "twist_evidence.json").get(
        "acceptance", {}
    )
    contact_feedback = (
        acceptance.get("phase") == "bounded_chunk"
        and acceptance.get("candidate_exhausted") is True
    )
    if (
        type(depth) not in (int, float)
        or not math.isfinite(depth)
        or height not in SUPPORT_HEIGHTS
    ):
        return report, selected, execution_root, attempts
    dimensional_attempts = [
        {
            "grip_depth_m": depth,
            "height_m": height,
            "bundle": str(selected.output_dir),
            "execution_root": str(execution_root),
            "status": str(report.get("status")),
            "size_rejected": True,
        }
    ]
    _save_scale_attempts(output, dimensional_attempts)
    for index, candidate_depth in enumerate(GRIP_DEPTHS, 1):
        if math.isclose(candidate_depth, depth, abs_tol=1e-8, rel_tol=0):
            continue
        bundle = _candidate_bundle_path(output, "d", index)
        next_execution = output.parent / f"{output.name}_d{index}"
        try:
            graph, generated = generate_task_program_bundle(
                preparation.semantic_task_graph,
                preparation.adaptation.prepared_scene,
                bundle,
                robot_profile=preparation.adaptation.scene_manifest["robot_profile"],
                max_episodes=planning.max_episodes,
                max_episode_steps=planning.max_episode_steps,
                fit_grasp_assets=planning.fit_grasp_assets,
                twist_support_lift_m=height,
                twist_grip_depth_m=candidate_depth,
                **({"twist_contact_feedback": True} if contact_feedback else {}),
                **({"twist_mass_source": True} if planning.twist_mass_source else {}),
            )
        except (TwistGripDimensionError, TwistSupportPenetrationError) as error:
            dimensional_attempts.append(
                {
                    "grip_depth_m": candidate_depth,
                    "height_m": height,
                    "status": "geometric_candidate_rejected",
                    "error": str(error),
                    "physical_attempt": False,
                }
            )
            _save_scale_attempts(output, dimensional_attempts)
            continue
        # Provider receipts remain attached to each newly qualified deployment.
        for path in preparation.output_dir.iterdir():
            if path.is_file() and not (bundle / path.name).exists():
                if path.name == "planner_report.json":
                    from ..semantic_graph import semantic_task_graph_hash

                    receipt = load_config(path)
                    receipt.update(
                        integration_fingerprint=graph["integration_fingerprint"],
                        semantic_task_graph_hash=semantic_task_graph_hash(graph),
                        source_preparation=str(preparation.output_dir),
                        support_lift_m=height,
                        grip_depth_m=candidate_depth,
                        contact_feedback=contact_feedback,
                    )
                    (bundle / path.name).write_text(
                        json.dumps(receipt, indent=2), encoding="utf-8"
                    )
                else:
                    shutil.copy2(path, bundle / path.name)
        candidate = replace(
            preparation,
            output_dir=bundle,
            semantic_task_graph=graph,
            generated_paths=generated,
        )
        row = {
            "grip_depth_m": candidate_depth,
            "height_m": height,
            "bundle": str(bundle),
            "execution_root": str(next_execution),
            "integration_fingerprint": graph.get("integration_fingerprint"),
            "status": "running",
            "seed": options.get("seed"),
        }
        dimensional_attempts.append(row)
        _save_scale_attempts(output, dimensional_attempts)
        try:
            report, selected, execution_root, height_attempts = (
                _execute_support_heights(
                    candidate,
                    next_execution,
                    executor,
                    options,
                    planning,
                    grip_depth_m=candidate_depth,
                )
            )
        except Exception as error:
            row.update(
                status="executor_error",
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            _save_scale_attempts(output, dimensional_attempts)
            raise
        rejected = size_failure(execution_root, report)
        height = _support_height(selected.output_dir)
        row.update(
            {
                "height_m": _support_height(selected.output_dir),
                "bundle": str(selected.output_dir),
                "execution_root": str(execution_root),
                "status": str(report.get("status")),
                "size_rejected": rejected,
                "support_attempts": height_attempts,
            }
        )
        _save_scale_attempts(output, dimensional_attempts)
        attempts.extend(height_attempts)
        if not rejected:
            break
    _save_scale_attempts(output, dimensional_attempts)
    return report, selected, execution_root, attempts
