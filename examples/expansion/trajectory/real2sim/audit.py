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

"""Independently audit persisted synthetic Real2Sim acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import yaml

from embodichain.data_pipeline.real2sim import load_prototype
from embodichain.data_pipeline.recording.journal import inspect_recording
from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    load_real2sim_tabletop_profile,
)
from embodichain.lab.sim.atomic_actions import MotionPolicy

__all__: list[str] = []


def _artifact(root: Path, item: dict[str, object]) -> Path:
    path = (root / str(item["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("artifact is missing or outside its dataset")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise ValueError("artifact digest mismatch")
    return path


def _check_case(
    root: Path, row: dict[str, object], profile: object
) -> dict[str, object]:
    receipt_path = (
        root
        / "meta"
        / "real2sim"
        / "receipts"
        / f"{uuid5(NAMESPACE_URL, row['commit_id'])}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not receipt["confirmed"]
        or receipt["candidate_id"] != row["candidate_id"]
        or receipt["episode_id"] != f"episode:{row['candidate_id']}:{row['attempt_id']}"
        or receipt["commit_id"] != row["commit_id"]
        or receipt["storage_id"] != f"{root}#{row['episode_uuid']}"
    ):
        raise ValueError("receipt identity does not match the committed episode")
    checks = row["validation_checks"]
    required = {
        "geometry",
        "motion_limits",
        "phase_mapping",
        "plan_collision",
        "observed_collision",
        "task",
    }
    if not required <= {item["id"] for item in checks} or any(
        item["status"] != "passed" for item in checks
    ):
        raise ValueError("a required validation check did not pass")
    evidence = row["real2sim_episode_evidence"]
    with np.load(_artifact(root, evidence), allow_pickle=False) as payload:
        actions = np.array(payload["actions"], copy=True)
        timestamps = np.array(payload["timestamps"], copy=True)
        names = evidence["observation_names"]
        if (
            actions.ndim != 2
            or timestamps.shape != (len(actions) + 1,)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("actual actions and timestamps do not form a T+1 stream")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("actual timestamps are invalid")
        for i, name in enumerate(names):
            value = payload[f"observation_{i}"]
            if value.shape[0] != len(actions) + 1 or not np.isfinite(value).all():
                raise ValueError(
                    f"observation {name!r} lacks an actual terminal sample"
                )
    with np.load(
        _artifact(root, row["real2sim_observed_geometry"]), allow_pickle=False
    ) as physical:
        objects = np.array(physical["object_pose"], copy=True)
        target = np.array(physical["target_pose"], copy=True)
        if (
            objects.shape != (len(actions) + 1, 7)
            or target.shape != (4, 4)
            or not np.isfinite(objects).all()
            or not np.isfinite(target).all()
        ):
            raise ValueError("measured object/support sidecar has the wrong shape")
        if not np.allclose(physical["timestamps"], timestamps, atol=1e-8, rtol=0):
            raise ValueError("object and robot observation clocks differ")
        if not np.allclose(
            np.linalg.norm(objects[:, 3:], axis=1), 1, atol=1e-4, rtol=0
        ):
            raise ValueError("measured object quaternion is invalid")
    metrics = next(item["metrics"] for item in checks if item["id"] == "task")
    xy_error = float(np.linalg.norm(objects[-1, :2] - target[:2, 3]))
    if abs(xy_error - metrics["placement_xy_error_m"]) > 1e-6:
        raise ValueError("placement metric differs from recorded object observations")
    drift = float(
        np.linalg.norm(objects[-1, :3] - objects[max(0, len(actions) - 19), :3])
    )
    if abs(drift - metrics["drift_m"]) > 1e-6:
        raise ValueError("stability metric differs from recorded object observations")
    if (
        xy_error > profile.placement_xy_tolerance_m
        or metrics["height_error_m"] > profile.placement_height_tolerance_m
        or drift > profile.maximum_drift_m
        or metrics["lift_m"] < profile.minimum_lift_m
        or metrics["held_translation_drift_m"] > profile.held_translation_tolerance_m
        or metrics["held_rotation_drift_deg"] > profile.held_rotation_tolerance_deg
    ):
        raise ValueError("measured task metrics exceed the explicit profile tolerances")
    phases = row["phase_duration_validation"]
    expected_phases = {
        (0, "approach"),
        (0, "close"),
        (0, "lift"),
        (1, "transport"),
        (2, "approach"),
        (2, "release"),
        (2, "retract"),
    }
    if (
        len(phases) != len(expected_phases)
        or {(x["call_index"], x["phase"]) for x in phases} != expected_phases
    ):
        raise ValueError("phase validation is missing a semantic phase")
    recipe = row["recipe"]
    scale = 1.15 if recipe == "time_115" else 1.30 if recipe == "time_130" else None
    free = {0: {"approach"}, 1: {"transport"}, 2: {"approach", "retract"}}
    for phase in phases:
        retimed = scale is not None and phase["phase"] in free[phase["call_index"]]
        expected = (
            math.ceil((phase["source_samples"] - 1) * scale - 1e-9) + 1
            if retimed
            else phase["source_samples"]
        )
        if (
            phase["final_samples"] != expected
            or not phase["passed"]
            or phase["retimed"] != retimed
        ):
            raise ValueError(
                "semantic phase duration does not match the selected recipe"
            )
    collisions = row["sampled_plan_collisions"]
    expected_checks = 3 if recipe == "nominal" else 6
    if len(collisions) != expected_checks or any(
        not item["accepted"] for item in collisions
    ):
        raise ValueError("nominal or expanded plan collision evidence is incomplete")
    if (
        not row["sampled_observed_collision"]["accepted"]
        or row["sampled_observed_collision"]["sample_count"] != len(actions) + 1
    ):
        raise ValueError("observed collision evidence is incomplete")
    ratios = row["sampled_motion_limits"]
    if len(ratios) != 3 or any(
        max(value.values()) > 1 + 1e-6 for value in ratios.values()
    ):
        raise ValueError("motion-limit ratios are incomplete or exceed the profile")
    lineage = row["real2sim_lineage"]
    calls = lineage["calls"]
    if len(calls) != 3 or {call["call_index"] for call in calls} != {0, 1, 2}:
        raise ValueError("source lineage must cover three distinct call occurrences")
    max_position, max_angle = 0.0, 0.0
    with np.load(_artifact(root, lineage["trace"]), allow_pickle=False) as trace:
        step_call, step_plan = trace["step_call_index"], trace["step_plan_index"]
        if step_call.shape != (len(actions),) or step_plan.shape != step_call.shape:
            raise ValueError("action-to-plan trace length is invalid")
        transport_steps = np.flatnonzero(step_call == 1)
        if not len(transport_steps):
            raise ValueError("trace has no measured transport boundary")
        lift = float(objects[int(transport_steps[0]), 2] - objects[0, 2])
        if abs(lift - metrics["lift_m"]) > 1e-6:
            raise ValueError("lift metric differs from recorded object observations")
        for call in calls:
            index = call["call_index"]
            bundle = row["real2sim_artifacts"][call["source_bundle"]]
            for item in bundle.values():
                _artifact(root, item)
            if bundle["prototype_manifest"]["sha256"] != call["prototype_sha256"]:
                raise ValueError("prototype identity differs from its archived bundle")
            prototype = load_prototype(root / bundle["prototype_manifest"]["path"])
            if prototype.unit.source.origin != "synthetic_reference":
                raise ValueError("unexpected source origin")
            source = prototype.unit.source
            if (
                source.source_id != row["source_id"]
                or source.source_revision != row["source_revision"]
                or source.robot_geometry_sha256 != collisions[0]["geometry_sha256"]
                or source.object_geometry_sha256 is None
                or (index != 0 and source.target_geometry_sha256 is None)
            ):
                raise ValueError("source revision or geometry identity is incomplete")
            if call["geometry_family_id"] != row["geometry_family_id"]:
                raise ValueError("call geometry family differs from its episode")
            if profile.motion_headroom_fraction is not None and recipe in {
                "joint_7",
                "joint_11",
            }:
                locked = call.get("locked_joint_indices")
                locked_names = call.get("locked_joint_names")
                if (
                    call.get("motion_headroom_fraction")
                    != profile.motion_headroom_fraction
                    or not isinstance(locked, list)
                    or len(set(locked)) != len(locked)
                    or any(
                        type(value) is not int
                        or not 0 <= value < len(prototype.unit.joint_names)
                        for value in locked
                    )
                    or locked_names
                    != [prototype.unit.joint_names[value] for value in locked]
                ):
                    raise ValueError("headroom-locked joint lineage is invalid")
            left, right, weight = (
                trace[f"call_{index}_{key}"] for key in ("left", "right", "weight")
            )
            src, dst = (
                trace[f"call_{index}_{key}"] for key in ("source_events", "plan_events")
            )
            if len(src) != len(dst) or np.any(dst < 0) or np.any(dst >= len(left)):
                raise ValueError("hard-event mapping is malformed")
            if any(
                left[d] != s or right[d] != s or weight[d] != 0
                for s, d in zip(src, dst)
            ):
                raise ValueError("hard event is not mapped to an exact source sample")
            observed = step_plan[(step_call == index) & (step_plan >= 0)]
            if set(observed.tolist()) != set(range(len(left))) or np.any(
                np.diff(observed) < 0
            ):
                raise ValueError("submitted actions do not cover the plan in order")
            if scale is None and (
                not np.array_equal(left, np.arange(len(left)))
                or not np.array_equal(right, left)
                or np.any(weight)
            ):
                raise ValueError("same-grid variant changed the plan time mapping")
            position, angle = call["fk_position_error_m"], call["fk_rotation_error_rad"]
            if (
                position is None
                or angle is None
                or position > 0.002
                or angle > math.radians(2)
            ):
                raise ValueError("IK residual is absent or exceeds its threshold")
            max_position, max_angle = max(max_position, position), max(max_angle, angle)
    return {
        "layout": row["layout"],
        "recipe": recipe,
        "steps": len(actions),
        "candidate_id": row["candidate_id"],
        "receipt_confirmed": True,
        "xy_error_m": xy_error,
        "lift_m": metrics["lift_m"],
        "held_translation_drift_m": metrics["held_translation_drift_m"],
        "held_rotation_drift_deg": metrics["held_rotation_drift_deg"],
        "fk_position_error_m": max_position,
        "fk_rotation_error_rad": max_angle,
        "maximum_speed_ratio": max(item["speed_ratio"] for item in ratios.values()),
        "maximum_acceleration_ratio": max(
            item["acceleration_ratio"] for item in ratios.values()
        ),
        "phase_duration_validation": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--execution-policy",
        type=Path,
        default=Path(__file__).with_name("execution_policy.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    root = options.dataset.resolve()
    profile = load_real2sim_tabletop_profile(options.profile)
    execution_policy = yaml.safe_load(
        options.execution_policy.read_text(encoding="utf-8")
    )
    arm_acceleration = execution_policy.get("motion", {}).get(
        "acceleration_limit", MotionPolicy().acceleration_limit
    )
    expected = {
        (layout["id"], recipe)
        for layout in profile.layouts
        for recipe in profile.recipes
    }
    journal = inspect_recording(root)
    rows = [
        json.loads(line)
        for line in (root / "meta/embodichain_episodes.jsonl").read_text().splitlines()
        if line
    ]
    cases, failures = [], []
    seen = set()
    for row in rows:
        key = (row.get("layout"), row.get("recipe"))
        if key in seen:
            failures.append(
                {
                    "layout": key[0],
                    "recipe": key[1],
                    "error": "duplicate schedule entry",
                }
            )
            continue
        seen.add(key)
        try:
            cases.append(_check_case(root, row, profile))
        except Exception as error:
            failures.append(
                {
                    "layout": key[0],
                    "recipe": key[1],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    for layout, recipe in sorted(expected - seen):
        failures.append(
            {"layout": layout, "recipe": recipe, "error": "missing episode"}
        )
    for layout, recipe in sorted(seen - expected):
        failures.append(
            {"layout": layout, "recipe": recipe, "error": "unexpected episode"}
        )
    if journal["errors"] or any(
        item["status"] != "complete" for item in journal["commits"]
    ):
        failures.append(
            {"error": "recording journal is incomplete", "details": journal["errors"]}
        )
    failed_dir = root / "meta/real2sim/failed-attempts"
    attempts = (
        [json.loads(path.read_text()) for path in sorted(failed_dir.glob("*.json"))]
        if failed_dir.exists()
        else []
    )
    report = {
        "origin": "synthetic_reference",
        "dataset": str(root),
        "profile": str(options.profile.resolve()),
        "profile_sha256": hashlib.sha256(options.profile.read_bytes()).hexdigest(),
        "execution_policy": str(options.execution_policy.resolve()),
        "execution_policy_sha256": hashlib.sha256(
            options.execution_policy.read_bytes()
        ).hexdigest(),
        "policy_hash_scope": "audited local configuration; episode metadata does not carry this digest",
        "velocity_limits": profile.velocity_limits,
        "acceleration_limits": profile.acceleration_limits,
        "arm_motion_policy_acceleration_limit": arm_acceleration,
        "expected_cases": len(expected),
        "validated_cases": len(cases),
        "failure_count": len(failures),
        "simulation_and_measured_validation_passed": len(cases),
        "sampled_collision_validated": len(cases),
        "persistence_receipts_verified": len(cases),
        "synthetic_matrix_complete": not failures and len(cases) == len(expected),
        "real_data_qualified": False,
        "continuous_collision_qualified": False,
        "metrics": {
            "maximum_fk_position_error_m": max(
                (item["fk_position_error_m"] for item in cases), default=None
            ),
            "maximum_fk_rotation_error_deg": (
                math.degrees(
                    max((item["fk_rotation_error_rad"] for item in cases), default=0.0)
                )
                if cases
                else None
            ),
            "minimum_pick_lift_m": min(
                (item["lift_m"] for item in cases), default=None
            ),
            "maximum_placement_xy_error_m": max(
                (item["xy_error_m"] for item in cases), default=None
            ),
            "maximum_held_translation_drift_m": max(
                (item["held_translation_drift_m"] for item in cases), default=None
            ),
            "maximum_held_rotation_drift_deg": max(
                (item["held_rotation_drift_deg"] for item in cases), default=None
            ),
            "maximum_speed_ratio": max(
                (item["maximum_speed_ratio"] for item in cases), default=None
            ),
            "maximum_acceleration_ratio": max(
                (item["maximum_acceleration_ratio"] for item in cases), default=None
            ),
        },
        "failures": failures,
        "failed_attempts": attempts,
        "collision_scope": "sampled nominal, expanded and measured states; no continuous or hardware qualification",
        "velocity_limit_origin": profile.velocity_limit_origin,
        "acceleration_limit_origin": profile.acceleration_limit_origin,
        "cases": cases,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"{len(cases)}/{len(expected)} cases validated; {len(failures)} failure(s): {options.output}"
    )
    if failures or len(cases) != len(expected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
