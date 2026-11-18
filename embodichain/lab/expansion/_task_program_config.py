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

"""Closed scene/motion declarations for native Task Program workloads."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import math
import json
import yaml

__all__: list[str] = []


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Read declarations without importing the simulator or its logging hooks."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


OPERATORS = (
    "object_pose",
    "affordance",
    "joint_residual",
    "via_points",
    "nullspace_residual",
)


def load_expansion_viewer(value: dict | None = None) -> dict:
    """Resolve presentation options without importing the native simulator.

    Args:
        value: Optional headless and renderer declarations.
    Returns:
        Owned normalized options; unsupported values reject.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {"headless", "renderer"}:
        raise ValueError("Expansion viewer accepts only headless and renderer")
    headless, renderer = value.get("headless", True), value.get("renderer", "rt")
    if type(headless) is not bool:
        raise ValueError("Expansion viewer.headless must be a boolean")
    if renderer not in ("rt", "fast-rt", "hybrid"):
        raise ValueError("Expansion viewer.renderer must be rt, fast-rt or hybrid")
    return {"headless": headless, "renderer": renderer}


def load_task_program_expansion(path: str | Path) -> dict[str, Any]:
    """Load bounded scene/motion declarations while retaining physical criteria.

    Args:
        path: YAML/JSON engine, scene, trajectory and validation declarations.
    Returns:
        Owned normalized declaration, including workload capacity.
    Raises:
        ValueError: For invalid scopes, protected phases, budgets or criteria.
    """
    path = Path(path)
    value = load_mapping(path)
    if (set(value) - {"scene_sampling", "scheduling", "pose_coverage"}) != {
        "schema_version",
        "engine",
        "scene",
        "trajectory",
        "validation",
    } or value["schema_version"] != 1:
        raise ValueError("Invalid demo expansion configuration")
    engine, motion = value["engine"], value["trajectory"]
    required_engine = {
        "episodes_per_operator",
        "max_attempts",
        "max_steps",
        "max_wall_time_s",
        "settling_steps",
        "capture_media",
    }
    if required_engine - set(engine) or set(engine) - required_engine - {
        "cpu_threads",
        "operator_workers",
        "workload_count",
        "batch_size",
        "collision_device",
        "grasp_nms_chunk_size",
        "collection_renderer",
        "collision_backend",
        "reuse_idle_rows",
        "nominal_seed",
    }:
        raise ValueError("Unknown or missing expansion engine settings")
    engine.setdefault("cpu_threads", 1)
    if "operator_workers" in engine:
        if "workload_count" in engine:
            raise ValueError(
                "Use engine.workload_count without legacy operator_workers"
            )
        # Compatibility is a capacity migration, never a process fallback.
        engine["workload_count"] = engine.pop("operator_workers")
    engine.setdefault("workload_count", 1)
    engine.setdefault("batch_size", 1)
    engine.setdefault("collision_device", "cpu")
    engine.setdefault("grasp_nms_chunk_size", 2048)
    engine.setdefault("collection_renderer", "auto")
    engine.setdefault("collision_backend", "newton")
    if "reuse_idle_rows" in engine and type(engine["reuse_idle_rows"]) is not bool:
        raise ValueError("engine.reuse_idle_rows must be a boolean")
    if "nominal_seed" in engine and (
        type(engine["nominal_seed"]) is not int or engine["nominal_seed"] < 0
    ):
        raise ValueError("engine.nominal_seed must be a nonnegative integer")
    if engine["collection_renderer"] not in ("auto", "rt", "fast-rt", "hybrid"):
        raise ValueError(
            "engine.collection_renderer must be auto, rt, fast-rt or hybrid"
        )
    if engine["collision_backend"] not in ("newton", "native_contacts"):
        raise ValueError("engine.collision_backend must be newton or native_contacts")
    for name in (
        "episodes_per_operator",
        "max_attempts",
        "max_steps",
        "settling_steps",
        "cpu_threads",
        "workload_count",
        "batch_size",
        "grasp_nms_chunk_size",
    ):
        if type(engine[name]) is not int or engine[name] < 1:
            raise ValueError(f"engine.{name} must be a positive integer")
    device = engine["collision_device"]
    if not isinstance(device, str) or not (
        device == "cpu"
        or device == "cuda"
        or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("engine.collision_device must be cpu, cuda or cuda:<index>")
    if (
        engine["max_attempts"] < engine["episodes_per_operator"]
        or type(engine["capture_media"]) is not bool
    ):
        raise ValueError("Invalid expansion attempt or media settings")
    required = {
        "operators",
        "joint_offset_scale",
        "via_count",
        "max_proposals",
        "minimum_joint_delta",
        "velocity_limit",
        "acceleration_limit",
        "nullspace_task_rows",
        "nullspace_translation_tolerance_m",
        "nullspace_rotation_tolerance_rad",
        "free_phases",
    }
    if required - set(motion) or set(motion) - required - {
        "affordance",
        "call_indices",
    }:
        raise ValueError("Unknown or missing trajectory settings")
    calls = motion.get("call_indices")
    if calls is not None and (
        not isinstance(calls, list)
        or not calls
        or any(type(index) is not int or index < 0 for index in calls)
        or len(set(calls)) != len(calls)
    ):
        raise ValueError(
            "Trajectory call_indices must be distinct non-negative integers"
        )
    operators = motion["operators"]
    if (
        not isinstance(operators, list)
        or not operators
        or len(set(operators)) != len(operators)
        or set(operators) - set(OPERATORS)
    ):
        raise ValueError(
            "trajectory.operators must contain distinct supported operators"
        )
    affordance = motion.get("affordance")
    if "affordance" in operators and affordance is None:
        raise ValueError("trajectory.affordance is required for affordance expansion")
    if affordance is not None:
        required_affordance = {"branches", "call_indices", "action_options"}
        if (
            not isinstance(affordance, dict)
            or required_affordance - set(affordance)
            or set(affordance) - required_affordance - {"grasp_generators"}
        ):
            raise ValueError("Invalid trajectory.affordance settings")
        if type(affordance["branches"]) is not int or affordance["branches"] < 2:
            raise ValueError("trajectory.affordance.branches must be at least two")
        calls = affordance["call_indices"]
        if (
            not isinstance(calls, list)
            or not calls
            or any(type(index) is not int or index < 0 for index in calls)
            or len(set(calls)) != len(calls)
        ):
            raise ValueError(
                "Affordance call_indices must be distinct non-negative integers"
            )
        if not isinstance(affordance["action_options"], dict) or any(
            not isinstance(name, str) or not isinstance(options, dict)
            for name, options in affordance["action_options"].items()
        ):
            raise ValueError(
                "Affordance action_options must map preset names to settings"
            )
        generators = affordance.get("grasp_generators", {})
        if not isinstance(generators, dict):
            raise ValueError("Affordance grasp_generators must be a mapping")
        for name, settings in generators.items():
            if (
                not isinstance(name, str)
                or not isinstance(settings, dict)
                or not settings
                or set(settings)
                - {
                    "max_candidates",
                    "approach_deviation_angle",
                    "kind",
                    "gripper",
                    "root_from_contact_frame",
                }
            ):
                raise ValueError("Invalid affordance grasp generator overrides")
            if {"kind", "gripper"} & settings.keys() and (
                not {"kind", "gripper"} <= settings.keys()
                or settings["kind"] != "antipodal_parallel_jaw"
                or not isinstance(settings["gripper"], dict)
            ):
                raise ValueError(
                    "A new grasp generator requires its kind and physical gripper description"
                )
            count = settings.get("max_candidates", 2)
            angle = settings.get("approach_deviation_angle", 0.0)
            if type(count) is not int or count < 1:
                raise ValueError("Affordance max_candidates must be positive")
            if type(angle) not in (int, float) or not math.isfinite(angle) or angle < 0:
                raise ValueError(
                    "Affordance approach_deviation_angle must be finite and non-negative"
                )
    for name in ("via_count", "max_proposals"):
        if type(motion[name]) is not int or motion[name] < 1:
            raise ValueError(f"trajectory.{name} must be a positive integer")
    for name in required - {
        "operators",
        "via_count",
        "max_proposals",
        "nullspace_task_rows",
        "free_phases",
    }:
        if (
            type(motion[name]) not in (float, int)
            or not math.isfinite(motion[name])
            or motion[name] <= 0
        ):
            raise ValueError(f"trajectory.{name} must be finite and positive")
    if motion["nullspace_task_rows"] not in ([0, 1, 2], [0, 1, 2, 3, 4, 5]):
        raise ValueError(
            "Nullspace tasks must explicitly constrain position or full pose"
        )
    # These are the upstream skills' conservative free phases, narrowed for these demos.
    allowed = {
        "hand_over": {"pickup_approach"},
        "pick_up": {"approach"},
        "place": {"retract"},
    }
    if not isinstance(motion["free_phases"], dict) or not motion["free_phases"]:
        raise ValueError("Explicit free_phases are required")
    for skill, phases in motion["free_phases"].items():
        if (
            skill not in allowed
            or not isinstance(phases, list)
            or not phases
            or set(phases) - allowed[skill]
        ):
            raise ValueError(f"Unsupported or protected free phase for {skill}")
    scene = value["scene"]
    if not isinstance(scene, dict) or not scene:
        raise ValueError("Explicit scene pose ranges are required")
    for uid, options in scene.items():
        if set(options) == {"follow"}:
            if options["follow"] not in scene or "follow" in scene[options["follow"]]:
                raise ValueError(
                    "Scene followers must reference an independent pose range"
                )
            continue
        if (
            set(options) != {"translation_m", "yaw_rad"}
            or len(options["translation_m"]) != 3
        ):
            raise ValueError(f"Invalid scene pose range for {uid}")
        for limits in (*options["translation_m"], options["yaw_rad"]):
            if (
                len(limits) != 2
                or any(
                    type(x) not in (float, int) or not math.isfinite(x) for x in limits
                )
                or limits[0] > limits[1]
            ):
                raise ValueError(f"Invalid finite ordered pose range for {uid}")
    if not math.isfinite(engine["max_wall_time_s"]) or engine["max_wall_time_s"] <= 0:
        raise ValueError("engine.max_wall_time_s must be finite and positive")
    required_validation = {
        "initial_checks",
        "final_checks",
        "events",
        "allowed_contacts",
        "mounting_contacts",
    }
    if required_validation - set(value["validation"]) or set(
        value["validation"]
    ) - required_validation - {"contact_hold"}:
        raise ValueError(
            "Explicit measured validation and contact permissions are required"
        )
    hold = value["validation"].get("contact_hold")
    if hold is not None:
        if (
            engine["collision_backend"] != "native_contacts"
            or not isinstance(hold, dict)
            or set(hold)
            != {
                "entity",
                "links",
                "duration_s",
                "max_translation_drift_m",
                "max_rotation_drift_rad",
            }
        ):
            raise ValueError(
                "contact_hold requires native_contacts and explicit hold settings"
            )
        if (
            not isinstance(hold["entity"], str)
            or not hold["entity"]
            or not isinstance(hold["links"], list)
            or len(hold["links"]) != 2
            or len(set(hold["links"])) != 2
            or any(not isinstance(link, str) or not link for link in hold["links"])
        ):
            raise ValueError(
                "contact_hold requires an entity and two distinct robot links"
            )
        for field in (
            "duration_s",
            "max_translation_drift_m",
            "max_rotation_drift_rad",
        ):
            if (
                type(hold[field]) not in (float, int)
                or not math.isfinite(hold[field])
                or hold[field] <= 0
            ):
                raise ValueError(f"contact_hold.{field} must be finite and positive")
    from ._pose_sampling_cfg import validate_sampling

    validate_sampling(value)
    return value
