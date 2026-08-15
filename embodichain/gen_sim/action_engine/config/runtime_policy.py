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

"""Load, resolve, snapshot, and hash package-owned Action Engine defaults."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from embodichain.gen_sim.action_engine.domain.motion import MOTION_MODIFIER_MODES
from embodichain.toolkits.graspkit.pg_grasp.profiles import (
    ParallelJawEefProfile,
    get_parallel_jaw_eef_profile,
)
from embodichain.utils import configclass
from embodichain.utils.utility import load_config

__all__ = [
    "ACTION_ENGINE_DEFAULTS_SCHEMA",
    "RUNTIME_POLICY_SCHEMA",
    "ArmSelectionPolicyCfg",
    "RuntimePolicyCfg",
    "default_runtime_policy",
    "generation_defaults",
    "resolve_agent_runtime_policy",
    "runtime_policy_hash",
]

ACTION_ENGINE_DEFAULTS_SCHEMA: Final = "action_engine_defaults_v1"
RUNTIME_POLICY_SCHEMA: Final = "action_engine_runtime_policy_v7"
_PRE_EEF_RUNTIME_POLICY_SCHEMA: Final = "action_engine_runtime_policy_v6"
_PREVIOUS_RUNTIME_POLICY_SCHEMA: Final = "action_engine_runtime_policy_v5"
_PRE_GRASP_RUNTIME_POLICY_SCHEMA: Final = "action_engine_runtime_policy_v4"
_PRE_PLANNER_RUNTIME_POLICY_SCHEMA: Final = "action_engine_runtime_policy_v3"
_LEGACY_RUNTIME_POLICY_SCHEMA: Final = "action_engine_runtime_policy_v1"
_DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")
_ARM_SELECTION_KEYS = (
    "crossing_deadband_ratio",
    "pickup_crossing_weight",
    "placement_crossing_weight",
    "motion_cost_scale",
    "fallback_workspace_half_width",
    "orient_object_preferred_arm_deadband",
)
_GROUNDING_KEYS = {
    "semantic_defaults": {
        "surface_clearance",
        "transport_clearance",
        "staging_lift_height",
        "relation_distance",
        "hover_height",
        "press_depth",
        "retreat_height",
        "maximum_eef_height",
    },
    "arrangement": {
        "slot_margin",
        "minimum_spacing",
        "layout_clearance",
        "row_search_step",
        "row_search_radius",
    },
    "placement": {
        "clearance",
        "candidate_count",
        "candidate_offset_fraction",
        "support_margin",
        "recovery_attempts",
    },
    "coordinated_grasp": {"inset_fraction", "minimum_inset"},
    "handover": {
        "retreat_height",
        "retreat_distance",
        "maximum_eef_height",
        "minimum_transfer_clearance",
        "minimum_transfer_lateral_clearance",
    },
    "joint_state": {
        "hand_close_sample_interval",
        "hand_open_sample_interval",
    },
}
_GRASP_KEYS = {
    "antipodal_n_sample",
    "antipodal_max_angle",
    "min_contact_span",
    "max_contact_span",
    "max_deviation_angle",
    "n_deviated_approach_directions",
    "n_top_grasps",
    "viser_port",
    "max_decomposition_hulls",
    "filter_support_collision",
    "force_grasp_reannotate",
}
_PLANNER_KEYS = {
    "backend",
    "single_arm_strategy",
    "coordinated_strategy",
    "fallback_strategy",
    "allow_fallback",
    "dynamic_collision",
    "static_obstacle_uids",
    "dynamic_obstacle_uids",
    "curobo",
}
_CUROBO_KEYS = {
    "log_level",
    "obstacle_representation",
    "multi_env",
    "use_cuda_graph",
    "preserve_plan_samples",
    "max_attempts",
    "collision_activation_distance",
}
_MOTION_DEFAULT_ACTIONS = {
    "CoordinatedPickment",
    "CoordinatedPlacement",
    "HandOver",
    "MoveEndEffector",
    "MoveHeldObject",
    "MoveJoints",
    "PickUp",
    "Place",
    "Press",
}
_PREDICATE_KEYS = {
    "held_position_tolerance",
    "held_gripper_tolerance",
    "position_tolerance",
    "xy_tolerance",
    "container_xy_radius",
    "container_min_z_offset",
    "container_max_z_offset",
    "support_xy_radius",
    "support_com_margin",
    "support_max_vertical_gap",
    "support_max_penetration",
    "support_min_overlap_ratio",
    "not_fallen_max_tilt",
    "upright_max_tilt",
    "axis_tolerance",
    "collinearity_tolerance",
    "ordering_tolerance",
    "minimum_lift_height",
    "arm_initial_qpos_tolerance",
    "gripper_state_tolerance",
    "gripper_clear_min_distance",
    "line_axis_tolerance",
    "line_perpendicular_tolerance",
    "preserve_orientation_tolerance",
    "payload_minimum_upright_cosine",
    "payload_position_tolerance",
    "payload_support_margin",
}
_DEPRECATED_PREDICATE_KEYS = {
    "support_min_z_offset",
    "support_max_z_offset",
}


@configclass
class ArmSelectionPolicyCfg:
    """Soft arm-allocation cost parameters resolved for one robot profile."""

    crossing_deadband_ratio: float = 0.08
    pickup_crossing_weight: float = 1.0
    placement_crossing_weight: float = 1.5
    motion_cost_scale: float = math.pi
    fallback_workspace_half_width: float = 0.5
    orient_object_preferred_arm_deadband: float = 0.02

    def __post_init__(self) -> None:
        for name in _ARM_SELECTION_KEYS:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= float(self.crossing_deadband_ratio) < 1.0:
            raise ValueError("crossing_deadband_ratio must be in [0, 1).")
        for name in _ARM_SELECTION_KEYS[1:3]:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        for name in _ARM_SELECTION_KEYS[3:]:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArmSelectionPolicyCfg:
        """Build a strict policy from a JSON/YAML mapping."""
        if set(value) != set(_ARM_SELECTION_KEYS):
            raise ValueError("arm_selection fields do not match the policy schema.")
        return cls(**{key: float(value[key]) for key in _ARM_SELECTION_KEYS})

    def as_mapping(self) -> dict[str, float]:
        """Return a stable JSON-compatible representation."""
        return {
            "crossing_deadband_ratio": float(self.crossing_deadband_ratio),
            "pickup_crossing_weight": float(self.pickup_crossing_weight),
            "placement_crossing_weight": float(self.placement_crossing_weight),
            "motion_cost_scale": float(self.motion_cost_scale),
            "fallback_workspace_half_width": float(self.fallback_workspace_half_width),
            "orient_object_preferred_arm_deadband": float(
                self.orient_object_preferred_arm_deadband
            ),
        }


@configclass
class RuntimePolicyCfg:
    """Effective runtime policy persisted in generated agent artifacts."""

    schema_version: str = RUNTIME_POLICY_SCHEMA
    end_effector_profile: ParallelJawEefProfile = get_parallel_jaw_eef_profile(
        "robotiq_arg2f_140"
    )
    arm_selection: ArmSelectionPolicyCfg = ArmSelectionPolicyCfg()
    execution: dict[str, Any] = {}
    planner: dict[str, Any] = {}
    grounding: dict[str, Any] = {}
    grasp: dict[str, Any] = {}
    motion_defaults: dict[str, dict[str, Any]] = {}
    motion_modifiers: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    predicate_fallbacks: dict[str, Any] = {}

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_POLICY_SCHEMA:
            raise ValueError(
                f"Unsupported runtime policy schema {self.schema_version!r}."
            )
        if not isinstance(self.arm_selection, ArmSelectionPolicyCfg):
            raise TypeError("arm_selection must be an ArmSelectionPolicyCfg.")
        if not isinstance(self.end_effector_profile, ParallelJawEefProfile):
            raise TypeError(
                "end_effector_profile must be a ParallelJawEefProfile."
            )
        for name in (
            "execution",
            "planner",
            "grounding",
            "grasp",
            "motion_defaults",
            "motion_modifiers",
            "predicate_fallbacks",
        ):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"{name} must be a mapping.")
            _validate_finite_numbers(getattr(self, name), name)
        if int(self.execution.get("max_transitions", 0)) <= 0:
            raise ValueError("execution.max_transitions must be positive.")
        if int(self.execution.get("semantic_step_settle_steps", -1)) < 0:
            raise ValueError(
                "execution.semantic_step_settle_steps must be non-negative."
            )
        for name in (
            "max_retries_per_action",
            "max_graph_revisions",
            "max_recovery_actions",
            "support_stability_interval_steps",
        ):
            if int(self.execution.get(name, -1)) < 0:
                raise ValueError(f"execution.{name} must be non-negative.")
        _require_keys(
            self.execution,
            {
                "max_transitions",
                "semantic_step_settle_steps",
                "max_retries_per_action",
                "max_graph_revisions",
                "max_recovery_actions",
                "support_stability_samples",
                "support_stability_interval_steps",
                "support_linear_velocity_tolerance",
                "support_angular_velocity_tolerance",
            },
            "execution",
        )
        if int(self.execution["support_stability_samples"]) <= 0:
            raise ValueError("execution.support_stability_samples must be positive.")
        for name in (
            "support_linear_velocity_tolerance",
            "support_angular_velocity_tolerance",
        ):
            if float(self.execution[name]) < 0.0:
                raise ValueError(f"execution.{name} must be non-negative.")
        _validate_planner(self.planner)
        _require_keys(self.grounding, set(_GROUNDING_KEYS), "grounding")
        for name, keys in _GROUNDING_KEYS.items():
            section = self.grounding.get(name)
            if not isinstance(section, Mapping):
                raise ValueError(f"grounding.{name} must be a mapping.")
            _require_keys(section, keys, f"grounding.{name}")
        placement = self.grounding["placement"]
        if not 1 <= int(placement["candidate_count"]) <= 9:
            raise ValueError("grounding.placement.candidate_count must be in [1, 9].")
        if int(placement["recovery_attempts"]) < 0:
            raise ValueError(
                "grounding.placement.recovery_attempts must be non-negative."
            )
        if not 0.0 <= float(placement["candidate_offset_fraction"]) <= 1.0:
            raise ValueError(
                "grounding.placement.candidate_offset_fraction must be in [0, 1]."
            )
        if float(placement["support_margin"]) < 0.0:
            raise ValueError("grounding.placement.support_margin must be non-negative.")
        _require_keys(self.grasp, _GRASP_KEYS, "grasp")
        _require_keys(
            self.motion_defaults,
            _MOTION_DEFAULT_ACTIONS,
            "motion_defaults",
        )
        if not all(
            isinstance(policy, Mapping) and policy
            for policy in self.motion_defaults.values()
        ):
            raise ValueError("Every motion default must be a non-empty mapping.")
        _validate_motion_modifiers(self.motion_modifiers)
        _require_keys(
            self.predicate_fallbacks,
            _PREDICATE_KEYS,
            "predicate_fallbacks",
        )
        minimum_span = float(self.grasp.get("min_contact_span", -1.0))
        if minimum_span < 0.0:
            raise ValueError("grasp.min_contact_span must be non-negative.")
        maximum_span = self.grasp.get("max_contact_span")
        if maximum_span is not None and float(maximum_span) <= minimum_span:
            raise ValueError(
                "grasp.max_contact_span must exceed min_contact_span."
            )
        for name in ("filter_support_collision", "force_grasp_reannotate"):
            if not isinstance(self.grasp.get(name), bool):
                raise ValueError(f"grasp.{name} must be a boolean.")
        direction_count = self.grasp.get("n_deviated_approach_directions")
        if (
            isinstance(direction_count, bool)
            or not isinstance(direction_count, int)
            or not 1 <= direction_count <= 16
        ):
            raise ValueError("grasp.n_deviated_approach_directions must be in [1, 16].")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuntimePolicyCfg:
        """Parse one fully resolved policy snapshot."""
        fields = {
            "schema_version",
            "end_effector_profile",
            "execution",
            "planner",
            "arm_selection",
            "grounding",
            "grasp",
            "motion_defaults",
            "motion_modifiers",
            "predicate_fallbacks",
        }
        if set(value) != fields:
            raise ValueError("Runtime policy fields do not match the policy schema.")
        if value.get("schema_version") != RUNTIME_POLICY_SCHEMA:
            raise ValueError("Runtime policy has an unexpected schema_version.")
        arm_selection = value.get("arm_selection")
        if not isinstance(arm_selection, Mapping):
            raise ValueError("Runtime policy requires an arm_selection mapping.")
        sections = {
            name: value.get(name)
            for name in fields
            if name not in {
                "schema_version",
                "arm_selection",
                "end_effector_profile",
            }
        }
        if not all(isinstance(section, Mapping) for section in sections.values()):
            raise ValueError("Runtime policy sections must be mappings.")
        resolved_sections = {
            name: deepcopy(dict(section)) for name, section in sections.items()
        }
        predicate_fallbacks = resolved_sections["predicate_fallbacks"]
        for key in _DEPRECATED_PREDICATE_KEYS:
            predicate_fallbacks.pop(key, None)
        return cls(
            schema_version=RUNTIME_POLICY_SCHEMA,
            end_effector_profile=ParallelJawEefProfile.from_mapping(
                value.get("end_effector_profile", {})
            ),
            arm_selection=ArmSelectionPolicyCfg.from_mapping(arm_selection),
            **resolved_sections,
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return the canonical artifact snapshot."""
        return {
            "schema_version": self.schema_version,
            "end_effector_profile": self.end_effector_profile.as_mapping(),
            "execution": deepcopy(self.execution),
            "planner": deepcopy(self.planner),
            "arm_selection": self.arm_selection.as_mapping(),
            "grounding": deepcopy(self.grounding),
            "grasp": deepcopy(self.grasp),
            "motion_defaults": deepcopy(self.motion_defaults),
            "motion_modifiers": deepcopy(self.motion_modifiers),
            "predicate_fallbacks": deepcopy(self.predicate_fallbacks),
        }


def default_runtime_policy(robot_profile: str) -> RuntimePolicyCfg:
    """Resolve a package policy for one canonical robot profile."""
    document = _load_defaults()
    runtime = document.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {"common", "profiles"}:
        raise ValueError("Runtime defaults require common and profiles mappings.")
    common, profiles = runtime["common"], runtime["profiles"]
    if not isinstance(common, Mapping) or not isinstance(profiles, Mapping):
        raise ValueError("Runtime common and profiles must be mappings.")
    override = profiles.get(str(robot_profile))
    if not isinstance(override, Mapping):
        raise ValueError(f"Unknown runtime robot profile {robot_profile!r}.")
    profile_override = deepcopy(dict(override))
    eef_profile_id = profile_override.pop("end_effector_profile_id", None)
    if not isinstance(eef_profile_id, str) or not eef_profile_id:
        raise ValueError(
            f"Runtime robot profile {robot_profile!r} requires an "
            "end_effector_profile_id."
        )
    resolved = _deep_merge(common, profile_override)
    return RuntimePolicyCfg.from_mapping(
        {
            "schema_version": RUNTIME_POLICY_SCHEMA,
            "end_effector_profile": get_parallel_jaw_eef_profile(
                eef_profile_id
            ).as_mapping(),
            **resolved,
        }
    )


def generation_defaults() -> dict[str, Any]:
    """Return a detached generation-policy mapping."""
    value = _load_defaults().get("generation")
    if not isinstance(value, Mapping):
        raise ValueError("Action Engine defaults require a generation mapping.")
    required = {
        "task",
        "environment",
        "scene",
        "physics",
        "randomization",
        "dataset",
    }
    if set(value) != required:
        raise ValueError("Generation defaults do not match the expected sections.")
    return deepcopy(dict(value))


def _load_defaults() -> dict[str, Any]:
    document = load_config(_DEFAULTS_PATH)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "generation",
        "runtime",
    }:
        raise ValueError("Action Engine defaults do not match the package schema.")
    if document.get("schema_version") != ACTION_ENGINE_DEFAULTS_SCHEMA:
        raise ValueError("Action Engine defaults have an unexpected schema_version.")
    return document


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        result[key] = (
            _deep_merge(current, value)
            if isinstance(current, Mapping) and isinstance(value, Mapping)
            else deepcopy(value)
        )
    return result


def _validate_finite_numbers(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_numbers(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite.")


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    path: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{path} fields do not match the defaults schema.")


def _validate_string_sequence(value: Any, path: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list of object UIDs.")
    normalized = [str(item) for item in value]
    if any(not item.strip() for item in normalized):
        raise ValueError(f"{path} entries must be non-empty strings.")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path} entries must be strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{path} must not contain duplicate object UIDs.")


def _validate_planner(value: Mapping[str, Any]) -> None:
    _require_keys(value, _PLANNER_KEYS, "planner")
    backend = value.get("backend")
    if backend not in {"curobo", "toppra"}:
        raise ValueError("planner.backend must be 'curobo' or 'toppra'.")
    for name in ("single_arm_strategy", "coordinated_strategy"):
        if value.get(name) not in {"motion_gen", "ik_interp"}:
            raise ValueError(f"planner.{name} must be 'motion_gen' or 'ik_interp'.")
    if value.get("fallback_strategy") != "ik_interp":
        raise ValueError("planner.fallback_strategy must be 'ik_interp'.")
    if value.get("coordinated_strategy") == "motion_gen" and backend == "curobo":
        raise ValueError(
            "planner.coordinated_strategy must be 'ik_interp' with cuRobo."
        )
    for name in ("allow_fallback", "dynamic_collision"):
        if not isinstance(value.get(name), bool):
            raise ValueError(f"planner.{name} must be a boolean.")
    if value.get("dynamic_collision") and backend != "curobo":
        raise ValueError("planner.dynamic_collision requires the cuRobo backend.")
    _validate_string_sequence(
        value.get("static_obstacle_uids"),
        "planner.static_obstacle_uids",
    )
    _validate_string_sequence(
        value.get("dynamic_obstacle_uids"),
        "planner.dynamic_obstacle_uids",
    )
    overlap = set(value["static_obstacle_uids"]) & set(value["dynamic_obstacle_uids"])
    if overlap:
        raise ValueError(
            "Planner obstacle UIDs cannot be both static and dynamic: "
            f"{sorted(overlap)}."
        )

    curobo = value.get("curobo")
    if not isinstance(curobo, Mapping):
        raise ValueError("planner.curobo must be a mapping.")
    _require_keys(curobo, _CUROBO_KEYS, "planner.curobo")
    if curobo.get("log_level") not in {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
    }:
        raise ValueError("planner.curobo.log_level is unsupported.")
    if curobo.get("obstacle_representation") not in {"sphere", "cuboid", "mesh"}:
        raise ValueError(
            "planner.curobo.obstacle_representation must be sphere, cuboid, or mesh."
        )
    for name in ("multi_env", "use_cuda_graph", "preserve_plan_samples"):
        if not isinstance(curobo.get(name), bool):
            raise ValueError(f"planner.curobo.{name} must be a boolean.")
    max_attempts = curobo.get("max_attempts")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise ValueError("planner.curobo.max_attempts must be positive.")
    activation_distance = curobo.get("collision_activation_distance")
    if (
        isinstance(activation_distance, bool)
        or not isinstance(activation_distance, (int, float))
        or float(activation_distance) < 0.0
    ):
        raise ValueError(
            "planner.curobo.collision_activation_distance must be non-negative."
        )


def _validate_motion_modifiers(value: Mapping[str, Any]) -> None:
    _require_keys(value, set(MOTION_MODIFIER_MODES), "motion_modifiers")
    for modifier_type, modes in MOTION_MODIFIER_MODES.items():
        configured_modes = value.get(modifier_type)
        if not isinstance(configured_modes, Mapping):
            raise ValueError(f"motion_modifiers.{modifier_type} must be a mapping.")
        _require_keys(
            configured_modes,
            set(modes),
            f"motion_modifiers.{modifier_type}",
        )
        for mode, patches in configured_modes.items():
            path = f"motion_modifiers.{modifier_type}.{mode}"
            if not isinstance(patches, Mapping) or not patches:
                raise ValueError(f"{path} must contain action-specific patches.")
            unknown_actions = set(patches) - _MOTION_DEFAULT_ACTIONS
            if unknown_actions:
                raise ValueError(
                    f"{path} references unknown actions: {sorted(unknown_actions)}."
                )
            if not all(
                isinstance(patch, Mapping) and patch for patch in patches.values()
            ):
                raise ValueError(f"Every {path} action patch must be non-empty.")


def runtime_policy_hash(policy: RuntimePolicyCfg | Mapping[str, Any]) -> str:
    """Hash the canonical effective policy independently of the Seed graph."""
    resolved = (
        policy
        if isinstance(policy, RuntimePolicyCfg)
        else RuntimePolicyCfg.from_mapping(policy)
    )
    return _mapping_hash(resolved.as_mapping())


def resolve_agent_runtime_policy(agent_config: Mapping[str, Any]) -> RuntimePolicyCfg:
    """Resolve a generated snapshot or fall back for a legacy v1 artifact."""
    snapshot = agent_config.get("runtime_policy")
    expected_hash = agent_config.get("runtime_policy_hash")
    bound_eef_profile_id = agent_config.get("end_effector_profile_id")
    if bound_eef_profile_id is not None and (
        not isinstance(bound_eef_profile_id, str) or not bound_eef_profile_id.strip()
    ):
        raise ValueError("end_effector_profile_id must be a non-empty string.")
    if snapshot is None:
        if expected_hash is not None:
            raise ValueError("runtime_policy_hash requires a runtime_policy snapshot.")
        policy = default_runtime_policy(
            str(agent_config.get("robot_profile", "dual_ur10"))
        )
        _validate_eef_binding(policy, bound_eef_profile_id)
        return policy
    if not isinstance(snapshot, Mapping):
        raise ValueError("agent_config.runtime_policy must be a mapping.")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "agent_config.runtime_policy requires a non-empty runtime_policy_hash."
        )
    if _mapping_hash(snapshot) != expected_hash:
        raise ValueError(
            "agent_config runtime policy hash does not match its snapshot."
        )
    snapshot_eef = snapshot.get("end_effector_profile")
    if bound_eef_profile_id is not None and isinstance(snapshot_eef, Mapping):
        snapshot_profile_id = snapshot_eef.get("profile_id")
        if snapshot_profile_id != bound_eef_profile_id:
            raise ValueError(
                "agent_config end-effector binding does not match its runtime "
                "policy snapshot."
            )
    if snapshot.get("schema_version") == _LEGACY_RUNTIME_POLICY_SCHEMA:
        if set(snapshot) != {"schema_version", "arm_selection"} or not isinstance(
            snapshot.get("arm_selection"), Mapping
        ):
            raise ValueError("Legacy runtime policy snapshot is malformed.")
        policy = default_runtime_policy(
            str(agent_config.get("robot_profile", "dual_ur10"))
        )
        merged = policy.arm_selection.as_mapping()
        merged.update(
            {key: float(value) for key, value in snapshot["arm_selection"].items()}
        )
        policy.arm_selection = ArmSelectionPolicyCfg.from_mapping(merged)
        return policy
    if snapshot.get("schema_version") == _PRE_EEF_RUNTIME_POLICY_SCHEMA:
        defaults = default_runtime_policy(
            str(agent_config.get("robot_profile", "dual_ur10"))
        )
        return _migrate_pre_eef_policy(snapshot, defaults)
    if snapshot.get("schema_version") == _PREVIOUS_RUNTIME_POLICY_SCHEMA:
        defaults = default_runtime_policy(
            str(agent_config.get("robot_profile", "dual_ur10"))
        )
        migrated = deepcopy(dict(snapshot))
        migrated["schema_version"] = RUNTIME_POLICY_SCHEMA
        migrated_execution = deepcopy(dict(migrated.get("execution", {})))
        for key in (
            "support_stability_samples",
            "support_stability_interval_steps",
            "support_linear_velocity_tolerance",
            "support_angular_velocity_tolerance",
        ):
            migrated_execution[key] = defaults.execution[key]
        migrated["execution"] = migrated_execution
        migrated_grounding = deepcopy(dict(migrated.get("grounding", {})))
        migrated_placement = deepcopy(dict(migrated_grounding.get("placement", {})))
        for key, value in defaults.grounding["placement"].items():
            migrated_placement.setdefault(key, value)
        migrated_grounding["placement"] = migrated_placement
        migrated["grounding"] = migrated_grounding
        migrated_predicates = deepcopy(dict(migrated.get("predicate_fallbacks", {})))
        for key in (
            "support_com_margin",
            "support_max_vertical_gap",
            "support_max_penetration",
            "support_min_overlap_ratio",
        ):
            migrated_predicates[key] = defaults.predicate_fallbacks[key]
        migrated["predicate_fallbacks"] = migrated_predicates
        return _migrate_pre_eef_policy(migrated, defaults)
    if snapshot.get("schema_version") == _PRE_GRASP_RUNTIME_POLICY_SCHEMA:
        defaults = default_runtime_policy(
            str(agent_config.get("robot_profile", "dual_ur10"))
        )
        migrated = deepcopy(dict(snapshot))
        migrated["schema_version"] = RUNTIME_POLICY_SCHEMA
        migrated_grasp = deepcopy(dict(migrated.get("grasp", {})))
        migrated_grasp["n_deviated_approach_directions"] = defaults.grasp[
            "n_deviated_approach_directions"
        ]
        migrated["grasp"] = migrated_grasp
        migrated_execution = deepcopy(dict(migrated.get("execution", {})))
        for key in (
            "support_stability_samples",
            "support_stability_interval_steps",
            "support_linear_velocity_tolerance",
            "support_angular_velocity_tolerance",
        ):
            migrated_execution[key] = defaults.execution[key]
        migrated["execution"] = migrated_execution
        migrated_grounding = deepcopy(dict(migrated.get("grounding", {})))
        migrated_placement = deepcopy(dict(migrated_grounding.get("placement", {})))
        for key, value in defaults.grounding["placement"].items():
            migrated_placement.setdefault(key, value)
        migrated_grounding["placement"] = migrated_placement
        migrated["grounding"] = migrated_grounding
        migrated_predicates = deepcopy(dict(migrated.get("predicate_fallbacks", {})))
        for key in (
            "support_com_margin",
            "support_max_vertical_gap",
            "support_max_penetration",
            "support_min_overlap_ratio",
        ):
            migrated_predicates[key] = defaults.predicate_fallbacks[key]
        migrated["predicate_fallbacks"] = migrated_predicates
        return _migrate_pre_eef_policy(migrated, defaults)
    if snapshot.get("schema_version") == _PRE_PLANNER_RUNTIME_POLICY_SCHEMA:
        expected_fields = {
            "schema_version",
            "execution",
            "arm_selection",
            "grounding",
            "grasp",
            "motion_defaults",
            "motion_modifiers",
            "predicate_fallbacks",
        }
        if set(snapshot) not in (
            expected_fields,
            expected_fields | {"end_effector_profile"},
        ):
            raise ValueError("Previous runtime policy snapshot is malformed.")
        defaults = default_runtime_policy(
            str(agent_config.get("robot_profile", "dual_ur10"))
        )
        migrated = deepcopy(dict(snapshot))
        migrated["schema_version"] = RUNTIME_POLICY_SCHEMA
        migrated["planner"] = deepcopy(defaults.planner)
        migrated_execution = deepcopy(dict(migrated.get("execution", {})))
        for key in (
            "support_stability_samples",
            "support_stability_interval_steps",
            "support_linear_velocity_tolerance",
            "support_angular_velocity_tolerance",
        ):
            migrated_execution[key] = defaults.execution[key]
        migrated["execution"] = migrated_execution
        migrated_grounding = deepcopy(dict(migrated.get("grounding", {})))
        migrated_placement = deepcopy(dict(migrated_grounding.get("placement", {})))
        for key, value in defaults.grounding["placement"].items():
            migrated_placement.setdefault(key, value)
        migrated_grounding["placement"] = migrated_placement
        migrated["grounding"] = migrated_grounding
        migrated_grasp = deepcopy(dict(migrated["grasp"]))
        migrated_grasp["n_deviated_approach_directions"] = defaults.grasp[
            "n_deviated_approach_directions"
        ]
        migrated["grasp"] = migrated_grasp
        migrated_predicates = deepcopy(dict(migrated["predicate_fallbacks"]))
        for key in (
            "support_com_margin",
            "support_max_vertical_gap",
            "support_max_penetration",
            "support_min_overlap_ratio",
        ):
            migrated_predicates[key] = defaults.predicate_fallbacks[key]
        migrated["predicate_fallbacks"] = migrated_predicates
        return _migrate_pre_eef_policy(migrated, defaults)
    policy = RuntimePolicyCfg.from_mapping(snapshot)
    return policy


def _migrate_pre_eef_policy(
    snapshot: Mapping[str, Any],
    defaults: RuntimePolicyCfg,
) -> RuntimePolicyCfg:
    """Upgrade v3-v6 grasp fields into separated EEF and sampling policy."""
    migrated = deepcopy(dict(snapshot))
    legacy_grasp = deepcopy(dict(migrated.get("grasp", {})))
    grasp = deepcopy(defaults.grasp)
    field_map = {
        "antipodal_n_sample": "antipodal_n_sample",
        "antipodal_max_angle": "antipodal_max_angle",
        "max_deviation_angle": "max_deviation_angle",
        "n_deviated_approach_directions": "n_deviated_approach_directions",
        "viser_port": "viser_port",
        "max_decomposition_hulls": "max_decomposition_hulls",
        "force_grasp_reannotate": "force_grasp_reannotate",
    }
    for old_name, new_name in field_map.items():
        if old_name in legacy_grasp:
            grasp[new_name] = deepcopy(legacy_grasp[old_name])
    if "min_open_length" in legacy_grasp:
        grasp["min_contact_span"] = float(legacy_grasp["min_open_length"])
    if "max_open_length" in legacy_grasp:
        grasp["max_contact_span"] = float(legacy_grasp["max_open_length"])

    eef_profile = defaults.end_effector_profile.as_mapping()
    if "max_open_length" in legacy_grasp:
        eef_profile["jaw_opening_max"] = float(legacy_grasp["max_open_length"])
    collision = eef_profile["collision_proxy"]
    if "finger_length" in legacy_grasp:
        collision["finger_length"] = float(legacy_grasp["finger_length"])
    if "point_sample_dense" in legacy_grasp:
        collision["point_sample_dense"] = float(
            legacy_grasp["point_sample_dense"]
        )

    migrated["schema_version"] = RUNTIME_POLICY_SCHEMA
    migrated["end_effector_profile"] = eef_profile
    migrated["grasp"] = grasp
    return RuntimePolicyCfg.from_mapping(migrated)


def _validate_eef_binding(
    policy: RuntimePolicyCfg,
    bound_profile_id: Any,
) -> None:
    if (
        bound_profile_id is not None
        and policy.end_effector_profile.profile_id != bound_profile_id
    ):
        raise ValueError(
            "agent_config end-effector binding does not match the resolved "
            "runtime policy."
        )


def _mapping_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
