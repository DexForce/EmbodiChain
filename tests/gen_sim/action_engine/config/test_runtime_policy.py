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

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from embodichain.gen_sim.action_engine.config import (
    ArmSelectionPolicyCfg,
    RuntimePolicyCfg,
    default_runtime_policy,
    generation_defaults,
    resolve_agent_runtime_policy,
    runtime_policy_hash,
)
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceOptions


def test_default_runtime_policy_preserves_current_arm_selection_behavior() -> None:
    policy = default_runtime_policy("dual_ur10")

    assert policy.arm_selection.as_mapping() == {
        "crossing_deadband_ratio": 0.08,
        "allow_cross_side_fallback": False,
        "pickup_crossing_weight": 1.0,
        "placement_crossing_weight": 1.5,
        "motion_cost_scale": pytest.approx(3.141592653589793),
        "fallback_workspace_half_width": 0.5,
        "orient_object_preferred_arm_deadband": 0.02,
    }


def test_defaults_cover_current_execution_and_generation_policy() -> None:
    runtime = default_runtime_policy("dual_ur10")
    generation = generation_defaults()

    assert runtime.execution == {
        "max_transitions": 1000,
        "semantic_step_settle_steps": 10,
        "max_retries_per_action": 2,
        "max_graph_revisions": 8,
        "max_recovery_actions": 12,
        "support_stability_samples": 3,
        "support_stability_interval_steps": 5,
        "support_linear_velocity_tolerance": pytest.approx(0.02),
        "support_angular_velocity_tolerance": pytest.approx(0.2),
    }
    assert runtime.planner == {
        "backend": "curobo",
        "single_arm_strategy": "motion_gen",
        "coordinated_strategy": "ik_interp",
        "fallback_strategy": "ik_interp",
        "allow_fallback": True,
        "dynamic_collision": False,
        "static_obstacle_uids": [],
        "dynamic_obstacle_uids": [],
        "curobo": {
            "log_level": "error",
            "obstacle_representation": "cuboid",
            "multi_env": False,
            "use_cuda_graph": True,
            "preserve_plan_samples": False,
            "max_attempts": 5,
            "collision_activation_distance": pytest.approx(0.01),
        },
    }
    assert runtime.grounding["arrangement"]["row_search_radius"] == 0.25
    assert runtime.grasp["antipodal_n_sample"] == 10000
    assert "max_open_length" not in runtime.grasp
    assert "min_open_length" not in runtime.grasp
    assert "finger_length" not in runtime.grasp
    assert runtime.motion_modifiers["orientation"]["upright"]["MoveHeldObject"][
        "surface_clearance"
    ] == pytest.approx(0.05)
    assert (
        "upright_yaw_samples"
        not in runtime.motion_modifiers["orientation"]["upright"]["PickUp"]
    )
    assert (
        "upright_yaw_samples"
        not in runtime.motion_modifiers["orientation"]["upright"]["MoveHeldObject"]
    )
    assert runtime.motion_modifiers["handover_role"]["transfer"]["PickUp"] == {
        "sample_interval": 80,
        "hand_interp_steps": 5,
        "pick_object_part": "top",
    }
    assert runtime.motion_defaults["HandOver"]["receive_pick_object_part"] == "bottom"
    assert runtime.motion_defaults["CoordinatedPickment"][
        "middle_empty_ratio"
    ] == pytest.approx(0.4)
    assert (
        runtime.motion_defaults["CoordinatedPickment"]["is_filter_ground_collision"]
        is False
    )
    assert (
        runtime.motion_defaults["CoordinatedPickment"]["release_sample_interval"] == 60
    )
    assert runtime.motion_defaults["CoordinatedPickment"][
        "release_gripper_tolerance"
    ] == pytest.approx(0.08)
    assert runtime.predicate_fallbacks["upright_max_tilt"] == pytest.approx(
        0.2617993877991494
    )
    assert generation["physics"]["rigid_object"]["mass"] == pytest.approx(0.1)
    assert generation["task"]["default_gripper_model"] == "pgi"
    assert generation["environment"]["arm_aim_yaw_offset"] == {
        "left": pytest.approx(0.0),
        "right": pytest.approx(0.0),
    }
    assert generation["environment"]["recording"] == {
        "enabled": True,
        "resolution": [640, 360],
        "interval_step": 1,
    }
    assert generation["scene"]["object_length_sample_points"] == 5000
    assert generation["dataset"]["control_frequency"] == 25
    assert generation["randomization"]["table_height_delta_range"] == [
        [-0.05],
        [0.05],
    ]


def test_e1_motion_defaults_match_atomic_action_tutorial_cadence() -> None:
    policy = default_runtime_policy("dual_franka")
    motion = policy.motion_defaults

    assert motion["PickUp"] == {
        "pre_grasp_distance": pytest.approx(0.15),
        "lift_height": pytest.approx(0.16),
        "sample_interval": 120,
        "hand_interp_steps": 12,
    }
    assert motion["MoveHeldObject"]["sample_interval"] == 120
    assert motion["Place"] == {
        "sample_interval": 120,
        "lift_height": pytest.approx(0.14),
        "post_hold_steps": 60,
        "cartesian_waypoint_count": 2,
        "hand_interp_steps": 12,
    }
    assert policy.motion_modifiers["orientation"]["upright"]["Place"] == {
        "sample_interval": 120,
        "post_hold_steps": 60,
        "hand_interp_steps": 12,
    }


def test_axis_align_defaults_match_atomic_action_tutorial_cadence() -> None:
    axis_align = default_runtime_policy("dual_franka").motion_defaults["AxisAlign"]

    assert axis_align == {
        "sample_interval": 180,
        "pre_grasp_distance": pytest.approx(0.15),
        "lift_height": pytest.approx(0.16),
        "lower_distance": pytest.approx(0.03),
        "hand_interp_steps": 12,
    }


def test_place_defaults_fit_the_mainline_motion_sample_budget() -> None:
    place = default_runtime_policy("dual_ur10").motion_defaults["Place"]
    sample_count = int(place["sample_interval"])
    hand_steps = PlaceOptions().hand_interp_steps
    motion_steps = sample_count - hand_steps
    down_steps = int(round(motion_steps) * 0.6)
    back_steps = motion_steps - down_steps
    cartesian_count = int(place["cartesian_waypoint_count"])

    assert 1 + 2 * cartesian_count <= down_steps
    assert 1 + cartesian_count <= back_steps


def test_default_runtime_policy_returns_detached_profile_snapshots() -> None:
    first = default_runtime_policy("dual_ur10")
    second = default_runtime_policy("dual_ur10")
    franka = default_runtime_policy("dual_franka")

    first.arm_selection.pickup_crossing_weight = 9.0
    first.motion_defaults["PickUp"]["lift_height"] = 9.0

    assert second.arm_selection.pickup_crossing_weight == 1.0
    assert second.motion_defaults["PickUp"]["lift_height"] == 0.16
    assert franka.arm_selection.pickup_crossing_weight == 1.0
    assert franka.motion_defaults["MoveEndEffector"]["retreat_height"] == 0.10


def test_generation_defaults_return_detached_values() -> None:
    first = generation_defaults()
    second = generation_defaults()

    first["physics"]["rigid_object"]["mass"] = 9.0

    assert second["physics"]["rigid_object"]["mass"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("crossing_deadband_ratio", 1.0, "crossing_deadband_ratio"),
        ("pickup_crossing_weight", -0.1, "pickup_crossing_weight"),
        ("placement_crossing_weight", -0.1, "placement_crossing_weight"),
        ("motion_cost_scale", 0.0, "motion_cost_scale"),
        ("fallback_workspace_half_width", 0.0, "fallback_workspace_half_width"),
    ],
)
def test_arm_selection_policy_rejects_invalid_values(
    field: str,
    value: float,
    message: str,
) -> None:
    values = default_runtime_policy("dual_ur10").arm_selection.as_mapping()
    values[field] = value

    with pytest.raises(ValueError, match=message):
        ArmSelectionPolicyCfg.from_mapping(values)


def test_arm_selection_policy_requires_boolean_cross_side_fallback() -> None:
    values = default_runtime_policy("dual_ur10").arm_selection.as_mapping()
    values["allow_cross_side_fallback"] = "false"

    with pytest.raises(TypeError, match="allow_cross_side_fallback"):
        ArmSelectionPolicyCfg.from_mapping(values)


def test_arm_selection_policy_loads_old_snapshot_without_fallback_field() -> None:
    snapshot = default_runtime_policy("dual_ur10").as_mapping()
    snapshot["arm_selection"].pop("allow_cross_side_fallback")
    payload = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    policy = resolve_agent_runtime_policy(
        {
            "robot_profile": "dual_ur10",
            "runtime_policy": snapshot,
            "runtime_policy_hash": hashlib.sha256(payload).hexdigest(),
        }
    )

    assert policy.arm_selection.allow_cross_side_fallback is False


def test_agent_policy_snapshot_is_hash_verified_and_legacy_config_falls_back() -> None:
    policy = default_runtime_policy("dual_ur5")
    snapshot = policy.as_mapping()
    config = {
        "robot_profile": "dual_ur5",
        "runtime_policy": snapshot,
        "runtime_policy_hash": runtime_policy_hash(policy),
    }

    resolved = resolve_agent_runtime_policy(config)
    assert resolved.as_mapping() == snapshot

    tampered = deepcopy(config)
    tampered["runtime_policy"]["motion_defaults"]["PickUp"]["lift_height"] = 8.0
    with pytest.raises(ValueError, match="hash does not match"):
        resolve_agent_runtime_policy(tampered)

    legacy = resolve_agent_runtime_policy({"robot_profile": "dual_ur5"})
    assert legacy.as_mapping() == snapshot


def test_v6_policy_snapshot_adds_axis_align_defaults_without_rewriting_e1() -> None:
    snapshot = default_runtime_policy("dual_franka").as_mapping()
    snapshot["schema_version"] = "action_engine_runtime_policy_v6"
    snapshot["motion_defaults"].pop("AxisAlign")
    snapshot["motion_defaults"]["PickUp"]["lift_height"] = 0.11
    payload = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    resolved = resolve_agent_runtime_policy(
        {
            "robot_profile": "dual_franka",
            "runtime_policy": snapshot,
            "runtime_policy_hash": hashlib.sha256(payload).hexdigest(),
        }
    )

    assert resolved.schema_version == "action_engine_runtime_policy_v8"
    assert resolved.motion_defaults["AxisAlign"]["sample_interval"] == 180
    assert resolved.motion_defaults["PickUp"]["lift_height"] == pytest.approx(0.11)


def test_v7_policy_snapshot_drops_legacy_gripper_geometry() -> None:
    snapshot = default_runtime_policy("dual_ur10").as_mapping()
    snapshot["schema_version"] = "action_engine_runtime_policy_v7"
    snapshot["grasp"].update(
        {
            "min_open_length": 0.01,
            "max_open_length": 0.15,
            "finger_length": 0.13,
        }
    )
    payload = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    resolved = resolve_agent_runtime_policy(
        {
            "robot_profile": "dual_ur10",
            "runtime_policy": snapshot,
            "runtime_policy_hash": hashlib.sha256(payload).hexdigest(),
        }
    )

    assert resolved.schema_version == "action_engine_runtime_policy_v8"
    assert "min_open_length" not in resolved.grasp
    assert "max_open_length" not in resolved.grasp
    assert "finger_length" not in resolved.grasp


def test_narrow_v1_policy_snapshot_is_migrated_to_complete_runtime_policy() -> None:
    snapshot = {
        "schema_version": "action_engine_runtime_policy_v1",
        "arm_selection": {
            "crossing_deadband_ratio": 0.08,
            "pickup_crossing_weight": 2.0,
            "placement_crossing_weight": 1.5,
            "motion_cost_scale": 3.141592653589793,
            "fallback_workspace_half_width": 0.5,
        },
    }
    payload = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    resolved = resolve_agent_runtime_policy(
        {
            "robot_profile": "dual_ur10",
            "runtime_policy": snapshot,
            "runtime_policy_hash": hashlib.sha256(payload).hexdigest(),
        }
    )

    assert resolved.arm_selection.pickup_crossing_weight == 2.0
    assert resolved.motion_defaults["PickUp"]["lift_height"] == 0.16


def test_v3_policy_snapshot_is_migrated_with_default_planner_policy() -> None:
    expected = default_runtime_policy("dual_ur10")
    snapshot = expected.as_mapping()
    snapshot.pop("planner")
    snapshot["schema_version"] = "action_engine_runtime_policy_v3"
    payload = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    resolved = resolve_agent_runtime_policy(
        {
            "robot_profile": "dual_ur10",
            "runtime_policy": snapshot,
            "runtime_policy_hash": hashlib.sha256(payload).hexdigest(),
        }
    )

    assert resolved.schema_version == "action_engine_runtime_policy_v8"
    assert resolved.planner == expected.planner


def test_curobo_policy_rejects_coordinated_motion_generation() -> None:
    snapshot = default_runtime_policy("dual_ur10").as_mapping()
    snapshot["planner"]["coordinated_strategy"] = "motion_gen"

    with pytest.raises(ValueError, match="coordinated_strategy"):
        RuntimePolicyCfg.from_mapping(snapshot)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"fallback_strategy": "motion_gen"}, "fallback_strategy"),
        ({"backend": "toppra", "dynamic_collision": True}, "dynamic_collision"),
    ],
)
def test_planner_policy_rejects_unsupported_combinations(
    patch: dict[str, object],
    message: str,
) -> None:
    snapshot = default_runtime_policy("dual_ur10").as_mapping()
    snapshot["planner"].update(patch)

    with pytest.raises(ValueError, match=message):
        RuntimePolicyCfg.from_mapping(snapshot)
