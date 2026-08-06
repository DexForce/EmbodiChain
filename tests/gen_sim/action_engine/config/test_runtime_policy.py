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
    default_runtime_policy,
    generation_defaults,
    resolve_agent_runtime_policy,
    runtime_policy_hash,
)


def test_default_runtime_policy_preserves_current_arm_selection_behavior() -> None:
    policy = default_runtime_policy("dual_ur10")

    assert policy.arm_selection.as_mapping() == {
        "crossing_deadband_ratio": 0.08,
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
    }
    assert runtime.grounding["arrangement"]["row_search_radius"] == 0.25
    assert runtime.grasp["antipodal_n_sample"] == 10000
    assert runtime.motion_policies["upright_in_place_transport"][
        "surface_clearance"
    ] == pytest.approx(0.05)
    assert runtime.predicate_fallbacks["upright_max_tilt"] == pytest.approx(
        0.2617993877991494
    )
    assert generation["physics"]["rigid_object"]["mass"] == pytest.approx(0.1)
    assert generation["environment"]["arm_aim_yaw_offset"] == {
        "left": pytest.approx(0.0),
        "right": pytest.approx(0.0),
    }
    assert generation["scene"]["object_length_sample_points"] == 5000
    assert generation["dataset"]["control_frequency"] == 25
    assert generation["randomization"]["table_height_delta_range"] == [
        [-0.05],
        [0.05],
    ]


def test_default_runtime_policy_returns_detached_profile_snapshots() -> None:
    first = default_runtime_policy("dual_ur10")
    second = default_runtime_policy("dual_ur10")
    franka = default_runtime_policy("dual_franka")

    first.arm_selection.pickup_crossing_weight = 9.0
    first.motion_policies["default_pickup"]["lift_height"] = 9.0

    assert second.arm_selection.pickup_crossing_weight == 1.0
    assert second.motion_policies["default_pickup"]["lift_height"] == 0.30
    assert franka.arm_selection.pickup_crossing_weight == 1.0


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
    tampered["runtime_policy"]["motion_policies"]["default_pickup"]["lift_height"] = 8.0
    with pytest.raises(ValueError, match="hash does not match"):
        resolve_agent_runtime_policy(tampered)

    legacy = resolve_agent_runtime_policy({"robot_profile": "dual_ur5"})
    assert legacy.as_mapping() == snapshot


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
    assert resolved.motion_policies["default_pickup"]["lift_height"] == 0.30
