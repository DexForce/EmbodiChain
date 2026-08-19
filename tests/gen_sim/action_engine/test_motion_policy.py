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

import pytest

from embodichain.gen_sim.action_engine.domain import motion_policy
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)


def test_upright_policy_matches_mature_runtime_across_robot_profiles() -> None:
    upright = motion_policy(("orientation", "upright"))
    franka = resolve_motion_policy("dual_franka", "PickUp", upright)
    ur10 = resolve_motion_policy("dual_ur10", "PickUp", upright)

    assert franka["lift_height"] == pytest.approx(0.30)
    assert ur10["lift_height"] == pytest.approx(0.30)
    assert ur10["rotate_upright"] == pytest.approx(0.7853981633974483)


def test_policy_resolution_returns_detached_values() -> None:
    upright = motion_policy(("orientation", "upright"))
    first = resolve_motion_policy("ur10", "MoveHeldObject", upright)
    first["surface_clearance"] = 123.0

    second = resolve_motion_policy("dual_ur10", "MoveHeldObject", upright)

    assert second["surface_clearance"] == pytest.approx(0.05)


def test_unknown_action_base_is_rejected_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="Unknown Action Engine motion base"):
        resolve_motion_policy("dual_franka", "TypoAction", motion_policy())


def test_upright_and_handover_role_modifiers_compose_without_named_cross_product() -> (
    None
):
    resolved = resolve_motion_policy(
        "dual_franka",
        "PickUp",
        motion_policy(
            ("orientation", "upright"),
            ("handover_role", "transfer"),
        ),
    )

    assert resolved["rotate_upright"] == pytest.approx(0.7853981633974483)
    assert resolved["pick_object_part"] == "top"
    assert "approach_direction_mode" not in resolved
    assert resolved["sample_interval"] == 80


def test_named_policy_strings_require_graph_regeneration() -> None:
    with pytest.raises(ValueError, match="named string policies are no longer"):
        resolve_motion_policy("dual_franka", "PickUp", "legacy_flat_name")
