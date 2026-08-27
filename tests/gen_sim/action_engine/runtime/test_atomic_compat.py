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

from dataclasses import dataclass
from types import SimpleNamespace

from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.atomic_compat import (
    ActionEngineMoveJoints,
    ActionEngineMoveJointsOptions,
    ExactTargetMoveHeldObject,
    ExactTargetMoveHeldObjectOptions,
)
from embodichain.lab.sim.atomic_actions import (
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    StateDelta,
)


def test_grounded_target_transport_uses_mainline_exact_target_contract() -> None:
    action = ExactTargetMoveHeldObject()
    assert type(action).__dict__["binding_contract"] is MoveHeldObject.binding_contract
    assert action._plan.__func__ is MoveHeldObject._plan
    assert "_apply_automatic_transport_rotation" not in type(action).__dict__


def test_semantic_transport_config_has_no_task_facing_rotation_switch() -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    action = SimpleNamespace(cfg={})
    capability = SimpleNamespace(
        config_type=MoveHeldObjectOptions,
        target_materializer="semantic_held_object",
    )

    options = adapter._build_single_arm_config(action, capability)

    assert isinstance(options, ExactTargetMoveHeldObjectOptions)
    assert not hasattr(options, "allow_automatic_transport_rotation")


def test_joint_config_materializes_single_release_only_when_requested() -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    capability = SimpleNamespace(
        config_type=MoveJointsOptions,
        target_materializer="joint_state",
    )

    release = adapter._build_single_arm_config(
        SimpleNamespace(cfg={"single_release": True}),
        capability,
    )
    ordinary = adapter._build_single_arm_config(
        SimpleNamespace(cfg={}),
        capability,
    )

    assert isinstance(release, ActionEngineMoveJointsOptions)
    assert release.single_release
    assert isinstance(ordinary, ActionEngineMoveJointsOptions)
    assert not ordinary.single_release


def test_single_release_binds_hand_motion_to_the_arm_held_state_key() -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    adapter._parts = lambda _arm: ("physical_left_arm", "physical_left_hand", 2)
    captured = {}

    class Engine:
        def bind_control_parts(self, skill_id, endpoints, *, task_state_keys=None):
            captured.update(
                skill_id=skill_id,
                endpoints=endpoints,
                task_state_keys=task_state_keys,
            )
            return object()

    adapter._binding(
        SimpleNamespace(
            arm="left_arm",
            control="hand",
            cfg={"single_release": True},
        ),
        SimpleNamespace(
            action_type=MoveJoints,
            config_materializer="single_arm",
        ),
        engine=Engine(),
    )

    assert captured == {
        "skill_id": "move_joints",
        "endpoints": {"primary": {"motion": "physical_left_hand"}},
        "task_state_keys": {"primary": "physical_left_arm"},
    }


def test_single_release_plan_removes_only_the_bound_arm_attachment(monkeypatch) -> None:
    @dataclass(frozen=True)
    class Plan:
        expected_effects: object

    monkeypatch.setattr(
        MoveJoints,
        "_plan",
        lambda _self, _request, _context: Plan(expected_effects=object()),
    )
    action = ActionEngineMoveJoints()
    request = SimpleNamespace(
        binding=SimpleNamespace(
            endpoint=lambda _slot, _endpoint: SimpleNamespace(task_state_key="left_arm")
        ),
        skill_options=ActionEngineMoveJointsOptions(single_release=True),
    )
    context = SimpleNamespace(
        task=SimpleNamespace(
            get_held_object=lambda key: object() if key == "left_arm" else None
        )
    )

    plan = action._plan(request, context)

    assert isinstance(plan.expected_effects, StateDelta)
    assert dict(plan.expected_effects.held_object_updates) == {"left_arm": None}
