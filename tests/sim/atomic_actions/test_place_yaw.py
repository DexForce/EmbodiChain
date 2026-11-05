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

"""Source parity of bounded yaw search with real floor/footprint geometry."""

from __future__ import annotations
from dataclasses import replace
import hashlib, json, random
from pathlib import Path
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions.primitives import _place_yaw
from embodichain.lab.sim.motion.search import search_cost_scope
from ._place_yaw_fixture import yaw_fixture


def digest(value):
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("mode", ["accepted", "late", "rejected", "preview", "nominal"])
@pytest.mark.parametrize("lazy", [False, True])
def test_yaw_search_matches_source_geometry_candidate_order_and_rng(
    tmp_path, monkeypatch, mode, lazy
):
    action, request, context, first, build, inspect, calls, _, _ = yaw_fixture(
        tmp_path, mode
    )
    acquired = []
    if lazy:
        original = action.planning_services.place_scene_provider

        class LazyProvider:
            def prepare(self, *args):
                raise AssertionError(
                    "Recovery must use its late native acquisition port."
                )

            def prepare_inside_landing(self, *args):
                acquired.append(True)
                return original.prepare(*args)

        action.planning_services.place_scene_provider = LazyProvider()
    monkeypatch.setattr(_place_yaw, "inspect_emitted", inspect)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    with search_cost_scope() as cost:
        result = _place_yaw.recover_inside_yaw(action, request, context, first, build)
    if lazy:
        assert len(acquired) == (0 if mode in {"preview", "nominal"} else 1)
    actual = {
        "success": result.plan_success.tolist(),
        "endpoint": None if result.endpoint is None else result.endpoint.tolist(),
        "metadata": result.diagnostics.metadata,
        "builds": calls,
        "counters": dict(cost.counters),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    expected = json.loads(
        (Path(__file__).parent / "data/place_yaw_reference.json").read_text()
    )[mode]
    assert actual == expected


def test_yaw_restores_rng_on_build_error(tmp_path, monkeypatch):
    action, request, context, first, build, inspect, *_ = yaw_fixture(
        tmp_path, "exception"
    )
    monkeypatch.setattr(_place_yaw, "inspect_emitted", inspect)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(ValueError, match="controlled planner unavailable"):
        _place_yaw.recover_inside_yaw(action, request, context, first, build)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_preview_role_survives_projection_independently_of_state(tmp_path):
    _, _, observed, *_ = yaw_fixture(tmp_path, "accepted")
    original = replace(observed.robot, qvel=torch.ones_like(observed.robot.qvel))
    observed = replace(observed, robot=original)
    projected = observed.project(qpos=observed.robot.qpos + 0.01, task=observed.task)
    assert not projected.is_preview
    assert observed.observed_robot is None
    assert projected.observed_robot is original
    assert torch.equal(projected.robot.qvel, torch.zeros_like(original.qvel))
    assert torch.equal(projected.observed_robot.qvel, torch.ones_like(original.qvel))
    second = projected.project(qpos=projected.robot.qpos + 0.01, task=projected.task)
    assert second.observed_robot is original
    with pytest.raises(ValueError, match="Original observation"):
        replace(
            projected,
            observed_robot=replace(original, timestamp=original.timestamp + 0.01),
        )
    preview = replace(observed, is_preview=True)
    assert preview.project(qpos=preview.robot.qpos + 0.01, task=preview.task).is_preview
    with pytest.raises(TypeError, match="is_preview"):
        replace(observed, is_preview=1)


def test_standard_engine_applies_yaw_after_failed_keyframe_preflight(tmp_path):
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        ControlPartCommandProfile,
        PlacePlanningRecovery,
        RecoveryPolicy,
        PlaceOptions,
        MotionPolicy,
    )

    action, request, context, *_ = yaw_fixture(tmp_path, "rejected")
    action.robot.compute_ik.side_effect = lambda pose, joint_seed, **kw: (
        torch.tensor([False]),
        joint_seed.clone(),
    )
    engine = AtomicActionEngine(
        action.motion_generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        place_scene_provider=action.planning_services.place_scene_provider,
    )
    invocation = engine.make_invocation(
        "place",
        request.goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
        ),
        skill_options=PlaceOptions(
            linear_segments=True,
            qualify_required_waypoints=True,
            preserve_current_object_orientation=True,
            lift_height=0.05,
        ),
        recovery_policy=RecoveryPolicy(
            planning_recovery=PlacePlanningRecovery(inside_world_yaw=True)
        ),
    )
    with search_cost_scope() as cost:
        plan = engine.plan(invocation, context)
    assert not plan.plan_success.any()
    assert plan.diagnostics.failure.code == "place_keyframes_unqualified"
    attempts = plan.diagnostics.metadata["inside_world_yaw"]["attempts"]
    assert [row["angle_deg"] for row in attempts] == [
        45.0,
        -45.0,
        90.0,
        -90.0,
        135.0,
        -135.0,
        180.0,
    ]
    assert cost.counters["inside_place_yaw_candidate"] == 2
    assert not plan.diagnostics.metadata["inside_world_yaw"]["physical_evidence"]


def test_replaced_native_binding_rejects_instead_of_measuring_later_geometry(tmp_path):
    action, request, context, first, build, _, calls, sim, _ = yaw_fixture(
        tmp_path, "accepted"
    )
    sim.get_rigid_object = lambda uid: object()
    with pytest.raises(ValueError, match="native bindings"):
        _place_yaw.recover_inside_yaw(action, request, context, first, build)
    assert calls == []
