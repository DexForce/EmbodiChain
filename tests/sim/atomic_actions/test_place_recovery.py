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

"""Shared ordinary Place recovery against the source bounded control flow."""

from __future__ import annotations
from dataclasses import replace
import hashlib, json, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    PlaceOptions,
    PlacePlanningRecovery,
    PlaceSceneInputs,
    RecoveryPolicy,
)
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from embodichain.lab.sim.motion import _cartesian as cartesian
from embodichain.lab.sim.atomic_actions.primitives import _place_scene, _place_recovery
from ._place_recovery_fixture import recovery_fixture, early_geometry_oracle

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/place_recovery_reference.json").read_text()
)["cases"]


def inputs(tmp_path, mode, monkeypatch, *, provider=True):
    generator, context, goal, geometry, calls, ik, generate = recovery_fixture(
        tmp_path, mode
    )
    generator._generate_once = generate
    scene = PlaceSceneInputs(
        geometry=SceneMeshSnapshot(geometry), floor_entity_id="table"
    )
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        place_scene_provider=(
            SimpleNamespace(prepare=lambda request, context: scene)
            if provider
            else None
        ),
    )
    invocation = engine.make_invocation(
        "place",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            segment_sample_count_retries=(260, 320),
            segment_continuous_ik_recovery=True,
        ),
        skill_options=PlaceOptions(
            hand_interp_steps=12,
            release_settle_steps=0,
            lift_height=0.1,
            cartesian_waypoint_count=2,
            preserve_current_object_orientation=True,
            linear_segments=True,
            qualify_required_waypoints=True,
            monitor_scene_until_approach=True,
        ),
        recovery_policy=RecoveryPolicy(planning_recovery=PlacePlanningRecovery()),
        invocation_id="place",
    )
    robot_clear = lambda robot, positions, part: {
        "valid": True,
        "frames": len(positions),
        "scope": "discrete_robot_meshes",
    }
    monkeypatch.setattr(cartesian, "resolve_continuous_ik_path", lambda *a, **k: None)
    monkeypatch.setattr(_place_scene, "inspect_robot_motion", robot_clear)
    monkeypatch.setattr(_place_recovery, "inspect_robot_motion", robot_clear)
    if mode == "early":
        monkeypatch.setattr(
            _place_recovery, "inspect_place_plan", early_geometry_oracle()
        )
    return engine, invocation, context, calls, ik, scene


def source_metadata(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {
            key: source_metadata(item)
            for key, item in value.items()
            if key not in {"place_recovery_cost", "segment_sampling"}
        }
    if isinstance(value, (list, tuple)):
        return [source_metadata(item) for item in value]
    return value


@pytest.mark.parametrize("reference", _REFERENCE, ids=lambda item: item["mode"])
def test_place_waypoint_and_clearance_recovery_preserve_source_budget_rng_and_release(
    tmp_path, monkeypatch, reference
):
    engine, invocation, context, calls, ik, _ = inputs(
        tmp_path, reference["mode"], monkeypatch
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    actual = {
        "mode": reference["mode"],
        "calls": calls,
        "ik": ik,
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "metadata": source_metadata(plan.diagnostics.metadata),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    assert actual == reference
    cost = plan.diagnostics.metadata["place_recovery_cost"]
    assert (
        cost["complete_plans"]
        == {"nominal": 1, "waypoint": 2, "via": 10, "exhaust": 10, "early": 8}[
            reference["mode"]
        ]
    )
    assert cost["waypoint_batches"] <= 2 and cost["clearance_routes"] <= 2
    if plan.plan_success.any():
        assert dict(plan.expected_effects.held_object_updates) == {"arm": None}


def test_place_recovery_requires_scene_inputs_before_any_ik_or_motion(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, ik, _ = inputs(
        tmp_path, "waypoint", monkeypatch, provider=False
    )
    with pytest.raises(ValueError, match="scene input provider"):
        engine.plan(invocation, context)
    assert not calls and not ik


def test_place_velocity_failure_retains_pre_velocity_kinematic_evidence(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, ik, _ = inputs(tmp_path, "nominal", monkeypatch)
    generator = engine.motion_generator
    callback = generator._generate_once
    del generator._generate_once
    generator._generate_ik_interpolation = callback
    urdf = Path(generator.robot.cfg.fpath)
    urdf.write_text(urdf.read_text().replace('velocity="1"', 'velocity="0.01"'))
    invocation = replace(
        invocation,
        motion_policy=replace(invocation.motion_policy, mask_generation_overspeed=True),
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)

    plan = engine.plan(invocation, context)

    assert plan.plan_success.tolist() == [False]
    # Source records normalized generation success before the speed mask.
    # This fixture's IK succeeds even when every path exceeds the speed cap.
    assert plan.diagnostics.metadata["place_waypoint_probe"]["kinematic_valid"] is True
    assert plan.diagnostics.metadata["place_recovery_cost"] == {
        "complete_plans": 10,
        "waypoint_batches": 2,
        "clearance_routes": 2,
        "scope": "one_complete_action_planning_attempt",
    }
    assert len(calls) == 16 and len(ik) == 1


def test_place_scene_owns_arrays_and_uses_explicit_floor_identity(
    tmp_path, monkeypatch
):
    _, _, _, _, _, scene = inputs(tmp_path, "nominal", monkeypatch)
    raw = SceneMeshSnapshot(
        {"payload": scene.geometry["object"], "work_surface": scene.geometry["table"]}
    )
    value = PlaceSceneInputs(geometry=raw, floor_entity_id="work_surface")
    before = value.geometry["payload"].clone()
    raw["payload"].fill_(8)
    assert torch.equal(value.geometry["payload"], before)
    copied = value.snapshot()
    value.geometry["payload"].fill_(9)
    assert torch.equal(copied.geometry["payload"], before)


def test_unsupported_place_recovery_variant_is_rejected_statically():
    with pytest.raises(ValueError, match="Unsupported Place recovery variant"):
        PlacePlanningRecovery(variant="support")


def test_place_recovery_floor_role_uses_supplied_identity_not_name(
    tmp_path, monkeypatch
):
    from embodichain.lab.sim.atomic_actions import SceneSnapshot

    engine, invocation, context, calls, ik, scene = inputs(
        tmp_path, "waypoint", monkeypatch
    )
    renamed = PlaceSceneInputs(
        geometry=SceneMeshSnapshot(
            {
                "object": scene.geometry["object"],
                "work_surface": scene.geometry["table"],
            },
            collision_roles={"work_surface": "none"},
        ),
        floor_entity_id="work_surface",
    )
    engine.planning_services._place_scene_provider = SimpleNamespace(
        prepare=lambda request, current: renamed
    )
    entities = dict(context.scene.entities)
    entities["work_surface"] = entities.pop("table")
    context = replace(
        context,
        scene=SceneSnapshot(context.scene.timestamp, context.scene.version, entities),
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = engine.plan(invocation, context)
    reference = next(item for item in _REFERENCE if item["mode"] == "waypoint")
    assert plan.plan_success.all()
    assert (
        hashlib.sha256(plan.joint_trajectory.positions.numpy().tobytes()).hexdigest()
        == reference["positions_sha256"]
    )
    assert (
        plan.diagnostics.metadata["place_waypoint_recovery"]["minimum_table_gap_m"]
        == reference["metadata"]["place_waypoint_recovery"]["minimum_table_gap_m"]
    )


def test_lateral_recovery_requires_declared_payload_service_before_ik(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, ik, _ = inputs(tmp_path, "nominal", monkeypatch)
    invocation = replace(
        invocation,
        recovery_policy=RecoveryPolicy(
            planning_recovery=PlacePlanningRecovery(variant="lateral")
        ),
    )
    with pytest.raises(ValueError, match="observed payload query service"):
        engine.plan(invocation, context)
    assert not calls and not ik
