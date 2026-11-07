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

"""Complete native receiver qualification matches frozen source composition."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from importlib.metadata import version
import math
import json, hashlib, random
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    HandOverOptions,
    HandOverPlanningRecovery,
    RecoveryPolicy,
)
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from embodichain.lab.sim.scene.native_hand_scene import (
    NativeHandSceneQuery,
    NativeSceneBody,
)
from ._handover_native_fixture import native_transfer_fixture, transfer_scene
from ._handover_adaptive_fixture import plain_metadata as plain

_NATIVE_RECEIPT_FIELDS = {
    "collision_backend",
    "collision_backend_version",
    "distance_semantics",
    "geometry_representation",
    "native_queries",
    "reported_distance_semantics",
    "minimum_checked_distance_interval_m",
    "broadphase_scope",
    "broadphase_culled_pairs",
    "last_native_contact_result",
}


def _assert_native_receiver_receipt(actual, expected):
    """Check the new native protocol while retaining the Source decision fields."""
    native_queries = actual["native_queries"]
    assert type(native_queries) is int and native_queries == expected["fcl_queries"]
    assert set(actual) == (set(expected) - {"fcl_queries"}) | (
        _NATIVE_RECEIPT_FIELDS
        - ({"last_native_contact_result"} if native_queries == 0 else set())
    )
    assert actual["collision_backend"] == "newton.CollisionPipeline"
    assert actual["collision_backend_version"] == version("newton")
    assert actual["distance_semantics"] == "capped_conversion_interval/v1"
    assert actual["geometry_representation"] == "captured_native_triangle_surfaces"
    assert actual["reported_distance_semantics"] == "conservative_clearance_lower_bound"
    assert actual["broadphase_scope"] == "canonical_native_conversion_lower_bound"
    culled = actual["broadphase_culled_pairs"]
    assert type(culled) is int and culled >= 0
    # These fixtures have one receiver link and one obstacle, in this order.
    # Every preceding frame is bound-checked; a rejected frame queries native.
    assert culled == (actual["frames"] if actual["valid"] else actual["frame"])
    if native_queries == 0:
        assert actual["valid"]
        assert actual["minimum_checked_distance_interval_m"] is None
        assert actual["minimum_checked_distance_m"] is None
        return
    assert native_queries == 1 and not actual["valid"]
    lower, upper = actual["minimum_checked_distance_interval_m"]
    threshold = actual["native_pair_contact_offset_m"]
    assert math.isfinite(lower) and math.isfinite(upper)
    assert lower <= upper <= threshold
    for key in ("minimum_checked_distance_m", "minimum_distance_m"):
        assert actual[key] == lower
        # Historical FCL scalars are retained, but not relabelled as Newton
        # measurements. They must lie inside the declared native interval.
        assert lower <= expected[key] <= upper
    contact = actual["last_native_contact_result"]
    assert set(contact) == {
        "backend",
        "predicate",
        "distance_semantics",
        "lower_bound_m",
        "upper_bound_m",
        "upper_unbounded",
        "signed_surface_separation_m",
        "representation_error_m",
        "detection_gap_m",
        "native_contact_count",
        "pair_ids",
        "pairs",
        "clearance_unavailable_reason",
        "continuous",
        "physical_acquisition",
    }
    assert isinstance(contact["clearance_unavailable_reason"], str)
    assert contact["backend"] == "newton.CollisionPipeline"
    assert contact["predicate"] == "native_surface_separation<=threshold"
    assert contact["distance_semantics"] == actual["distance_semantics"]
    assert contact["lower_bound_m"] == lower and contact["upper_bound_m"] == upper
    assert contact["upper_unbounded"] is False
    assert math.isfinite(contact["signed_surface_separation_m"])
    assert contact["signed_surface_separation_m"] <= threshold
    assert math.isfinite(contact["representation_error_m"])
    assert contact["representation_error_m"] >= 0
    assert contact["detection_gap_m"] >= threshold
    assert type(contact["native_contact_count"]) is int
    assert contact["native_contact_count"] > 0
    assert contact["continuous"] is contact["physical_acquisition"] is False
    assert contact["pair_ids"] is None and len(contact["pairs"]) == 1
    pair = contact["pairs"][0]
    assert pair["pair_ids"] == ["a:0", "b:0"] and pair["pairs"] == []
    assert {k: v for k, v in pair.items() if k not in {"pair_ids", "pairs"}} == {
        k: v for k, v in contact.items() if k not in {"pair_ids", "pairs"}
    }
    json.dumps(contact, allow_nan=False)


def _assert_source_record(actual, expected):
    """Compare all fields; only scoped native receipts have separate assertions."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        native = (
            expected.get("scope")
            in {
                "receiver_open_hand_nominal_approach_non_target_scene",
                "receiver_hand_non_target_scene_discrete_native_shapes",
            }
            and "fcl_queries" in expected
        )
        if native:
            _assert_native_receiver_receipt(actual, expected)
            separately_checked = _NATIVE_RECEIPT_FIELDS | {
                "fcl_queries",
                "minimum_checked_distance_m",
                "minimum_distance_m",
            }
            keys = set(expected) - separately_checked
        else:
            assert set(actual) == set(expected)
            keys = expected
        for key in keys:
            _assert_source_record(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_source_record(value, reference)
    else:
        assert actual == expected


def native_inputs(tmp_path, mode):
    gen, context, goal, grasp, grasps, ik, fk = native_transfer_fixture(tmp_path, mode)
    context = replace(
        context,
        world_source_stamp=PhysicalObservationStamp(
            "test-source", "default.contacts", 7, 0.0
        ),
    )
    queries = []

    class Provider:
        robot = gen.robot

        def prepare(self, request, current):
            parts, obstacles, inventory = transfer_scene(NativeSceneBody, mode)
            query = NativeHandSceneQuery(
                scope=NativeQueryScope("query-" + str(len(queries)), 0.0, 0),
                robot_uid=self.robot.uid,
                joint_names=self.robot.joint_names,
                root_pose=np.eye(4),
                parts=parts,
                obstacles=obstacles,
                inventory=inventory,
                native_revision=7,
                read_native_revision=lambda: 7,
            )
            queries.append(query)
            return query

    provider = Provider()
    engine = AtomicActionEngine(
        gen,
        control_profiles={
            name: ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.04]), grasp=torch.tensor([0.0])
            )
            for name in ("source_hand", "destination_hand")
        },
        grasp_pose_generators={"source_hand": grasp, "destination_hand": grasp},
        handover_scene_provider=provider,
    )
    invocation = engine.make_invocation(
        "hand_over",
        goal,
        control_parts={
            "source": {"motion": "source_arm", "grasp": "source_hand"},
            "destination": {"motion": "destination_arm", "grasp": "destination_hand"},
        },
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
        ),
        skill_options=HandOverOptions(
            release_at_target=False,
            arm_selection="bound",
            receive_pick_object_part="center",
            adaptive_receive=True,
            receive_region="supported_end",
            existing_hold_hand_source="grasp_command",
            require_receiver_scene=True,
        ),
        recovery_policy=RecoveryPolicy(planning_recovery=HandOverPlanningRecovery()),
        invocation_id="transfer",
    )
    return engine, invocation, context, queries, grasps, ik, fk, provider


@pytest.mark.parametrize(
    "mode", ["clear", "exchange", "reject", "bad_command", "late_reject"]
)
def test_native_receiver_station_and_command_guards_match_source(tmp_path, mode):
    engine, invocation, context, queries, grasps, ik, fk, provider = native_inputs(
        tmp_path, mode
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    record = {
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "metadata": plain(plan.diagnostics.metadata),
        "captures": len(queries),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    for name, value in (("grasps", grasps), ("ik", ik), ("fk", fk)):
        record[name + "_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()
    expected = json.loads(
        (
            Path(__file__).parent / f"data/handover_native_{mode}_reference.json"
        ).read_text()
    )
    _assert_source_record(record, expected)
    assert all(query.closed for query in queries)


def test_receiver_geometry_never_falls_back_when_observation_stamp_is_missing(tmp_path):
    engine, invocation, context, queries, grasps, ik, _, _ = native_inputs(
        tmp_path, "clear"
    )
    plan = engine.plan(invocation, replace(context, world_source_stamp=None))
    assert not plan.plan_success.any() and not grasps and not ik
    assert len(queries) == 1 and queries[0].closed


def test_receiver_guard_rejects_command_trajectory_or_invocation_drift(tmp_path):
    from embodichain.lab.sim.atomic_actions.primitives._handover_scene import (
        HandOverEnvironmentScreen,
    )

    engine, invocation, context, queries, _, _, _, provider = native_inputs(
        tmp_path, "clear"
    )
    plan = engine.plan(invocation, context)
    request = engine.resolve(invocation)
    with provider.prepare(request, context) as query:
        screen = HandOverEnvironmentScreen(engine.robot, request, context, query)
        with pytest.raises(ValueError, match="command_identity"):
            screen.inspect(replace(plan, invocation_id="different"))
        positions = plan.joint_trajectory.positions.clone()
        positions[:, 0, 6] += 0.01
        changed = replace(
            plan, joint_trajectory=replace(plan.joint_trajectory, positions=positions)
        )
        with pytest.raises(ValueError, match="command_payload_mismatch"):
            screen.inspect(changed)
    assert all(q.closed for q in queries)


@pytest.mark.parametrize(
    "mode", ["clear", "exchange", "reject", "bad_command", "late_reject"]
)
def test_complete_cartesian_handover_matches_source_outer_scope(tmp_path, mode):
    engine, invocation, context, queries, grasps, ik, fk, _ = native_inputs(
        tmp_path, mode
    )
    invocation = replace(
        invocation,
        skill_options=replace(invocation.skill_options, linear_segments=True),
        motion_policy=replace(
            invocation.motion_policy, segment_sample_count_retries=(260, 320)
        ),
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    record = {
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "metadata": plain(plan.diagnostics.metadata),
        "captures": len(queries),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    for name, value in (("grasps", grasps), ("ik", ik), ("fk", fk)):
        record[name + "_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()
    expected = json.loads(
        (Path(__file__).parent / f"data/handover_cartesian_{mode}_b0.json").read_text()
    )["record"]
    _assert_source_record(record, expected)
    assert all(query.closed for query in queries)


def test_native_receiver_bounds_support_proposals_without_bypassing_inspection(
    tmp_path, monkeypatch
):
    engine, invocation, context, queries, _, _, _, _ = native_inputs(tmp_path, "clear")
    inspected = []
    original = NativeHandSceneQuery.inspect

    def observe(query, poses):
        inspected.append(len(poses))
        bounds = query.obstacle_bounds
        assert len(bounds) == len(query.obstacles)
        for index, value in enumerate(bounds):
            if value is not None:
                before = query.obstacle_bounds
                value[0][:] = -123.0
                assert np.array_equal(query.obstacle_bounds[index][0], before[index][0])
                break
        return original(query, poses)

    monkeypatch.setattr(NativeHandSceneQuery, "inspect", observe)
    plan = engine.plan(invocation, context)
    assert plan.plan_success.all()
    assert queries and inspected and max(inspected) > 1
    assert all(query.closed for query in queries)
    with pytest.raises(RuntimeError, match="closed"):
        queries[0].obstacle_bounds
