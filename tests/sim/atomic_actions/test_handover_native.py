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
    assert record == expected
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
    assert record == expected
    assert all(query.closed for query in queries)
