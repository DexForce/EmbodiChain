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

"""Lateral self-collision recovery retains source contact/payload composition."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import json, hashlib, random
import numpy as np
import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    PlaceOptions,
    PlacePlanningRecovery,
    RecoveryPolicy,
    PlaceSceneInputs,
)
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from embodichain.lab.sim.atomic_actions.primitives import _place_scene, _place_recovery
from ._place_lateral_fixture import lateral_fixture, payload_query_fixture


@pytest.mark.parametrize("prefix_valid", [True, False])
def test_lateral_branch_matches_source_ik_payload_grid_and_rng(
    tmp_path, monkeypatch, prefix_valid
):
    (
        generator,
        context,
        goal,
        geometry,
        calls,
        batches,
        collision,
        collision_calls,
        generate_prefix,
        prefix_calls,
    ) = lateral_fixture(tmp_path, prefix_valid=prefix_valid)
    queries, observed, closed = payload_query_fixture(generator.robot, generate_prefix)
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
        joint_prefix_planner=queries,
        place_scene_provider=SimpleNamespace(prepare=lambda request, context: scene),
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
        ),
        skill_options=PlaceOptions(
            hand_interp_steps=12,
            cartesian_waypoint_count=2,
            preserve_current_object_orientation=True,
            linear_segments=True,
            qualify_required_waypoints=True,
            monitor_scene_until_approach=True,
        ),
        recovery_policy=RecoveryPolicy(
            action_timeout=30,
            planning_recovery=PlacePlanningRecovery(variant="lateral"),
        ),
        invocation_id="place",
    )
    monkeypatch.setattr(_place_scene, "inspect_robot_motion", collision)
    monkeypatch.setattr(_place_recovery, "inspect_robot_motion", collision)
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
        "ik_sha256": hashlib.sha256(
            json.dumps(calls, sort_keys=True).encode()
        ).hexdigest(),
        "batch_sha256": hashlib.sha256(
            json.dumps(batches, sort_keys=True).encode()
        ).hexdigest(),
        "collision_hashes": [digest(t) for t in collision_calls],
        "payload": observed,
        "prefix_calls": prefix_calls,
        "closed": closed,
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
        "branch": dict(plan.diagnostics.metadata.get("place_branch", {})),
    }
    expected = json.loads(
        (
            Path(__file__).parent
            / (
                "data/place_lateral_reference.json"
                if prefix_valid
                else "data/place_lateral_failed_reference.json"
            )
        ).read_text()
    )
    assert record == expected
    assert plan.plan_success.tolist() == [True]
    # Projected-held payload must work without the actual subject in the scene.
    assert "object" not in context.scene.entities
    assert dict(plan.expected_effects.held_object_updates) == {"arm": None}
