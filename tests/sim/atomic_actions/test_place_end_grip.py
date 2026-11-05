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

"""Retained end-grip release keeps source velocity/contact phase semantics."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
    EntityState,
    SceneSnapshot,
)
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from embodichain.lab.sim.motion import _cartesian as cartesian
from embodichain.lab.sim.atomic_actions.primitives import _place_scene
from ._place_lateral_fixture import lateral_fixture, payload_query_fixture


@pytest.mark.parametrize("count", [1, 2])
def test_retained_end_grip_matches_source_velocity_or_payload_phase_grid(
    tmp_path, monkeypatch, count
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
    ) = lateral_fixture(tmp_path, initial_ik_failures=0, self_collision_probes=0)
    object_pose = torch.eye(4)[None]
    object_pose[0, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    object_pose[0, :3, 3] = torch.tensor([0.02, 0.0, 0.4])
    context = replace(
        context,
        scene=SceneSnapshot(
            0.0, 0, {**context.scene.entities, "object": EntityState(object_pose)}
        ),
    )
    original = generator._generate_once
    dense = []

    def checked(targets, options=None):
        result = original(targets, options=options)
        if options.preserve_cartesian_samples:
            dense.append(options.sample_count)
            return replace(result, success=torch.zeros_like(result.success))
        return result

    generator._generate_once = checked
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
        joint_prefix_planner=queries if count == 2 else None,
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
            segment_sample_count_retries=(260, 320),
            segment_continuous_ik_recovery=True,
        ),
        skill_options=PlaceOptions(
            hand_interp_steps=12,
            cartesian_waypoint_count=count,
            preserve_current_object_orientation=True,
            linear_segments=True,
            monitor_scene_until_approach=True,
        ),
        recovery_policy=RecoveryPolicy(
            action_timeout=30,
            planning_recovery=PlacePlanningRecovery(variant="retained_end_grip"),
        ),
        invocation_id="place",
    )
    monkeypatch.setattr(cartesian, "resolve_continuous_ik_path", lambda *a, **k: None)
    monkeypatch.setattr(_place_scene, "inspect_robot_motion", collision)
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
        "dense_counts": dense,
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
        "branch": dict(plan.diagnostics.metadata.get("place_branch", {})),
    }
    expected = json.loads(
        (
            Path(__file__).parent / f"data/place_end_grip_{count}_reference.json"
        ).read_text()
    )
    assert record == expected
    assert plan.plan_success.tolist() == [True]
    assert dict(plan.expected_effects.held_object_updates) == {"arm": None}
    if count == 2:
        assert observed[0]["pose"] == object_pose.tolist()
