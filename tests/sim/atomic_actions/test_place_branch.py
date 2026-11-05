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

"""Source-equivalent goal-seeded Place through the ordinary Atomic engine."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, random
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
from embodichain.lab.sim.atomic_actions.primitives import _place_scene
from ._place_recovery_fixture import recovery_fixture


def branch_fixture(tmp_path, mode, monkeypatch):
    generator, context, goal, geometry, _, batchcalls, _ = recovery_fixture(
        tmp_path, "nominal"
    )
    del generator._generate_once
    geometry["target"] = geometry["object"].clone()
    ikcalls = []

    def inverse(pose, joint_seed, **kw):
        ikcalls.append({"pose": pose.tolist(), "seed": joint_seed.tolist()})
        torch.rand(())
        np.random.rand()
        random.random()
        valid = mode == "nominal" or (
            mode == "fallback"
            and bool(torch.equal(pose, context.scene.entities["target"].pose))
        )
        return torch.tensor([valid]), pose[:, :3, 3].clone()

    generator.robot.compute_ik.side_effect = inverse
    scene = PlaceSceneInputs(
        geometry=SceneMeshSnapshot(geometry),
        floor_entity_id="table",
        support_owner_id="target",
    )
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
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
            linear_segments=True,
            monitor_scene_until_approach=True,
        ),
        recovery_policy=RecoveryPolicy(
            planning_recovery=PlacePlanningRecovery(variant="goal_seeded_on")
        ),
        invocation_id="place",
    )
    monkeypatch.setattr(
        _place_scene,
        "inspect_robot_motion",
        lambda robot, positions, part: {
            "valid": True,
            "frames": len(positions),
            "scope": "discrete_robot_meshes",
        },
    )
    return engine, invocation, context, ikcalls, batchcalls


@pytest.mark.parametrize("mode", ["nominal", "fallback", "failed"])
def test_common_seed_branch_preserves_source_grid_rng_and_mandatory_samples(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, ikcalls, batchcalls = branch_fixture(
        tmp_path, mode, monkeypatch
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    actual = {
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "ik_sha256": hashlib.sha256(
            json.dumps(ikcalls, sort_keys=True).encode()
        ).hexdigest(),
        "batch_sha256": hashlib.sha256(
            json.dumps(batchcalls, sort_keys=True).encode()
        ).hexdigest(),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
        "branch": dict(plan.diagnostics.metadata.get("place_branch", {})),
    }
    reference = json.loads(
        (Path(__file__).parent / "data/place_branch_reference.json").read_text()
    )[mode]
    assert actual == reference
    if plan.plan_success.all():
        assert dict(plan.expected_effects.held_object_updates) == {"arm": None}
        assert plan.scene_dependency_monitor_until["target"] == plan.segments[0].stop


def test_goal_seeded_variant_rejects_preserved_orientation_before_ik(
    tmp_path, monkeypatch
):
    engine, invocation, context, ikcalls, batchcalls = branch_fixture(
        tmp_path, "fallback", monkeypatch
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options, preserve_current_object_orientation=True
        ),
    )
    with pytest.raises(ValueError, match="Place waypoint recovery"):
        engine.plan(invocation, context)
    assert not ikcalls and not batchcalls
