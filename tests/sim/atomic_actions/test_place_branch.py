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


def test_place_scene_near_threshold_retains_native_unavailable(monkeypatch):
    """A raw tensorAABB cannot bypass the shared native capability decision."""
    from itertools import product

    points = torch.tensor(list(product((-0.125, 0.125), repeat=3)))[None]
    angle = torch.tensor(0.1)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, torch.tensor(0.0))),
            torch.stack((sine, cosine, torch.tensor(0.0))),
            torch.tensor((0.0, 0.0, 1.0)),
        )
    )
    fk = torch.eye(4).repeat(1, 2, 1, 1)
    fk[:, :, :3, :3] = rotation
    world = torch.einsum("btij,bvj->btvi", fk[:, :, :3, :3], points)
    threshold = 0.002 + 1e-6
    obstacle = torch.eye(4)[None]
    obstacle[:, 0, 3] = world[0, 0, :, 0].max() + 0.125 + threshold + 2e-7
    raw_gap = obstacle[0, 0, 3] - 0.125 - world[0, 0, :, 0].max()
    assert threshold < float(raw_gap) < threshold + 1e-6
    floor = torch.eye(4)[None]
    floor[:, 2, 3] = -1.0
    geometry = SceneMeshSnapshot({"object": points, "wall": points, "floor": points})
    endpoint = SimpleNamespace(
        task_state_key="arm",
        require_target=lambda kind: SimpleNamespace(control_part="arm", joint_ids=(0,)),
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="object"), object_to_eef=torch.eye(4)[None]
    )
    context = SimpleNamespace(
        last_qpos=torch.zeros(1, 1),
        get_held_object=lambda key: held,
        scene=SimpleNamespace(
            entities={
                "object": SimpleNamespace(pose=torch.eye(4)[None], confidence=1.0),
                "wall": SimpleNamespace(pose=obstacle, confidence=1.0),
                "floor": SimpleNamespace(pose=floor, confidence=1.0),
            }
        ),
    )
    robot = SimpleNamespace(
        get_qpos_limits=lambda **kwargs: torch.tensor([[[-1.0, 1.0]]]),
        compute_batch_fk=lambda **kwargs: fk,
        get_solver=lambda *args: SimpleNamespace(
            get_iteration_params=lambda: {"pos_eps": 1e-5, "rot_eps": 1e-5}
        ),
    )
    plan = SimpleNamespace(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(
            positions=torch.zeros(1, 2, 1), dt=torch.full((1, 2), 0.04)
        ),
        segments=(SimpleNamespace(name="approach", stop=2),),
    )
    monkeypatch.setattr(
        _place_scene, "authored_joint_velocity_limits", lambda *args: torch.ones(1, 1)
    )
    monkeypatch.setattr(
        _place_scene, "inspect_robot_motion", lambda *args: {"valid": True}
    )
    calls = []

    def native_unavailable(*args, **kwargs):
        calls.append((args[4].tolist(), kwargs))
        raise RuntimeError("native precision interval unavailable")

    monkeypatch.setattr(_place_scene, "convex_input_clearance", native_unavailable)
    result = _place_scene.inspect_place_plan(
        SimpleNamespace(robot=robot),
        SimpleNamespace(binding=SimpleNamespace(endpoint=lambda *args: endpoint)),
        context,
        plan,
        PlaceSceneInputs(
            geometry=geometry, floor_entity_id="floor", support_owner_id="floor"
        ),
    )
    assert calls == [([0, 1], {"clearance_m": 0.002, "numerical_allowance_m": 1e-6})]
    assert result == {
        "valid": False,
        "reason": "scene_geometry_unavailable",
        "unavailable": "native precision interval unavailable",
    }
