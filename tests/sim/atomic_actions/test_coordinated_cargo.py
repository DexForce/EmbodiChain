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

"""Observed cargo preference retains source bounds, fallback and sampler inputs."""

from __future__ import annotations
from dataclasses import replace
from unittest.mock import Mock
import json
from pathlib import Path
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    CoordinatedPickmentOptions,
    EntityState,
    SceneSnapshot,
    TaskState,
    MotionPolicy,
)
from embodichain.lab.sim.atomic_actions.primitives._coordinated_cargo import (
    cargo_preference_bounds,
)
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from ._coordinated_continuation_fixture import coordinated_fixture


@pytest.mark.parametrize(
    "mode",
    ["enabled", "disabled", "unobserved", "missing", "missing_after", "empty", "env1"],
)
def test_engine_forwards_only_available_same_context_cargo_bounds(tmp_path, mode):
    gen, context, goal, grasp, _, _ = coordinated_fixture(tmp_path, "hold")
    # Keep the real candidate entry point and stop at an unavailable population.
    # Candidate ranking itself has pinned source coverage in test_coordinated_grasp.
    grasp.get_dual_arm_valid_grasp_poses = Mock(return_value=[None])
    carrier = context.scene.entities["object"].pose
    cargo_pose = carrier.clone()
    cargo_pose[:, 2, 3] += 0.1
    far_pose = carrier.clone()
    far_pose[:, 0, 3] += 3
    mesh = torch.tensor([[[-0.02, -0.03, 0.0], [0.02, 0.03, 0.5]]])
    geometry = SceneMeshSnapshot(
        {
            "object": torch.tensor([[[-1.0, -1.0, 0.0], [1.0, 1.0, 0.2]]]),
            "cargo": mesh,
            "outside": mesh,
        }
    )
    context = replace(
        context,
        task=TaskState(1, "cpu"),
        scene=SceneSnapshot(
            0.0,
            0,
            {
                "object": EntityState(carrier),
                "cargo": EntityState(
                    cargo_pose, confidence=0.0 if mode == "unobserved" else 1.0
                ),
                "outside": EntityState(far_pose),
            },
        ),
    )
    if mode == "missing":
        context = replace(
            context, scene=SceneSnapshot(0.0, 0, {"object": EntityState(carrier)})
        )
    if mode == "missing_after":
        context = replace(
            context,
            scene=SceneSnapshot(
                0.0,
                0,
                {"object": EntityState(carrier), "cargo": EntityState(cargo_pose)},
            ),
        )
    if mode == "env1":
        context = replace(context, env_ids=torch.tensor([1]))
    engine = AtomicActionEngine(
        gen,
        coordinated_cargo_geometry=geometry if mode != "empty" else None,
        control_profiles={
            name: ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.04]), grasp=torch.tensor([0.0])
            )
            for name in ("source_hand", "destination_hand")
        },
        grasp_pose_generators={"source_hand": grasp, "destination_hand": grasp},
    )
    # Assembly owns its geometry; changing the caller's snapshot cannot affect planning.
    geometry["cargo"].fill_(100.0)
    invocation = engine.make_invocation(
        "coordinated_pickment",
        goal,
        control_parts={
            "left": {"motion": "source_arm", "grasp": "source_hand"},
            "right": {"motion": "destination_arm", "grasp": "destination_hand"},
        },
        motion_policy=MotionPolicy(sample_count=140, dynamic_collision_mode="off"),
        skill_options=CoordinatedPickmentOptions(
            prefer_cargo_clearance=mode != "disabled"
        ),
    )
    rng = torch.get_rng_state().clone()
    cargo_preference_bounds(context, geometry, "object")
    assert torch.equal(rng, torch.get_rng_state())
    plan = engine.plan(invocation, context)
    assert not plan.plan_success.any()
    assert grasp.get_dual_arm_valid_grasp_poses.called
    for call in grasp.get_dual_arm_valid_grasp_poses.call_args_list:
        if mode == "enabled":
            torch.testing.assert_close(
                call.kwargs["cargo_bounds"],
                torch.tensor([[[-0.02, -0.03, 0.5], [0.02, 0.03, 1.0]]]),
                rtol=0,
                atol=0,
            )
        else:
            assert "cargo_bounds" not in call.kwargs
