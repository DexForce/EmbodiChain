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

"""Pure Pour projections qualify sampled Pick candidates before full planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import pytest

from embodichain.compute.geometry.pouring import PourGraspProjection, pour_staging_pose
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
)
from embodichain.lab.sim.utility.grasp_geometry import ordered_pad_line_world
from ._pour_fixture import hand_cfg
from ._pick_projection_fixture import pick_fixture, ProjectionCandidates


class _ProjectionProvider:
    def __init__(self, robot, vertices, receiving, returned):
        self.robot, self.vertices, self.receiving, self.returned = (
            robot,
            vertices,
            receiving,
            returned,
        )
        self.calls = 0
        self.enabled = True

    def prepare(self, request, context):
        self.calls += 1
        if not self.enabled:
            return None
        source = context.scene.entities["object"].pose[0].numpy()
        receiver = context.scene.entities["receiver"].pose[0].numpy()
        stage = pour_staging_pose(
            self.vertices.numpy(),
            self.receiving.numpy(),
            source,
            receiver,
            clearance_m=0.05,
            gravity_world=(0.0, 0.0, -9.81),
        )
        tcp = self.robot.compute_fk(
            qpos=context.robot.qpos[:, :2], name="left_arm", to_matrix=True
        )
        jaw = ordered_pad_line_world(self.robot, cfg=hand_cfg())
        local = (tcp[:, :3, :3].transpose(1, 2) @ jaw[..., None])[0, :, 0]
        return PourGraspProjection(
            source_vertices=self.vertices.numpy(),
            receiver_vertices=self.receiving.numpy(),
            staging=stage,
            receiver_pose=receiver,
            return_pose=self.returned.numpy(),
            pad_line_tcp=local,
            gravity_world=(0.0, 0.0, -9.81),
            angle_rad=0.8,
            sample_count=7,
        )


def _inputs(tmp_path):
    gen, context, sem, vertices, receiving, returned, ik, motion = pick_fixture(
        tmp_path
    )
    provider = _ProjectionProvider(gen.robot, vertices, receiving, returned)
    engine = AtomicActionEngine(
        gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(2), grasp=torch.full((2,), 0.1)
            )
        },
        grasp_pose_generators={"hand": ProjectionCandidates()},
        pour_grasp_projection_provider=provider,
    )
    invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics=sem),
        control_parts={"primary": {"motion": "left_arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(sample_count=12, dynamic_collision_mode="off"),
        skill_options=PickUpOptions(
            hand_interp_steps=2,
            grasp_settle_steps=1,
            downstream_object_target_poses=(returned, returned),
        ),
        invocation_id="pick",
    )
    return engine, invocation, context, provider, ik, motion


def test_shared_pick_projection_matches_source_ik_order_rng_and_complete_plan(tmp_path):
    reference = json.loads(
        (
            Path(__file__).parent / "data/pick_projection_source_reference.json"
        ).read_text()
    )
    engine, invocation, context, provider, ik, motion = _inputs(tmp_path)
    torch.manual_seed(883)
    plan = engine.plan(invocation, context)
    actual = dict(
        ik_calls=ik,
        motion_calls=motion,
        positions=plan.joint_trajectory.positions.tolist(),
        dt=plan.joint_trajectory.dt.tolist(),
        success=plan.plan_success.tolist(),
        segments=[
            {"name": s.name, "start": s.start, "stop": s.stop} for s in plan.segments
        ],
        active_masks=[frame.active_mask.tolist() for frame in plan.commands.frames],
        effect_transform={
            key: value.object_to_eef.tolist()
            for key, value in plan.expected_effects.held_object_updates.items()
        },
        scene_dependencies=list(plan.scene_dependencies),
        rng_after_sha256=hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest(),
    )
    assert actual == {key: reference[key] for key in actual}
    assert provider.calls == 1 and len(ik) == 8 and motion[0]["ik_before_motion"] == 8
    # A subsequent unscoped call retains ordinary downstream targets; no context
    # variable may leak the previous observation's five-role projection.
    provider.enabled = False
    ik.clear()
    motion.clear()
    engine.plan(invocation, context)
    assert len(ik) == 5 and motion[0]["ik_before_motion"] == 5


def test_projected_pick_nominal_entry_rejects_before_source_or_ik(tmp_path):
    engine, invocation, context, provider, ik, motion = _inputs(tmp_path)

    class Source:
        def instantiate(self, request, current):
            raise AssertionError(
                "required projection must not be bypassed by nominal motion"
            )

    with pytest.raises(ValueError, match="External nominal Pick cannot bypass"):
        engine.plan(invocation, context, nominal_source=Source())
    assert provider.calls == 1 and ik == [] and motion == []
