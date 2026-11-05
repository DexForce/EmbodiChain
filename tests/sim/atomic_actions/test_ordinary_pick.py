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

"""Complete shared ordinary Pick paths against the frozen source algorithm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    PickPlanningRecovery,
    PickUpOptions,
    RecoveryPolicy,
)
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from ._ordinary_pick_fixture import ordinary_fixture, OrdinaryCandidates


def _inputs(tmp_path, monkeypatch, *, ik_failure=False, engine_kwargs=None):
    from embodichain.lab.sim.atomic_actions.primitives import _manipulation_checks

    generator, context, semantics, calls, ik = ordinary_fixture(tmp_path)
    if ik_failure:
        generator.robot.compute_batch_ik.side_effect = lambda pose, joint_seed, **kw: (
            torch.zeros(pose.shape[:2], dtype=torch.bool),
            joint_seed.clone(),
        )
        generator.robot.compute_ik.side_effect = lambda pose, joint_seed, **kw: (
            torch.zeros(pose.shape[0], dtype=torch.bool),
            joint_seed.clone(),
        )
    query = NS(
        robot=generator.robot,
        inspect=lambda positions, **kw: MotionCollisionReport(
            True, frames=len(positions)
        ),
    )
    monkeypatch.setattr(
        _manipulation_checks,
        "inspect_robot_motion",
        lambda robot, positions, part: {
            "valid": True,
            "frames": len(positions),
            "scope": "discrete_robot_meshes",
        },
    )
    supplied = dict(engine_kwargs or {})
    generators = supplied.pop("grasp_pose_generators", {"hand": OrdinaryCandidates()})
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators=generators,
        free_motion_query=query,
        **supplied,
    )
    invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics=semantics),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
        ),
        skill_options=PickUpOptions(
            lift_height=0.16,
            pre_grasp_distance=0.15,
            approach_alignment_max_angle=0.1,
            hand_interp_steps=12,
            adaptive_grasp="ordinary",
            full_opening_clearance=True,
            require_qualified_approach=True,
        ),
        recovery_policy=RecoveryPolicy(planning_recovery=PickPlanningRecovery()),
        invocation_id="pick",
    )
    return engine, invocation, context, calls, ik


@pytest.mark.parametrize("ik_failure", [False, True])
def test_ordinary_pick_preserves_source_sampling_contact_path_and_commands(
    tmp_path, monkeypatch, ik_failure
):
    engine, invocation, context, calls, ik = _inputs(
        tmp_path, monkeypatch, ik_failure=ik_failure
    )
    torch.manual_seed(147)
    plan = engine.plan(invocation, context)
    digest = lambda tensor: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    reference = json.loads(
        (
            Path(__file__).parent
            / (
                "data/ordinary_pick_failure_source_reference.json"
                if ik_failure
                else "data/ordinary_pick_source_reference.json"
            )
        ).read_text()
    )
    actual = dict(
        calls=calls,
        ik=ik,
        success=plan.plan_success.tolist(),
        positions_sha256=digest(plan.joint_trajectory.positions),
        dt_sha256=digest(plan.joint_trajectory.dt),
        segments=[list((s.name, s.start, s.stop)) for s in plan.segments],
        metadata=dict(plan.diagnostics.metadata),
        rng=digest(torch.get_rng_state()),
    )
    assert actual == {key: reference[key] for key in actual}
    assert bool(plan.plan_success.all()) is not ik_failure
    if not ik_failure:
        assert plan.diagnostics.metadata["pick_path"]["close_lift_unchanged"] is True
    else:
        assert (
            len(plan.diagnostics.metadata["pick_raw_candidate_retry"]["attempts"]) == 2
        )


def test_pick_transform_qualification_rejects_without_replanning(tmp_path, monkeypatch):
    engine, invocation, context, calls, ik = _inputs(tmp_path, monkeypatch)
    at_transform = []
    original = []

    def transform(request, current, plan):
        at_transform.append((len(calls), len(ik)))
        original.append(plan.joint_trajectory.positions.clone())
        engine.planning_services.free_motion_query.inspect = (
            lambda positions, **kw: MotionCollisionReport(
                False, frame=0, kind="scene", pairs=(("arm", "wall"),)
            )
        )
        return plan

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert at_transform == [(len(calls), len(ik))]
    assert not plan.plan_success.any()
    assert torch.equal(
        plan.joint_trajectory.positions, original[0][:, :1].expand_as(original[0])
    )
    assert torch.equal(plan.rejected_candidate.positions, original[0])
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)


def test_new_pick_options_reject_external_nominal_before_source(tmp_path, monkeypatch):
    engine, invocation, context, calls, _ = _inputs(tmp_path, monkeypatch)

    class Source:
        def instantiate(self, request, current):
            raise AssertionError("The unsupported source must not be called.")

    with pytest.raises(ValueError, match="External nominal"):
        engine.plan(invocation, context, nominal_source=Source())
    assert not calls
