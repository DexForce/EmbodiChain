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

"""Source-order held heading screening, fallback, command checks and RNG."""

from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    RecoveryPolicy,
    MoveHeldObjectPlanningRecovery,
    MoveHeldObjectOptions,
)
from embodichain.lab.sim.motion import mesh_collision
from ._held_alignment_fixture import alignment_fixture


def _inputs(tmp_path, monkeypatch, mode, *, linear=False):
    gen, context, goal, generate, check, calls, checks, ik = alignment_fixture(
        tmp_path, mode
    )
    gen._generate_once = generate
    captures = []

    def factory(robot, simulation, part, *, include_scene):
        captures.append((part, include_scene))
        return check

    monkeypatch.setattr(mesh_collision, "_make_checker", factory)
    engine = AtomicActionEngine(
        gen,
        control_profiles={
            "source_hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.04]), grasp=torch.zeros(1)
            )
        },
    )
    invocation = engine.make_invocation(
        "move_held_object",
        goal,
        control_parts={"primary": {"motion": "source_arm", "grasp": "source_hand"}},
        motion_policy=MotionPolicy(
            sample_count=45,
            strategy="motion_gen" if mode == "unsupported" else "ik_interp",
            dynamic_collision_mode="off",
            interpolation_mode="adaptive" if mode == "unsupported" else "fixed_grid",
            ik_seed_mode="batched" if mode == "unsupported" else "sequential",
            acceleration_limit=None,
            segment_sample_count_retries=(260, 320) if linear else (),
        ),
        skill_options=MoveHeldObjectOptions(
            report_retained_attachment=True,
            linear_segments=linear,
            lift_clearance=0.06 if mode == "lift_fallback" else None,
        ),
        recovery_policy=RecoveryPolicy(
            planning_recovery=MoveHeldObjectPlanningRecovery()
        ),
        invocation_id="alignment",
    )
    return engine, invocation, context, calls, checks, ik, captures


@pytest.mark.parametrize(
    "mode",
    [
        "safe",
        "later",
        "fallback",
        "lift_fallback",
        "initial_collision",
        "raw_failed",
        "unsupported",
        "unavailable",
    ],
)
def test_held_alignment_matches_source_complete_commands_and_search(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, calls, checks, ik, captures = _inputs(
        tmp_path, monkeypatch, mode
    )
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    hashed = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    actual = {
        "mode": mode,
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "metadata": dict(plan.diagnostics.metadata),
        "failure": (
            None if plan.diagnostics.failure is None else plan.diagnostics.failure.code
        ),
        "calls_sha256": hashed(calls),
        "checks_sha256": hashed(checks),
        "ik_sha256": hashed(ik),
        "held_keys": list(plan.expected_effects.held_object_updates),
        "rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    expected = json.loads(
        (
            Path(__file__).parent / f"data/held_alignment_{mode}_reference.json"
        ).read_text()
    )
    assert actual == expected
    assert captures == ([] if mode == "unsupported" else [("source_arm", False)])


def test_transformed_held_plan_is_rechecked_without_replanning(tmp_path, monkeypatch):
    from dataclasses import replace

    engine, invocation, context, calls, checks, ik, captures = _inputs(
        tmp_path, monkeypatch, "safe"
    )

    def transform(request, current, plan):
        positions = plan.joint_trajectory.positions.clone()
        positions[
            :, 1:, 6
        ] += 0.01  # Passive destination arm is outside this skill's binding.
        return replace(
            plan, joint_trajectory=replace(plan.joint_trajectory, positions=positions)
        )

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert len(calls) == 1
    assert not plan.plan_success.any()
    assert plan.diagnostics.failure.code == "held_alignment_safety_failed"
    assert (
        plan.diagnostics.metadata["final_qualification"]["held_alignment"][
            "final_command_check"
        ]["reason"]
        == "Alignment changed an uncommanded joint."
    )
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)


def test_observed_hand_mode_keeps_its_posture_and_source_command_rejection(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    engine, invocation, context, calls, checks, ik, captures = _inputs(
        tmp_path, monkeypatch, "safe"
    )
    invocation = replace(
        invocation,
        skill_options=replace(invocation.skill_options, hand_hold="observed"),
    )
    plan = engine.plan(invocation, context)
    assert not plan.plan_success.any()
    assert plan.diagnostics.failure.code == "held_alignment_safety_failed"
    assert (
        plan.diagnostics.metadata["held_alignment"]["final_command_check"]["reason"]
        == "Alignment changed the retained grasp command."
    )


def test_drawer_transport_requires_its_implemented_owner():
    from embodichain.lab.sim.atomic_actions import MoveHeldObjectOptions

    with pytest.raises(ValueError, match="Drawer transport is not installed"):
        MoveHeldObjectOptions(drawer_transport=True)
