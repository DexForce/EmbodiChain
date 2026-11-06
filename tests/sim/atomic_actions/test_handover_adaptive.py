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

"""Adaptive HandOver preserves source receiving, retries and transfer effects."""

from __future__ import annotations
from pathlib import Path
from dataclasses import replace
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
    TaskState,
)
from embodichain.lab.sim.atomic_actions.primitives import _handover_recovery
from ._handover_adaptive_fixture import (
    handover_fixture,
    plain_metadata as plain,
    reference_record,
)


def configured_handover(tmp_path, mode):
    gen, context, goal, grasp, grasps, ik = handover_fixture(tmp_path, mode)
    engine = AtomicActionEngine(
        gen,
        control_profiles={
            name: ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.04]), grasp=torch.tensor([0.0])
            )
            for name in ("source_hand", "destination_hand")
        },
        grasp_pose_generators={"source_hand": grasp, "destination_hand": grasp},
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
            source_retreat_mode="root_separation" if mode == "exit" else "approach",
        ),
        recovery_policy=RecoveryPolicy(planning_recovery=HandOverPlanningRecovery()),
        invocation_id="transfer",
    )
    return engine, invocation, context, grasps, ik


@pytest.mark.parametrize("mode", ["nominal", "retry", "failed", "exit"])
def test_adaptive_handover_matches_source_direction_rng_and_held_commands(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, grasps, ik = configured_handover(tmp_path, mode)
    monkeypatch.setattr(
        _handover_recovery,
        "inspect_robot_motion",
        lambda *a: {"valid": True, "scope": "controlled_mesh_adapter"},
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
        "grasps_sha256": hashlib.sha256(
            json.dumps(grasps, sort_keys=True).encode()
        ).hexdigest(),
        "ik_sha256": hashlib.sha256(
            json.dumps(ik, sort_keys=True).encode()
        ).hexdigest(),
        "metadata": plain(plan.diagnostics.metadata),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    expected = json.loads(
        (Path(__file__).parent / "data/handover_reference.json").read_text()
    )["adaptive"][mode]
    assert reference_record(actual) == expected
    assert context.task.get_held_object("source_arm") is not None
    assert context.task.get_held_object("destination_arm") is None
    if plan.plan_success.all():
        release = next(s for s in plan.segments if s.name == "handover_release")
        assert bool((plan.joint_trajectory.positions[:, : release.start, 6] == 0).all())
        assert plan.expected_effects.held_object_updates["source_arm"] is None
        received = plan.expected_effects.held_object_updates["destination_arm"]
        assert received.semantics.entity_id == "object"


@pytest.mark.parametrize("missing", ["recovery", "attachment"])
def test_adaptive_handover_rejects_incomplete_declaration_before_sampling(
    tmp_path, missing
):
    engine, invocation, context, grasps, ik = configured_handover(tmp_path, "nominal")
    if missing == "recovery":
        invocation = replace(invocation, recovery_policy=RecoveryPolicy())
        message = "declared together"
    elif missing == "attachment":
        context = replace(context, task=TaskState.empty(batch_size=1, device="cpu"))
        message = "exclusive source attachment"
    with pytest.raises(ValueError, match=message):
        engine.plan(invocation, context)
    assert not grasps and not ik


def test_adaptive_handover_rejects_incompatible_options_during_construction():
    with pytest.raises(ValueError, match="source closing command"):
        HandOverOptions(
            adaptive_receive=True, release_at_target=False, arm_selection="bound"
        )
