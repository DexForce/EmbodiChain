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

"""Coordinated verified-grasp continuation keeps the source synchronized path."""

from __future__ import annotations
from pathlib import Path
import json, hashlib, random
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    CoordinatedPickmentOptions,
)
from ._coordinated_continuation_fixture import coordinated_fixture


@pytest.mark.parametrize(
    "mode", ["hold", "release", "missing", "continuous", "partial"]
)
def test_verified_dual_continuation_matches_source_without_resampling(tmp_path, mode):
    gen, context, goal, grasp, grasps, ik = coordinated_fixture(tmp_path, mode)
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
        "coordinated_pickment",
        goal,
        control_parts={
            "left": {"motion": "source_arm", "grasp": "source_hand"},
            "right": {"motion": "destination_arm", "grasp": "destination_hand"},
        },
        motion_policy=MotionPolicy(sample_count=140, dynamic_collision_mode="off"),
        skill_options=CoordinatedPickmentOptions(
            release=mode == "release", continue_verified_holds=True
        ),
        invocation_id="dual_move",
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    effects = {
        key: (
            None
            if value is None
            else {
                "entity": value.semantics.entity_id,
                "offset": value.object_to_eef.tolist(),
                "grasp": value.grasp_xpos.tolist(),
            }
        )
        for key, value in plan.expected_effects.held_object_updates.items()
    }
    actual = {
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "ik_sha256": hashlib.sha256(
            json.dumps(ik, sort_keys=True).encode()
        ).hexdigest(),
        "grasps": grasps,
        "effects": effects,
        "monitor": dict(plan.scene_dependency_monitor_until),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    expected = json.loads(
        (Path(__file__).parent / f"data/coordinated_{mode}_reference.json").read_text()
    )
    assert actual == expected
    assert not grasps
