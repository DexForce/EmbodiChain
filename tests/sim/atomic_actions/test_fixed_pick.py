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

"""Source fixed-grasp qualification never resamples or consumes caller RNG."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace as NS
import json
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
    RecoveryPolicy,
)
from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from ._fixed_pick_fixture import fixed_fixture
from ._ordinary_pick_fixture import OrdinaryCandidates, stable_closing_report
from ._constrained_pick_fixture import record
from .test_pick_native_closing import QueryProvider


class NoSamples(OrdinaryCandidates):
    def get_valid_grasp_poses(self, **kwargs):
        pytest.fail("Fixed-grasp qualification must not resample candidates.")


def _inputs(tmp_path, monkeypatch, mode):
    from embodichain.lab.sim.atomic_actions.primitives import _manipulation_checks

    (
        gen,
        context,
        sem,
        fixed,
        calls,
        ik,
        robot_query,
        scene_query,
        robot_checks,
        scene_checks,
    ) = fixed_fixture(tmp_path, mode)
    monkeypatch.setattr(_manipulation_checks, "inspect_robot_motion", robot_query)

    def inspect_scene(positions, **kwargs):
        result = scene_query(gen.robot, None, positions, "arm")
        return MotionCollisionReport(
            **{key: value for key, value in result.items() if key != "scope"}
        )

    engine = AtomicActionEngine(
        gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators={"hand": NoSamples()},
        free_motion_query=NS(robot=gen.robot, inspect=inspect_scene),
        pick_closing_query_provider=QueryProvider(),
        hand_geometry_cfgs={
            "hand": RobotHandGeometryCfg(
                motion_part="arm",
                hand_part="hand",
                hand_root_link="hand_root",
                pad_link_names=("pad_a", "pad_b"),
                open_qpos=(0.04, 0.04),
                closed_qpos=(0.0, 0.0),
                sample_count=65,
                body_max_edge=0.1,
                pad_max_edge=0.01,
                max_subdivisions=1,
            )
        },
    )
    invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics=sem),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
        ),
        skill_options=PickUpOptions(
            qualify_fixed_grasp=True,
            fixed_object_to_eef=fixed,
            native_closing_revision=1,
            lift_height=0.16,
            pre_grasp_distance=0.15,
            approach_alignment_max_angle=0.1,
            hand_interp_steps=12,
        ),
        recovery_policy=RecoveryPolicy(
            action_timeout=0.1 if mode == "budget" else 90.0
        ),
        invocation_id="pick",
    )
    return engine, invocation, context, calls, ik, robot_checks, scene_checks


@pytest.mark.parametrize(
    "mode", ["accepted", "alignment", "scene_failed", "ik_failed", "budget"]
)
def test_fixed_pick_retains_source_main_path_and_failure_receipts(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, calls, ik, robot_checks, scene_checks = _inputs(
        tmp_path, monkeypatch, mode
    )
    torch.manual_seed(17)
    plan = engine.plan(invocation, context)
    actual = stable_closing_report(record(plan, calls, ik, robot_checks))
    actual["scene_checks"] = scene_checks
    actual = json.loads(
        json.dumps(
            actual,
            default=lambda value: (
                value.tolist() if isinstance(value, torch.Tensor) else str(value)
            ),
        )
    )
    expected = json.loads(
        (Path(__file__).parent / "data" / f"fixed_pick_{mode}_b0.json").read_text()
    )
    if "gen_sim_pick_contact_geometry" in expected["metadata"]:
        expected["metadata"]["pick_contact_geometry"] = expected["metadata"].pop(
            "gen_sim_pick_contact_geometry"
        )
    assert actual == expected
    assert plan.diagnostics.metadata["gen_sim_fixed_pick"]["resampled"] is False


def test_fixed_grasp_transform_rechecks_commands_without_resampling(
    tmp_path, monkeypatch
):
    from embodichain.lab.sim.atomic_actions.primitives import _manipulation_checks

    engine, invocation, context, calls, ik, checks, _ = _inputs(
        tmp_path, monkeypatch, "accepted"
    )
    before = []

    def transform(request, current, plan):
        before.append((len(calls), len(ik)))
        monkeypatch.setattr(
            _manipulation_checks,
            "inspect_robot_motion",
            lambda *a: {
                "valid": False,
                "frame": 0,
                "kind": "self",
                "pairs": [["hand", "arm"]],
            },
        )
        return plan

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert not plan.plan_success.any()
    assert before == [(len(calls), len(ik))]
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)


def test_fixed_qualification_rejects_incompatible_sampling_flags_before_capture(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    engine, invocation, context, calls, ik, _, _ = _inputs(
        tmp_path, monkeypatch, "accepted"
    )
    for changed in (
        {"adaptive_grasp": "ordinary"},
        {"constrained_grasp": True},
        {"full_opening_clearance": True},
    ):
        with pytest.raises(ValueError, match="Fixed Pick qualification"):
            engine.plan(
                replace(
                    invocation,
                    skill_options=replace(invocation.skill_options, **changed),
                ),
                context,
            )
    assert calls == [] and ik == []
    assert engine.planning_services.pick_closing_query_provider.queries == []
