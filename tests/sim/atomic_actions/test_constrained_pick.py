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

"""Full constrained Pick source parity through the ordinary Atomic engine."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
    PickPlanningRecovery,
    RecoveryPolicy,
)
from embodichain.toolkits.graspkit.pg_grasp.constraints import (
    GraspGeometryConstraint,
    grasp_geometry_digest,
)
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from ._constrained_pick_fixture import (
    constrained_fixture,
    ConstrainedCandidates,
    record,
)


def _inputs(tmp_path, monkeypatch, mode):
    from embodichain.lab.sim.atomic_actions.primitives import (
        _manipulation_checks,
        _pick_constrained,
    )

    gen, context, sem, calls, ik, inspect, inspections = constrained_fixture(
        tmp_path, mode
    )
    monkeypatch.setattr(_manipulation_checks, "inspect_robot_motion", inspect)
    monkeypatch.setattr(_pick_constrained, "inspect_robot_motion", inspect)
    if mode == "opening_prefix":
        from ._constrained_pick_fixture import force_opening_jump

        original = _pick_constrained.recover_constrained_pick
        monkeypatch.setattr(
            _pick_constrained,
            "recover_constrained_pick",
            lambda action, request, context, plan: original(
                action,
                request,
                context,
                force_opening_jump(action, request, context, plan),
            ),
        )
    constraint = GraspGeometryConstraint(
        grasp_geometry_digest(
            sem.affordance.mesh_vertices, sem.affordance.mesh_triangles
        ),
        (0.0, 0.0, 1.0),
        0.0,
        True,
    )
    native_kwargs = {}
    revision = None
    if mode.startswith("native"):
        from .test_pick_native_closing import QueryProvider
        from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

        revision = int(mode.removeprefix("native"))
        native_kwargs = dict(
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
    engine = AtomicActionEngine(
        gen,
        **native_kwargs,
        **(
            {
                "free_motion_query": SimpleNamespace(
                    robot=gen.robot,
                    inspect=lambda positions, **kwargs: MotionCollisionReport(
                        True, frames=len(positions)
                    ),
                )
            }
            if mode.startswith("raw_")
            else {}
        ),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators={"hand": ConstrainedCandidates(mode)},
    )
    invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics=sem, grasp_constraint=constraint),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            segment_sample_count_retries=(
                (260, 320) if mode.startswith("cartesian_") else ()
            ),
        ),
        skill_options=PickUpOptions(
            constrained_grasp=True,
            linear_segments=mode.startswith("cartesian_"),
            native_closing_revision=revision,
            lift_height=0.16,
            pre_grasp_distance=0.15,
            approach_alignment_max_angle=0.1,
            hand_interp_steps=12,
        ),
        recovery_policy=RecoveryPolicy(
            planning_recovery=PickPlanningRecovery(constrained_roll=True)
        ),
        invocation_id="pick",
    )
    return engine, invocation, context, calls, ik, inspections


@pytest.mark.parametrize(
    "mode",
    [
        "safe",
        "alternate",
        "exhausted",
        "start_rejected",
        "unavailable",
        "opening",
        "opening_prefix",
        "initial_failed",
        "direction_fallback",
        "native5",
        "native8",
        "cartesian_safe",
        "cartesian_dense",
        "cartesian_exhausted",
        "raw_after_limits",
        "raw_exhausted",
    ],
)
def test_shared_constrained_pick_preserves_source_commands_queries_and_rng(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, calls, ik, inspections = _inputs(
        tmp_path, monkeypatch, mode
    )
    torch.manual_seed(17)
    plan = engine.plan(invocation, context)
    from ._ordinary_pick_fixture import stable_closing_report

    actual = json.loads(
        json.dumps(
            stable_closing_report(record(plan, calls, ik, inspections)),
            default=lambda v: v.tolist() if isinstance(v, torch.Tensor) else str(v),
        )
    )
    expected = json.loads(
        (
            Path(__file__).parent / "data" / f"constrained_pick_{mode}_b0.json"
        ).read_text()
    )
    # Shared owners use canonical diagnostic names; every numeric query,
    # counter, complete plan and RNG state below these keys stays source-exact.
    for source, shared in (
        ("gen_sim_pick_contact_geometry", "pick_contact_geometry"),
        ("gen_sim_pick_raw_candidate_retry", "pick_raw_candidate_retry"),
    ):
        if source in expected["metadata"]:
            expected["metadata"][shared] = expected["metadata"].pop(source)
    assert actual == expected


def test_constrained_transform_rechecks_robot_without_recovery(tmp_path, monkeypatch):
    from embodichain.lab.sim.atomic_actions.primitives import _manipulation_checks

    engine, invocation, context, calls, ik, _ = _inputs(tmp_path, monkeypatch, "safe")
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
                "pairs": (("arm", "other"),),
                "scope": "discrete_robot_meshes",
            },
        )
        return plan

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert not plan.plan_success.any()
    assert before == [(len(calls), len(ik))]
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)


def test_constrained_raw_recovery_keeps_explicit_object_pose_outside_source_scope(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    engine, invocation, context, _, _, _ = _inputs(
        tmp_path, monkeypatch, "raw_after_limits"
    )
    engine.planning_services.free_motion_query.inspect = (
        lambda *args, **kwargs: pytest.fail(
            "Source raw recovery does not own an explicit object-pose override."
        )
    )
    goal = replace(
        invocation.goal, object_pose=context.scene.entities["object"].pose.clone()
    )
    plan = engine.plan(replace(invocation, goal=goal), context)
    assert not plan.plan_success.any()
    assert "pick_raw_candidate_retry" not in plan.diagnostics.metadata


def test_constrained_invalid_policy_rejects_before_candidate_generation(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    engine, invocation, context, calls, ik, _ = _inputs(tmp_path, monkeypatch, "safe")
    with pytest.raises(ValueError, match="explicit bounded roll"):
        engine.plan(replace(invocation, recovery_policy=RecoveryPolicy()), context)
    assert calls == [] and ik == []

    class Source:
        def instantiate(self, *args):
            pytest.fail("nominal source must not run")

    with pytest.raises(ValueError, match="External nominal"):
        engine.plan(
            replace(invocation, recovery_policy=RecoveryPolicy()),
            context,
            nominal_source=Source(),
        )
