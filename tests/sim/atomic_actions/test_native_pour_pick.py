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

"""Native Pour candidate/IK/order source parity without ordinary scene screens."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace as NS
import json, hashlib, random
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    PickUpOptions,
    GraspGoal,
    PickPlanningRecovery,
    RecoveryPolicy,
    PourGraspProjectionInputs,
)
from embodichain.compute.geometry.pouring import pour_staging_pose
from embodichain.lab.sim.utility.grasp_geometry import (
    RobotHandGeometryCfg,
    ordered_pad_line_world,
)
from ._native_pour_pick_fixture import native_pour_fixture
from ._ordinary_pick_fixture import stable_closing_report
from .test_pick_native_closing import QueryProvider


class Closing(QueryProvider):
    def prepare(self, request, context, *, hand_geometry_cfg):
        query = super().prepare(request, context, hand_geometry_cfg=hand_geometry_cfg)
        query.identity = {"simulation_uid": "source"}
        return query


class Projection:
    def __init__(self, robot, cfg, sem, receiving, returned):
        self.robot, self.cfg, self.sem, self.receiving, self.returned = (
            robot,
            cfg,
            sem,
            receiving,
            returned,
        )

    def prepare(self, request, context):
        source = self.sem.affordance.mesh_vertices.numpy()
        receiver = self.receiving.numpy()
        source_pose = context.scene.entities["object"].pose[0].numpy()
        receiver_pose = context.scene.entities["receiver"].pose[0].numpy()
        stage = pour_staging_pose(
            source,
            receiver,
            source_pose,
            receiver_pose,
            clearance_m=0.05,
            gravity_world=(0.0, 0.0, -9.81),
        )
        tcp = self.robot.compute_fk(
            context.robot.qpos[:, :3], name="left_arm", to_matrix=True
        )
        jaw = ordered_pad_line_world(self.robot, cfg=self.cfg)
        local = (tcp[:, :3, :3].transpose(1, 2) @ jaw[..., None])[0, :, 0]
        return PourGraspProjectionInputs(
            pick_id="pick",
            stage_id="stage",
            pour_id="pour",
            object_id="object",
            receiver_id="receiver",
            resource="direct.primary",
            source_vertices=source,
            receiver_vertices=receiver,
            staging=stage,
            receiver_pose=receiver_pose,
            return_pose=self.returned,
            pad_line_tcp=local,
            gravity_world=(0.0, 0.0, -9.81),
            clearance_m=0.05,
            angle_rad=0.8,
            sample_count=7,
        )


def inputs(tmp_path, revision=8, mode="accept"):
    gen, context, sem, receiving, returned, grasp, calls, ik = native_pour_fixture(
        tmp_path, mode
    )
    cfg = RobotHandGeometryCfg(
        motion_part="left_arm",
        hand_part="hand",
        hand_root_link="hand_root",
        pad_link_names=("left_inner_finger_pad", "left_right_inner_finger_pad"),
        open_qpos=(0.04, 0.04),
        closed_qpos=(0.0, 0.0),
        sample_count=65,
        body_max_edge=0.1,
        pad_max_edge=0.01,
        max_subdivisions=1,
    )
    closing = Closing()

    def no_scene(*args, **kwargs):
        raise AssertionError(
            "Canonical Pour must not borrow ordinary/end-grasp scene screening."
        )

    engine = AtomicActionEngine(
        gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators={"hand": grasp},
        hand_geometry_cfgs={"hand": cfg},
        pick_closing_query_provider=closing,
        pour_grasp_projection_provider=Projection(
            gen.robot, cfg, sem, receiving, returned
        ),
        free_motion_query=NS(robot=gen.robot, inspect=no_scene),
    )
    invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics=sem),
        control_parts={"primary": {"motion": "left_arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
        ),
        skill_options=PickUpOptions(
            adaptive_grasp="pour",
            native_closing_revision=revision,
            full_opening_clearance=revision <= 4,
            lift_height=0.16,
            pre_grasp_distance=0.15,
            approach_alignment_max_angle=0.1,
            hand_interp_steps=12,
            downstream_object_target_poses=(returned, returned),
        ),
        recovery_policy=RecoveryPolicy(planning_recovery=PickPlanningRecovery()),
        invocation_id="pick",
    )
    return engine, invocation, context, calls, ik, closing


@pytest.mark.parametrize(
    "revision,mode",
    [
        (4, "accept"),
        (6, "accept"),
        (7, "accept"),
        (8, "accept"),
        (9, "accept"),
        (8, "downstream_fail"),
        (8, "path_fail"),
        (8, "degenerate"),
        (8, "audit_limit"),
    ],
)
def test_native_pour_pick_matches_source_projection_audit_budget_and_rng(
    tmp_path, revision, mode
):
    engine, invocation, context, calls, ik, closing = inputs(tmp_path, revision, mode)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        random.seed(147)
        np.random.seed(147)
        torch.manual_seed(147)
        plan = engine.plan(invocation, context)
        digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
        hashed = lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        metadata = stable_closing_report(plan.diagnostics.metadata)
        actual = {
            "revision": revision,
            "mode": mode,
            "success": plan.plan_success.tolist(),
            "positions_sha256": digest(plan.joint_trajectory.positions),
            "dt_sha256": digest(plan.joint_trajectory.dt),
            "segments": [[s.name, s.start, s.stop] for s in plan.segments],
            "metadata_sha256": hashed(metadata),
            "calls_sha256": hashed(calls),
            "ik_sha256": hashed(ik),
            "rng": [
                digest(torch.get_rng_state()),
                float(np.random.rand()),
                random.random(),
            ],
        }
        expected = json.loads(
            (
                Path(__file__).parent
                / f"data/native_pour_pick_{revision}_{mode}_reference.json"
            ).read_text()
        )
        assert actual == expected
        assert len(closing.queries) == 1 and closing.queries[0].closed
        assert closing.queries[0].rigid_scene_screen is None
        assert "pick_raw_candidate_retry" not in metadata
        if mode == "audit_limit":
            audit = metadata["pour_pick_ik_audit"]
            assert len(audit["groups"]) == 8
            assert audit["omitted_groups"] == 2 and audit["omitted_calls"] == 16
            assert audit["trace_complete"] is False and audit["truncated"] is True
        if mode == "degenerate":
            assert not ik and not calls
            assert "pour_pick_ik_audit" not in metadata
        if mode == "path_fail":
            assert metadata["pick_native_depth_search"]["extra_full_paths"] == 4
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("field", ["pick_id", "object_id"])
def test_bound_pour_inputs_reject_another_occurrence_before_native_capture(
    tmp_path, monkeypatch, field
):
    from dataclasses import replace

    engine, invocation, context, calls, ik, closing = inputs(tmp_path)
    provider = engine.planning_services.pour_grasp_projection_provider
    original = provider.prepare
    monkeypatch.setattr(
        provider,
        "prepare",
        lambda request, current: replace(
            original(request, current), **{field: "different"}
        ),
    )
    with pytest.raises(ValueError, match="exact Pick identity"):
        engine.plan(invocation, context)
    assert not closing.queries and not calls and not ik


def test_bound_pour_raw_inputs_and_numeric_projection_own_caller_arrays(tmp_path):
    from dataclasses import replace

    engine, invocation, context, calls, ik, closing = inputs(tmp_path)
    value = engine.planning_services.pour_grasp_projection_provider.prepare(
        engine.resolve(invocation), context
    )
    vertices = value.source_vertices.copy()
    returned = value.return_pose.clone()
    owned = replace(value, source_vertices=vertices, return_pose=returned)
    vertices.fill(0.99)
    returned.zero_()
    assert np.array_equal(owned.source_vertices, value.source_vertices)
    assert np.array_equal(
        owned.projection.source_vertices, value.projection.source_vertices
    )
    assert torch.equal(owned.recipe()["return_pose"], value.return_pose)
    assert not closing.queries and not calls and not ik
