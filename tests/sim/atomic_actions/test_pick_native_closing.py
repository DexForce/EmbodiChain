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

"""Owned native closing lifetimes through the actual shared Atomic engine."""

from __future__ import annotations
from dataclasses import replace
import hashlib, json
from pathlib import Path
import pytest
import torch
import numpy as np
from embodichain.lab.sim.atomic_actions.source_native_contact import (
    NativeClosingQuery,
    NativeContactPart,
    NativeQueryScope,
)
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
from embodichain.lab.sim.atomic_actions import RecoveryPolicy, TimedTrajectory
from .test_ordinary_pick import _inputs
from ._ordinary_pick_fixture import closing_query_parts, stable_closing_report


class QueryProvider:
    def __init__(self, *, stale=False):
        self.queries = []
        self.cfgs = []
        self.stale = stale

    def prepare(self, request, context, *, hand_geometry_cfg):
        parts, target = closing_query_parts(NativeContactPart)
        query = NativeClosingQuery(
            parts,
            target,
            0.0,
            scope=NativeQueryScope(
                f"query-{len(self.queries)}",
                context.robot.timestamp + (1 if self.stale else 0),
                context.scene.version,
            ),
            revision=request.skill_options.native_closing_revision,
            target_contact_offset=0.002,
            **(
                {"gravity_in_object": np.array([0.0, 0.0, -1.0])}
                if request.skill_options.native_closing_revision == 9
                else {}
            ),
        )
        if query.revision >= 7:
            query.identity = {"simulation_uid": "source"}
        query.object_pose_at_capture = context.scene.entities["object"].pose.clone()
        self.cfgs.append(hand_geometry_cfg)
        self.queries.append(query)
        return query


def native_inputs(
    tmp_path,
    monkeypatch,
    *,
    count=65,
    stale=False,
    provider_type=QueryProvider,
    generator=None,
):
    pytest.importorskip(
        "fcl",
        reason="Native closing source qualification needs planning-geometry/python-fcl.",
    )
    cfg = RobotHandGeometryCfg(
        motion_part="arm",
        hand_part="hand",
        hand_root_link="hand_root",
        pad_link_names=("pad_a", "pad_b"),
        open_qpos=(0.04, 0.04),
        closed_qpos=(0.0, 0.0),
        sample_count=count,
        body_max_edge=0.1,
        pad_max_edge=0.01,
        max_subdivisions=1,
    )
    provider = provider_type(stale=stale)
    engine, invocation, context, calls, ik = _inputs(
        tmp_path,
        monkeypatch,
        engine_kwargs={
            "hand_geometry_cfgs": {"hand": cfg},
            "pick_closing_query_provider": provider,
            **(
                {"grasp_pose_generators": {"hand": generator}}
                if generator is not None
                else {}
            ),
        },
    )
    invocation = replace(
        invocation,
        skill_options=replace(invocation.skill_options, native_closing_revision=4),
    )
    return engine, invocation, context, calls, ik, provider


def test_native_closing_pick_matches_source_plan_and_query_report(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path, monkeypatch
    )
    torch.manual_seed(147)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    reference = json.loads(
        (
            Path(__file__).parent / "data/native_closing_pick_source_reference.json"
        ).read_text()
    )
    actual = dict(
        calls=calls,
        ik=ik,
        success=plan.plan_success.tolist(),
        positions_sha256=digest(plan.joint_trajectory.positions),
        dt_sha256=digest(plan.joint_trajectory.dt),
        segments=[[s.name, s.start, s.stop] for s in plan.segments],
        metadata=stable_closing_report(plan.diagnostics.metadata),
        rng=digest(torch.get_rng_state()),
    )
    assert actual == {key: reference[key] for key in actual}
    assert len(provider.queries) == 1 and provider.queries[0].closed
    assert provider.cfgs[0].sample_count == 65


@pytest.mark.parametrize("case", ["budget", "scope", "exception"])
def test_native_query_preflight_and_exception_cleanup(tmp_path, monkeypatch, case):
    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path,
        monkeypatch,
        count=33 if case == "budget" else 65,
        stale=case == "scope",
    )
    if case == "exception":
        engine.robot.compute_batch_ik.side_effect = RuntimeError("solver unavailable")
    with pytest.raises(
        (ValueError, RuntimeError), match="65|observation scope|solver unavailable"
    ):
        engine.plan(invocation, context)
    if case == "budget":
        assert provider.queries == [] and calls == []
    else:
        assert len(provider.queries) == 1 and provider.queries[0].closed


def test_transformed_native_closing_uses_fresh_query_and_preserves_rejected_samples(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path, monkeypatch
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options, require_qualified_approach=False
        ),
        recovery_policy=RecoveryPolicy(),
    )
    captured = []

    def transform(request, current, plan):
        positions = plan.joint_trajectory.positions.clone()
        close = next(s for s in plan.segments if s.name == "close")
        positions[:, close.start : close.stop, 3:] = 0.02
        captured.append(positions.clone())
        return engine.rebuild_plan_from_trajectory(
            request,
            current,
            plan,
            TimedTrajectory.from_positions(
                positions,
                dt=plan.joint_trajectory.dt,
                env_ids=plan.joint_trajectory.env_ids,
            ),
        )

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert len(provider.queries) == 2 and all(q.closed for q in provider.queries)
    assert len({q.scope.query_id for q in provider.queries}) == 2
    assert not plan.plan_success.any()
    assert torch.equal(plan.rejected_candidate.positions, captured[0])
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)


def test_invalid_pick_options_reject_before_pour_provider_preparation(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    called = []
    provider = SimpleNamespace(prepare=lambda request, context: called.append(request))
    engine, invocation, context, calls, ik = _inputs(
        tmp_path,
        monkeypatch,
        engine_kwargs={"pour_grasp_projection_provider": provider},
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options, fixed_object_to_eef=torch.eye(4)
        ),
    )
    with pytest.raises(ValueError, match="unconstrained"):
        engine.plan(invocation, context)
    assert not called and not calls and not ik


@pytest.mark.parametrize("revision", [0, 10, True])
def test_native_closing_revision_decoder_rejects_unimplemented_search_modes(revision):
    from embodichain.lab.task_program.integrations.configured import (
        _decode_action_options,
    )

    with pytest.raises((TypeError, ValueError), match="native_closing_revision"):
        _decode_action_options(
            {"kind": "pick_up", "native_closing_revision": revision}, path="options"
        )


@pytest.mark.parametrize(
    "option",
    [
        {"native_closing_revision": 4},
        {"native_scene_trial": True},
        {"adaptive_grasp": "ordinary"},
        {"full_opening_clearance": True},
        {"require_qualified_approach": True},
        {"linear_segments": True},
        {"qualify_fixed_grasp": True},
        {"constrained_grasp": True},
        {"measured_preshape": True},
        {"qualify_downstream_targets": False},
    ],
)
def test_axis_align_cannot_silently_ignore_native_pick_requirement(option):
    from embodichain.lab.sim.atomic_actions import AxisAlignOptions

    with pytest.raises(ValueError, match="Pick-only"):
        AxisAlignOptions(**option)


def test_axis_align_generated_signature_resolves_inherited_pick_types():
    from typing import get_type_hints
    from embodichain.lab.sim.atomic_actions import AxisAlignOptions, PickUpOptions

    assert get_type_hints(AxisAlignOptions.__init__)["adaptive_grasp"] == (
        get_type_hints(PickUpOptions.__init__)["adaptive_grasp"]
    )


@pytest.mark.parametrize("exhaust", [False, True])
def test_native_depth_reuses_source_pool_rng_and_shared_four_path_budget(
    tmp_path, monkeypatch, exhaust
):
    from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport

    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path, monkeypatch
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options,
            native_closing_revision=6,
            full_opening_clearance=False,
        ),
    )

    def scene(positions, **kwargs):
        valid = not exhaust and bool((positions[:, 2] > 0.102).all())
        return (
            MotionCollisionReport(True, frames=len(positions))
            if valid
            else MotionCollisionReport(
                False, frame=0, kind="scene", pairs=(("hand", "obstacle"),)
            )
        )

    engine.planning_services.free_motion_query.inspect = scene
    torch.manual_seed(147)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    reference = json.loads(
        (
            Path(__file__).parent
            / "data"
            / (
                "native_depth_exhaust_reference.json"
                if exhaust
                else "native_depth_accept_reference.json"
            )
        ).read_text()
    )
    actual = dict(
        calls=calls,
        ik=ik,
        success=plan.plan_success.tolist(),
        positions_sha256=digest(plan.joint_trajectory.positions),
        dt_sha256=digest(plan.joint_trajectory.dt),
        segments=[[s.name, s.start, s.stop] for s in plan.segments],
        metadata=stable_closing_report(plan.diagnostics.metadata),
        rng=digest(torch.get_rng_state()),
    )
    assert actual == {key: reference[key] for key in actual}
    assert provider.queries[0].closed
    assert plan.diagnostics.metadata["pick_native_depth_search"][
        "extra_full_paths"
    ] == (4 if exhaust else 1)
    if exhaust:
        assert not plan.commands.frames


def test_native_depth_requires_its_declared_recovery_before_prepare(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path, monkeypatch
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options,
            native_closing_revision=6,
            full_opening_clearance=False,
        ),
        recovery_policy=RecoveryPolicy(),
    )
    with pytest.raises(ValueError, match="revisions 6-9 require"):
        engine.plan(invocation, context)
    assert not provider.queries and not calls and not ik


@pytest.mark.parametrize("exhaust", [False, True])
def test_native_precontact_preserves_source_filtering_depth_order_and_commands(
    tmp_path, monkeypatch, exhaust
):
    from types import SimpleNamespace
    from embodichain.lab.sim.motion import mesh_collision
    from embodichain.lab.sim.motion.mesh_collision import SimulationFreeMotionQuery

    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path, monkeypatch
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options,
            native_closing_revision=7,
            full_opening_clearance=False,
        ),
    )
    prepare = provider.prepare

    def scoped(request, current, *, hand_geometry_cfg):
        query = prepare(request, current, hand_geometry_cfg=hand_geometry_cfg)
        query.identity = {"simulation_uid": "source"}
        return query

    provider.prepare = scoped
    world = SimpleNamespace(
        num_envs=1,
        get_rigid_object_uid_list=lambda: ["source", "obstacle"],
        get_rigid_object=lambda uid: uid,
        get_articulation_uid_list=lambda: [],
        get_articulation=lambda uid: uid,
    )
    engine.planning_services._free_motion_query = SimulationFreeMotionQuery(
        engine.robot, world
    )

    def make_checker(robot, selected, part, *, include_scene):
        def inspect(positions):
            valid = ("source" in selected.get_rigid_object_uid_list()) or (
                not exhaust and bool((positions[:, 2] > 0.102).all())
            )
            return (
                {"valid": True, "frames": len(positions)}
                if valid
                else {
                    "valid": False,
                    "frame": 0,
                    "kind": "scene",
                    "pairs": [["hand", "obstacle"]],
                }
            )

        return inspect

    monkeypatch.setattr(mesh_collision, "_make_checker", make_checker)
    torch.manual_seed(147)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    reference = json.loads(
        (
            Path(__file__).parent
            / "data"
            / (
                "native_precontact_exhaust_reference.json"
                if exhaust
                else "native_precontact_accept_reference.json"
            )
        ).read_text()
    )

    def source_report(value):
        if isinstance(value, dict):
            return {
                key: source_report(item)
                for key, item in value.items()
                if key != "excluded_rigid_uid"
                and not (
                    key == "scope"
                    and value.get("valid") is False
                    and item == "discrete_precontact_meshes"
                )
            }
        if isinstance(value, (list, tuple)):
            return [source_report(item) for item in value]
        return (
            "discrete_free_motion_meshes"
            if value == "discrete_precontact_meshes"
            else value
        )

    actual = dict(
        calls=calls,
        ik=ik,
        success=plan.plan_success.tolist(),
        positions_sha256=digest(plan.joint_trajectory.positions),
        dt_sha256=digest(plan.joint_trajectory.dt),
        segments=[[s.name, s.start, s.stop] for s in plan.segments],
        metadata=source_report(stable_closing_report(plan.diagnostics.metadata)),
        rng=digest(torch.get_rng_state()),
    )
    assert actual == {key: reference[key] for key in actual}
    assert provider.queries[0].closed
    assert plan.diagnostics.metadata["pick_closing_scene_search"]


def test_native_precontact_requires_capability_before_geometry_prepare(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path, monkeypatch
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options,
            native_closing_revision=7,
            full_opening_clearance=False,
        ),
    )
    with pytest.raises(ValueError, match="precontact scene query capability"):
        engine.plan(invocation, context)
    assert not provider.queries and not calls and not ik


class RigidQueryProvider(QueryProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scenes = []

    def prepare_rigid_scene(self, query):
        from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
            NativeRigidRestScene,
        )

        assert query.rigid_scene_screen is None
        scene = NativeRigidRestScene(
            query.parts, [], scope=query.scope, collision_factory=_ConvexCollection
        )
        query.rigid_scene_screen = scene
        self.scenes.append(scene)
        return scene


def normalize_source_scope(value):
    if isinstance(value, dict):
        return {
            key: normalize_source_scope(item)
            for key, item in value.items()
            if key != "excluded_rigid_uid"
            and not (
                key == "scope"
                and value.get("valid") is False
                and item == "discrete_precontact_meshes"
            )
        }
    if isinstance(value, (list, tuple)):
        return [normalize_source_scope(item) for item in value]
    return (
        "discrete_free_motion_meshes"
        if value == "discrete_precontact_meshes"
        else value
    )


@pytest.mark.parametrize(
    "revision,purpose,mode",
    [
        (8, "ordinary", "accept"),
        (9, "ordinary", "accept"),
        (8, "handover_source", "accept"),
        (8, "handover_source", "raw"),
        (8, "handover_source", "raw_success"),
        (8, "handover_source", "shared_raw"),
        (8, "handover_source", "shared_raw_success"),
        (8, "handover_source", "exhaust"),
    ],
)
def test_native_proposals_preserve_cap_chunks_lazy_expansion_rng_and_rest_queries(
    tmp_path, monkeypatch, revision, purpose, mode
):
    from types import SimpleNamespace
    from ._ordinary_pick_fixture import native_candidates, native_end_candidates
    from embodichain.lab.sim.motion import mesh_collision
    from embodichain.lab.sim.motion.mesh_collision import SimulationFreeMotionQuery

    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path,
        monkeypatch,
        provider_type=RigidQueryProvider,
        generator=(
            native_end_candidates()
            if purpose == "handover_source"
            else native_candidates()
        ),
    )
    case = mode.removeprefix("shared_")
    if purpose == "handover_source":
        engine.robot.get_solver("arm").supports_continuous_batch_ik = False
    if case in {"raw", "raw_success"}:
        original_batch = engine.robot.compute_batch_ik.side_effect

        def fail_batch(*args, **kwargs):
            valid, positions = original_batch(*args, **kwargs)
            return (
                valid
                if case == "raw_success" and positions.shape[1] == 1
                else torch.zeros_like(valid)
            ), positions

        engine.robot.compute_batch_ik.side_effect = fail_batch
        if case == "raw":
            original_single = engine.robot.compute_ik.side_effect

            def fail_single(*args, **kwargs):
                valid, positions = original_single(*args, **kwargs)
                return torch.zeros_like(valid), positions

            engine.robot.compute_ik.side_effect = fail_single
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options,
            native_closing_revision=revision,
            adaptive_grasp=purpose,
            require_qualified_approach=purpose == "ordinary",
            full_opening_clearance=False,
        ),
    )
    if mode.startswith("shared_"):
        invocation = replace(
            invocation,
            motion_policy=replace(
                invocation.motion_policy,
                interpolation_mode="adaptive",
                ik_seed_mode="batched",
                acceleration_limit=0.5,
            ),
        )
    world = SimpleNamespace(
        num_envs=1,
        get_rigid_object_uid_list=lambda: ["source", "obstacle"],
        get_rigid_object=lambda uid: uid,
        get_articulation_uid_list=lambda: [],
        get_articulation=lambda uid: uid,
    )
    engine.planning_services._free_motion_query = SimulationFreeMotionQuery(
        engine.robot, world
    )

    def make_checker(robot, selected, part, *, include_scene):
        def inspect(positions):
            valid = mode != "exhaust" and bool((positions[:, 2] > 0.102).all())
            return (
                {"valid": True, "frames": len(positions)}
                if valid
                else {
                    "valid": False,
                    "frame": 0,
                    "kind": "scene",
                    "pairs": [["hand", "obstacle"]],
                }
            )

        return inspect

    monkeypatch.setattr(mesh_collision, "_make_checker", make_checker)
    torch.manual_seed(147)
    plan = engine.plan(invocation, context)
    tensor_digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    digest = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected = json.loads(
        (
            Path(__file__).parent
            / "data"
            / (
                (
                    "native_handover_source_reference.json"
                    if mode == "accept"
                    else f"native_handover_source_{mode}_reference.json"
                )
                if purpose == "handover_source"
                else f"native_proposals_{revision}_reference.json"
            )
        ).read_text()
    )
    metadata = normalize_source_scope(stable_closing_report(plan.diagnostics.metadata))
    actual = {
        "calls_sha256": digest(calls),
        "ik_sha256": digest(ik),
        "metadata_sha256": digest(metadata),
        "positions_sha256": tensor_digest(plan.joint_trajectory.positions),
        "dt_sha256": tensor_digest(plan.joint_trajectory.dt),
        "rng": tensor_digest(torch.get_rng_state()),
        "success": plan.plan_success.tolist(),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
    }
    assert actual == {key: expected[key] for key in actual}
    assert len(provider.scenes) == (1 if case in {"accept", "raw_success"} else 0)
    assert all(scene.closed for scene in provider.scenes)
    assert provider.queries[0].closed
    caps = [
        row
        for row in metadata.get("pick_closing_scene_search", [])
        if row["stage"] == "native_candidate_limit"
    ]
    if purpose == "handover_source":
        assert all(not row["first_feasible_batch"] for row in caps)
        return
    assert any(row["retained"] == 50 and row["first_feasible_batch"] for row in caps)
    assert any(
        row["retained"] == 80 and not row["first_feasible_batch"] for row in caps
    )


@pytest.mark.parametrize("failure", ["raise", "wrong_scope"])
def test_native_rigid_scene_partial_failure_closes_only_owned_queries(
    tmp_path, monkeypatch, failure
):
    from ._ordinary_pick_fixture import native_candidates
    from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
    from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
        NativeRigidRestScene,
    )

    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path,
        monkeypatch,
        provider_type=RigidQueryProvider,
        generator=native_candidates(),
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options,
            native_closing_revision=8,
            full_opening_clearance=False,
        ),
    )
    engine.planning_services.free_motion_query.inspect_precontact_closure = (
        lambda positions, **kw: MotionCollisionReport(
            True,
            frames=3,
            scope="discrete_precontact_meshes",
            excluded_rigid_uid="source",
        )
    )
    created = []

    def invalid(query):
        if failure == "raise":
            raise RuntimeError("capture failed")
        scene = NativeRigidRestScene(
            query.parts,
            [],
            collision_factory=_ConvexCollection,
            scope=NativeQueryScope(
                "other", query.scope.observation_timestamp, query.scope.scene_version
            ),
        )
        query.rigid_scene_screen = scene
        created.append(scene)
        return scene

    provider.prepare_rigid_scene = invalid
    with pytest.raises(
        (ValueError, RuntimeError), match="capture failed|exact closing-query scope"
    ):
        engine.plan(invocation, context)
    assert provider.queries and provider.queries[0].closed
    assert all(scene.closed for scene in created)


@pytest.mark.parametrize("revision", [7, 9])
def test_native_handover_source_does_not_enable_other_depth_revisions(
    tmp_path, monkeypatch, revision
):
    engine, invocation, context, calls, ik, provider = native_inputs(
        tmp_path, monkeypatch
    )
    invocation = replace(
        invocation,
        skill_options=replace(
            invocation.skill_options,
            adaptive_grasp="handover_source",
            native_closing_revision=revision,
            require_qualified_approach=False,
            full_opening_clearance=False,
        ),
    )
    with pytest.raises(ValueError, match="qualified approach"):
        engine.plan(invocation, context)
    assert not provider.queries and not calls and not ik
