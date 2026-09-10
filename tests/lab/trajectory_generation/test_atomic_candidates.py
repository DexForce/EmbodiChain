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

"""CPU source orchestration contracts; physics qualification is separate."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    JointPositionTarget,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
    TimedTrajectory,
)
from embodichain.lab.sim.motion.expansion import (
    GenerationSession,
    MotionSnapshot,
    SceneCase,
    TrajectoryGenerationJobCfg,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.integrations.atomic_candidates import (
    AtomicCandidateGenerationCfg,
    AtomicTrajectoryGenerator,
)
from embodichain.lab.trajectory_generation.replicas import SceneReplicaPool
from embodichain.toolkits.graspkit import GraspCandidateBatch


def _context(count: int = 4) -> PlanningContext:
    q = torch.zeros(count, 2)
    return PlanningContext(
        RobotObservation(0.0, q, q.clone(), root_pose=torch.eye(4).repeat(count, 1, 1)),
        TaskState.empty(batch_size=count, device=q.device),
        SceneSnapshot.empty(),
        torch.arange(count),
        control_dt=0.05,
    )


def _pool(context: PlanningContext) -> SceneReplicaPool:
    case = SceneCase("cube", "initial", "same-geometry", "pickup", "test-arm")
    return SceneReplicaPool(
        tuple(
            MotionSnapshot(
                case,
                ("arm", "finger"),
                context.robot.qpos[row],
                context.robot.qvel[row],
                context.robot.root_pose[row],
            )
            for row in range(context.batch_size)
        ),
        fixed_condition_signatures=("fixed-controller",) * context.batch_size,
    )


def _grasps(count: int = 8) -> GraspCandidateBatch:
    poses = torch.eye(4).repeat(1, count, 1, 1)
    poses[0, :, 0, 3] = torch.arange(1, count + 1).float() / 10
    return GraspCandidateBatch(
        poses, torch.arange(count).float()[None], torch.ones(1, count, dtype=torch.bool)
    )


class _Engine:
    def __init__(
        self, context, *, ik_fail=(), path_fail=(), crash=False, variable=False
    ):
        self.robot = SimpleNamespace(
            num_instances=context.batch_size,
            dof=2,
            joint_names=("arm", "finger"),
            mimic_ids=(),
            mimic_parents=(),
            mimic_multipliers=(),
            mimic_offsets=(),
            get_qpos_limits=lambda: torch.tensor([[-2.0, 2.0], [-2.0, 2.0]]).repeat(
                context.batch_size, 1, 1
            ),
        )
        self.ik_fail, self.path_fail = set(ik_fail), set(path_fail)
        self.crash, self.variable = crash, variable
        self.starts, self.waves, self.enumerations = [], [], 0

    def enumerate_candidates(
        self, invocation, context, *, grasp_candidates, active_mask
    ):
        self.enumerations += 1
        if self.crash:
            raise RuntimeError("backend exploded")
        values = grasp_candidates.poses[:, 0, 0, 3]
        ids = (values * 10).round().long().tolist()
        good = active_mask & torch.tensor([index not in self.ik_fail for index in ids])
        valid = good[:, None].expand(-1, 2).clone()
        batch = SimpleNamespace(
            valid_mask=valid,
            failure_stages=tuple(
                (None, None) if item else ("pre_grasp", "pre_grasp") for item in good
            ),
            failure_reasons=tuple(
                (None, None) if item else ("IK_NOT_FOUND", "IK_NOT_FOUND")
                for item in good
            ),
            select=lambda indices, active_mask: SimpleNamespace(
                success=good & active_mask,
                values=values,
                indices=indices,
            ),
        )
        return batch

    def compile(self, invocations, context, *, eligible_mask, candidate_selections):
        self.starts.append(context.robot.qpos.clone())
        self.waves.append(int(eligible_mask.sum()))
        selection = candidate_selections[invocations[-1].invocation_id](
            None, context, eligible_mask
        )
        success = selection.success & torch.tensor(
            [
                int(round(float(value) * 10)) not in self.path_fail
                for value in selection.values
            ]
        )
        horizon = 3 + (len(self.waves) if self.variable else 0)
        q = context.robot.qpos[:, None].repeat(1, horizon, 1)
        q[:, 1:, 0] = selection.values[:, None]
        q = torch.where(success[:, None, None], q, context.robot.qpos[:, None])
        trajectory = TimedTrajectory.from_uniform_step(
            q, env_ids=context.env_ids, step_dt=0.05
        )
        plan = SimpleNamespace(
            skill_id="pick_up",
            plan_success=success,
            joint_trajectory=trajectory,
            segments=(SimpleNamespace(name="approach", start=0, stop=horizon),),
            diagnostics=SimpleNamespace(metadata={}, failure=None),
        )
        return SimpleNamespace(
            plan_success=success, trajectory=trajectory, action_plans=(plan,)
        )


def _run(
    *,
    grasps=None,
    cfg=None,
    ik_fail=(),
    path_fail=(),
    crash=False,
    variable=False,
    session=None,
    validator=None,
):
    context = _context()
    engine = _Engine(
        context, ik_fail=ik_fail, path_fail=path_fail, crash=crash, variable=variable
    )
    source = AtomicTrajectoryGenerator(
        engine, replica_pool=_pool(context), validator=validator
    )
    invocation = SimpleNamespace(
        skill_id="pick_up",
        invocation_id="pick",
        motion_policy=SimpleNamespace(strategy="ik_interp"),
    )
    result = source.generate(
        invocations=(invocation,),
        context=context,
        candidate_inputs={("pick", "primary"): _grasps() if grasps is None else grasps},
        cfg=cfg or AtomicCandidateGenerationCfg(variants="original"),
        session=session,
    )
    return result, engine


def test_single_case_eight_grasps_compile_in_two_real_batch_waves():
    result, engine = _run()
    assert engine.waves == [4, 4]
    assert result.trajectories.positions.shape == (8, 3, 2)
    assert result.summary["status"] == "complete"
    assert all(
        identity.scene_case_id == "cube" for identity in result.trajectories.identities
    )
    assert (
        len({identity.candidate_id for identity in result.trajectories.identities}) == 8
    )
    assert all(torch.equal(start, torch.zeros(4, 2)) for start in engine.starts)
    assert not result.summary["physical_validation"]


def test_mixed_grasp_ik_and_path_failures_export_only_complete_successes():
    grasps = _grasps()
    poses = grasps.poses.clone()
    poses[0, 0, 0, 0] = float("nan")
    poses[0, 1, 0, 0] = 2.0
    invalid = GraspCandidateBatch(poses, grasps.costs, grasps.valid_mask)
    result, _ = _run(grasps=invalid, ik_fail=(3,))
    assert result.trajectories.positions.shape[0] == 5
    assert result.summary["status"] == "partial"
    assert result.summary["rejection_counts"]["IK_NOT_FOUND"] == 1
    assert sum(result.summary["rejection_counts"].values()) == 3
    assert torch.isfinite(result.trajectories.positions).all()
    torch.testing.assert_close(
        result.trajectories.positions[:, -1, 0], torch.arange(4, 9).float() / 10
    )
    assert len(result.branch_metadata) == len(result.planning_checks) == 5


def test_refill_keeps_candidates_not_selected_in_a_short_wave():
    cfg = AtomicCandidateGenerationCfg(variants="original", max_output_trajectories=1)
    result, engine = _run(cfg=cfg, ik_fail=(1, 2), path_fail=(3,))
    assert engine.waves == [1, 1, 1, 1]
    assert result.summary["output_count"] == 1
    assert result.trajectories.positions[0, -1, 0] == pytest.approx(0.4)


@pytest.mark.parametrize("empty", [True, False])
def test_empty_or_all_ik_failed_returns_a_legal_empty_batch(empty):
    result, engine = _run(
        grasps=_grasps(0) if empty else None, ik_fail=tuple(range(1, 9))
    )
    assert result.trajectories.positions.shape == (0, 0, 2)
    assert result.trajectories.valid_mask.shape == (0, 0)
    assert result.summary["status"] == "empty"
    if empty:
        assert engine.enumerations == 0
        assert result.summary["rejection_counts"] == {"NO_GRASP_CANDIDATES": 1}


def test_different_horizons_are_padded_with_last_sample_and_zero_time():
    result, _ = _run(variable=True)
    batch = result.trajectories
    assert batch.valid_length.tolist() == [4] * 4 + [5] * 4
    torch.testing.assert_close(batch.positions[:4, -1], batch.positions[:4, -2])
    assert (batch.dt[:4, -1] == 0).all()
    assert not batch.valid_mask[:4, -1].any()


def test_failure_audit_is_bounded_but_counts_are_not_truncated():
    cfg = AtomicCandidateGenerationCfg(variants="original", max_rejection_records=2)
    result, _ = _run(cfg=cfg, ik_fail=tuple(range(1, 9)))
    assert len(result.rejections) == 2
    assert result.summary["rejection_counts"]["IK_NOT_FOUND"] == 8


def test_backend_exception_is_not_a_normal_empty_result():
    with pytest.raises(RuntimeError, match="backend exploded"):
        _run(crash=True)


def test_shared_session_requires_real_collision_evidence_and_releases_failures():
    session = GenerationSession(TrajectoryGenerationJobCfg())
    validator = lambda row, snapshot: ValidationResult(
        (ValidationCheck("path_collision", "passed"),)
    )
    result, _ = _run(session=session, validator=validator, ik_fail=(1,))
    assert result.summary["output_count"] == 7
    snapshot = session.snapshot()
    assert snapshot["counts"]["planned_valid"] == 7
    assert snapshot["counts"]["committed"] == 0


def test_unsupported_options_and_unknown_fields_fail_before_planning():
    with pytest.raises(ValueError, match="unknown"):
        AtomicCandidateGenerationCfg.from_mapping({"typo": 1})
    with pytest.raises(ValueError, match="alternate"):
        AtomicCandidateGenerationCfg(ik_max_attempts_per_candidate=2)
    with pytest.raises(ValueError, match="env_rows"):
        AtomicCandidateGenerationCfg(batch_mode="candidate_buckets")


def test_duplicate_geometry_is_not_exported_as_two_grasps():
    grasps = _grasps(2)
    poses = grasps.poses.clone()
    poses[:, 1] = poses[:, 0]
    result, _ = _run(grasps=GraspCandidateBatch(poses, grasps.costs, grasps.valid_mask))
    assert result.summary["output_count"] == 1
    assert result.summary["rejection_counts"]["DUPLICATE_GRASP"] == 1


def test_shared_session_proposal_limit_returns_partial_without_overproposing():
    session = GenerationSession(
        TrajectoryGenerationJobCfg.from_mapping({"collection": {"max_proposals": 1}})
    )
    validator = lambda row, snapshot: ValidationResult(
        (ValidationCheck("path_collision", "passed"),)
    )
    result, engine = _run(session=session, validator=validator)
    assert engine.waves == [1]
    assert result.summary["output_count"] == 1
    assert result.summary["stop_reason"] == "proposal_budget_exhausted"
    assert session.remaining_proposals == 0


def test_ready_pool_backpressure_keeps_already_admitted_rows():
    session = GenerationSession(
        TrajectoryGenerationJobCfg.from_mapping(
            {"execution": {"ready_low_watermark": 0, "ready_high_watermark": 1}}
        )
    )
    validator = lambda row, snapshot: ValidationResult(
        (ValidationCheck("path_collision", "passed"),)
    )
    result, _ = _run(session=session, validator=validator)
    assert result.summary["output_count"] == 1
    assert result.summary["stop_reason"] == "ready_budget_exhausted"
    assert session.snapshot()["counts"]["ready"] == 1
    assert all(state != "proposed" for _, state, _ in session.snapshot()["audit"])


def test_source_deadline_applies_to_a_shared_session(monkeypatch):
    from embodichain.lab.trajectory_generation.integrations import atomic_candidates

    now = [0.0]
    monkeypatch.setattr(
        atomic_candidates, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    original = _Engine.compile

    def delayed_compile(self, *args, **kwargs):
        compiled = original(self, *args, **kwargs)
        now[0] = 2.0
        return compiled

    monkeypatch.setattr(_Engine, "compile", delayed_compile)
    validator = lambda row, snapshot: ValidationResult(
        (ValidationCheck("path_collision", "passed"),)
    )
    result, engine = _run(
        session=GenerationSession(TrajectoryGenerationJobCfg()),
        validator=validator,
        cfg=AtomicCandidateGenerationCfg(variants="original", max_wall_time_s=1.0),
    )
    assert engine.waves == [4]
    assert result.summary["output_count"] == 4
    assert result.summary["stop_reason"] == "wall_time_exhausted"


def test_grasp_limit_selects_by_cost_after_invalid_filtering():
    grasps = _grasps(3)
    poses = grasps.poses.clone()
    poses[0, 0, 0, 0] = float("nan")
    ordered = GraspCandidateBatch(
        poses, torch.tensor([[0.0, 8.0, 1.0]]), grasps.valid_mask
    )
    result, _ = _run(
        grasps=ordered,
        cfg=AtomicCandidateGenerationCfg(
            variants="original", max_grasps_per_case=1, max_output_trajectories=1
        ),
    )
    assert result.trajectories.positions[0, -1, 0] == pytest.approx(0.3)


class _PrefixEngine(_Engine):
    """Model compile's alive projection while giving each prefix a distinct fault."""

    def __init__(self, context, *, all_failed=False, structured_failure=False):
        super().__init__(context)
        self.all_failed = all_failed
        self.structured_failure = structured_failure
        self.enumeration_masks = []

    def enumerate_candidates(
        self, invocation, context, *, grasp_candidates, active_mask
    ):
        self.enumeration_masks.append(active_mask.clone())
        return super().enumerate_candidates(
            invocation,
            context,
            grasp_candidates=grasp_candidates,
            active_mask=active_mask,
        )

    def compile(self, invocations, context, *, eligible_mask, candidate_selections):
        projected, alive = context, eligible_mask.clone()
        plans, trajectories = [], []
        for index, invocation in enumerate(invocations[:-1]):
            success = alive.clone()
            if self.all_failed:
                success[:] = False
            else:
                success[index] = False
            qpos = projected.robot.qpos[:, None].repeat(1, 2, 1)
            qpos[success, -1, 0] += 0.01
            trajectory = TimedTrajectory.from_uniform_step(
                qpos, env_ids=context.env_ids, step_dt=context.require_control_dt()
            )
            plans.append(
                SimpleNamespace(
                    skill_id=invocation.skill_id,
                    plan_success=success,
                    joint_trajectory=trajectory,
                    segments=(SimpleNamespace(name="prefix", start=0, stop=2),),
                    diagnostics=SimpleNamespace(
                        metadata=(
                            {}
                            if self.structured_failure
                            else {
                                "candidate_failure_reasons": tuple(
                                    (
                                        f"prefix_{index}:PREFIX_PATH_FAILED"
                                        if alive[row] and not success[row]
                                        else None
                                    )
                                    for row in range(context.batch_size)
                                )
                            }
                        ),
                        failure=(
                            SimpleNamespace(code=f"PREFIX_{index}_NO_ROUTE")
                            if self.structured_failure
                            else None
                        ),
                    ),
                )
            )
            trajectories.append(trajectory)
            projected = projected.project(qpos=qpos[:, -1], task=projected.task)
            alive = success
            if not alive.any():
                break
        if alive.any():
            suffix = super().compile(
                invocations[-1:],
                projected,
                eligible_mask=alive,
                candidate_selections=candidate_selections,
            )
            plans.extend(suffix.action_plans)
            trajectories.append(suffix.trajectory)
            alive &= suffix.plan_success
        return SimpleNamespace(
            plan_success=alive,
            trajectory=TimedTrajectory.concatenate(trajectories, empty_like=context),
            action_plans=tuple(plans),
        )


def _prefix_invocation(skill_id, invocation_id, *, control_part="arm", joint_ids=(0,)):
    target = JointPositionTarget(control_part, joint_ids)
    binding = SimpleNamespace(
        endpoint=lambda *_: SimpleNamespace(require_target=lambda _: target)
    )
    return SimpleNamespace(
        skill_id=skill_id,
        invocation_id=invocation_id,
        binding=binding,
        motion_policy=SimpleNamespace(strategy="ik_interp"),
    )


@pytest.mark.parametrize("all_failed", (False, True))
@pytest.mark.parametrize("structured_failure", (False, True))
def test_prefix_failure_is_attributed_before_inactive_pickup_diagnostics(
    all_failed, structured_failure
):
    context = _context()
    engine = _PrefixEngine(
        context, all_failed=all_failed, structured_failure=structured_failure
    )
    source = AtomicTrajectoryGenerator(engine, replica_pool=_pool(context))
    prefixes = (
        _prefix_invocation("move_joints", "prefix-a"),
        _prefix_invocation("move_end_effector", "prefix-b"),
    )
    result = source.generate(
        invocations=(*prefixes, _prefix_invocation("pick_up", "pick")),
        context=context,
        candidate_inputs={("pick", "primary"): _grasps(3)},
        cfg=AtomicCandidateGenerationCfg(
            variants="original", max_output_trajectories=3
        ),
    )
    assert result.summary["output_count"] == (0 if all_failed else 1)
    expected_invocations = ["prefix-a"] * 3 if all_failed else ["prefix-a", "prefix-b"]
    assert [item.invocation_id for item in result.rejections] == expected_invocations
    if structured_failure:
        assert {item.stage for item in result.rejections} == {"path"}
        assert [item.reason_code for item in result.rejections] == (
            ["PREFIX_0_NO_ROUTE"] * 3
            if all_failed
            else ["PREFIX_0_NO_ROUTE", "PREFIX_1_NO_ROUTE"]
        )
    else:
        assert [item.stage for item in result.rejections] == (
            ["prefix_0"] * 3 if all_failed else ["prefix_0", "prefix_1"]
        )
        assert {item.reason_code for item in result.rejections} == {
            "PREFIX_PATH_FAILED"
        }
    assert "IK_NOT_FOUND" not in result.summary["rejection_counts"]
    if all_failed:
        assert engine.enumerations == 0
    else:
        assert len(engine.enumeration_masks) == 1
        assert engine.enumeration_masks[0].tolist() == [False, False, True, False]
        assert len(result.rejections) == 2  # No audit entry for the unused fourth slot.


@pytest.mark.parametrize(
    ("control_part", "joint_ids"),
    (("torso", (0,)), ("arm", (0, 1))),
)
def test_source_rejects_prefixes_outside_pickup_arm_before_compilation(
    control_part, joint_ids
):
    context = _context()
    engine = _PrefixEngine(context)
    source = AtomicTrajectoryGenerator(engine, replica_pool=_pool(context))
    prefix = _prefix_invocation(
        "move_joints", "prefix", control_part=control_part, joint_ids=joint_ids
    )
    with pytest.raises(NotImplementedError, match="same PickUp arm"):
        source.generate(
            invocations=(prefix, _prefix_invocation("pick_up", "pick")),
            context=context,
            candidate_inputs={("pick", "primary"): _grasps(1)},
        )
    assert not engine.starts and engine.enumerations == 0


@pytest.mark.parametrize("initial_finger", (0.0, 0.05))
def test_mimic_export_rejects_changed_frozen_initial_qpos(initial_finger):
    context = _context()
    qpos = context.robot.qpos.clone()
    qpos[:, 1] = initial_finger
    context = context.project(qpos=qpos, task=context.task)
    engine = _Engine(context)
    engine.robot.mimic_ids = (1,)
    engine.robot.mimic_parents = (0,)
    engine.robot.mimic_multipliers = (0.5,)
    engine.robot.mimic_offsets = (0.05,)
    source = AtomicTrajectoryGenerator(engine, replica_pool=_pool(context))
    result = source.generate(
        invocations=(_prefix_invocation("pick_up", "pick"),),
        context=context,
        candidate_inputs={("pick", "primary"): _grasps(1)},
        cfg=AtomicCandidateGenerationCfg(variants="original"),
    )
    if initial_finger == 0.0:
        assert result.summary["output_count"] == 0
        assert result.summary["rejection_counts"] == {"INITIAL_QPOS_MISMATCH": 1}
        assert len(result.rejections) == 1
        assert result.rejections[0].stage == "export"
    else:
        assert result.summary["output_count"] == 1
        torch.testing.assert_close(result.trajectories.positions[0, 0], qpos[0])
        torch.testing.assert_close(
            result.trajectories.positions[0, :, 1],
            result.trajectories.positions[0, :, 0] * 0.5 + 0.05,
        )
