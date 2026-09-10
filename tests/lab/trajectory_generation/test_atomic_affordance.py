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

"""CPU contracts for distinct-raw-grasp planning over physical environment rows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
    TimedTrajectory,
)
from embodichain.lab.trajectory_generation.integrations.atomic_affordance import (
    plan_affordance_batch,
)
from embodichain.toolkits.graspkit import GraspCandidateBatch

CONTROL_DT = 0.05


def _context(rows: int) -> PlanningContext:
    qpos = torch.zeros(rows, 2)
    return PlanningContext(
        RobotObservation(
            0.0, qpos, qpos.clone(), root_pose=torch.eye(4).repeat(rows, 1, 1)
        ),
        TaskState.empty(batch_size=rows, device=qpos.device),
        SceneSnapshot.empty(),
        torch.arange(rows) + 5,
        control_dt=CONTROL_DT,
    )


def _grasps(rows: int, count: int, *, ids=None, valid=None) -> GraspCandidateBatch:
    poses = torch.eye(4).repeat(rows, count, 1, 1)
    poses[:, :, 0, 3] = torch.arange(count).float()[None] / 10 + 0.1
    return GraspCandidateBatch(
        poses=poses,
        costs=torch.arange(count).float()[None].repeat(rows, 1),
        valid_mask=(
            torch.ones(rows, count, dtype=torch.bool) if valid is None else valid
        ),
        grasp_ids=(
            tuple(
                tuple(f"raw-{column}" for column in range(count)) for _ in range(rows)
            )
            if ids is None
            else ids
        ),
    )


class _Engine:
    def __init__(
        self,
        context,
        *,
        variants=2,
        ik_fail=(),
        path_fail=(),
        corrupt=None,
        crash=False,
        variable=False,
    ):
        self.robot = SimpleNamespace(
            num_instances=context.batch_size,
            dof=2,
            mimic_ids=(),
            mimic_parents=(),
            mimic_multipliers=(),
            mimic_offsets=(),
            get_qpos_limits=lambda: torch.tensor([[-2.0, 2.0], [-2.0, 2.0]]).repeat(
                context.batch_size, 1, 1
            ),
        )
        self.variants = variants
        self.ik_fail = set(ik_fail)
        self.path_fail = set(path_fail)
        self.corrupt = corrupt
        self.crash = crash
        self.variable = variable
        self.calls = []
        self.enumerations = 0

    def enumerate_candidates(
        self, invocation, context, *, grasp_candidates, active_mask
    ):
        self.enumerations += 1
        if self.crash:
            raise RuntimeError("backend exploded")
        self.raw = grasp_candidates
        valid = (
            grasp_candidates.valid_mask.repeat_interleave(self.variants, dim=1)
            & active_mask[:, None]
        )
        for row, index in self.ik_fail:
            valid[row, index] = False
        identities = tuple(
            tuple(identity for identity in row for _ in range(self.variants))
            for row in grasp_candidates.grasp_ids
        )
        return SimpleNamespace(
            env_ids=context.env_ids,
            valid_mask=valid,
            costs=grasp_candidates.costs.repeat_interleave(self.variants, dim=1),
            grasp_ids=identities,
            failure_stages=tuple(
                tuple(None if item else "pre_grasp" for item in row) for row in valid
            ),
            failure_reasons=tuple(
                tuple(None if item else "IK_NOT_FOUND" for item in row) for row in valid
            ),
            select=lambda indices, active_mask: SimpleNamespace(
                indices=indices, active_mask=active_mask
            ),
        )

    def compile(self, invocations, context, *, eligible_mask, candidate_selections):
        choice = candidate_selections[invocations[0].invocation_id]
        indices = choice.indices
        self.calls.append(
            (eligible_mask.clone(), indices.clone(), context.robot.qpos.clone())
        )
        success = eligible_mask.clone()
        for row, index in self.path_fail:
            if indices[row] == index:
                success[row] = False
        horizon = 3 + len(self.calls) if self.variable else 3
        positions = context.robot.qpos[:, None].repeat(1, horizon, 1)
        for row in torch.nonzero(eligible_mask).flatten().tolist():
            positions[row, 1:, 0] = self.raw.poses[
                row, int(indices[row]) // self.variants, 0, 3
            ]
        trajectory = TimedTrajectory.from_uniform_step(
            positions, env_ids=context.env_ids, step_dt=CONTROL_DT
        )
        if self.corrupt is not None:
            self.corrupt(trajectory, indices)
        plan = SimpleNamespace(
            plan_success=success, diagnostics=SimpleNamespace(metadata={}, failure=None)
        )
        return SimpleNamespace(
            plan_success=success, trajectory=trajectory, action_plans=(plan,)
        )


def _run(
    *,
    rows=3,
    count=4,
    grasps=None,
    active=None,
    max_rounds=32,
    skill="pick_up",
    **kwargs,
):
    context = _context(rows)
    engine = _Engine(context, **kwargs)
    invocation = SimpleNamespace(
        skill_id=skill,
        invocation_id="grasp",
        motion_policy=SimpleNamespace(strategy="ik_interp"),
    )
    result = plan_affordance_batch(
        engine,
        invocation,
        context,
        _grasps(rows, count) if grasps is None else grasps,
        active_mask=active,
        max_rounds=max_rounds,
    )
    return result, engine


def test_distinct_raw_grasps_not_symmetric_rolls_fill_real_environment_rows():
    result, engine = _run()
    assert len(set(result.grasp_ids)) == 3
    assert result.trajectory.positions.shape == (3, 3, 2)
    assert result.success_mask.all()
    assert engine.enumerations == 1
    assert len(engine.calls) == 1
    assert result.compact_env_ids.tolist() == [5, 6, 7]
    assert not result.summary["physical_validation"]


def test_matching_preserves_constrained_row_instead_of_greedy_starvation():
    # Row 1 can use only raw 0; a greedy row-0-first allocation would lose it.
    result, _ = _run(rows=2, count=2, ik_fail=((1, 2), (1, 3)))
    assert result.grasp_ids == ("raw-1", "raw-0")
    assert result.success_mask.all()


def test_single_raw_with_two_rolls_produces_only_one_success_and_safe_hold():
    result, _ = _run(rows=3, count=1)
    assert int(result.success_mask.sum()) == 1
    assert result.summary["status"] == "partial"
    assert torch.count_nonzero(result.trajectory.positions[~result.success_mask]) == 0
    assert result.candidate_indices[~result.success_mask].tolist() == [-1, -1]
    assert result.compact_positions.shape[0] == 1


def test_path_failure_tries_other_roll_then_next_raw_without_reactivating_success():
    result, engine = _run(rows=2, count=3, path_fail=((0, 2), (0, 3)), variable=True)
    # Maximum matching assigns row 0 raw 1 and row 1 raw 0 initially.
    assert result.success_mask.all()
    assert result.grasp_ids == ("raw-2", "raw-0")
    assert [mask.tolist() for mask, _, _ in engine.calls] == [
        [True, True],
        [True, False],
        [True, False],
    ]
    assert result.valid_length.tolist() == [6, 4]
    assert torch.equal(
        result.trajectory.positions[1, 3:],
        result.trajectory.positions[1, 3].expand(3, -1),
    )
    assert all(torch.equal(start, torch.zeros(2, 2)) for _, _, start in engine.calls)
    assert torch.allclose(result.dt[:, 1:], torch.full((2, 5), CONTROL_DT))


def test_round_budget_returns_partial_result_without_duplicate_fill():
    result, engine = _run(rows=2, count=3, path_fail=((0, 2),), max_rounds=1)
    assert result.success_mask.tolist() == [False, True]
    assert len(engine.calls) == 1
    assert result.summary["stop_reason"] == "round_budget_exhausted"
    assert torch.count_nonzero(result.trajectory.positions[0]) == 0


def test_failed_input_and_padding_produce_empty_safe_hold_without_enumeration():
    grasps = GraspCandidateBatch.from_ragged(
        [(torch.empty(0, 4, 4), torch.empty(0)), (torch.empty(0, 4, 4), torch.empty(0))]
    )
    result, engine = _run(rows=2, grasps=grasps)
    assert not result.success_mask.any()
    assert result.trajectory.positions.shape == (2, 1, 2)
    assert result.compact_positions.shape == (0, 0, 2)
    assert result.summary["rejection_counts"] == {"NO_GRASP_CANDIDATES": 2}
    assert engine.enumerations == 0


def test_all_ik_failures_are_empty_not_backend_errors():
    result, engine = _run(rows=2, count=1, ik_fail=((0, 0), (0, 1), (1, 0), (1, 1)))
    assert not result.success_mask.any()
    assert result.summary["rejection_counts"] == {"IK_NOT_FOUND": 4}
    assert not engine.calls


def test_backend_exception_propagates_instead_of_becoming_ik_failure():
    with pytest.raises(RuntimeError, match="backend exploded"):
        _run(crash=True)


def test_push_keeps_each_rows_own_single_raw_and_inactive_rows_stay_idle():
    grasps = _grasps(3, 1, ids=(("left",), ("middle",), ("right",)))
    result, engine = _run(
        rows=3,
        grasps=grasps,
        active=torch.tensor([True, False, True]),
        variants=1,
        skill="slide",
    )
    assert result.grasp_ids == ("left", None, "right")
    assert result.success_mask.tolist() == [True, False, True]
    assert torch.equal(result.grasp_poses[[0, 2]], grasps.poses[[0, 2], 0])
    assert engine.calls[0][0].tolist() == [True, False, True]


@pytest.mark.parametrize(
    "corruption,reason",
    [
        ("nan", "NONFINITE_QPOS"),
        ("start", "INITIAL_QPOS_MISMATCH"),
        ("limit", "JOINT_LIMIT_FAILED"),
        ("timing", "NONUNIFORM_CONTROL_DT"),
    ],
)
def test_bad_output_row_is_filtered_without_contaminating_successful_peer(
    corruption, reason
):
    def corrupt(trajectory, indices):
        if corruption == "nan":
            trajectory.positions[0, 1, 0] = torch.nan
        elif corruption == "start":
            trajectory.positions[0, 0, 0] = 0.2
        elif corruption == "limit":
            trajectory.positions[0, 1, 0] = 3.0
        else:
            trajectory.dt[0, 1] = CONTROL_DT * 2

    result, _ = _run(rows=2, count=2, corrupt=corrupt, max_rounds=1)
    assert result.success_mask.tolist() == [False, True]
    assert result.summary["rejection_counts"] == {reason: 1}
    assert torch.isfinite(result.trajectory.positions).all()
    assert torch.count_nonzero(result.trajectory.positions[0]) == 0


def test_mimic_geometry_is_expanded_without_commanding_an_independent_joint():
    context = _context(1)
    engine = _Engine(context, variants=1)
    engine.robot.mimic_ids = (1,)
    engine.robot.mimic_parents = (0,)
    engine.robot.mimic_multipliers = (-1.0,)
    engine.robot.mimic_offsets = (0.0,)
    invocation = SimpleNamespace(
        skill_id="slide",
        invocation_id="grasp",
        motion_policy=SimpleNamespace(strategy="ik_interp"),
    )
    result = plan_affordance_batch(engine, invocation, context, _grasps(1, 1))
    assert result.success_mask.all()
    assert torch.equal(
        result.trajectory.positions[:, :, 1], -result.trajectory.positions[:, :, 0]
    )


def test_observed_passive_residual_is_preserved_only_at_initial_sample():
    context = _context(1)
    # Live physics need not satisfy the passive affine relationship exactly.
    context.robot.qpos[0, 1] = 0.02
    engine = _Engine(context, variants=1)
    engine.robot.mimic_ids = (1,)
    engine.robot.mimic_parents = (0,)
    engine.robot.mimic_multipliers = (-1.0,)
    engine.robot.mimic_offsets = (0.0,)
    invocation = SimpleNamespace(
        skill_id="slide",
        invocation_id="grasp",
        motion_policy=SimpleNamespace(strategy="ik_interp"),
    )
    result = plan_affordance_batch(engine, invocation, context, _grasps(1, 1))
    assert result.success_mask.all()
    assert torch.equal(result.trajectory.positions[:, 0], context.robot.qpos)
    assert torch.equal(
        result.trajectory.positions[:, 1:, 1],
        -result.trajectory.positions[:, 1:, 0],
    )
    assert context.robot.qpos[0, 1] == pytest.approx(0.02)


@pytest.mark.parametrize("wrong_joint", [0, 1], ids=["active", "passive"])
def test_mimic_expansion_never_hides_wrong_compiled_initial_joint(wrong_joint):
    context = _context(1)
    context.robot.qpos[0, 1] = 0.02

    def corrupt(trajectory, indices):
        # A wrong passive value must be rejected even if it matches the ideal
        # affine geometry; the first waypoint represents the real observation.
        trajectory.positions[0, 0, wrong_joint] = 0.1 if wrong_joint == 0 else 0.0

    engine = _Engine(context, variants=1, corrupt=corrupt)
    engine.robot.mimic_ids = (1,)
    engine.robot.mimic_parents = (0,)
    engine.robot.mimic_multipliers = (-1.0,)
    engine.robot.mimic_offsets = (0.0,)
    invocation = SimpleNamespace(
        skill_id="slide",
        invocation_id="grasp",
        motion_policy=SimpleNamespace(strategy="ik_interp"),
    )
    result = plan_affordance_batch(engine, invocation, context, _grasps(1, 1))
    assert not result.success_mask.any()
    assert result.summary["rejection_counts"] == {"INITIAL_QPOS_MISMATCH": 1}
    assert torch.equal(result.trajectory.positions[:, 0], context.robot.qpos)


def test_invalid_raw_pose_does_not_prevent_other_raw_grasp_success():
    grasps = _grasps(2, 3)
    poses = grasps.poses.clone()
    poses[:, 0, 0, 0] = torch.nan
    invalid = GraspCandidateBatch(
        poses, grasps.costs, grasps.valid_mask, grasp_ids=grasps.grasp_ids
    )
    result, _ = _run(rows=2, grasps=invalid)
    assert result.success_mask.all()
    assert set(result.grasp_ids) == {"raw-1", "raw-2"}
    assert result.summary["rejection_counts"]["INVALID_GRASP_POSE"] == 2


def test_repeated_zero_time_boundary_merges_without_inventing_motion_time():
    def corrupt(trajectory, indices):
        trajectory.positions[:, 1] = trajectory.positions[:, 0]
        trajectory.dt[:, 1] = 0

    result, _ = _run(rows=1, count=1, corrupt=corrupt)
    assert result.success_mask.all()
    assert result.valid_length.tolist() == [2]
    assert result.dt.tolist()[0] == pytest.approx([0.0, CONTROL_DT])


def test_zero_time_jump_is_rejected_instead_of_retimed():
    def corrupt(trajectory, indices):
        trajectory.dt[:, 1] = 0

    result, _ = _run(rows=1, count=1, corrupt=corrupt, max_rounds=1)
    assert not result.success_mask.any()
    assert result.summary["rejection_counts"] == {"DISCONTINUOUS_ZERO_TIME_BOUNDARY": 1}


def test_distinct_input_ids_with_duplicate_columns_never_fill_by_aliasing():
    grasps = _grasps(2, 2, ids=(("same", "same"), ("same", "same")))
    result, _ = _run(rows=2, grasps=grasps)
    assert int(result.success_mask.sum()) == 1
    assert result.summary["unique_raw_grasps"] == 1


def test_result_owns_tensors_and_compact_views_do_not_mutate_replay():
    grasps = _grasps(1, 1)
    result, _ = _run(rows=1, grasps=grasps)
    compact = result.compact_positions
    compact.fill_(1.0)
    grasps.poses.fill_(2.0)
    assert result.trajectory.positions[0, 0, 0] == 0
    assert result.grasp_poses[0, 0, 3] == pytest.approx(0.1)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_round_budget_requires_positive_integer(value):
    with pytest.raises(ValueError, match="max_rounds"):
        _run(max_rounds=value)
