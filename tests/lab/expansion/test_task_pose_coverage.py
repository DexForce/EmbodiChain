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

"""Measured-cell accounting, sampling compatibility and feedback regressions."""

from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
import json
import math
import numpy as np
import pytest
import torch
from embodichain.lab.expansion.task_pose_coverage import TaskPoseCoverage
from embodichain.lab.expansion.contracts import SceneCase
from embodichain.lab.expansion._pose_sampling import (
    _BoundedPoseSource,
    _AdaptivePoseSource,
)
from embodichain.lab.expansion._pose_sampling_cfg import validate_sampling
from embodichain.lab.sim.scene.feasibility import FeasibilityBudgetExhausted


def test_admitted_planning_retains_count_and_cancel_limits_after_admission_time():
    from threading import Event
    import time
    from embodichain.lab.expansion._pose_sampling import (
        _PoseBudget,
        _AdmittedPoseBudget,
    )
    from embodichain.lab.sim.scene.feasibility import SceneRobotFeasibilityCfg

    cancellation = Event()
    budget = _PoseBudget(
        SceneRobotFeasibilityCfg(max_wall_time_s=0.01, max_ik_calls=2),
        cancelled=cancellation,
    )
    budget.validation = True
    admitted = _AdmittedPoseBudget(budget)
    time.sleep(0.03)
    with pytest.raises(FeasibilityBudgetExhausted, match="wall_time"):
        budget.charge("candidates")
    admitted.check()
    admitted.charge("ik")
    assert budget.counts["ik"] == 1 and budget.counts["candidates"] == 0
    cancellation.set()
    with pytest.raises(FeasibilityBudgetExhausted, match="cancelled"):
        admitted.check()
    cancellation.clear()
    with pytest.raises(FeasibilityBudgetExhausted, match="ik_budget"):
        admitted.charge("ik", 2)
    assert budget.counts["ik"] == 1


def grids():
    return {
        phase: {
            "lower": [-1, -1, -math.pi / 2],
            "upper": [1, 1, math.pi / 2],
            "cell_size": [0.5, 0.5, math.pi / 2],
            "yaw_period": math.pi,
        }
        for phase in ("pick", "place")
    }


def coverage():
    cov = TaskPoseCoverage("domain", "object", grids())
    for name in ("scene-a", "scene-b"):
        cov.register_case(
            SceneCase(name, "initial", "signature", "task", "robot"),
            torch.tensor([[-1.0, 1.0]]),
            ("joint",),
            None,
        )
    return cov


def episode(commit="c1", case="scene-a", x=0.1, yaw=0.0, accepted=True):
    state = torch.tensor(
        [
            [x, 0.1, 0.02, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
            [x + 0.1, 0.1, 0.02, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
        ]
    )
    return SimpleNamespace(
        commit_id=commit,
        identity=SimpleNamespace(scene_case_id=case),
        validation=SimpleNamespace(accepted=accepted),
        timestamps=torch.tensor([0.0, 0.1]),
        observations={"rigid/object/state": state},
    )


def test_receipts_aggregate_cases_without_double_credit_and_own_cells():
    cov = coverage()
    e = episode()
    assert cov.reserve(e)
    assert cov.snapshot()[0] == {"domain": 0}
    e.observations["rigid/object/state"].fill_(99)
    cov.confirm("scene-a", "c1")
    cov.confirm("scene-a", "c1")
    assert cov.phase_counts() == {"pick": 1, "place": 1}
    assert cov.snapshot()[1] == {}
    assert cov.reserve(episode("c2", "scene-b", yaw=math.pi))
    cov.confirm("scene-b", "c2")
    assert cov.snapshot()[0] == {
        "domain": 2
    }  # Symmetry and scene changes aren't novelty.
    assert not cov.reserve(episode("bad", accepted=False))
    cov.reserve(episode("c3", x=-0.9))
    cov.release("scene-a", "c3")
    with pytest.raises(ValueError):
        cov.confirm("scene-a", "c3")
    assert cov.snapshot()[0] == {"domain": 2}


def test_checkpoint_keeps_pending_unconfirmed_and_rejects_foreign_domain():
    cov = coverage()
    cov.reserve(episode())
    cov.confirm("scene-a", "c1")
    cov.reserve(episode("pending", x=-0.9))
    state = json.loads(json.dumps(cov.state_dict()))
    restored = TaskPoseCoverage("domain", "object", grids())
    restored.load_state_dict(state)
    assert restored.snapshot() == cov.snapshot()
    restored.confirm("scene-a", "pending")
    assert restored.snapshot()[0] == {"domain": 4}
    assert cov.snapshot()[0] == {"domain": 2}
    with pytest.raises(ValueError):
        TaskPoseCoverage("another", "object", grids()).load_state_dict(state)
    with pytest.raises(ValueError):
        restored.confirm("scene-b", "pending")


def test_cells_boundaries_periodicity_and_measured_not_requested_pose():
    cov = coverage()
    assert cov.cells([-1, -1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1]) == (
        ("pick", 0, 0, 1),
        ("place", 3, 3, 1),
    )
    assert cov.cells([-1.1, 0, 0, 0, 0, 0, 1], [1.1, 0, 0, 0, 0, 0, 1]) == ()
    with pytest.raises(ValueError):
        cov.cells([0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1])
    e = episode(x=-0.9)
    e.metadata = {"requested_pose": [0.9, 0.9, 0]}
    cov.reserve(e)
    cov.confirm("scene-a", "c1")
    assert cov.gain((("pick", 0, 2, 1),)) == 0.5


def test_named_boundaries_use_real_artifacts_and_fixed_categories(tmp_path):
    from embodichain.lab.expansion import ArtifactEpisodeSink, load_expansion_episode
    from embodichain.lab.expansion.contracts import (
        CandidateIdentity,
        ExpertEpisode,
        ValidationCheck,
        ValidationResult,
    )

    measurements = {
        "initial": {"entity": "object", "boundary": "initial"},
        "handover": {
            "entity": "object",
            "boundary": "segment_end",
            "segment": "handover",
            "fixed": True,
        },
        "handover_pose": {
            "entity": "object",
            "boundary": "segment_end",
            "segment": "handover",
        },
        "held": {"entity": "object", "boundary": "terminal", "fixed": True},
    }
    grid = grids()["pick"]
    cov = TaskPoseCoverage(
        "domain",
        "object",
        {name: grid for name in measurements},
        measurements=measurements,
    )
    cov.register_case(
        SceneCase("case", "initial", "signature", "task", "robot"),
        torch.tensor([[-1.0, 1.0]]),
        ("joint",),
        None,
    )
    sink = ArtifactEpisodeSink(tmp_path / "episodes")
    for ordinal, x in enumerate((-0.7, 0.7)):
        state = torch.tensor(
            [
                [x, 0.1, 0.1, 0, 0, 0, 1],
                [0.1, 0.1, 0.1, 0, 0, 0, 1],
                [0.1 + ordinal * 0.6, 0.1, 0.1, 0, 0, 0, 1],
            ]
        )
        episode = ExpertEpisode(
            CandidateIdentity(
                "case",
                "initial",
                f"candidate{ordinal}",
                "family",
                "source",
                "v1",
                "plan",
            ),
            {"rigid/object/state": state},
            torch.zeros(2, 1),
            torch.tensor([0.0, 0.1, 0.2]),
            "qpos",
            ValidationResult((ValidationCheck("task", "passed"),)),
            f"episode{ordinal}",
            f"commit{ordinal}",
            metadata={
                "program_segments": [
                    {
                        "name": "handover",
                        "start_step": 0,
                        "end_step": 1,
                        "success": True,
                    }
                ]
            },
        )
        assert cov.reserve(episode)
        assert cov.phase_counts()["initial"] == ordinal
        receipt = sink.submit(episode, submission_id=ordinal)
        restored = load_expansion_episode(receipt.storage_id)
        assert cov.reserve(restored)
        cov.confirm("case", episode.commit_id)
        cov.confirm("case", episode.commit_id)
    assert cov.phase_counts() == {
        "initial": 2,
        "handover": 1,
        "handover_pose": 1,
        "held": 1,
    }
    assert ("handover_pose", 2, 2, 1) in cov.state_dict()["confirmed"]["commit1"][1]
    assert cov.proposal_cells({"object": np.eye(4)}) == (("initial", 2, 2, 1),)
    snapshot = cov.state_dict()
    restored = TaskPoseCoverage(
        "domain",
        "object",
        {name: grid for name in measurements},
        measurements=measurements,
    )
    restored.load_state_dict(snapshot)
    assert restored.phase_counts() == cov.phase_counts()
    with pytest.raises(ValueError, match="segment name"):
        TaskPoseCoverage(
            "domain",
            "object",
            {"bad": grid},
            measurements={"bad": {"entity": "object", "boundary": "segment_end"}},
        )


class _Robot:
    control_parts = {"arm": [0, 1, 2]}
    device = torch.device("cpu")
    uid = "robot"
    cfg = SimpleNamespace(to_dict=lambda: {"uid": "robot"})

    def get_local_pose(self, to_matrix=False):
        return torch.eye(4)[None]

    def get_qpos(self, **kwargs):
        return torch.tensor([[0.0, 0.0, 0.02]])

    def get_qpos_limits(self, **kwargs):
        return torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [0.02, 0.02]]])

    def compute_batch_fk(self, qpos, **kwargs):
        pose = torch.eye(4).expand(*qpos.shape[:2], 4, 4).clone()
        pose[..., :3, 3] = qpos
        return pose

    def compute_ik(self, pose, **kwargs):
        return torch.ones(len(pose), dtype=torch.bool), pose[:, :3, 3].clone()

    def compute_fk(self, qpos, **kwargs):
        pose = torch.eye(4).repeat(len(qpos), 1, 1)
        pose[:, :3, 3] = qpos
        return pose


def environment():
    pose = torch.eye(4)[None]
    pose[:, 2, 3] = 0.02
    obj = SimpleNamespace(get_local_pose=lambda to_matrix=True: pose.clone())
    return SimpleNamespace(
        robot=_Robot(),
        sim=SimpleNamespace(
            get_rigid_object=lambda uid: obj,
            sim_config=SimpleNamespace(
                physics_cfg=SimpleNamespace(to_dict=lambda: {"physics": "default"})
            ),
        ),
    )


def options(strategy="bounded_pose", policy="fifo"):
    value = {
        "engine": {"max_attempts": 8, "max_wall_time_s": 120},
        "validation": {},
        "scene": {
            uid: {
                "translation_m": [[-0.9, 0.9], [-0.9, 0.9], [0, 0]],
                "yaw_rad": [0, 0],
            }
            for uid in ("object", "goal")
        },
        "scene_sampling": {
            "strategy": strategy,
            "sample_count": 64,
            "max_candidates": 64,
            "targets": (
                {uid: np.eye(4).tolist() for uid in ("object", "goal")}
                if strategy == "workspace_joint"
                else {}
            ),
        },
        "scheduling": {"policy": policy, "pool_size": 8},
        "pose_coverage": {
            "entity": "object",
            "place_reference": "goal",
            "grids": grids(),
        },
    }
    validate_sampling(value)
    return value


def test_legacy_config_and_simple_fifo_keep_identical_poses():
    old = {"engine": {}, "scene": {}}
    expected = deepcopy(old)
    validate_sampling(old)
    assert old == expected
    env = environment()
    opts = options()
    simple = _BoundedPoseSource(env, opts["scene"], seed=7, revision="revision")
    new = _AdaptivePoseSource(env, opts, seed=7, revision="revision")
    for ordinal in range(4):
        assert (
            new.candidate(ordinal).scene.pose_changes
            == simple.candidate(ordinal).scene.pose_changes
        )


@pytest.mark.parametrize("strategy", ["bounded_pose", "workspace_joint"])
@pytest.mark.parametrize("policy", ["fifo", "task_pose_coverage"])
def test_candidate_is_frozen_after_feedback_and_source_is_seeded(strategy, policy):
    opts = options(strategy, policy)
    a = _AdaptivePoseSource(environment(), opts, seed=7, revision="revision")
    b = _AdaptivePoseSource(environment(), opts, seed=7, revision="revision")
    first = a.candidate(0)
    assert first == b.candidate(0)
    assert a.validate(0).accepted
    a.finish(0, None, None, executed=True)
    assert a.candidate(0) is first
    assert a.budget.counts["candidates"] <= 64
    assert len(a._predicted) == 0


def test_bounded_budget_never_bypasses_candidate_quota():
    a = _AdaptivePoseSource(environment(), options(), seed=7, revision="r")
    a.budget.charge("candidates", 64)
    with pytest.raises(FeasibilityBudgetExhausted):
        a.budget.charge("candidates")
    assert a.budget.counts["candidates"] == 64


def test_invalid_joint_config_does_not_fall_back():
    opt = options()
    opt["scene_sampling"]["strategy"] = "workspace_joint"
    with pytest.raises(ValueError, match="target transforms"):
        validate_sampling(opt)


def test_sampler_checkpoint_replays_choices_after_feedback():
    opts = options("workspace_joint", "task_pose_coverage")
    a = _AdaptivePoseSource(environment(), opts, seed=7, revision="r")
    first = a.candidate(0)
    a.finish(0, None, None, executed=True)
    state = json.loads(json.dumps(a.state_dict()))
    b = _AdaptivePoseSource(environment(), opts, seed=7, revision="r")
    b.load_state_dict(state)
    assert b.candidate(0) == first
    assert a.candidate(1) == b.candidate(1)
    assert a.budget.counts == b.budget.counts


def test_relative_pose_validation_preserves_yaw_symmetry_and_tilt():
    from embodichain.lab.task_program.integrations.feasibility import (
        MeasuredTaskMonitor,
    )

    actual = torch.eye(4).repeat(2, 1, 1)
    target = actual.clone()
    target[:, 2, 3] = -0.1
    actual[:, 2, 3] = 0.02
    actual[0, 0, 0] = actual[0, 1, 1] = -1
    actual[1, 0, 0] = actual[1, 1, 1] = 0
    actual[1, 0, 1] = -1
    actual[1, 1, 0] = 1

    class Entity:
        def __init__(self, poses):
            self.poses = poses

        def get_local_pose(self, to_matrix=False):
            if to_matrix:
                return self.poses.clone()
            return self.poses[:, :3, 3].clone()

    objects = {"object": Entity(actual), "target": Entity(target)}
    env = SimpleNamespace(sim=SimpleNamespace(get_rigid_object=objects.get))
    condition = {
        "kind": "pose_relative",
        "entity": "object",
        "other": "target",
        "position_offset": [0, 0, 0.12],
        "position_tolerance": [0.02, 0.02, 0.01],
        "yaw_period_rad": math.pi,
        "yaw_tolerance_rad": 0.18,
        "tilt_tolerance_rad": 0.18,
    }
    monitor = MeasuredTaskMonitor([condition], [], env_id=0)
    assert monitor.predicate(env, condition)
    monitor.env_id = 1
    assert not monitor.predicate(env, condition)
    actual[1] = actual[0]
    actual[1, 2, 3] += 0.02
    assert not monitor.predicate(env, condition)
    with pytest.raises(ValueError):
        MeasuredTaskMonitor([{**condition, "yaw_period_rad": 0}], [])


def test_workspace_fifo_does_not_require_coverage_policy():
    opts = options("workspace_joint", "fifo")
    del opts["pose_coverage"]
    validate_sampling(opts)
    source = _AdaptivePoseSource(environment(), opts, seed=7, revision="r")
    assert source.coverage is None
    source.candidate(0)
    assert source.validate(0).accepted
    source.finish(0, None, None, executed=False)
    opts["scheduling"]["policy"] = "task_pose_coverage"
    with pytest.raises(ValueError, match="grids"):
        validate_sampling(opts)


def test_checkpoint_rejects_out_of_grid_cell_without_partial_restore():
    cov = coverage()
    cov.reserve(episode())
    state = cov.state_dict()
    state["pending"]["c1"] = ("scene-a", (("pick", 99, 0, 0),))
    restored = TaskPoseCoverage("domain", "object", grids())
    with pytest.raises(ValueError):
        restored.load_state_dict(state)
    assert not restored._cases and not restored._pending


def test_sampling_proposal_budget_counts_discarded_pool_candidates():
    opts = options("bounded_pose", "task_pose_coverage")
    source = _AdaptivePoseSource(environment(), opts, seed=7, revision="r")
    source.candidate(0)
    assert source.budget.counts["candidates"] == 8
    assert len(source._candidates) == 1
    # Re-reading the chosen immutable candidate consumes no extra budget.
    source.candidate(0)
    assert source.budget.counts["candidates"] == 8


def test_coverage_gain_uses_confirmed_pending_and_predicted_separately():
    cov = coverage()
    e = episode()
    cells = cov.cells(
        e.observations["rigid/object/state"][0],
        e.observations["rigid/object/state"][-1],
    )
    assert cov.gain(cells) == 2
    cov.reserve(e)
    assert cov.gain(cells) == 1
    cov.release("scene-a", "c1")
    assert cov.gain(cells) == 2
    assert cov.gain(cells, {c: 1 for c in cells}) == 1


def test_workspace_branch_must_match_execution_policy():
    source = _AdaptivePoseSource(
        environment(), options("workspace_joint"), seed=7, revision="r"
    )
    source.env.cfg = SimpleNamespace(
        task_program=SimpleNamespace(
            integration=SimpleNamespace(runtime_preset="trajectory")
        )
    )
    pick = SimpleNamespace(fixed_object_to_eef=torch.eye(4))
    profile = SimpleNamespace(
        presets={
            "trajectory": SimpleNamespace(skill_option_templates={"pick_up": pick})
        }
    )
    adapter = SimpleNamespace(
        _factory=SimpleNamespace(create_robot_skill_profile=lambda: profile)
    )
    from embodichain.lab.task_program.semantics.calls import Pick
    from embodichain.lab.task_program.semantics.scene import SceneObjectRef

    program = SimpleNamespace(
        iter_segments=lambda: [
            SimpleNamespace(
                calls=[SimpleNamespace(call=Pick(object=SceneObjectRef("object")))]
            )
        ]
    )
    source.bind_program(adapter, program)
    pick.fixed_object_to_eef[0, 3] = 0.1
    with pytest.raises(ValueError, match="match"):
        source.bind_program(adapter, program)
    pick.fixed_object_to_eef = None
    with pytest.raises(ValueError, match="fixed_object"):
        source.bind_program(adapter, program)


def test_budget_expiry_before_first_command_has_no_fabricated_hold():
    from embodichain.lab.expansion._workload_cursor import _WorkloadCursor

    class Bridge:
        def __iter__(self):
            raise AssertionError("Expired planning must not enter the bridge")
            yield

    class Budget:
        def check(self):
            raise FeasibilityBudgetExhausted("ik_budget_exhausted")

    cursor = _WorkloadCursor(
        Bridge(), SimpleNamespace(device=torch.device("cpu")), 7, budget=Budget()
    )
    assert cursor.next() is None
    assert cursor.finished and not cursor.accepted and cursor.length == 0
    assert cursor.reason == "ik_budget_exhausted" and cursor.last_command is None


def test_coverage_rejects_nonphysical_yaw_period_and_identifies_branch_cut():
    from embodichain.lab.expansion._pose_cells import _grid

    with pytest.raises(ValueError, match="complete turn"):
        _grid(
            {
                "lower": [-1, -1, -0.5],
                "upper": [1, 1, 0.5],
                "cell_size": [0.5, 0.5, 0.1],
                "yaw_period": 1,
            }
        )
    cov = coverage()
    a = [0, 0, 0, 0, 0, 1, 0]
    b = [0, 0, 0, 0, 0, -1, 0]
    assert cov.cells(a, a) == cov.cells(b, b)


def test_cache_configuration_is_used_and_empty_qualified_cache_fails():
    from unittest.mock import Mock

    env = environment()
    opts = options("workspace_joint")
    opts["scene_sampling"]["workspace_source"] = "cache"
    cache = SimpleNamespace(
        sample_indices=Mock(side_effect=ValueError("No samples meet min_score"))
    )
    env.robot.cfg = SimpleNamespace(
        to_dict=lambda: {"uid": "robot"},
        workspace_cfg={"arm": SimpleNamespace(strategy="point_uniform", min_score=0.8)},
    )
    env.robot.get_workspace = lambda part: cache
    with pytest.raises(ValueError, match="min_score"):
        _AdaptivePoseSource(env, opts, seed=7, revision="r")
    assert cache.sample_indices.call_args.kwargs["strategy"] == "point_uniform"
    assert cache.sample_indices.call_args.kwargs["min_score"] == 0.8


def test_fifo_without_coverage_still_checks_grasp_correspondence():
    from embodichain.lab.task_program.semantics.calls import Pick
    from embodichain.lab.task_program.semantics.scene import SceneObjectRef

    opts = options("workspace_joint")
    del opts["pose_coverage"]
    source = _AdaptivePoseSource(environment(), opts, seed=7, revision="r")
    source.env.cfg = SimpleNamespace(
        task_program=SimpleNamespace(
            integration=SimpleNamespace(runtime_preset="trajectory")
        )
    )
    fixed = torch.eye(4)
    fixed[2, 3] = 0.2
    profile = SimpleNamespace(
        presets={
            "trajectory": SimpleNamespace(
                skill_option_templates={
                    "pick_up": SimpleNamespace(fixed_object_to_eef=fixed)
                }
            )
        }
    )
    adapter = SimpleNamespace(
        _factory=SimpleNamespace(create_robot_skill_profile=lambda: profile)
    )
    program = SimpleNamespace(
        iter_segments=lambda: [
            SimpleNamespace(
                calls=[SimpleNamespace(call=Pick(object=SceneObjectRef("object")))]
            )
        ]
    )
    with pytest.raises(ValueError, match="match"):
        source.bind_program(adapter, program)


def test_transport_guard_remembers_a_drop_and_resets_per_episode():
    from embodichain.lab.task_program.integrations.feasibility import (
        MeasuredTaskMonitor,
    )

    state = torch.tensor([[0.0, 0.0, 0.02]])
    goal = torch.tensor([[0.5, 0.0, 0.02]])
    objects = {
        "object": SimpleNamespace(get_local_pose=lambda: state.clone()),
        "goal": SimpleNamespace(get_local_pose=lambda: goal.clone()),
    }
    env = SimpleNamespace(
        sim=SimpleNamespace(
            get_rigid_object=objects.get,
            get_rigid_object_uid_list=lambda: tuple(objects),
        )
    )
    condition = {
        "kind": "transport_guard",
        "entity": "object",
        "other": "goal",
        "lift_height_m": 0.05,
        "minimum_transit_height_m": 0.025,
        "goal_tolerance_m": 0.03,
    }
    monitor = MeasuredTaskMonitor([condition], [])
    monitor.reset(env)
    assert not monitor.validation(env).accepted
    state[0, 2] = 0.08
    monitor.observe(env, None)
    assert monitor.validation(env).accepted
    state[0, 2] = 0.02
    monitor.observe(env, None)
    state[0] = goal[0]
    monitor.observe(env, None)
    assert not monitor.validation(env).accepted
    state[0] = torch.tensor([0.0, 0.0, 0.02])
    monitor.reset(env)
    state[0, 2] = 0.08
    monitor.observe(env, None)
    state[0] = goal[0]
    monitor.observe(env, None)
    assert monitor.validation(env).accepted


def test_new_feedback_hook_preserves_legacy_positional_extension_constructor():
    from embodichain.lab.expansion.host_extensions import GymExpansionHostExtensions

    factory = lambda env, row: None
    extensions = GymExpansionHostExtensions(None, None, None, None, factory, True)
    assert (
        extensions.require_plan_transform
        and extensions.plan_transform_factory is factory
    )
    assert extensions.feedback_factory is None


def test_bounded_source_owns_input_ranges():
    cfg = options()["scene"]
    source = _BoundedPoseSource(environment(), cfg, seed=7, revision="r")
    expected = source.candidate(3)
    cfg["object"]["translation_m"][0][:] = [0.5, 0.5]
    assert source.candidate(3) == expected


def test_transport_guard_rejects_nonfinite_measured_state():
    from embodichain.lab.task_program.integrations.feasibility import (
        MeasuredTaskMonitor,
    )

    state = torch.tensor([[0.0, 0.0, 0.02]])
    goal = torch.tensor([[0.5, 0.0, 0.02]])
    objects = {
        "object": SimpleNamespace(get_local_pose=lambda: state.clone()),
        "goal": SimpleNamespace(get_local_pose=lambda: goal.clone()),
    }
    env = SimpleNamespace(
        sim=SimpleNamespace(
            get_rigid_object=objects.get,
            get_rigid_object_uid_list=lambda: tuple(objects),
        )
    )
    condition = {
        "kind": "transport_guard",
        "entity": "object",
        "other": "goal",
        "lift_height_m": 0.05,
        "minimum_transit_height_m": 0.025,
        "goal_tolerance_m": 0.03,
    }
    monitor = MeasuredTaskMonitor([condition], [])
    monitor.reset(env)
    state[0, 2] = 0.08
    monitor.observe(env, None)
    assert monitor.validation(env).accepted
    state[0, 2] = float("nan")
    monitor.observe(env, None)
    assert not monitor.validation(env).accepted
