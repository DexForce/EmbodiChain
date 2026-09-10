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

"""Real-batch mapping and failure isolation for the free-motion adapter."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners import CollisionWorldInfo
from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    MotionSnapshot,
    SceneCase,
    TrajectoryPhase,
)
from embodichain.lab.trajectory_generation.integrations.planning import (
    EEFPath,
    EnvRowMotionPlanner,
)


@pytest.mark.gpu
@pytest.mark.slow
def test_real_curobo_free_path_and_dynamic_obstacle_with_cpu_physics():
    """Exercise the actual joint-bound, self, and world checker on one Panda."""
    pytest.importorskip("curobo")
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
    from embodichain.lab.sim.motion.motion_generator import MotionGenCfg
    from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
        CuroboPlannerCfg,
        CuroboWorldCfg,
    )
    from embodichain.lab.sim.objects import RigidObjectCfg
    from embodichain.lab.sim.robots import FrankaPandaCfg
    from embodichain.lab.sim.shapes import CubeCfg

    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=1)
    )
    generator = None
    try:
        robot = sim.add_robot(
            cfg=FrankaPandaCfg.from_dict(
                {"uid": "free_path_panda", "robot_type": "panda"}
            )
        )
        robot.set_qpos(
            torch.tensor([robot.cfg.init_qpos], device=robot.device), target=False
        )
        obstacle = sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="path_obstacle",
                shape=CubeCfg(size=(0.3, 0.3, 0.3)),
                attrs=RigidBodyAttributesCfg(),
                body_type="kinematic",
                init_pos=(3.0, 0.0, 1.0),
            )
        )
        generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=CuroboPlannerCfg(
                    robot_uid=robot.uid,
                    world=CuroboWorldCfg(
                        rigid_objects={"obstacle": obstacle},
                        dynamic_obstacle_names=["obstacle"],
                        obstacle_representation="cuboid",
                    ),
                    use_cuda_graph=False,
                    cuda_device=0,
                )
            )
        )
        planner = EnvRowMotionPlanner(
            generator,
            control_part="arm",
            held_object_ids=(),
            max_joint_step=0.01,
            max_validation_samples=8,
        )
        snapshot = MotionSnapshot(
            SceneCase("real-case", "real-initial", "real-scene", "move", "panda"),
            tuple(robot.joint_names),
            robot.get_qpos()[0],
            robot.get_qvel()[0],
            robot.get_local_pose(to_matrix=True)[0],
            {"obstacle": obstacle.get_local_pose(to_matrix=True)[0]},
        )
        positions = snapshot.joint_positions.repeat(1, 3, 1)
        positions[0, :, planner.joint_ids[0]] += torch.tensor([0.0, 0.01, 0.02])
        identity = CandidateIdentity(
            "real-case",
            "real-initial",
            "real-candidate",
            "real-family",
            "source",
            "r1",
            "template",
        )
        batch = CandidateTrajectoryBatch(
            positions,
            torch.tensor([[0.0, 0.1, 0.1]]),
            torch.tensor([3]),
            (identity,),
            snapshot.joint_names,
            ((TrajectoryPhase("free", 0, 3),),),
            source_row_indices=torch.tensor([0]),
        )
        result = planner.validate_qpos(batch, (snapshot,))[0]
        assert result.accepted, (
            result.checks,
            snapshot.joint_positions,
            robot.cfg.init_qpos,
            planner.fixed_ids,
        )
        assert robot.device.type == "cpu"
        assert generator.planner._curobo_device.type == "cuda"
        assert len(generator.planner._backend_cache) == 1

        blocked_pose = robot.compute_fk(
            qpos=robot.get_qpos(name="arm"), name="arm", to_matrix=True
        )
        obstacle.set_local_pose(blocked_pose)
        blocked = replace(snapshot, entity_poses={"obstacle": blocked_pose[0]})
        rejected = planner.validate_qpos(batch, (blocked,))[0]
        assert rejected.checks[0].status == "failed", rejected.checks
        assert len(generator.planner._backend_cache) == 1
    finally:
        if generator is not None:
            generator.planner.close()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


class _Robot:
    device = torch.device("cpu")
    num_instances = 4
    joint_names = ("arm", "gripper")
    cfg = SimpleNamespace(init_qpos=[0.0, 0.25])

    def __init__(self):
        self.roots = torch.eye(4).repeat(self.num_instances, 1, 1)
        self.roots[:, 0, 3] = torch.arange(self.num_instances) * 10.0
        self.ik_calls = []
        self.fk_calls = []
        self.failed_ik_rows = set()

    def get_joint_ids(self, name):
        assert name == "arm"
        return [0]

    def get_local_pose(self, *, to_matrix):
        assert to_matrix
        return self.roots.clone()

    def get_qpos_limits(self):
        return torch.tensor([[-10.0, 10.0], [0.0, 1.0]]).repeat(
            self.num_instances, 1, 1
        )

    def compute_fk(self, *, qpos, name, to_matrix):
        assert qpos.shape == (self.num_instances, 1)
        assert torch.isfinite(qpos).all()
        self.fk_calls.append(qpos.clone())
        poses = self.roots.clone()
        poses[:, 0, 3] += qpos[:, 0]
        return poses

    def compute_ik(self, *, pose, name, joint_seed):
        assert pose.shape == (self.num_instances, 4, 4)
        assert torch.isfinite(joint_seed).all()
        self.ik_calls.append((pose.clone(), joint_seed.clone()))
        solved = (pose[:, 0, 3] - self.roots[:, 0, 3]).unsqueeze(1)
        mask = torch.ones(self.num_instances, dtype=torch.bool)
        for row in self.failed_ik_rows:
            solved[row] = float("nan")
            mask[row] = False
        return mask, solved


class _Backend:
    supports_joint_trajectory_validation = True
    collision_world_info = CollisionWorldInfo(
        ("obstacle",), ("obstacle",), "per_env", True
    )

    def __init__(self):
        self.queries = []
        self.block_middle = False
        self.reject_padding_after = {}

    def validate_joint_trajectory(self, trajectory, *, control_part, obstacle_poses):
        assert trajectory.shape[0] == 4
        assert torch.isfinite(trajectory).all()
        self.queries.append((trajectory.clone(), obstacle_poses))
        mask = torch.ones(trajectory.shape[:2], dtype=torch.bool)
        if self.block_middle:
            mask &= ~((trajectory[..., 0] > 0.04) & (trajectory[..., 0] < 0.06))
        for row, start in self.reject_padding_after.items():
            mask[row, start:] = False
        return mask


def _fixture(**kwargs):
    robot, backend = _Robot(), _Backend()
    generator = object.__new__(MotionGenerator)
    generator.robot = robot
    generator.planner = backend
    planner = EnvRowMotionPlanner(
        generator, control_part="arm", held_object_ids=(), **kwargs
    )
    snapshots = tuple(
        MotionSnapshot(
            SceneCase(f"case-{row}", f"initial-{row}", f"scene-{row}", "move", "robot"),
            robot.joint_names,
            torch.tensor([0.0, 0.25]),
            torch.zeros(2),
            robot.roots[row],
            {"obstacle": robot.roots[row]},
        )
        for row in range(robot.num_instances)
    )
    return planner, robot, backend, snapshots


def _identity(index, row):
    return CandidateIdentity(
        f"case-{row}",
        f"initial-{row}",
        f"candidate-{index}",
        f"family-{index}",
        "source",
        "revision",
        "template",
    )


def _batch(rows, *, displacements=None, phases=None):
    count = len(rows)
    values = [0.1] * count if displacements is None else displacements
    q = torch.tensor([[[0.0, 0.25], [value, 0.25]] for value in values]).reshape(
        count, 2, 2
    )
    return CandidateTrajectoryBatch(
        q,
        torch.tensor([[0.0, 0.1]] * count).reshape(count, 2),
        torch.full((count,), 2, dtype=torch.int64),
        tuple(_identity(index, row) for index, row in enumerate(rows)),
        ("arm", "gripper"),
        (
            tuple((TrajectoryPhase("free", 0, 2),) for _ in rows)
            if phases is None
            else phases
        ),
        source_row_indices=torch.tensor(rows, dtype=torch.int64),
    )


def _eef_paths(rows, robot):
    paths = []
    for index, row in enumerate(rows):
        poses = robot.roots[row].repeat(3, 1, 1)
        poses[:, 0, 3] += torch.tensor([0.0, 0.02, 0.04 + index * 0.003])
        paths.append(
            EEFPath(
                _identity(index, row),
                row,
                poses,
                torch.tensor([0.0, 0.1, 0.2]),
                (TrajectoryPhase("free", 0, 3),),
            )
        )
    return paths


def test_eleven_candidates_use_four_real_rows_across_three_rounds():
    planner, robot, backend, snapshots = _fixture()
    rows = [0, 0, 1, 2, 3, 3, 3, 1, 2, 0, 2]
    batch = _batch(rows, displacements=[0.03 + index * 0.001 for index in range(11)])
    original = batch.positions.clone()
    results = planner.validate_qpos(batch, snapshots)
    assert all(result.accepted for result in results)
    assert len(backend.queries) == 3
    assert robot.num_instances == 4
    assert torch.equal(batch.positions, original)
    expected_groups = [
        {0: 0, 1: 2, 2: 3, 3: 4},
        {0: 1, 1: 7, 2: 8, 3: 5},
        {0: 9, 2: 10, 3: 6},
    ]
    for (query, obstacles), expected in zip(backend.queries, expected_groups):
        assert query.shape[0] == 4
        assert torch.equal(obstacles["obstacle"], robot.roots)
        for row in range(4):
            assert query[row, -1, 0] == pytest.approx(
                batch.positions[expected[row], -1, 0].item() if row in expected else 0.0
            )


def test_eef_mapping_retains_all_rows_times_phases_and_uncontrolled_joints():
    planner, robot, backend, snapshots = _fixture()
    rows = [0, 0, 1, 2, 3, 3, 3, 1, 2, 0, 2]
    paths = _eef_paths(rows, robot)
    batch, results = planner.plan_eef(paths, snapshots)
    assert all(result.accepted for result in results)
    assert len(robot.ik_calls) == 6
    assert batch.source_row_indices.tolist() == rows
    for index, path in enumerate(paths):
        assert torch.allclose(
            batch.positions[index, :, 0],
            path.poses[:, 0, 3] - robot.roots[rows[index], 0, 3],
        )
        assert torch.equal(batch.dt[index], path.dt)
        assert batch.phases[index] == path.phases
    assert torch.equal(batch.positions[..., 1], torch.full((11, 3), 0.25))


def test_fixed_solved_joint_samples_skip_ik_and_remain_exact():
    planner, robot, backend, snapshots = _fixture()
    path = _eef_paths([2], robot)[0]
    solved = torch.tensor([[0.0], [0.02], [0.04]])
    path = replace(path, solved_joint_targets=solved)
    batch, results = planner.plan_eef([path], snapshots)
    assert results[0].accepted
    assert not robot.ik_calls
    assert torch.equal(batch.positions[0, :, 0], solved[:, 0])
    bad_path = replace(path, solved_joint_targets=solved + 0.1)
    _, rejected = planner.plan_eef([bad_path], snapshots)
    assert not rejected[0].accepted
    assert not robot.ik_calls


def test_failed_ik_row_never_contaminates_later_seeds_or_other_rows():
    planner, robot, backend, snapshots = _fixture()
    robot.failed_ik_rows = {1}
    batch, results = planner.plan_eef(_eef_paths([0, 1, 2], robot), snapshots)
    assert [value.accepted for value in results] == [True, False, True]
    assert torch.equal(batch.positions[1], snapshots[1].joint_positions.expand(3, -1))
    assert all(torch.isfinite(seed).all() for _, seed in robot.ik_calls)


def test_collision_between_anchors_and_across_phase_boundary_is_rejected():
    planner, robot, backend, snapshots = _fixture(max_joint_step=0.01)
    backend.block_middle = True
    phases = ((TrajectoryPhase("first", 0, 1), TrajectoryPhase("second", 1, 2)),)
    result = planner.validate_qpos(_batch([0], phases=phases), snapshots)[0]
    assert result.checks[0].status == "failed"
    assert backend.queries[0][0].shape[1] > 2


def test_inactive_rows_and_padded_samples_do_not_affect_acceptance():
    planner, robot, backend, snapshots = _fixture(max_joint_step=0.02)
    backend.reject_padding_after = {0: 2, 2: 0, 3: 0}
    results = planner.validate_qpos(
        _batch([0, 1], displacements=[0.01, 0.1]), snapshots
    )
    assert all(value.accepted for value in results)
    query = backend.queries[0][0]
    assert torch.equal(query[2:], torch.zeros_like(query[2:]))


@pytest.mark.parametrize("kind", ["contact", "hold", "missing"])
def test_unsupported_phase_semantics_never_enter_collision_backend(kind):
    planner, robot, backend, snapshots = _fixture()
    phases = ((),) if kind == "missing" else ((TrajectoryPhase("phase", 0, 2, kind),),)
    result = planner.validate_qpos(_batch([0], phases=phases), snapshots)[0]
    assert result.checks[0].status == "unavailable"
    assert not backend.queries


def test_held_objects_and_moving_or_noninitial_grippers_are_unavailable():
    planner, robot, backend, snapshots = _fixture()
    batch = _batch([0])
    planner.held_object_ids = ("payload",)
    assert planner.validate_qpos(batch, snapshots)[0].checks[0].status == "unavailable"
    planner.held_object_ids = ()
    batch.positions[0, 1, 1] = 0.4
    assert planner.validate_qpos(batch, snapshots)[0].checks[0].status == "unavailable"
    batch.positions[0, :, 1] = 0.4
    snapshots = (
        replace(snapshots[0], joint_positions=torch.tensor([0.0, 0.4])),
        *snapshots[1:],
    )
    assert planner.validate_qpos(batch, snapshots)[0].checks[0].status == "unavailable"
    assert not backend.queries


def test_sampling_budget_and_missing_backend_capability_are_unavailable():
    planner, robot, backend, snapshots = _fixture(max_validation_samples=2)
    assert (
        planner.validate_qpos(_batch([0]), snapshots)[0].checks[0].status
        == "unavailable"
    )
    backend.supports_joint_trajectory_validation = False
    assert (
        planner.validate_qpos(_batch([0]), snapshots)[0].checks[0].status
        == "unavailable"
    )
    assert not backend.queries


def test_empty_qpos_and_eef_inputs_do_not_invoke_backend():
    planner, robot, backend, snapshots = _fixture()
    assert planner.validate_qpos(_batch([]), snapshots) == ()
    batch, results = planner.plan_eef([], snapshots)
    assert batch.positions.shape == (0, 0, 2)
    assert results == ()
    assert not backend.queries and not robot.ik_calls


def test_nan_candidate_is_failed_before_backend_and_eef_constructor_rejects_nan():
    planner, robot, backend, snapshots = _fixture()
    batch = _batch([0])
    batch.positions[0, 1, 0] = float("nan")
    assert planner.validate_qpos(batch, snapshots)[0].checks[0].status == "failed"
    path = _eef_paths([0], robot)[0]
    poses = path.poses.clone()
    poses[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        replace(path, poses=poses)
    assert not backend.queries


def test_missing_source_mapping_foreign_case_and_stale_roots_are_rejected():
    planner, robot, backend, snapshots = _fixture()
    with pytest.raises(ValueError, match="source_row_indices"):
        planner.validate_qpos(replace(_batch([0]), source_row_indices=None), snapshots)
    with pytest.raises(ValueError, match="identity"):
        planner.validate_qpos(
            replace(_batch([0]), source_row_indices=torch.tensor([1])), snapshots
        )
    robot.roots[0, 0, 3] += 1
    with pytest.raises(ValueError, match="root pose"):
        planner.validate_qpos(_batch([0]), snapshots)
    assert not backend.queries


def test_collision_densification_is_interleaved_with_real_batch_queries(monkeypatch):
    """The adapter never materializes dense samples for every logical candidate."""
    planner, robot, backend, snapshots = _fixture()
    batch = _batch([0, 0, 1, 2, 3, 3, 3, 1, 2, 0, 2])
    events = []
    real_arange = torch.arange
    real_validate = backend.validate_joint_trajectory

    def arange(*args, **kwargs):
        events.append("densify")
        return real_arange(*args, **kwargs)

    def validate(*args, **kwargs):
        events.append("query")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", arange)
    monkeypatch.setattr(backend, "validate_joint_trajectory", validate)
    assert all(result.accepted for result in planner.validate_qpos(batch, snapshots))
    dense_since_query = 0
    for event in events:
        if event == "densify":
            dense_since_query += 1
            assert dense_since_query <= robot.num_instances
        else:
            dense_since_query = 0
    assert events[-1] == "query"


def test_eef_keeps_float64_arrival_intervals_and_solved_joint_samples():
    planner, robot, backend, snapshots = _fixture()
    path = _eef_paths([0], robot)[0]
    dt = torch.tensor([0.0, 0.10000000001, 0.20000000003], dtype=torch.float64)
    solved = torch.tensor(
        [[0.0], [0.02000000001], [0.04000000003]], dtype=torch.float64
    )
    batch, results = planner.plan_eef(
        [replace(path, dt=dt, solved_joint_targets=solved)], snapshots
    )
    assert results[0].accepted
    assert batch.dt.dtype == torch.float64
    assert torch.equal(batch.dt[0], dt)
    assert torch.equal(batch.positions[0, :, 0], solved[:, 0])


def test_missing_locked_joint_configuration_is_unavailable():
    planner, robot, backend, snapshots = _fixture()
    robot.cfg = SimpleNamespace(init_qpos=None)
    result = planner.validate_qpos(_batch([0]), snapshots)[0]
    assert result.checks[0].status == "unavailable"
    assert not backend.queries


def test_post_rollout_qpos_validation_uses_snapshots_without_reading_current_joints():
    planner, robot, backend, snapshots = _fixture()

    def forbid_live_qpos(*args, **kwargs):
        raise AssertionError("post-rollout current joints do not represent the path")

    robot.get_qpos = forbid_live_qpos
    measured = _batch([0])
    result = planner.validate_qpos(measured, snapshots)[0]
    assert result.accepted
    query, obstacles = backend.queries[0]
    assert query[0, -1, 0] == measured.positions[0, -1, 0]
    assert torch.equal(obstacles["obstacle"], robot.roots)


@pytest.mark.parametrize(
    "start_error, accepted", [(0.0, True), (5e-7, True), (5e-6, False)]
)
def test_measured_initial_anchor_has_a_stricter_tolerance_than_host_default(
    start_error, accepted
):
    planner, robot, backend, snapshots = _fixture()
    measured = _batch([0])
    measured.positions[0, 0, 0] += start_error
    result = planner.validate_qpos(measured, snapshots)[0]
    assert result.accepted is accepted
    if not accepted:
        assert result.checks[0].status == "failed"
        assert "does not start" in result.checks[0].detail
        assert not backend.queries


def test_changed_runtime_lock_model_is_unavailable_even_for_matching_new_snapshot():
    from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
        CuroboPlanner,
        CuroboPlannerCfg,
    )
    from embodichain.lab.sim.motion.planners import MoveType

    planner, robot, _, snapshots = _fixture()
    robot.control_parts = {"arm": ["arm"]}
    curobo = object.__new__(CuroboPlanner)
    curobo.robot = robot
    curobo.cfg = CuroboPlannerCfg(robot_uid="cached-model")
    curobo._curobo_device = torch.device("cpu")
    curobo._backend_cache = {
        ("arm", 4, False, MoveType.JOINT_MOVE): SimpleNamespace(
            robot_lock_signature=curobo._locked_joint_signature("arm")
        )
    }
    planner.motion_generator.planner = curobo
    robot.cfg = SimpleNamespace(init_qpos=[0.0, 0.3])
    snapshots = tuple(
        replace(snapshot, joint_positions=torch.tensor([0.0, 0.3]))
        for snapshot in snapshots
    )
    candidate = _batch([0])
    candidate.positions[:, :, 1] = 0.3
    result = planner.validate_qpos(candidate, snapshots)[0]
    assert result.checks[0].status == "unavailable"
    assert "locked-joint configuration changed" in result.checks[0].detail
