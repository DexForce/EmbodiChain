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

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.atomic_actions.state import (
    EntityState,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.motion.expansion import (
    MotionSnapshot,
    SceneCase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.initial_state import (
    FixedSceneHost,
    InitialStateProfile,
)
from embodichain.lab.trajectory_generation.replicas import (
    CandidateSlotAssignment,
    SceneReplicaPool,
)


def _snapshots(
    source_indices: tuple[int, ...] = (0, 0, 0, 0)
) -> tuple[MotionSnapshot, ...]:
    values = []
    for source in source_indices:
        pose = torch.eye(4)
        pose[0, 3] = source * 0.1
        values.append(
            MotionSnapshot(
                scene_case=SceneCase(
                    f"case-{source}", "initial", "scene", "pick", "robot"
                ),
                joint_names=("arm", "finger"),
                joint_positions=torch.tensor([float(source), 0.04]),
                joint_velocities=torch.zeros(2),
                root_pose=torch.eye(4),
                entity_poses={"cube": pose},
                dependency_revisions={"cube": 1},
            )
        )
    return tuple(values)


def _pool(
    snapshots: tuple[MotionSnapshot, ...] | None = None, **kwargs: object
) -> SceneReplicaPool:
    values = _snapshots() if snapshots is None else snapshots
    kwargs.setdefault("fixed_condition_signatures", ("fixed-conditions",) * len(values))
    return SceneReplicaPool(values, **kwargs)


def _context(snapshots: tuple[MotionSnapshot, ...] | None = None) -> PlanningContext:
    values = _snapshots() if snapshots is None else snapshots
    return PlanningContext(
        robot=RobotObservation(
            timestamp=0.0,
            qpos=torch.stack([value.joint_positions for value in values]),
            qvel=torch.stack([value.joint_velocities for value in values]),
            root_pose=torch.stack([value.root_pose for value in values]),
        ),
        task=TaskState.empty(len(values), "cpu"),
        scene=SceneSnapshot(
            timestamp=0.0,
            version=0,
            entities={
                "cube": EntityState(
                    torch.stack([value.entity_poses["cube"] for value in values])
                )
            },
            collision_entity_ids=("cube",),
        ),
        env_ids=torch.arange(len(values)),
        control_dt=0.01,
    )


def test_one_case_four_physical_replicas_schedule_eight_candidates_in_two_batches() -> (
    None
):
    pool = _pool()
    ids = tuple(f"grasp-{index}" for index in range(8))
    batches = tuple(pool.rounds((0,) * 8, candidate_ids=ids))
    assert [len(batch) for batch in batches] == [4, 4]
    assert pool.source_count == 1 and pool.batch_size == 4
    assert pool.source_rows == (0,)
    assert pool.source_indices == (0, 0, 0, 0)
    for wave, batch in enumerate(batches):
        assert [entry.slot_index for entry in batch] == [0, 1, 2, 3]
        assert [entry.env_id for entry in batch] == [0, 1, 2, 3]
        assert [entry.candidate_index for entry in batch] == list(
            range(wave * 4, (wave + 1) * 4)
        )
        assert all(entry.source_index == entry.source_row == 0 for entry in batch)
        assert all(entry.candidate_id == ids[entry.candidate_index] for entry in batch)
        assert all(entry.host_epoch is None for entry in batch)


def test_multiple_canonical_sources_only_use_their_equivalent_physical_slots() -> None:
    pool = _pool(_snapshots((0, 1, 0, 1)))
    batches = tuple(pool.rounds((0, 1, 0, 1, 0, 1)))
    assert pool.source_rows == (0, 1)
    assert pool.source_indices == (0, 1, 0, 1)
    assert [[entry.candidate_index for entry in batch] for batch in batches] == [
        [0, 1, 2, 3],
        [4, 5],
    ]
    for batch in batches:
        for entry in batch:
            assert pool.source_indices[entry.slot_index] == entry.source_index
            assert entry.source_row == pool.source_rows[entry.source_index]


def test_uneven_sources_do_not_borrow_incompatible_idle_rows() -> None:
    pool = _pool(_snapshots((0, 1, 0, 1)))
    batches = tuple(pool.rounds((1, 1, 1, 1, 1)))
    assert [[entry.slot_index for entry in batch] for batch in batches] == [
        [1, 3],
        [1, 3],
        [1],
    ]
    assert [entry.candidate_index for batch in batches for entry in batch] == list(
        range(5)
    )


def test_no_candidates_returns_no_scheduled_batches() -> None:
    assert tuple(_pool().rounds(())) == ()


def test_signatures_cannot_be_inferred_by_splitting_one_string() -> None:
    with pytest.raises(ValueError, match="signature per row"):
        _pool(fixed_condition_signatures="aaaa")


@pytest.mark.parametrize("source", [-1, 1, True, 0.0, "0"])
def test_invalid_logical_source_is_rejected(source: object) -> None:
    with pytest.raises(ValueError, match="canonical"):
        tuple(_pool().rounds((source,)))


@pytest.mark.parametrize(
    "ids", [("same", "same"), ("only-one",), ("valid", ""), ("valid", None)]
)
def test_candidate_ids_must_match_jobs_without_duplicates(
    ids: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="candidate_ids"):
        tuple(_pool().rounds((0, 0), candidate_ids=ids))


@pytest.mark.parametrize(
    "field",
    [
        "joint_positions",
        "joint_velocities",
        "root_pose",
        "entity_pose",
        "entity_ids",
        "revision",
        "case",
        "signature",
    ],
)
def test_same_case_metadata_or_initial_state_mismatch_is_rejected(field: str) -> None:
    values = list(_snapshots())
    signatures = ["fixed"] * len(values)
    if field in ("joint_positions", "joint_velocities", "root_pose"):
        value = getattr(values[1], field).clone()
        if field == "root_pose":
            value[0, 3] += 0.02
        else:
            value[0] += 0.02
        values[1] = replace(values[1], **{field: value})
    elif field == "entity_pose":
        pose = torch.eye(4)
        pose[0, 3] = 0.02
        values[1] = replace(values[1], entity_poses={"cube": pose})
    elif field == "entity_ids":
        values[1] = replace(values[1], entity_poses={"other-cube": torch.eye(4)})
    elif field == "revision":
        values[1] = replace(values[1], dependency_revisions={"cube": 2})
    elif field == "case":
        values[1] = replace(
            values[1],
            scene_case=replace(values[1].scene_case, scene_signature="other-mesh"),
        )
    else:
        signatures[1] = "different-material"
    with pytest.raises(ValueError, match="replica"):
        _pool(tuple(values), fixed_condition_signatures=signatures)


@pytest.mark.parametrize(
    "env_ids", [(0, 0, 1, 2), (1, 0, 2, 3), (4, 5, 6, 7), (0, 1), (False, 1, 2, 3)]
)
def test_fake_repeated_reordered_or_partial_env_ids_are_rejected(
    env_ids: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="physical env IDs"):
        _pool(env_ids=env_ids)


@pytest.mark.parametrize("atol", [True, -0.1, float("inf"), float("nan")])
def test_invalid_replica_tolerance_is_rejected(atol: object) -> None:
    with pytest.raises(ValueError, match="atol"):
        _pool(atol=atol)


def test_pool_owns_snapshots_and_returns_independent_copies() -> None:
    values = _snapshots()
    pool = _pool(values)
    values[0].joint_positions.fill_(20)
    exported = pool.snapshots
    exported[1].entity_poses["cube"][0, 3] = 30
    source = pool.source_snapshot(0)
    source.root_pose[0, 3] = 40
    assert pool.source_snapshot(0).joint_positions[0] == 0
    assert pool.snapshots[1].entity_poses["cube"][0, 3] == 0
    assert pool.source_snapshot(0).root_pose[0, 3] == 0


def test_matching_context_can_be_reused_without_mutation() -> None:
    pool = _pool()
    context = _context()
    pool.assert_current(context)
    pool.assert_current(context)
    assert torch.equal(context.env_ids, torch.arange(4))
    assert torch.equal(context.robot.qpos[:, 0], torch.zeros(4))


@pytest.mark.parametrize(
    "field", ["qpos", "qvel", "root_pose", "entity_pose", "missing_entity"]
)
def test_context_inputs_must_match_captured_initial_state(field: str) -> None:
    pool, context = _pool(), _context()
    if field in ("qpos", "qvel", "root_pose"):
        value = getattr(context.robot, field).clone()
        if field == "root_pose":
            value[1, 0, 3] += 0.01
        else:
            value[1, 0] += 0.01
        context = replace(context, robot=replace(context.robot, **{field: value}))
    elif field == "entity_pose":
        pose = context.scene.entities["cube"].pose.clone()
        pose[1, 0, 3] += 0.01
        context = replace(
            context, scene=replace(context.scene, entities={"cube": EntityState(pose)})
        )
    else:
        context = replace(context, scene=SceneSnapshot.empty())
    with pytest.raises(ValueError, match="context"):
        pool.assert_current(context)


def test_planning_only_context_requires_root_evidence() -> None:
    context = _context()
    context = replace(context, robot=replace(context.robot, root_pose=None))
    with pytest.raises(ValueError, match="root_pose"):
        _pool().assert_current(context)


def test_fixed_control_dt_and_context_structure_cannot_drift_between_waves() -> None:
    pool, context = _pool(), _context()
    pool.assert_current(context)
    with pytest.raises(RuntimeError, match="context drifted"):
        pool.assert_current(replace(context, control_dt=0.02))


def test_context_extra_entities_must_also_be_replica_equivalent() -> None:
    pool, context = _pool(), _context()
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[2, 0, 3] = 0.1
    context = replace(
        context,
        scene=replace(
            context.scene,
            entities={**context.scene.entities, "goal": EntityState(poses)},
        ),
    )
    with pytest.raises(ValueError, match="replica scene.goal"):
        pool.assert_current(context)


def test_nonfinite_extra_scene_input_is_rejected_even_for_one_replica() -> None:
    snapshots = _snapshots((0,))
    pool, context = _pool(snapshots), _context(snapshots)
    pose = torch.eye(4)
    pose[0, 3] = float("nan")
    context = replace(
        context,
        scene=replace(
            context.scene,
            entities={**context.scene.entities, "goal": EntityState(pose)},
        ),
    )
    with pytest.raises(ValueError, match="finite initial values"):
        pool.assert_current(context)


def test_explicit_physical_uid_to_semantic_entity_mapping_is_verified() -> None:
    context = _context()
    context = replace(
        context,
        scene=replace(
            context.scene,
            entities={"target": context.scene.entities["cube"]},
            collision_entity_ids=("target",),
        ),
    )
    _pool(entity_id_map={"cube": "target"}).assert_current(context)
    with pytest.raises(ValueError, match="missing captured entity"):
        _pool().assert_current(context)


def test_assignment_cannot_invent_an_environment_id() -> None:
    with pytest.raises(ValueError, match="physical slot"):
        CandidateSlotAssignment(0, 0, 0, 0, 100)


class _Adapter:
    """Small read/write state owner for exercising real host token contracts."""

    def __init__(self) -> None:
        self.sim = SimpleNamespace(num_envs=4)
        self.robot = object()
        self.positions = torch.tensor([[0.0, 0.04]]).repeat(4, 1)

    def signature(self) -> str:
        return "physical-robot-and-geometry"

    def capture(self) -> SimpleNamespace:
        return SimpleNamespace(
            joint_names=("arm", "finger"),
            robot={
                "qpos": self.positions.clone(),
                "qvel": torch.zeros_like(self.positions),
                "root_pose": torch.eye(4).repeat(4, 1, 1),
            },
            rigid_objects={"cube": {"pose": torch.eye(4).repeat(4, 1, 1)}},
        )

    def restore(self, state: SimpleNamespace) -> None:
        self.positions.copy_(state.robot["qpos"])

    def verify(self, state: SimpleNamespace) -> ValidationResult:
        status = (
            "passed" if torch.equal(self.positions, state.robot["qpos"]) else "failed"
        )
        return ValidationResult((ValidationCheck("physical_initial", status),))


def _host(adapter: _Adapter, signature: dict[str, str] | None = None) -> FixedSceneHost:
    state = {"value": "trusted-fixed-conditions"} if signature is None else signature
    profile = InitialStateProfile(
        profile_id="replicated-pick",
        prepare=lambda: None,
        signature=lambda: state["value"],
        verify=lambda cases: ValidationResult(
            (ValidationCheck("case_geometry_and_task", "passed"),)
        ),
    )
    return FixedSceneHost(adapter, profile)


def test_host_pool_tracks_epoch_and_rejects_old_assignments_after_restore() -> None:
    adapter = _Adapter()
    with _host(adapter) as host:
        batch = host.acquire_case(tuple(value.scene_case for value in _snapshots()))
        pool = SceneReplicaPool.from_host(host, batch)
        context = _context()
        pool.assert_current(
            replace(context, robot=replace(context.robot, root_pose=None))
        )
        iterator = pool.rounds((0,) * 8)
        first = next(iterator)
        assert all(entry.host_epoch == batch.epoch for entry in first)
        host.restore_initial()
        with pytest.raises(RuntimeError, match="obsolete"):
            next(iterator)


def test_host_pool_checks_live_initial_state_before_every_wave() -> None:
    adapter = _Adapter()
    with _host(adapter) as host:
        batch = host.acquire_case(tuple(value.scene_case for value in _snapshots()))
        pool = SceneReplicaPool.from_host(host, batch)
        iterator = pool.rounds((0,) * 8)
        next(iterator)
        adapter.positions[0, 0] = 0.2
        with pytest.raises(RuntimeError, match="drifted"):
            next(iterator)


def test_host_pool_does_not_accept_fixed_condition_drift() -> None:
    adapter = _Adapter()
    signature = {"value": "original-material"}
    with _host(adapter, signature) as host:
        batch = host.acquire_case(tuple(value.scene_case for value in _snapshots()))
        pool = SceneReplicaPool.from_host(host, batch)
        signature["value"] = "changed-material"
        with pytest.raises(RuntimeError, match="conditions changed"):
            pool.assert_current()


def test_host_pool_cannot_be_created_from_unverified_live_state() -> None:
    adapter = _Adapter()
    with _host(adapter) as host:
        batch = host.acquire_case(tuple(value.scene_case for value in _snapshots()))
        adapter.positions[0, 0] = 0.2
        with pytest.raises(RuntimeError, match="verified host initial"):
            SceneReplicaPool.from_host(host, batch)
