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

"""Object-local workspace sampling and actual manipulation-pose checks."""

from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from embodichain.lab.sim.motion.workspace.runtime import WorkspaceSample
from embodichain.lab.sim.scene_expansion import ScenePoseChange
from embodichain.lab.task_program.integrations.simulation.workspace import (
    RobotSceneWorkspace,
    SceneWorkspaceCandidate,
)

_PART = "right_arm"
_DOF = 2


class _Robot:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls = []
        self.invalidate_last = False
        self.base_x = 3.0

    def get_qpos(self) -> torch.Tensor:
        return torch.zeros(1, _DOF)

    def get_solver(self, *, name: str) -> object:
        assert name == _PART
        return SimpleNamespace(dof=_DOF)

    def sample_reachable_pose(self, **kwargs) -> WorkspaceSample:
        self.calls.append(("sample", kwargs))
        count = kwargs["num_samples"]
        qpos = torch.rand(1, count, _DOF, generator=kwargs["generator"])
        pose = torch.eye(4).repeat(1, count, 1, 1)
        # A rotated TCP and translated current base expose frame mistakes.
        pose[:, :, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        pose[:, :, 0, 3] = qpos[:, :, 0] + self.base_x
        pose[:, :, 1, 3] = 2.0
        valid = torch.ones(1, count, dtype=torch.bool)
        indices = torch.arange(count).unsqueeze(0)
        if self.invalidate_last:
            valid[0, -1] = False
            indices[0, -1] = -1
            pose[0, -1] = torch.eye(4)
            qpos[0, -1] = 0
        return WorkspaceSample(pose, qpos, indices, valid)

    def compute_ik(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(("ik", kwargs))
        # Orientation matters even when the object's center is reachable.
        success = kwargs["pose"][:, 0, 0] > 0.5
        return success, torch.zeros(1, _DOF)


def _grasp() -> torch.Tensor:
    grasp = torch.eye(4)
    grasp[0, 3] = 0.25  # Grasp is displaced along the object's local x axis.
    return grasp


def test_workspace_proposals_respect_object_local_transform_and_current_base() -> None:
    robot = _Robot()
    workspace = RobotSceneWorkspace(robot, control_part=_PART)
    candidates = workspace.sample_object_poses("cube", _grasp(), num_samples=2, seed=7)
    kwargs = robot.calls[0][1]
    assert kwargs["name"] == _PART
    assert kwargs["env_ids"] == [0]
    assert len(candidates) == 2
    for candidate in candidates:
        object_pose = torch.tensor(candidate.pose_change.arena_pose)
        tcp = object_pose @ _grasp()
        assert tcp[0, 3].item() == pytest.approx(robot.base_x + candidate.joint_seed[0])
        assert object_pose[1, 3].item() == pytest.approx(1.75)
    robot.base_x += 1
    shifted = workspace.sample_object_poses("cube", _grasp(), num_samples=2, seed=7)
    assert shifted[0].pose_change.arena_pose[0][3] - candidates[
        0
    ].pose_change.arena_pose[0][3] == pytest.approx(1)


def test_invalid_workspace_padding_never_becomes_a_scene_candidate() -> None:
    robot = _Robot()
    robot.invalidate_last = True
    candidates = RobotSceneWorkspace(robot, control_part=_PART).sample_object_poses(
        "cube", _grasp(), num_samples=3, seed=4
    )
    assert len(candidates) == 2
    assert all(candidate.cache_index >= 0 for candidate in candidates)


def test_seeded_sampling_is_reproducible_without_changing_global_rng() -> None:
    workspace = RobotSceneWorkspace(_Robot(), control_part=_PART)
    before = torch.random.get_rng_state().clone()
    first = workspace.sample_object_poses("cube", _grasp(), num_samples=3, seed=24)
    second = workspace.sample_object_poses("cube", _grasp(), num_samples=3, seed=24)
    assert first == second
    assert torch.equal(before, torch.random.get_rng_state())


def test_bounds_filter_object_origin_after_inverse_grasp_composition() -> None:
    workspace = RobotSceneWorkspace(_Robot(), control_part=_PART)
    candidates = workspace.sample_object_poses(
        "cube",
        _grasp(),
        num_samples=2,
        seed=1,
        position_bounds=((3.0, 1.7, -0.1), (4.0, 1.8, 0.1)),
        max_attempts=8,
    )
    assert len(candidates) == 2
    empty = workspace.sample_object_poses(
        "cube",
        _grasp(),
        num_samples=2,
        seed=1,
        position_bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        max_attempts=8,
    )
    assert empty == ()


def test_actual_pose_ik_checks_grasp_orientation_with_no_cache_lookup() -> None:
    robot = _Robot()
    workspace = RobotSceneWorkspace(robot, control_part=_PART)
    pose = torch.eye(4)
    pose[0, 3] = 0.4
    change = ScenePoseChange("cube", pose.tolist())
    assert workspace.check_object_pose(change, _grasp(), joint_seed=(0.1, 0.2)).accepted
    grasp = _grasp()
    grasp[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert not workspace.check_object_pose(change, grasp).accepted
    assert all(call[0] == "ik" for call in robot.calls)
    kwargs = robot.calls[0][1]
    assert torch.allclose(kwargs["pose"][0], pose @ _grasp())
    assert kwargs["name"] == _PART
    assert kwargs["env_ids"] == [0]
    assert torch.allclose(kwargs["joint_seed"], torch.tensor([[0.1, 0.2]]))


def test_modified_workspace_proposal_is_rechecked_instead_of_inheriting_validity() -> (
    None
):
    workspace = RobotSceneWorkspace(_Robot(), control_part=_PART)
    sampled = workspace.sample_object_poses("cube", _grasp(), num_samples=1, seed=0)[0]
    # The sampled TCP has a 90-degree yaw; adding a -90-degree grasp makes
    # the actual target orientation solvable by the fixture's IK criterion.
    assert not workspace.check_object_pose(sampled.pose_change, _grasp()).accepted
    modified = torch.tensor(sampled.pose_change.arena_pose)
    modified[:3, :3] = torch.eye(3)
    assert workspace.check_object_pose(
        ScenePoseChange("cube", modified.tolist()), _grasp()
    ).accepted


@pytest.mark.parametrize("part", ["", " right_arm", None])
def test_control_part_must_be_explicit(part: str | None) -> None:
    with pytest.raises(ValueError, match="control_part"):
        RobotSceneWorkspace(_Robot(), control_part=part)


def test_malformed_grasp_or_seed_fails_before_robot_query() -> None:
    robot = _Robot()
    workspace = RobotSceneWorkspace(robot, control_part=_PART)
    change = ScenePoseChange("cube", torch.eye(4).tolist())
    with pytest.raises(ValueError, match="rigid transform"):
        workspace.check_object_pose(change, torch.eye(3))
    with pytest.raises(ValueError, match="joint_seed"):
        workspace.check_object_pose(change, _grasp(), joint_seed=(0.1,))
    assert robot.calls == []


def test_direct_candidate_construction_owns_mutable_joint_seed() -> None:
    seed = [0.1, 0.2]
    candidate = SceneWorkspaceCandidate(
        ScenePoseChange("cube", torch.eye(4).tolist()), seed, 0, 0.7
    )
    seed[0] = 9.0
    assert candidate.joint_seed == (0.1, 0.2)


@pytest.mark.parametrize(
    "seed", [[], None, [True], [float("nan")], [float("inf")], ["0.1"]]
)
def test_direct_candidate_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="joint_seed"):
        SceneWorkspaceCandidate(
            ScenePoseChange("cube", torch.eye(4).tolist()), seed, 0, None
        )


@pytest.mark.parametrize("index", [-1, True, 0.0])
def test_direct_candidate_rejects_invalid_cache_index(index: object) -> None:
    with pytest.raises(ValueError, match="cache_index"):
        SceneWorkspaceCandidate(
            ScenePoseChange("cube", torch.eye(4).tolist()), (0.1,), index, None
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.7"])
def test_direct_candidate_rejects_invalid_score(score: object) -> None:
    with pytest.raises(ValueError, match="score"):
        SceneWorkspaceCandidate(
            ScenePoseChange("cube", torch.eye(4).tolist()), (0.1,), 0, score
        )


def test_direct_candidate_requires_scene_pose_change() -> None:
    with pytest.raises(TypeError, match="ScenePoseChange"):
        SceneWorkspaceCandidate(object(), (0.1,), 0, None)


def test_sampling_rejects_nonfinite_valid_workspace_score() -> None:
    robot = _Robot()
    sample = robot.sample_reachable_pose

    def nonfinite_score(**kwargs) -> WorkspaceSample:
        result = sample(**kwargs)
        return replace(result, score=torch.full(result.indices.shape, float("nan")))

    robot.sample_reachable_pose = nonfinite_score
    workspace = RobotSceneWorkspace(robot, control_part=_PART)
    with pytest.raises(ValueError, match="scores must be finite"):
        workspace.sample_object_poses("cube", _grasp(), num_samples=1, seed=0)
