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

import json
import types

import numpy as np
import pytest
import torch

from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.workspace import RobotWorkspace, RobotWorkspaceCfg


def test_load_joint_workspace_cache(tmp_path):
    """Joint-space cache points remain aligned with joint configurations."""
    entry = tmp_path / "cache-entry"
    entry.mkdir()
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    qpos = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    np.savez(
        entry / "results.npz", workspace_points=positions, joint_configurations=qpos
    )
    (entry / "meta.json").write_text(
        json.dumps({"mode": "joint_space"}), encoding="utf-8"
    )

    workspace = RobotWorkspace.from_cache(entry)

    assert workspace.num_samples == 2
    assert torch.allclose(workspace.positions, torch.from_numpy(positions))
    assert torch.allclose(workspace.qpos, torch.from_numpy(qpos))
    assert workspace.metadata["mode"] == "joint_space"


def test_load_cartesian_workspace_uses_reachable_points_and_scores(tmp_path):
    """Cartesian caches select reachable points and aligned success rates."""
    all_points = np.arange(12, dtype=np.float32).reshape(4, 3)
    reachable = all_points[[1, 3]]
    mask = np.array([False, True, False, True])
    scores = np.array([0.0, 0.8, 0.0, 0.9], dtype=np.float32)
    qpos = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    np.savez(
        tmp_path / "results.npz",
        workspace_points=all_points,
        reachable_points=reachable,
        joint_configurations=qpos,
        success_rates=scores,
        reachability_mask=mask,
    )

    workspace = RobotWorkspace.from_cache(tmp_path / "results.npz")

    assert torch.allclose(workspace.positions, torch.from_numpy(reachable))
    assert torch.allclose(workspace.scores, torch.tensor([0.8, 0.9]))


def test_workspace_rejects_unaligned_cache(tmp_path):
    """A cache without a point array aligned to qpos is rejected."""
    np.savez(
        tmp_path / "results.npz",
        workspace_points=np.zeros((3, 3), dtype=np.float32),
        joint_configurations=np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="aligned"):
        RobotWorkspace.from_cache(tmp_path / "results.npz")


def test_voxel_uniform_sampling_returns_cache_indices():
    """Voxel-uniform sampling returns valid indices with the requested shape."""
    workspace = RobotWorkspace(
        positions=torch.tensor(
            [[0.00, 0.00, 0.00], [0.01, 0.00, 0.00], [1.00, 0.00, 0.00]]
        ),
        qpos=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        voxel_size=0.1,
    )
    generator = torch.Generator().manual_seed(7)

    indices = workspace.sample_indices(
        20, strategy="voxel_uniform", generator=generator
    )

    assert indices.shape == (20,)
    assert torch.logical_and(indices >= 0, indices < workspace.num_samples).all()


def test_robot_sample_reachable_pose_applies_runtime_fk_bounds():
    """Robot sampling filters using per-environment FK positions."""
    workspace = RobotWorkspace(
        positions=torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        qpos=torch.tensor([[0.1, 0.0], [0.9, 0.0]]),
    )
    robot = object.__new__(Robot)
    robot.device = torch.device("cpu")
    robot._all_indices = torch.tensor([0, 1])
    robot._workspaces = {"arm": workspace}
    robot.cfg = types.SimpleNamespace(
        control_parts={"arm": ["joint_1", "joint_2"]},
        workspace_cfg={
            "arm": RobotWorkspaceCfg(
                cache_path="unused",
                strategy="point_uniform",
            )
        },
    )
    fk_call = {}

    def fake_compute_batch_fk(
        self, qpos, name, env_ids=None, to_matrix=False
    ) -> torch.Tensor:
        del self, env_ids
        fk_call["name"] = name
        poses = (
            torch.eye(4, dtype=qpos.dtype)
            .reshape(1, 1, 4, 4)
            .repeat(qpos.shape[0], qpos.shape[1], 1, 1)
        )
        poses[:, :, 0, 3] = qpos[:, :, 0]
        return poses if to_matrix else poses[:, :, :3, 3]

    robot.compute_batch_fk = types.MethodType(fake_compute_batch_fk, robot)

    samples = robot.sample_reachable_pose(
        env_ids=torch.tensor([0, 1]),
        num_samples=1,
        strategy="point_uniform",
        position_bounds=([0.8, -0.1, -0.1], [1.0, 0.1, 0.1]),
        max_attempts=64,
        generator=torch.Generator().manual_seed(11),
    )

    assert samples.valid.all()
    assert fk_call["name"] == "arm"
    assert torch.all(samples.eef_pose[:, 0, 0, 3] >= 0.8)


def test_robot_sample_reachable_pose_marks_failed_bounds_invalid():
    """Filters that reject every cached pose return an explicit invalid mask."""
    workspace = RobotWorkspace(
        positions=torch.tensor([[0.1, 0.0, 0.0]]),
        qpos=torch.tensor([[0.1, 0.0]]),
    )
    robot = object.__new__(Robot)
    robot.device = torch.device("cpu")
    robot._all_indices = torch.tensor([0])
    robot._workspaces = {"arm": workspace}
    robot.cfg = types.SimpleNamespace(
        control_parts={"arm": ["joint_1", "joint_2"]},
        workspace_cfg={
            "arm": RobotWorkspaceCfg(
                cache_path="unused",
                strategy="point_uniform",
            )
        },
    )

    def fake_compute_batch_fk(
        self, qpos, name, env_ids=None, to_matrix=False
    ) -> torch.Tensor:
        del self, name, env_ids
        poses = (
            torch.eye(4, dtype=qpos.dtype)
            .reshape(1, 1, 4, 4)
            .repeat(qpos.shape[0], qpos.shape[1], 1, 1)
        )
        poses[:, :, 0, 3] = qpos[:, :, 0]
        return poses

    robot.compute_batch_fk = types.MethodType(fake_compute_batch_fk, robot)

    samples = robot.sample_reachable_pose(
        name="arm",
        env_ids=torch.tensor([0]),
        position_bounds=([2.0, -0.1, -0.1], [3.0, 0.1, 0.1]),
        max_attempts=4,
    )

    assert not samples.valid.any()
    assert samples.indices.item() == -1
