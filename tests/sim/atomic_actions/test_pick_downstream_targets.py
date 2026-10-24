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

"""The lazy candidate-target hook preserves legacy Pick roll and IK ordering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    JointPositionTarget,
    PickUp,
    PickUpOptions,
)


class _LegacyProjection(PickUp):
    def _iter_downstream_eef_targets(self, variants, options):
        # Reference the pre-extraction projection, including lazy shape errors.
        num_envs = variants.shape[0]
        for pose in options.downstream_object_target_poses:
            pose = pose.to(device=self.device, dtype=torch.float32)
            if pose.shape == (4, 4):
                pose = pose.unsqueeze(0).repeat(num_envs, 1, 1)
            if pose.shape != (num_envs, 4, 4):
                raise ValueError(
                    "downstream_object_target_poses entries must have shape "
                    f"(4, 4) or ({num_envs}, 4, 4), but got {pose.shape}."
                )
            yield torch.matmul(pose[:, None, None], variants)


def _bound(action_type, batch):
    action = action_type()
    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"),
        robot=SimpleNamespace(
            compute_fk=lambda **kwargs: torch.eye(4).repeat(batch, 1, 1)
        ),
    )
    return action


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("batched", [False, True])
def test_projection_preserves_targets_seed_chain_rolls_and_rng(batch, count, batched):
    poses = torch.eye(4).repeat(batch, 3, 1, 1)
    poses[..., 0, 3] = torch.arange(batch * 3).reshape(batch, 3) * 0.01
    targets = []
    for index in range(count):
        target = torch.eye(4, dtype=torch.float64)
        target[1, 3] = 0.2 + index * 0.1
        targets.append(target.repeat(batch, 1, 1) if batched else target)
    options = PickUpOptions(downstream_object_target_poses=tuple(targets))
    args = (
        poses,
        torch.zeros(batch, 6),
        torch.eye(4).repeat(batch, 1, 1),
        JointPositionTarget("arm", tuple(range(6))),
        options,
        torch.tensor([0.0, 0.0, 1.0]),
    )
    results, records, states = [], [], []
    for action_type in (_LegacyProjection, PickUp):
        action = _bound(action_type, batch)
        calls = []

        def solve(target, seed, manipulator):
            calls.append((target.clone(), seed.clone(), manipulator))
            valid = torch.ones(target.shape[:3], dtype=torch.bool)
            # Only the less-preferred roll of one column reaches the suffix.
            if len(calls) >= 3:
                valid[:, 1, 0] = False
            expanded = seed[:, None, None] if seed.ndim == 2 else seed
            return valid, expanded.expand(*target.shape[:3], 6) + torch.rand(())

        action._compute_batch_candidate_ik = solve
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            results.append(action._select_feasible_grasp_variants(*args))
            states.append(torch.get_rng_state())
        records.append(calls)
    assert len(records[0]) == len(records[1]) == 3 + count
    for first, second in zip(records[0], records[1], strict=True):
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        assert first[2] == second[2]
    for first, second in zip(*results, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(*states)
    assert results[1][1].all()
    torch.testing.assert_close(
        results[1][0][:, 1, :3, :2], -poses[:, 1, :3, :2], rtol=0, atol=0
    )


@pytest.mark.parametrize("batch", [1, 2])
def test_projection_is_lazy_and_retains_legacy_shape_error(batch):
    action = _bound(PickUp, batch)
    variants = torch.eye(4).repeat(batch, 2, 2, 1, 1)
    options = SimpleNamespace(
        downstream_object_target_poses=(
            torch.eye(4),
            torch.eye(4).repeat(batch + 1, 1, 1),
        )
    )
    targets = action._iter_downstream_eef_targets(variants, options)
    torch.testing.assert_close(next(targets), variants, rtol=0, atol=0)
    with pytest.raises(ValueError, match="downstream_object_target_poses entries"):
        next(targets)


def test_candidate_hook_runs_before_roll_selection():
    class CandidateTargets(PickUp):
        def _iter_downstream_eef_targets(self, variants, options):
            assert variants.shape == (1, 1, 2, 4, 4)
            yield variants

    action = _bound(CandidateTargets, 1)
    calls = []

    def solve(target, seed, manipulator):
        calls.append(target.clone())
        valid = torch.ones(target.shape[:3], dtype=torch.bool)
        if len(calls) == 4:
            valid[..., 0] = False
        return valid, torch.zeros(*target.shape[:3], 6)

    action._compute_batch_candidate_ik = solve
    selected, success = action._select_feasible_grasp_variants(
        torch.eye(4).repeat(1, 1, 1, 1),
        torch.zeros(1, 6),
        torch.eye(4).repeat(1, 1, 1),
        JointPositionTarget("arm", tuple(range(6))),
        PickUpOptions(),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert len(calls) == 4 and success.all()
    torch.testing.assert_close(selected[0, 0, :3, :2], -torch.eye(3)[:, :2])
