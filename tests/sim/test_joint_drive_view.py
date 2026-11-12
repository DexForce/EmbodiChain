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

"""Batched PD updates select names and rows without altering reset defaults."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

from embodichain.lab.sim.physics import _joint_drive_view as implementation

pytestmark = pytest.mark.no_sim


def test_batched_gains_preserve_other_rows_and_notify_once(monkeypatch) -> None:
    model = SimpleNamespace(
        device="cpu",
        joint_label=["arena_0/a", "arena_0/b", "arena_1/a", "arena_1/b"],
        joint_qd_start=wp.array([0, 1, 2, 3, 4], dtype=wp.int32, device="cpu"),
        joint_target_ke=wp.array([1, 2, 3, 4], dtype=wp.float32, device="cpu"),
        joint_target_kd=wp.array([0.1, 0.2, 0.3, 0.4], dtype=wp.float32, device="cpu"),
    )
    result = SimpleNamespace(
        world=object(),
        topology_revision=2,
        arenas=SimpleNamespace(names=("arena_0", "arena_1")),
    )
    runtime = SimpleNamespace(notify_model_changed=MagicMock())
    manager = SimpleNamespace(spawn_result=result, is_newton_backend=True, num_envs=2)
    backend = SimpleNamespace(model=model, _runtime=runtime, solver_type="mujoco_warp")
    monkeypatch.setattr(implementation, "get_newton_backend", lambda _world: backend)
    view = implementation._JointDriveGainView(manager, ("b", "a"))
    view.apply(
        torch.tensor([[8.0, 7.0]]), torch.tensor([[0.8, 0.7]]), torch.tensor([1])
    )
    np.testing.assert_allclose(model.joint_target_ke.numpy(), [1, 2, 7, 8])
    np.testing.assert_allclose(model.joint_target_kd.numpy(), [0.1, 0.2, 0.7, 0.8])
    runtime.notify_model_changed.assert_called_once()
    runtime.notify_model_changed.reset_mock()
    with pytest.raises(ValueError):
        view.apply(
            torch.tensor([[float("nan"), 1.0]]), torch.ones(1, 2), torch.tensor([0])
        )
    np.testing.assert_allclose(model.joint_target_ke.numpy(), [1, 2, 7, 8])
    runtime.notify_model_changed.assert_not_called()
    result.topology_revision += 1
    with pytest.raises(RuntimeError, match="stale"):
        view.apply(torch.ones(2, 2), torch.ones(2, 2))
    view.close()
    with pytest.raises(RuntimeError, match="closed"):
        view.apply(torch.ones(2, 2), torch.ones(2, 2))
