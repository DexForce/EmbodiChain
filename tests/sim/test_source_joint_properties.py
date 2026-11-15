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

"""Source damping remains passive and never changes actuator velocity gains."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp

from embodichain.lab.sim.physics import _source_joint_properties as implementation

pytestmark = pytest.mark.no_sim


def test_source_passive_damping_preserves_actuator_gains(monkeypatch) -> None:
    model = SimpleNamespace(
        device="cpu",
        joint_label=["arena_0/fixed", "arena_0/finger", "arena_0/free"],
        joint_qd_start=wp.array([0, 0, 1, 7], dtype=wp.int32, device="cpu"),
        joint_damping=wp.zeros(7, dtype=wp.float32, device="cpu"),
        joint_target_kd=wp.array([0.4] * 7, dtype=wp.float32, device="cpu"),
    )
    runtime = SimpleNamespace(notify_model_changed=MagicMock())
    backend = SimpleNamespace(model=model, solver_type="mujoco_warp", _runtime=runtime)
    manager = SimpleNamespace(
        spawn_result=SimpleNamespace(world=object()), is_newton_backend=True, num_envs=1
    )
    monkeypatch.setattr(implementation, "get_newton_backend", lambda _world: backend)
    implementation._apply_source_joint_damping(manager, {"finger": 0.003})
    np.testing.assert_allclose(model.joint_damping.numpy(), [0.003, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(model.joint_target_kd.numpy(), [0.4] * 7)
    runtime.notify_model_changed.assert_called_once()
    runtime.notify_model_changed.reset_mock()
    with pytest.raises(ValueError, match="one source joint"):
        implementation._apply_source_joint_damping(manager, {"missing": 0.01})
    np.testing.assert_allclose(model.joint_damping.numpy(), [0.003, 0, 0, 0, 0, 0, 0])
    runtime.notify_model_changed.assert_not_called()
