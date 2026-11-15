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

"""Cylinder variants preserve physical sizes/inertias and reject unsafe expansion."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp
from newton import GeoType

from embodichain.lab.sim.physics import _cylinder_variants as implementation

pytestmark = pytest.mark.no_sim


def test_conservative_cylinder_variants_validate_before_writing(monkeypatch) -> None:
    model = SimpleNamespace(
        device="cpu",
        shape_type=wp.array(
            [int(GeoType.CYLINDER), int(GeoType.BOX), int(GeoType.CYLINDER)],
            dtype=wp.int32,
            device="cpu",
        ),
        shape_body=wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
        shape_scale=wp.array(
            [[0.03, 0.024, 0], [1, 2, 3], [0.03, 0.024, 0]], dtype=wp.vec3, device="cpu"
        ),
        body_inertia=wp.array(
            np.stack([np.eye(3) * 4] * 3), dtype=wp.mat33, device="cpu"
        ),
        body_inv_inertia=wp.array(
            np.stack([np.eye(3) * 0.25] * 3), dtype=wp.mat33, device="cpu"
        ),
    )
    result = SimpleNamespace(
        world=object(), arenas=SimpleNamespace(names=("default", "arena_0", "arena_1"))
    )
    manager = SimpleNamespace(spawn_result=result, is_newton_backend=True, num_envs=2)
    notify = MagicMock()
    backend = SimpleNamespace(
        model=model,
        solver_type="mujoco_warp",
        _runtime=SimpleNamespace(notify_model_changed=notify),
    )
    monkeypatch.setattr(implementation, "get_newton_backend", lambda _: backend)
    monkeypatch.setattr(
        implementation,
        "resolve_contact_shape_ids",
        lambda _, path: [0] if path.startswith("arena_0/") else [2],
    )
    for ratios in [
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.01]),
        torch.tensor([1.0, float("nan")]),
        torch.ones(1),
    ]:
        with pytest.raises(ValueError):
            implementation._shrink_cylinder_variants(manager, "cylinder", ratios)
    notify.assert_not_called()
    np.testing.assert_allclose(
        model.shape_scale.numpy(), [[0.03, 0.024, 0], [1, 2, 3], [0.03, 0.024, 0]]
    )
    implementation._shrink_cylinder_variants(
        manager, "cylinder", torch.tensor([0.5, 1.0])
    )
    np.testing.assert_allclose(
        model.shape_scale.numpy(), [[0.015, 0.012, 0], [1, 2, 3], [0.03, 0.024, 0]]
    )
    np.testing.assert_allclose(
        model.body_inertia.numpy(), np.stack([np.eye(3), np.eye(3) * 4, np.eye(3) * 4])
    )
    np.testing.assert_allclose(
        model.body_inv_inertia.numpy(),
        np.stack([np.eye(3), np.eye(3) * 0.25, np.eye(3) * 0.25]),
    )
    notify.assert_called_once()
    model.shape_type = wp.array([int(GeoType.BOX)] * 3, dtype=wp.int32, device="cpu")
    with pytest.raises(ValueError, match="analytic cylinder"):
        implementation._shrink_cylinder_variants(manager, "cylinder", torch.ones(2))
    notify.assert_called_once()
