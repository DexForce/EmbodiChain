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

"""Reset contact scaling preserves source ratios, selective rows and lifetime."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

from embodichain.lab.sim.physics import _contact_material_view as implementation

pytestmark = pytest.mark.no_sim


@pytest.fixture
def contact_view(monkeypatch):
    model = SimpleNamespace(
        device="cpu", shape_count=5, mujoco=SimpleNamespace(geom_solimp=None)
    )
    values = (
        [1.0, 2.0, 10.0, 3.0, 4.0],
        [0.1, 0.2, 1.0, 0.3, 0.4],
        [0.01, 0.02, 0.1, 0.03, 0.04],
    )
    for name, value in zip(
        implementation._ContactMaterialScaleView._FIELDS, values, strict=True
    ):
        setattr(model, name, wp.array(value, dtype=wp.float32, device="cpu"))
    result = SimpleNamespace(
        world=object(),
        topology_revision=4,
        arenas=SimpleNamespace(names=("arena_0", "arena_1")),
    )
    runtime = SimpleNamespace(notify_model_changed=MagicMock())
    backend = SimpleNamespace(model=model, _runtime=runtime, solver_type="mujoco_warp")
    manager = SimpleNamespace(spawn_result=result, is_newton_backend=True, num_envs=2)
    monkeypatch.setattr(implementation, "get_newton_backend", lambda _world: backend)
    # The unowned index2 must remain unchanged. Each row owns two shapes.
    monkeypatch.setattr(
        implementation,
        "resolve_contact_shape_ids",
        lambda _result, path: np.array(
            [0, 1] if path.startswith("arena_0/") else [3, 4]
        ),
    )
    view = implementation._ContactMaterialScaleView(manager, ("hand",))
    yield view, model, manager, runtime
    view.close()


def test_contact_scaling_is_source_relative_and_selective(contact_view) -> None:
    view, model, _, runtime = contact_view
    view.apply(torch.tensor([0.5, 2.0]))
    np.testing.assert_allclose(model.shape_material_mu.numpy(), [0.5, 1, 10, 6, 8])
    np.testing.assert_allclose(
        model.shape_material_mu_torsional.numpy(), [0.05, 0.1, 1, 0.6, 0.8]
    )
    np.testing.assert_allclose(
        model.shape_material_mu_rolling.numpy(), [0.005, 0.01, 0.1, 0.06, 0.08]
    )
    view.apply(torch.tensor([0.75]), torch.tensor([0]))
    np.testing.assert_allclose(model.shape_material_mu.numpy(), [0.75, 1.5, 10, 6, 8])
    view.apply(torch.tensor([1.0, 1.0]))
    np.testing.assert_allclose(model.shape_material_mu.numpy(), [1, 2, 10, 3, 4])
    assert runtime.notify_model_changed.call_count == 3


def test_contact_scaling_rejects_stale_or_closed_views(contact_view) -> None:
    view, model, manager, runtime = contact_view
    manager.spawn_result.topology_revision += 1
    with pytest.raises(RuntimeError, match="stale"):
        view.apply(torch.ones(2))
    np.testing.assert_allclose(model.shape_material_mu.numpy(), [1, 2, 10, 3, 4])
    runtime.notify_model_changed.assert_not_called()
    view.close()
    with pytest.raises(RuntimeError, match="closed"):
        view.apply(torch.ones(2))


def test_contact_scaling_rejects_invalid_rows_without_writes(contact_view) -> None:
    view, model, _, runtime = contact_view
    for scales, ids in [
        (torch.tensor([0.0]), torch.tensor([0])),
        (torch.tensor([float("nan")]), torch.tensor([0])),
        (torch.ones(2), torch.tensor([0, 0])),
        (torch.ones(1), torch.tensor([-1])),
        (torch.ones(1), torch.tensor([2])),
    ]:
        with pytest.raises(ValueError):
            view.apply(scales, ids)
    view.apply(torch.empty(0), torch.empty(0, dtype=torch.long))
    np.testing.assert_allclose(model.shape_material_mu.numpy(), [1, 2, 10, 3, 4])
    runtime.notify_model_changed.assert_not_called()


def test_contact_impedance_preserves_unowned_shapes(contact_view) -> None:
    view, model, _, runtime = contact_view
    view.set_impedance((0.95, 0.99, 0.001, 0.5, 2.0))
    values = model.mujoco.geom_solimp.numpy()
    np.testing.assert_allclose(
        values[[0, 1, 3, 4]], np.tile([0.95, 0.99, 0.001, 0.5, 2], (4, 1))
    )
    np.testing.assert_allclose(values[2], [0.9, 0.95, 0.001, 0.5, 2])
    with pytest.raises(ValueError):
        view.set_impedance((0.95, 0.99, float("inf"), 0.5, 2.0))
    assert runtime.notify_model_changed.call_count == 1
