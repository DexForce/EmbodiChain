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

from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.scene import SceneView
from embodichain.lab.sim.scene.physics import (
    capture_rigid_physics,
    apply_rigid_physics,
    restore_rigid_physics,
)
from ._runtime_ports import Sim, Rigid


class PhysicsBody(Rigid):
    def __init__(self):
        super().__init__(4)
        self.mass = torch.arange(1, 5).float()
        self.inertia = self.mass[:, None].expand(-1, 3).clone()
        self.friction = torch.full((4,), 0.5)
        self.body_data = SimpleNamespace(body_view=SimpleNamespace(is_ready=True))

    def get_mass(self, env_ids):
        return self.mass[env_ids]

    def get_inertia(self, env_ids):
        return self.inertia[env_ids]

    def get_friction(self, env_ids):
        return self.friction[env_ids]

    def set_mass(self, value, env_ids):
        self.mass[env_ids] = value
        self.writes += 1

    def set_inertia(self, value, env_ids):
        self.inertia[env_ids] = value
        self.writes += 1

    def set_friction(self, value, env_ids):
        self.friction[env_ids] = value
        self.writes += 1


def fixture():
    sim = Sim(4)
    sim.rigid = {"a": PhysicsBody(), "b": PhysicsBody()}
    view = SceneView(sim, env_ids=(3, 1))
    return sim, view, capture_rigid_physics(view, ("a", "b"))


def test_scaled_inertia_readback_and_restore_preserve_neighbor_rows():
    sim, view, baseline = fixture()
    kwargs = {
        "mass_scale": {"a": torch.tensor([2.0, 0.5])},
        "friction": {"b": torch.tensor([0.1, 0.9])},
    }
    first = apply_rigid_physics(view, baseline, **kwargs)
    assert first == apply_rigid_physics(view, baseline, **kwargs)
    assert sim.rigid["a"].mass.tolist() == [1, 1, 3, 8]
    torch.testing.assert_close(sim.rigid["a"].inertia[:, 0], sim.rigid["a"].mass)
    restore_rigid_physics(view, baseline)
    assert capture_rigid_physics(view, ("a", "b")) == baseline
    assert sim.steps == 0


@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan"), float("inf")])
def test_later_invalid_body_cannot_partially_modify_earlier_body(value):
    sim, view, baseline = fixture()
    with pytest.raises(ValueError):
        apply_rigid_physics(
            view,
            baseline,
            mass_scale={"a": torch.ones(2), "b": torch.tensor([1.0, value])},
        )
    assert all(body.writes == 0 for body in sim.rigid.values())


def test_unsupported_parameter_and_foreign_baseline_fail_before_writes():
    sim, view, baseline = fixture()
    with pytest.raises(TypeError):
        apply_rigid_physics(view, baseline, damping={"a": torch.ones(2)})
    other, other_view, _ = fixture()
    with pytest.raises(ValueError, match="ownership"):
        apply_rigid_physics(other_view, baseline)
    assert all(
        body.writes == 0 for body in (*sim.rigid.values(), *other.rigid.values())
    )


def test_noop_backend_write_is_detected():
    sim, view, baseline = fixture()
    sim.rigid["b"].set_friction = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="partially modified"):
        apply_rigid_physics(view, baseline, friction={"b": torch.ones(2)})
