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

"""Fluid state contracts exercised without native simulation."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest
import torch
from embodichain.lab.sim.cfg import FluidObjectCfg, FluidSpawnCfg
from embodichain.lab.sim.objects.fluid import FluidObject

pytestmark = pytest.mark.no_sim


class ParticleBatch:
    def __init__(self, scene, entities):
        self.scene = scene
        self.ids = [e.index for e in entities]

    def fetch_particle_positions(self, out):
        out.copy_(self.scene.state[self.ids, :, :3].reshape(-1, 3))

    def fetch_particle_velocities(self, out):
        out.copy_(self.scene.state[self.ids, :, 3:].reshape(-1, 3))

    def apply_particle_positions(self, value):
        self.scene.state[self.ids, :, :3] = value.reshape(len(self.ids), -1, 3)

    def apply_particle_velocities(self, value):
        self.scene.state[self.ids, :, 3:] = value.reshape(len(self.ids), -1, 3)


@pytest.fixture
def fluid():
    points = np.array([[0.01, 0, 0], [0, 0.01, 0]], dtype=np.float32)
    scene = SimpleNamespace(state=torch.zeros(2, 2, 6))
    scene.state[:, :, :3] = torch.from_numpy(points)
    scene.create_particle_set_batch = lambda entities: ParticleBatch(scene, entities)
    result = FluidObject(
        FluidObjectCfg(uid="water", spawn=FluidSpawnCfg(positions=points)),
        points,
        np.eye(4, dtype=np.float32),
        torch.device("cpu"),
        on_write=lambda ids: None,
    )
    result._initialize_spawn_declaration(2)
    result.attach_spawn_handles(
        [SimpleNamespace(index=i, particle_count=2) for i in range(2)]
    )
    result.bind_spawn(scene)
    return result


def test_reads_do_not_alias_live_or_default_state(fluid):
    before = fluid.data.particle_state_w
    detached = fluid.data.particle_pos_w
    detached.fill_(12)
    torch.testing.assert_close(fluid.data.particle_state_w, before)
    default = fluid.data.default_particle_state_w
    default.fill_(13)
    torch.testing.assert_close(fluid.data.default_particle_state_w, before)


def test_selected_write_and_reset_preserve_other_instance(fluid):
    original = fluid.data.particle_state_w
    changed = original[[1]].clone() + 4
    fluid.write_particle_state_to_sim(changed, [1])
    torch.testing.assert_close(fluid.data.particle_state_w[0], original[0])
    torch.testing.assert_close(fluid.data.particle_state_w[1], changed[0])
    fluid.reset([1])
    torch.testing.assert_close(fluid.data.particle_state_w, original)


def test_reset_to_pose_rotates_template_without_changing_default(fluid):
    original = fluid.data.default_particle_state_w
    q = 2**-0.5
    fluid.reset_to_pose(torch.tensor([[1, 2, 3, 0, 0, q, q]]), [1])
    expected = torch.tensor([[1, 2.01, 3], [0.99, 2, 3]])
    torch.testing.assert_close(fluid.data.particle_pos_w[1], expected)
    torch.testing.assert_close(fluid.data.default_particle_state_w, original)


@pytest.mark.parametrize("ids", [[0, 0], [-1], [2], [0.5]])
def test_invalid_selection_does_not_mutate(fluid, ids):
    before = fluid.data.particle_state_w
    with pytest.raises(ValueError):
        fluid.write_particle_state_to_sim(torch.zeros(len(ids), 2, 6), ids)
    torch.testing.assert_close(fluid.data.particle_state_w, before)


def test_empty_selection_and_invalid_state(fluid):
    fluid.write_particle_state_to_sim(torch.empty(0, 2, 6), [])
    with pytest.raises(ValueError, match="finite"):
        fluid.write_particle_state_to_sim(torch.full((1, 2, 6), float("nan")), [0])
    with pytest.raises(ValueError, match="shape"):
        fluid.write_particle_state_to_sim(torch.zeros(1, 3, 6), [0])


def test_integer_initial_velocity_is_normalized():
    points = np.array([[0, 0, 0]], dtype=np.float32)
    scene = SimpleNamespace(state=torch.zeros(1, 1, 6))
    scene.create_particle_set_batch = lambda entities: ParticleBatch(scene, entities)
    fluid = FluidObject(
        FluidObjectCfg(
            uid="water",
            spawn=FluidSpawnCfg(positions=points, initial_velocity=(1, 0, 0)),
        ),
        points,
        np.eye(4, dtype=np.float32),
        torch.device("cpu"),
        on_write=lambda ids: None,
    )
    fluid._initialize_spawn_declaration(1)
    fluid.attach_spawn_handles([SimpleNamespace(index=0, particle_count=1)])
    fluid.bind_spawn(scene)
    torch.testing.assert_close(fluid.data.particle_vel_w, torch.tensor([[[1.0, 0, 0]]]))
    fluid.reset_to_pose(torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]]))
    torch.testing.assert_close(fluid.data.particle_vel_w, torch.tensor([[[1.0, 0, 0]]]))


@pytest.mark.parametrize("inside_frame", [False, True])
def test_fluid_write_respects_render_frame_cache_scope(inside_frame):
    from embodichain.lab.sim import SimulationManager

    calls = []
    sim = SimpleNamespace(
        physics=SimpleNamespace(invalidate_fluid_state=lambda **kwargs: None),
        _render_state_published=True if inside_frame else None,
        sync_render_state=lambda: calls.append("publish"),
    )
    SimulationManager._invalidate_fluid_state(sim)
    SimulationManager._ensure_render_state(sim)
    SimulationManager._ensure_render_state(sim)
    assert len(calls) == (1 if inside_frame else 2)
    assert sim._render_state_published is (True if inside_frame else None)
