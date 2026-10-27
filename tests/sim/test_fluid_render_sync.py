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

"""Fluid render clipping preserves physics state and publication ownership."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp

from embodichain.lab.sim.physics.newton import NewtonPhysicsBackend

pytestmark = pytest.mark.no_sim


def _backend() -> NewtonPhysicsBackend:
    return NewtonPhysicsBackend(SimpleNamespace(has_native_renderer=True))


@pytest.mark.parametrize(
    "ranges",
    [((0, 5, 0.0),), ((1, 3, 0.0),), ((0, 2, 0.0), (3, 5, 1.0))],
)
def test_clip_only_hides_selected_particles_below_world_height(ranges):
    source = torch.tensor(
        [[1, 2, -1], [2, 3, 0], [3, 4, -2], [4, 5, 0.5], [5, 6, 1]],
        dtype=torch.float32,
    )
    before = source.clone()
    expected = source.clone()
    for start, end, height in ranges:
        for index in range(start, end):
            if source[index, 2] < height:
                expected[index] = float("nan")

    rendered = _backend()._clip_fluid_render_positions(source, ranges)

    torch.testing.assert_close(rendered, expected, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert rendered.data_ptr() != source.data_ptr()


def test_repeated_publications_do_not_overwrite_previous_render_input():
    backend = _backend()
    source = torch.tensor([[1.0, 2, -1], [3, 4, 1], [5, 6, -2]])
    ranges = ((0, 1, 0.0), (2, 3, -3.0))
    first = backend._clip_fluid_render_positions(source, ranges)
    saved = first.clone()
    source[:, 2] += 5

    second = backend._clip_fluid_render_positions(source, ranges)

    torch.testing.assert_close(first, saved, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(second, source, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()


def test_clip_layout_cache_tracks_height_ranges_shape_and_dtype():
    backend = _backend()
    source = torch.tensor([[1.0, 2, -1], [3, 4, -2], [5, 6, 1]])
    first_ranges = ((0, 1, 0.0), (2, 3, 2.0))
    backend._clip_fluid_render_positions(source, first_ranges)
    cached = backend._fluid_render_clip_z
    backend._clip_fluid_render_positions(source + 1, first_ranges)
    assert backend._fluid_render_clip_z is cached

    changed = backend._clip_fluid_render_positions(source, ((1, 2, -3.0),))
    torch.testing.assert_close(changed, source, rtol=0, atol=0)
    assert backend._fluid_render_clip_z is not cached

    for resized in (source[:2], source.to(torch.float64)):
        actual = backend._clip_fluid_render_positions(resized, ((0, 1, -3.0),))
        torch.testing.assert_close(actual, resized, rtol=0, atol=0)
        assert actual.dtype == resized.dtype


def test_single_fluid_needs_no_per_particle_height_cache():
    backend = _backend()
    source = torch.ones(3, 3)
    backend._clip_fluid_render_positions(source, ((1, 2, 0.0),))

    result = backend._clip_fluid_render_positions(source, ((0, 3, 0.0),))

    torch.testing.assert_close(result, source)
    assert backend._fluid_render_clip_z is None
    assert backend._fluid_render_clip_key is None


def test_native_publication_preserves_body_state_time_and_unclipped_ranges(monkeypatch):
    from dexsim.engine.newton_physics import backend_registry

    wp.init()
    source = torch.tensor([[1.0, 2, -1], [3, 4, -2], [5, 6, -3], [7, 8, 1]])
    before = source.clone()
    published = []

    def publish(world, body_q, particle_q, *, simulation_time):
        published.append((world, body_q, wp.to_torch(particle_q), simulation_time))

    native = SimpleNamespace(
        state_0=SimpleNamespace(
            particle_q=wp.from_torch(source, dtype=wp.vec3), body_q=object()
        ),
        sim_time=4.5,
        sync_to_dexsim=Mock(),
        render_sync=SimpleNamespace(sync=publish),
    )
    monkeypatch.setattr(backend_registry, "get_newton_backend", lambda world: native)
    backend = _backend()
    backend._manager._fluid_objects = {
        "clipped": SimpleNamespace(
            cfg=SimpleNamespace(render=SimpleNamespace(clip_below_z=0.0)),
            _entities=[
                SimpleNamespace(
                    _physics_binding=SimpleNamespace(particle_start=start),
                    particle_count=1,
                )
                for start in (0, 3)
            ],
        ),
        "unclipped": SimpleNamespace(
            cfg=SimpleNamespace(render=SimpleNamespace(clip_below_z=None))
        ),
    }
    scene = SimpleNamespace(world=object())

    backend.sync_render_state(scene)

    world, body_q, particles, sim_time = published[0]
    assert world is scene.world
    assert body_q is native.state_0.body_q
    assert sim_time == native.sim_time
    assert torch.isnan(particles[0]).all()
    torch.testing.assert_close(particles[1:], source[1:], rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    native.sync_to_dexsim.assert_not_called()

    backend._manager._fluid_objects["clipped"].cfg.render.clip_below_z = None
    backend.sync_render_state(scene)
    native.sync_to_dexsim.assert_called_once_with(scene.world)
    assert backend._fluid_render_clip_z is None


def test_teardown_releases_clip_metadata_even_without_a_world():
    backend = _backend()
    backend._clip_fluid_render_positions(torch.ones(3, 3), ((1, 2, 0.0),))

    backend.prepare_for_teardown()

    assert backend._fluid_render_clip_z is None
    assert backend._fluid_render_clip_key is None


@pytest.mark.gpu
def test_clip_cache_is_safe_across_nondefault_cuda_streams():
    backend = _backend()
    source = torch.tensor([[1.0, 2, -1], [3, 4, 1], [5, 6, -2]], device="cuda")
    ranges = ((0, 1, 0.0), (2, 3, -3.0))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    initial = torch.cuda.current_stream()
    results = []
    for index in (0, 1, 0):
        stream = streams[index]
        stream.wait_stream(initial)
        with torch.cuda.stream(stream):
            results.append(backend._clip_fluid_render_positions(source, ranges))
    torch.cuda.synchronize()

    expected = source.clone()
    expected[0] = float("nan")
    for result in results:
        torch.testing.assert_close(result, expected, equal_nan=True, rtol=0, atol=0)
