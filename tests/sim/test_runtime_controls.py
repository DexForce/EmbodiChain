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

"""Tests for manager-owned Newton runtime-control adapters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from embodichain.lab.sim._runtime_controls import (
    _KinematicNodalTrajectoryControl,
)

pytestmark = pytest.mark.no_sim


class _ArrayView:
    """Host-array stand-in for a borrowed Warp particle view."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        """Return a host snapshot matching Warp's ``numpy`` method."""
        return self.values.copy()


class _ParticleSet:
    """Minimal particle-set facade used by the runtime-control tests."""

    def __init__(self) -> None:
        self.positions = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        self.fixed_indices: np.ndarray | None = None

    @property
    def particle_count(self) -> int:
        """Return the number of test particles."""
        return len(self.positions)

    def get_particle_positions(self) -> _ArrayView:
        """Return the current particle positions."""
        return _ArrayView(self.positions)

    def set_particle_positions(self, positions: np.ndarray) -> None:
        """Store one complete position snapshot."""
        self.positions = np.asarray(positions, dtype=np.float32).copy()

    def fix_particles(self, particle_indices: np.ndarray) -> None:
        """Record which particles were made kinematic."""
        self.fixed_indices = np.asarray(particle_indices, dtype=np.int32).copy()


def test_kinematic_nodal_control_interpolates_offsets_per_substep() -> None:
    particle_set = _ParticleSet()
    solver = SimpleNamespace(rebuild_bvh=MagicMock())
    current_state = object()
    context = SimpleNamespace(
        result=SimpleNamespace(get_particle_set=lambda _target: particle_set),
        solver=solver,
        current_state=current_state,
    )
    offsets = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    control = _KinematicNodalTrajectoryControl(
        "arena_0/cloth",
        np.asarray([1], dtype=np.int32),
        offsets,
        fps=10.0,
        rebuild_self_contact_bvh=True,
    )

    control.initialize(context)
    control(context, substep_index=0, substep_count=2, substep_dt=0.05)
    np.testing.assert_allclose(particle_set.positions[1], [2.0, 0.0, 0.0])
    control(context, substep_index=1, substep_count=2, substep_dt=0.05)

    assert particle_set.fixed_indices is None
    np.testing.assert_allclose(particle_set.positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(particle_set.positions[1], [2.0, 1.0, 0.0])
    solver.rebuild_bvh.assert_called_once_with(current_state)
    assert control.exclusive_resource_claims() == (
        ("kinematic_nodal_trajectory", "arena_0/cloth"),
    )


def test_kinematic_nodal_control_holds_last_unrated_sample() -> None:
    particle_set = _ParticleSet()
    context = SimpleNamespace(
        result=SimpleNamespace(get_particle_set=lambda _target: particle_set),
        solver=object(),
        current_state=object(),
    )
    offsets = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    control = _KinematicNodalTrajectoryControl(
        "arena_0/cloth",
        np.asarray([1], dtype=np.int32),
        offsets,
        fps=None,
        rebuild_self_contact_bvh=False,
    )
    control.initialize(context)

    for _ in range(3):
        control(context, substep_index=0, substep_count=1, substep_dt=0.01)

    np.testing.assert_allclose(particle_set.positions[1], [3.0, 0.0, 0.0])
