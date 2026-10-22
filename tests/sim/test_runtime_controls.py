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
import torch

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


def test_initial_joint_seed_uses_source_order_and_never_drives_runtime(monkeypatch):
    from embodichain.lab.sim._runtime_controls import _InitialJointState

    initializer = MagicMock()
    factory = MagicMock(return_value=initializer)
    monkeypatch.setattr(
        "dexsim.engine.newton_physics.KinematicJointTrajectoryControl", factory
    )
    handle = SimpleNamespace(
        get_actived_joint_names=lambda: ["source_a", "source_b"],
        joint_dof_layout=[
            SimpleNamespace(name="source_b"),
            SimpleNamespace(name="source_a"),
        ],
    )
    batch = MagicMock()
    result = SimpleNamespace(
        get_articulation=lambda target: handle,
        create_articulation_batch=MagicMock(return_value=batch),
    )
    control = _InitialJointState("arena/tool", np.array([0.1, 0.2]))
    control.initialize_model(object(), result)
    np.testing.assert_allclose(factory.call_args.args[1], [[0.2, 0.1]])
    assert factory.call_args.kwargs["initialize_model_pose"] is True
    control.initialize(SimpleNamespace(result=result))
    np.testing.assert_allclose(
        batch.apply_state.call_args.kwargs["joint_target_position"], [[0.2, 0.1]]
    )
    batch.reset_mock()
    control(None, 0, 4, 1 / 240)
    control.prepare_episode_reset((0,))()
    batch.apply_state.assert_not_called()
    control.close()
    initializer.close.assert_called_once()


@pytest.mark.parametrize(
    "value",
    [[0.1, 0.2], np.array([0.1, 0.2]), torch.tensor([0.1, 0.2], requires_grad=True)],
)
def test_initial_joint_seed_copies_supported_inputs(value):
    from embodichain.lab.sim._runtime_controls import _InitialJointState

    control = _InitialJointState("arena/tool", value)
    np.testing.assert_allclose(control.positions, [[0.1, 0.2]])
    assert control.positions.dtype == np.float32
    if isinstance(value, torch.Tensor):
        with torch.no_grad():
            value.zero_()
    elif isinstance(value, np.ndarray):
        value[:] = 0
    else:
        value[:] = [0, 0]
    np.testing.assert_allclose(control.positions, [[0.1, 0.2]])


@pytest.mark.parametrize(
    "value", [[[0.1]], [float("nan")], [float("inf")], [1e100], [True], ["0.1"]]
)
def test_initial_joint_seed_rejects_invalid_vectors(value):
    from embodichain.lab.sim._runtime_controls import _InitialJointState

    with pytest.raises(ValueError, match="init_qpos"):
        _InitialJointState("arena/tool", value)


def test_initial_joint_seed_accepts_bfloat16_tensor():
    from embodichain.lab.sim._runtime_controls import _InitialJointState

    control = _InitialJointState(
        "arena/tool", torch.tensor([0.125, 0.25], dtype=torch.bfloat16)
    )
    np.testing.assert_array_equal(control.positions, [[0.125, 0.25]])


@pytest.mark.gpu
def test_initial_joint_seed_accepts_cuda_tensor():
    from embodichain.lab.sim._runtime_controls import _InitialJointState

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    control = _InitialJointState("arena/tool", torch.tensor([0.025], device="cuda"))
    np.testing.assert_allclose(control.positions, [[0.025]])
