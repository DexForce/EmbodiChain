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

import numpy as np
import pytest
import torch
from dexsim.types import DriveType

from embodichain.lab.sim.objects.articulation import Articulation

pytestmark = pytest.mark.no_sim


def test_newton_target_modes_map_to_portable_drive_types() -> None:
    target_modes = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    entity = SimpleNamespace(
        get_newton_drive=lambda: (None, None, None, None, None, None, target_modes)
    )
    articulation = object.__new__(Articulation)
    articulation._data = SimpleNamespace(
        is_newton_backend=True,
        dof=len(target_modes),
    )
    articulation._all_indices = np.asarray([0], dtype=np.int32)
    articulation._entities = [entity]

    assert articulation.get_joint_drive_type() == [
        [
            DriveType.NONE,
            DriveType.FORCE,
            DriveType.FORCE,
            DriveType.FORCE,
            DriveType.NONE,
        ]
    ]


def test_newton_drive_type_query_honors_joint_selection() -> None:
    target_modes = np.asarray([0, 3, 0], dtype=np.int32)
    entity = SimpleNamespace(
        get_newton_drive=lambda: (None, None, None, None, None, None, target_modes)
    )
    articulation = object.__new__(Articulation)
    articulation._data = SimpleNamespace(is_newton_backend=True, dof=3)
    articulation._all_indices = np.asarray([0], dtype=np.int32)
    articulation._entities = [entity]

    assert articulation.get_joint_drive_type(joint_ids=[2, 1]) == [
        [DriveType.NONE, DriveType.FORCE]
    ]


def test_runtime_effort_mode_disables_pd_gains_on_newton() -> None:
    calls: list[dict[str, object]] = []
    entity = SimpleNamespace(set_newton_drive=lambda **kwargs: calls.append(kwargs))
    articulation = object.__new__(Articulation)
    articulation._spawn_result = object()
    articulation._entities = [entity]
    articulation._all_indices = np.asarray([0], dtype=np.int32)
    articulation._data = SimpleNamespace(is_newton_backend=True, dof=1)
    articulation.device = torch.device("cpu")

    articulation.set_joint_drive(
        stiffness=torch.tensor([[12.0]]),
        damping=torch.tensor([[4.0]]),
        drive_type="force",
        target_mode="effort",
    )

    assert len(calls) == 1
    assert calls[0]["target_mode"] == 4
    assert calls[0]["target_ke"] == 0.0
    assert calls[0]["target_kd"] == 0.0


@pytest.mark.parametrize("override_impedance", [False, True])
def test_mimic_profile_only_changes_selected_articulation_equalities(
    monkeypatch, override_impedance: bool
) -> None:
    from dexsim.engine.newton_physics import backend_registry
    from embodichain.lab.sim.objects.backends.newton import (
        _configure_newton_mimic_compliance,
    )

    class Array:
        def __init__(self, values) -> None:
            self.values = np.asarray(values).copy()

        def numpy(self):
            return self.values

        def assign(self, values) -> None:
            self.values[:] = values

    initial_ref = np.tile([0.02, 1.0], (1, 3, 1))
    initial_imp = np.tile([0.9, 0.95, 0.001, 0.5, 2.0], (1, 3, 1))
    mjw_model = SimpleNamespace(
        eq_solref=Array(initial_ref), eq_solimp=Array(initial_imp)
    )
    cpu = SimpleNamespace(
        eq_solref=initial_ref[0].copy(), eq_solimp=initial_imp[0].copy()
    )
    solver = SimpleNamespace(
        mjc_eq_to_newton_mimic=Array([[-1, 0, 1]]), mjw_model=mjw_model, mj_model=cpu
    )
    model = SimpleNamespace(
        requires_grad=False,
        joint_target_ke=Array([100.0, 0.0, 100.0, 0.0]),
        joint_target_kd=Array([10.0, 0.0, 10.0, 0.0]),
        joint_target_mode=Array([1, 0, 1, 0]),
        constraint_mimic_joint0=Array([1, 3]),
        constraint_mimic_joint1=Array([0, 2]),
    )
    runtime = SimpleNamespace(model=model, solver=solver)
    binding = SimpleNamespace(
        _runtime=runtime,
        joints=[
            SimpleNamespace(name=name, joint_id=i, qd_size=1, qd_start=i)
            for i, name in enumerate(["leader", "follower"])
        ],
    )
    entity = SimpleNamespace(
        _physics_binding=binding, set_newton_drive=lambda **kwargs: 1
    )
    backend = SimpleNamespace(
        solver_type="mujoco_warp",
        model=model,
        cfg=SimpleNamespace(requires_grad=False, dt=0.001, num_substeps=1),
    )
    monkeypatch.setattr(backend_registry, "get_newton_backend", lambda world: backend)
    impedance = [0.99, 0.99, 0.001, 0.5, 2.0]
    assert _configure_newton_mimic_compliance(
        result=SimpleNamespace(backend="newton", world=object()),
        entities=[entity],
        state_joint_names=["leader", "follower"],
        mimic_ids=[1],
        mimic_parents=[0],
        solref=[0.002, 1.0],
        solimp=impedance if override_impedance else None,
    )
    expected_ref = initial_ref.copy()
    expected_ref[0, 1] = [0.002, 1.0]
    expected_imp = initial_imp.copy()
    if override_impedance:
        expected_imp[0, 1] = impedance
    np.testing.assert_allclose(mjw_model.eq_solref.numpy(), expected_ref)
    np.testing.assert_allclose(mjw_model.eq_solimp.numpy(), expected_imp)
    np.testing.assert_allclose(cpu.eq_solref, expected_ref[0])
    np.testing.assert_allclose(cpu.eq_solimp, expected_imp[0])
