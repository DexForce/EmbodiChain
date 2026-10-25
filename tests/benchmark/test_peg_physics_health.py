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
"""CPU fake-state checks for pre-reset native physics diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark.assembly.peg_reachability import _PhysicsHealth


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        return self.values


class _Solver:
    def __init__(self, data: SimpleNamespace) -> None:
        self.mjw_data = data


def _environment() -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build readable native-like buffers without any simulation runtime."""
    data = SimpleNamespace(
        overflow=_Array(np.zeros(3, dtype=np.int32)),
        qpos=_Array(np.zeros((3, 4), dtype=np.float32)),
        qvel=_Array(np.zeros((3, 3), dtype=np.float32)),
        qacc=_Array(np.zeros((3, 3), dtype=np.float32)),
    )
    solver = _Solver(data)
    part = SimpleNamespace(
        _entities=[
            SimpleNamespace(
                _physics_binding=SimpleNamespace(
                    _runtime=SimpleNamespace(solver=solver)
                )
            )
        ]
    )
    env = SimpleNamespace(
        physics_dt=0.001,
        num_envs=3,
        sim=SimpleNamespace(
            get_rigid_object=lambda uid: part, update=lambda *args, **kwargs: None
        ),
    )
    return env, data


@pytest.mark.parametrize(
    "field,value", [("qpos", np.nan), ("qvel", np.inf), ("qacc", -np.inf)]
)
def test_strict_health_rejects_nonfinite_native_state_before_reset(
    field: str, value: float
) -> None:
    env, data = _environment()
    getattr(data, field).values[1, 0] = value
    original = getattr(data, field).values.copy()
    health = _PhysicsHealth(
        env, scope="control", strict=True, include_world_flags=False
    )
    with pytest.raises(
        RuntimeError, match="nonfinite qpos/qvel/qacc before auto-reset"
    ):
        env.sim.update(0.001, step=20)
    state = health.values["native_state"]
    assert state["nonfinite_seen"] is True
    assert state["all_samples_finite"] is False
    assert state["first_nonfinite"]["field"] == field
    assert state["first_nonfinite"]["first_bad_world_ids"] == [1]
    assert state["field_summaries"][field]["nonfinite_world_observations"] == 1
    assert state["field_summaries"][field]["nonfinite_value_observations"] == 1
    assert health.values["control_boundaries_observed"] == 1
    np.testing.assert_equal(getattr(data, field).values, original)


def test_strict_health_requires_all_native_finite_state_fields() -> None:
    env, data = _environment()
    del data.qacc
    with pytest.raises(RuntimeError, match="finite-state fields.*qacc"):
        _PhysicsHealth(env, scope="control", strict=True)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_nonstrict_health_records_missing_and_bad_state_without_raising(
    nonfinite: bool,
) -> None:
    env, data = _environment()
    del data.qacc
    if nonfinite:
        data.qvel.values[2, 0] = np.inf
    health = _PhysicsHealth(env, scope="control")
    env.sim.update(0.001, step=20)
    state = health.values["native_state"]
    assert state["available"] is False
    assert state["missing_fields"] == ["qacc"]
    assert state["complete_observations"] == 0
    assert state["nonfinite_seen"] is nonfinite
    assert state["all_samples_finite"] is (False if nonfinite else None)
    assert health.values["overflow_seen"] is False


def test_strict_health_counts_complete_finite_control_samples() -> None:
    env, _ = _environment()
    health = _PhysicsHealth(env, scope="control", strict=True)
    for _ in range(2):
        env.sim.update(0.001, step=20)
    state = health.values["native_state"]
    assert state["available"] is True
    assert state["complete_observations"] == 2
    assert state["all_samples_finite"] is True
    assert state["first_nonfinite"] is None
    assert [item["observations"] for item in state["field_summaries"].values()] == [
        2,
        2,
        2,
    ]
