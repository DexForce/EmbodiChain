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
from dataclasses import replace
import pytest
import torch
from embodichain.lab.sim.scene.query import SceneView
from embodichain.lab.sim.scene.state import capture_state, restore_state
from embodichain.lab.sim.scene.validation import compare_states
from ._runtime_ports import Sim, Rigid


def test_capture_owns_batches_and_restores_targets_effort_and_velocity():
    sim = Sim()
    view = SceneView(sim)
    saved = capture_state(view)
    sim.rigid["cube"].value[:, 0] = 9
    for value in sim.articulations["drawer"].state.values():
        value.zero_()
    assert saved.env_ids == (0,) and len(saved.batches) == 1
    restore_state(view, saved)
    assert compare_states(saved, capture_state(view)).passed
    assert sim.steps == 0 and sim.syncs == 1
    assert sim.articulations["drawer"].state["qf"].item() == pytest.approx(0.5)


def test_foreign_and_rebound_snapshots_rejected_before_writes():
    sim = Sim()
    view = SceneView(sim)
    saved = capture_state(view)
    other = Sim()
    with pytest.raises(ValueError, match="another"):
        restore_state(SceneView(other), saved)
    assert other.rigid["cube"].writes == 0
    sim.rigid["other"] = Rigid()
    with pytest.raises(ValueError, match="bindings changed"):
        restore_state(view, saved)
    assert sim.rigid["cube"].writes == 0


def test_restore_preflights_all_entity_capabilities():
    sim = Sim()
    view = SceneView(sim)
    saved = capture_state(view)
    sim.rigid["other"].set_velocity = None
    with pytest.raises(ValueError, match="does not support"):
        restore_state(view, saved)
    assert sim.rigid["cube"].writes == 0


def test_state_comparison_allows_quaternion_sign_and_joint_limit_rounding():
    sim = Sim()
    view = SceneView(sim)
    saved = capture_state(view)
    sim.rigid["cube"].value[:, 3:7] *= -1
    sim.articulations["drawer"].state["qpos"] += 5e-5
    assert compare_states(saved, capture_state(view)).passed
    sim.articulations["drawer"].state["target_qpos"] += 5e-5
    assert not compare_states(saved, capture_state(view)).passed


def test_restore_native_failure_reports_partial_state_without_stepping():
    sim = Sim()
    view = SceneView(sim)
    saved = capture_state(view)

    def fail(*args, **kwargs):
        raise OSError("backend failure")

    sim.rigid["other"].set_local_pose = fail
    with pytest.raises(RuntimeError, match="partially restored"):
        restore_state(view, saved)
    assert sim.steps == 0


def test_restore_backend_overflow_preflight_does_not_clear_earlier_entities():
    sim = Sim()
    view = SceneView(sim)
    saved = capture_state(view)
    rows = tuple(
        replace(row, pose=(1e100, *row.pose[1:])) if row.uid == "other" else row
        for row in saved.batches[0]
    )
    altered = replace(saved, batches=(rows,))
    sim.articulations["drawer"].state["qf"][:] = 7.0
    with pytest.raises(ValueError, match="backend"):
        restore_state(view, altered)
    assert all(body.writes == 0 for body in sim.rigid.values())
    assert sim.articulations["drawer"].state["qf"].item() == 7.0
    assert sim.syncs == 0


@pytest.mark.parametrize("field", ["qpos", "target_qpos"])
@pytest.mark.parametrize("position", [-1.001, 1.001])
def test_restore_rejects_limit_excursions_before_any_write(
    field: str, position: float
) -> None:
    sim = Sim()
    articulation = sim.articulations["drawer"]
    articulation.state[field][:] = position
    view = SceneView(sim)
    saved = capture_state(view)
    articulation.state[field][:] = 0.0
    before = capture_state(view)

    def unexpected_write(**kwargs):
        pytest.fail("Snapshot rejection must precede all articulation writes")

    articulation.set_state = unexpected_write
    with pytest.raises(ValueError, match=rf"drawer\.{field} exceeds"):
        restore_state(view, saved)
    assert compare_states(before, capture_state(view)).passed
    assert all(body.writes == 0 for body in sim.rigid.values())
    assert sim.syncs == 0 and sim.steps == 0


def test_restore_rechecks_limits_changed_since_capture() -> None:
    sim = Sim()
    articulation = sim.articulations["drawer"]
    view = SceneView(sim)
    saved = capture_state(view)
    articulation.body_data.qpos_limits[..., 1] = 0.1
    with pytest.raises(ValueError, match="drawer.qpos exceeds"):
        restore_state(view, saved)
    assert compare_states(saved, capture_state(view)).passed
    assert sim.syncs == 0


@pytest.mark.parametrize("position,tolerance", [(-1.0, 0.0), (-1.00005, 1e-4)])
def test_boundary_positions_round_trip_through_production_articulation_setter(
    position: float, tolerance: float
) -> None:
    from types import MethodType, SimpleNamespace
    from embodichain.lab.sim.objects.articulation import Articulation

    sim = Sim()
    articulation = sim.articulations["drawer"]
    articulation.state["qpos"][:] = position
    articulation.state["target_qpos"][:] = 1.0
    articulation.device = sim.device
    articulation._resolve_env_ids = lambda ids: torch.tensor([0])
    articulation._resolve_joint_ids = lambda ids: torch.tensor([0])
    articulation._stabilize_newton_mimic_target_write = lambda *args, **kwargs: None
    articulation._world = SimpleNamespace(update=lambda dt: None)
    articulation.body_data.is_newton_backend = False
    port_write = articulation.set_state
    articulation.body_data.articulation_view = SimpleNamespace(
        apply_state=lambda rows, columns, **fields: port_write(env_ids=rows, **fields)
    )
    articulation._data = articulation.body_data
    articulation.set_state = MethodType(Articulation.set_state, articulation)
    view = SceneView(sim)
    saved = capture_state(view)
    articulation.state["qpos"][:] = 0.0
    articulation.state["target_qpos"][:] = 0.0
    restore_state(view, saved, qpos_tolerance=tolerance)
    assert compare_states(saved, capture_state(view)).passed


@pytest.mark.parametrize("tolerance", [True, -1.0, float("nan"), float("inf")])
def test_invalid_qpos_tolerance_rejected_before_writes(tolerance: object) -> None:
    sim = Sim()
    view = SceneView(sim)
    saved = capture_state(view)
    with pytest.raises(ValueError, match="qpos_tolerance"):
        restore_state(view, saved, qpos_tolerance=tolerance)
    assert all(body.writes == 0 for body in sim.rigid.values())
    assert sim.syncs == 0
