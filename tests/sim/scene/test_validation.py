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
import pytest
from embodichain.lab.sim.scene.query import SceneView
from embodichain.lab.sim.scene.validation import (
    SceneCheck,
    SettlingCfg,
    validate_scene,
    settle_scene,
)
from ._runtime_ports import Sim


def test_empty_or_missing_evidence_never_accepts():
    assert not validate_scene([]).passed
    assert not validate_scene([SceneCheck("support", "unverified")]).passed
    assert validate_scene([lambda: SceneCheck("support", "passed")]).passed


def test_fixed_steps_do_not_claim_stability():
    sim = Sim()
    report = settle_scene(SceneView(sim), sim.step, SettlingCfg(max_steps=7))
    assert sim.steps == 7 and report.checks[0].status == "unverified"


def test_convergence_requires_consecutive_measured_stability():
    sim = Sim()

    def step(count):
        sim.step(count)
        for rigid in sim.rigid.values():
            rigid.value[:, 7:] = 0
        sim.articulations["drawer"].state["qvel"][:] = 0

    report = settle_scene(
        SceneView(sim),
        step,
        SettlingCfg(mode="convergence", max_steps=8, consecutive_steps=3),
    )
    assert report.passed and sim.steps == 3


def test_moving_scene_exhausts_budget_and_fails():
    sim = Sim()
    report = settle_scene(
        SceneView(sim), sim.step, SettlingCfg(mode="convergence", max_steps=3)
    )
    assert report.checks[0].status == "failed" and sim.steps == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"max_steps": -1}, {"linear_threshold": float("nan")}, {"mode": "unknown"}],
)
def test_invalid_settling_rejected_before_physics(kwargs):
    with pytest.raises(ValueError):
        SettlingCfg(**kwargs)


def test_convergence_preflights_readable_state_before_advancing():
    sim = Sim()
    sim.rigid["other"].value[:, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        settle_scene(SceneView(sim), sim.step, SettlingCfg(mode="convergence"))
    assert sim.steps == 0


def test_empty_convergence_has_no_evidence_and_does_not_step():
    sim = Sim()
    sim.rigid.clear()
    sim.articulations.clear()
    report = settle_scene(SceneView(sim), sim.step, SettlingCfg(mode="convergence"))
    assert not report.passed
    assert report.checks[0].status == "unverified"
    assert sim.steps == 0


def test_world_that_becomes_parallel_is_rejected_before_stepping():
    sim = Sim()
    view = SceneView(sim)
    sim.num_envs = 2
    with pytest.raises(ValueError, match="B=1"):
        settle_scene(view, sim.step, SettlingCfg(mode="convergence"))
    assert sim.steps == 0
