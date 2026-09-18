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
"""Atomic-action Newton native-contact configuration tests."""

from __future__ import annotations

import importlib

import pytest

from embodichain.lab.sim.cfg.simulation import NewtonPhysicsCfg


@pytest.mark.no_sim
def test_atomic_action_tutorial_disables_external_collision_pipeline() -> None:
    module = importlib.import_module("scripts.tutorials.atomic_action.tutorial_utils")

    physics_cfg = module._tutorial_physics_cfg("newton")

    assert isinstance(physics_cfg, NewtonPhysicsCfg)
    assert physics_cfg.collision_cfg is None
    dexsim_cfg = physics_cfg.to_dexsim_cfg(gpu_id=0)
    assert dexsim_cfg.collision_pipeline_cfg is None
    assert dexsim_cfg.solver_cfg.use_mujoco_contacts is True
