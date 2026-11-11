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

"""Gym classifies authored USD actors after source resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.sim.cfg import RigidObjectCfg, UsdSourceCfg

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    "role,resolved_type,valid",
    [
        ("interactive", "dynamic", True),
        ("interactive", "kinematic", False),
        ("background", "kinematic", True),
        ("background", "dynamic", False),
    ],
)
def test_role_uses_resolved_usd_actor(role, resolved_type, valid):
    cfg = RigidObjectCfg(uid="usd", source=UsdSourceCfg(file_path="scene.usda"))
    assert cfg.body_type is None
    registered = []

    def declare(*, cfg, validate_resolved):
        cfg.body_type = resolved_type
        validate_resolved(cfg)
        registered.append(cfg)

    env = SimpleNamespace(
        cfg=SimpleNamespace(
            background=[cfg],
            rigid_object=[cfg],
            articulation=[],
            visual_object=[],
            rigid_object_group=[],
        ),
        sim=SimpleNamespace(add_rigid_object=declare),
    )
    if role == "interactive":
        setup = EmbodiedEnv._setup_interactive_objects
    else:
        setup = EmbodiedEnv._setup_background
    if valid:
        setup(env)
        assert registered == [cfg]
    else:
        with pytest.raises(RuntimeError, match="rigid object"):
            setup(env)
        assert registered == []


def test_resolved_role_failure_does_not_declare_a_rigid_object(monkeypatch):
    import torch
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.spawn.scene import SpawnScene

    cfg = RigidObjectCfg(uid="usd", source=UsdSourceCfg(file_path="scene.usda"))
    sim = object.__new__(SimulationManager)
    sim.device = torch.device("cpu")
    sim.physics = SimpleNamespace(name="default")
    sim._rigid_objects = {}
    scene = object.__new__(SpawnScene)
    scene.builder = SimpleNamespace(is_finalized=False, result=None)
    scene.declare = Mock()
    sim._spawn_scene = scene
    sim.notify_visualization_topology_changed = Mock()

    def resolve(cfg, **kwargs):
        cfg.body_type = "kinematic"
        return object(), {}

    resolver = Mock(side_effect=resolve)
    monkeypatch.setattr("embodichain.lab.sim.sim_manager.rigid_desc_from_usd", resolver)

    def require_dynamic(resolved):
        if resolved.body_type != "dynamic":
            raise ValueError("expected dynamic actor")

    with pytest.raises(ValueError, match="dynamic"):
        sim.add_rigid_object(cfg, validate_resolved=require_dynamic)
    assert sim._rigid_objects == {}
    scene.declare.assert_not_called()
    sim.notify_visualization_topology_changed.assert_not_called()
    resolver.assert_called_once()
