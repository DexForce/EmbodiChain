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
def test_role_uses_resolved_usd_actor(monkeypatch, role, resolved_type, valid):
    cfg = RigidObjectCfg(uid="usd", source=UsdSourceCfg(file_path="scene.usda"))
    assert cfg.body_type is None
    errors = Mock()
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.embodied_env.logger.log_error", errors
    )

    def declare(*, cfg):
        cfg.body_type = resolved_type

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
        EmbodiedEnv._setup_interactive_objects(env)
    else:
        EmbodiedEnv._setup_background(env)
    assert errors.called is not valid
