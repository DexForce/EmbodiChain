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

"""Cable scene composition through the public Gym configuration loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim.cfg import CableObjectCfg, CablePhysicsCfg

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("component", [False, True], ids=["inline", "component"])
def test_cable_config_is_declared_before_environment_preparation(
    tmp_path: Path, component: bool
) -> None:
    cable = {
        "uid": "wire",
        "points": [[0, 0, 1], [0.2, 0, 1]],
        "radius": 0.01,
        "fixed_segments": [0],
        "attrs": {"bend_stiffness": 0.1},
    }
    config = {
        "id": "EmbodiedEnv-v1",
        "physics": "newton",
        "env": {},
        "robot": {"uid": "robot"},
        "cable_object": [cable],
    }
    if component:
        (tmp_path / "env.yaml").write_text(
            yaml.safe_dump(
                {
                    "environment_id": "EmbodiedEnv-v1",
                    "physics": "newton",
                    "env": {},
                    "simulation": {"cable_object": [cable]},
                }
            )
        )
        config = {
            "id": "EmbodiedEnv-v1",
            "environment": {"component": "env.yaml"},
            "robot": {"uid": "robot"},
        }
    cfg = config_to_cfg(config, source_path=tmp_path / "task.yaml")
    assert len(cfg.cable_object) == 1
    assert isinstance(cfg.cable_object[0], CableObjectCfg)
    assert isinstance(cfg.cable_object[0].attrs, CablePhysicsCfg)
    assert cfg.cable_object[0].attrs.bend_stiffness == 0.1

    env = object.__new__(EmbodiedEnv)
    env.cfg = cfg
    env.sim = MagicMock()
    env._setup_interactive_objects()
    env.sim.add_cable_object.assert_called_once_with(cfg=cfg.cable_object[0])
    env.sim.prepare.assert_not_called()
