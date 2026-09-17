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

"""Validate runnable locomotion configurations without patching their fields."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import cfg as sim_cfg
from embodichain.utils.utility import load_config

_TASKS = (
    "locomotion/velocity/g1_flat",
    "locomotion/velocity/h1_2_flat",
    "locomotion/velocity/go1_flat",
    "locomotion/velocity/go2_flat",
    "locomotion/velocity/anymal_c_flat",
    "locomotion/velocity/microduck_flat",
    "classic_control/humanoid",
)


@pytest.mark.no_sim
@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("backend", ("default", "newton"))
def test_packaged_deployment_and_agent_config_agree(
    task: str, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real config decoding must work without smoke-script backend overrides."""
    monkeypatch.setattr(sim_cfg, "get_data_path", lambda value: value)
    directory = Path("embodichain_tasks/configs/tasks") / task
    suffix = "" if backend == "default" else ".newton"
    path = directory / f"env{suffix}.yaml"
    raw = load_config(str(path))
    agent = load_config(str(directory / f"agents/ppo{suffix}.yaml"))
    assert raw["physics"] == backend
    assert "physics_config_by_backend" not in raw
    cfg = config_to_cfg(
        raw,
        source_path=path,
        manager_modules=[
            f"embodichain_tasks.locomotion.managers.{module}"
            for module in ("observations", "rewards")
        ],
    )
    physics_type = (
        sim_cfg.DefaultPhysicsCfg if backend == "default" else sim_cfg.NewtonPhysicsCfg
    )
    assert isinstance(cfg.sim_cfg.physics_cfg, physics_type)
    assert cfg.sim_cfg.render_cfg.renderer in {"auto", "hybrid", "fast-rt", "rt"}
    assert agent["trainer"]["renderer"] == cfg.sim_cfg.render_cfg.renderer
    assert Path(agent["trainer"]["gym_config"]).resolve() == path.resolve()
    assert set(agent) == {"trainer", "policy", "algorithm"}
    assert agent["policy"]["initial_action_std"] > 0
