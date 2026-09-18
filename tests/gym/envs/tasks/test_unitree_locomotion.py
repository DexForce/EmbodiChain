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

import ast
from pathlib import Path

import pytest
import torch

from embodichain_tasks.locomotion.velocity.contracts.g1 import config as g1_config
from embodichain_tasks.locomotion.velocity.contracts.g1 import mdp as g1_mdp
from embodichain_tasks.locomotion.velocity.contracts.go1 import config as go1_config
from embodichain_tasks.locomotion.velocity.contracts.go1 import mdp as go1_mdp
from embodichain_tasks.locomotion.velocity.contracts.go2 import config as go2_config
from embodichain_tasks.locomotion.velocity.contracts.go2 import mdp as go2_mdp
from embodichain_tasks.locomotion.velocity.contracts.h1_2 import config as h1_config
from embodichain_tasks.locomotion.velocity.contracts.h1_2 import mdp as h1_mdp

from embodichain_tasks.locomotion.velocity.contracts.anymal_c import (
    config as anymal_config,
    mdp as anymal_mdp,
)
from embodichain_tasks.locomotion.velocity.contracts.microduck import (
    config as microduck_config,
    mdp as microduck_mdp,
)


@pytest.mark.parametrize(
    ("load", "action_target", "action_dim", "actor_dim", "critic_dim"),
    [
        (g1_config.load_config, g1_mdp.action_target, 29, 98, 113),
        (anymal_config.load_config, anymal_mdp.action_target, 12, 48, 48),
        (microduck_config.load_config, microduck_mdp.action_target, 14, 53, 68),
        (go1_config.load_config, go1_mdp.action_target, 12, 48, 72),
        (go2_config.load_config, go2_mdp.action_target, 12, 47, 74),
        (h1_config.load_config, h1_mdp.action_target, 27, 92, 107),
    ],
)
def test_contract_dimensions_and_action_mapping(
    load, action_target, action_dim: int, actor_dim: int, critic_dim: int
):
    config = load()
    action = torch.zeros((2, action_dim))

    target = action_target(config, action)

    assert target.shape == (2, action_dim)
    assert config.actor_observation_dim == actor_dim
    assert config.critic_observation_dim == critic_dim
    assert torch.allclose(
        target[0], torch.tensor(config.default_joint_position, dtype=torch.float32)
    )


def test_contract_package_has_no_backend_imports():
    root = Path("embodichain_tasks/embodichain_tasks/locomotion/velocity/contracts")
    forbidden = ("embodichain.lab", "dexsim")
    imports: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert not any(name.startswith(forbidden) for name in imports)


def test_heading_commands_consume_xyzw_quaternions() -> None:
    import math
    from types import SimpleNamespace
    from embodichain_tasks.locomotion.velocity._embodichain import (
        EmbodiChainVelocityEnv,
    )

    env = SimpleNamespace(
        velocity_task_config=SimpleNamespace(
            data={"commands": {"twist": {"heading_control_stiffness": 1.0}}}
        ),
        robot=SimpleNamespace(
            body_data=SimpleNamespace(
                root_pose=torch.tensor(
                    [[0.0, 0.0, 1.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]]
                )
            )
        ),
        _heading_target=torch.tensor([math.pi / 2]),
        _active_command_ranges=lambda: ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        _heading_env=torch.tensor([True]),
        _standing_env=torch.tensor([False]),
        command=torch.zeros((1, 3)),
    )
    EmbodiChainVelocityEnv._update_heading_commands(env)
    torch.testing.assert_close(env.command, torch.zeros((1, 3)), atol=1e-6, rtol=0)


@pytest.mark.parametrize("robot", ["g1", "h1_2", "go1", "go2"])
@pytest.mark.parametrize("backend", ["default", "newton"])
def test_unitree_deployments_preserve_task_physics(
    robot: str, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.sim import cfg as sim_cfg
    from embodichain.utils.utility import load_config

    monkeypatch.setattr(sim_cfg, "get_data_path", lambda value: value)
    name = "env.yaml" if backend == "default" else "env.newton.yaml"
    path = Path(
        f"embodichain_tasks/configs/tasks/locomotion/velocity/{robot}_flat/{name}"
    )
    config = config_to_cfg(
        load_config(str(path)),
        source_path=path,
        manager_modules=[
            f"embodichain_tasks.locomotion.managers.{module}"
            for module in ("actions", "events", "observations", "rewards")
        ],
    )
    physics_type = (
        sim_cfg.DefaultPhysicsCfg if backend == "default" else sim_cfg.NewtonPhysicsCfg
    )
    assert isinstance(config.sim_cfg.physics_cfg, physics_type)
    assert config.sim_cfg.physics_dt == pytest.approx(0.005)
    assert config.sim_cfg.scene_node_capacity == 262144
    if backend == "default":
        assert config.sim_cfg.physics_cfg.to_dexsim_args()["cache_material"] is True
    else:
        assert config.sim_cfg.physics_cfg.sync_to_renderer is False
    assert config.sim_steps_per_control == 4
    assert config.max_episode_steps == 1000
    assert config.robot.root_props.fixed_base is False
    assert config.robot.root_props.self_collision_enabled is (robot in {"g1", "h1_2"})
    assert config.robot.asset_physics_mode == "overlay"
    assert config.sensor[0].articulation_cfg_list[0].link_name_list == []
