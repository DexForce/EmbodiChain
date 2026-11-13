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

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.cli._task_catalog import _catalog_from_args
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
from embodichain.lab.sim import cfg as sim_cfg
from embodichain.utils.utility import load_config
from embodichain_tasks.manipulation.dexterous import _runtime, _shadow_task


def test_dexuni_scene_joint_ownership_assigns_all_joints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dexsim.engine.newton_physics import backend_registry
    from embodichain_tasks.manipulation.dexterous._runtime import (
        _DexUniSceneJointOwnership,
    )

    solver_cfg = SimpleNamespace(
        mujoco_articulations=None,
        mujoco_articulation_paths=None,
        mujoco_joints=None,
    )
    backend = SimpleNamespace(
        solver_type="dexuni", cfg=SimpleNamespace(solver_cfg=solver_cfg)
    )
    monkeypatch.setattr(backend_registry, "get_newton_backend", lambda world: backend)

    control = _DexUniSceneJointOwnership()
    control.initialize_model(
        SimpleNamespace(joint_count=12), SimpleNamespace(world=object())
    )
    assert solver_cfg.mujoco_joints == tuple(range(12))

    explicit_cfg = SimpleNamespace(
        mujoco_articulations=None,
        mujoco_articulation_paths=("/World/hand",),
        mujoco_joints=None,
    )
    backend.cfg = SimpleNamespace(solver_cfg=explicit_cfg)
    control.initialize_model(
        SimpleNamespace(joint_count=12), SimpleNamespace(world=object())
    )
    assert explicit_cfg.mujoco_joints is None


def test_dexterous_and_rough_catalog_configs_use_single_dexuni_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in (_runtime, _shadow_task):
        source = inspect.getsource(module)
        assert "from_xml_path" not in source
        assert "ElementTree" not in source
        assert ".xml" not in source

    worktree_config_root = Path.cwd() / "extensions/embodichain_tasks/configs/tasks"
    if worktree_config_root.is_dir():
        config_root = worktree_config_root
    else:
        config_root = Path(_runtime.__file__).resolve().parents[3] / "configs/tasks"
        if not config_root.is_dir():
            config_root = Path(_runtime.__file__).resolve().parents[2] / "configs/tasks"

    tasks = _catalog_from_args(
        argparse.Namespace(config_root=[f"embodichain_tasks={config_root}"])
    )
    catalog = {task.qualified_key: task for task in tasks}
    expected = {
        "embodichain_tasks:shadow_pen_fixed": "manipulation/dexterous/shadow_pen_fixed",
        "embodichain_tasks:shadow_pen_continuous": "manipulation/dexterous/shadow_pen_continuous",
        "embodichain_tasks:leap_cube_face_turn": "manipulation/dexterous/leap_cube_face_turn",
        "embodichain_tasks:leap_cube_reorient": "manipulation/dexterous/leap_cube_reorient",
        "embodichain_tasks:leap_cube_rotate_z": "manipulation/dexterous/leap_cube_rotate_z",
        "embodichain_tasks:g1_rough": "locomotion/velocity/g1_rough",
    }

    monkeypatch.setattr(sim_cfg, "get_data_path", lambda value: value)
    manager_modules = [
        *get_manager_modules(),
        "embodichain_tasks.locomotion.managers.observations",
        "embodichain_tasks.locomotion.managers.rewards",
    ]
    for selector, rel_dir in expected.items():
        assert selector in catalog
        entry = catalog[selector]
        assert entry.default_deployment == "default"
        deps = {d.name: d for d in entry.deployments}
        assert set(deps) == {"default"}
        dep = deps["default"]
        assert dep.physics == "newton"
        assert len(dep.agent_refs) == 1

        task_dir = config_root / rel_dir
        env_path = task_dir / "env.yaml"
        ppo_path = task_dir / "agents/ppo.yaml"
        assert env_path.is_file()
        assert ppo_path.is_file()
        assert not (task_dir / "env.newton.yaml").exists()
        assert not (task_dir / "env.mjwarp.yaml").exists()
        assert not (task_dir / "agents/ppo.newton.yaml").exists()
        assert not (task_dir / "agents/ppo.mjwarp.yaml").exists()

        env_raw = load_config(env_path)
        agent_raw = load_config(ppo_path)
        assert Path(agent_raw["trainer"]["gym_config"]).name == "env.yaml"
        algo = agent_raw["algorithm"]["cfg"]
        assert algo["batch_size"] == (
            agent_raw["trainer"]["num_envs"]
            * agent_raw["trainer"]["buffer_size"]
            // algo["num_mini_batches"]
        )
        cfg = config_to_cfg(
            env_raw, source_path=env_path, manager_modules=manager_modules
        )
        assert isinstance(cfg.sim_cfg.physics_cfg, sim_cfg.NewtonPhysicsCfg)
        solver = cfg.sim_cfg.physics_cfg.to_dexsim_cfg(gpu_id=0).solver_cfg
        assert solver.solver_type == "dexuni"
