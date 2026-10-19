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

from types import SimpleNamespace
from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch


def _native_layout_probe(backend: str) -> None:
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg, physics_cfg_for_backend
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.gym.envs.managers import EventCfg, SceneEntityCfg
    from embodichain.lab.gym.envs.managers.event_manager import EventManager
    from embodichain.lab.gym.envs.managers.randomization import RandomizeSceneLayout

    cfg = SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=4,
        physics_cfg=physics_cfg_for_backend(backend),
    )
    cfg.render_cfg.renderer = "no-render"
    sim = SimulationManager(cfg)
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="bench",
                shape=CubeCfg(size=(1.0, 1.0, 0.1)),
                body_type="kinematic",
                init_pos=(0.0, 0.0, 0.0),
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="holder",
                shape=CubeCfg(size=(0.2, 0.2, 0.02)),
                init_pos=(-0.2, 0.0, 0.061),
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="part",
                shape=CubeCfg(size=(0.05, 0.05, 0.05)),
                init_pos=(-0.2, 0.0, 0.096),
            )
        )
        sim.prepare()
        before = sim.get_rigid_object("part").get_local_pose(to_matrix=True).clone()
        env = SimpleNamespace(
            sim=sim,
            num_envs=4,
            device=sim.device,
            cfg=SimpleNamespace(seed=12),
            _profiler=None,
            event_manager=None,
            _demo_episode_metadata=[{} for _ in range(4)],
        )
        event = EventCfg(
            func=RandomizeSceneLayout,
            mode="reset",
            params={
                "entity_cfg": SceneEntityCfg(uid="bench"),
                "entity_cfgs": [
                    SceneEntityCfg(uid="holder"),
                    SceneEntityCfg(uid="part"),
                ],
                "parent_by_uid": {"holder": "bench", "part": "holder"},
                "policy": {
                    "intra_group": {
                        "targets": {"part": {"yaw_delta_range_deg": [-30.0, 30.0]}}
                    }
                },
                "candidate_count": 4,
                "max_attempts": 32,
            },
        )
        env.event_manager = EventManager({"layout": event}, env)
        sim.update = lambda *args, **kwargs: pytest.fail(
            "Layout application advanced physics"
        )
        env.event_manager.reset(env_ids=[3, 1])
        env.event_manager.apply("reset", env_ids=[3, 1])
        after = sim.get_rigid_object("part").get_local_pose(to_matrix=True)
        torch.testing.assert_close(after[[0, 2]], before[[0, 2]], rtol=0, atol=0)
        assert not torch.allclose(after[1], after[3])
        holder_z = sim.get_rigid_object("holder").get_local_pose(to_matrix=True)[
            [1, 3], 2, 3
        ]
        torch.testing.assert_close(
            after[[1, 3], 2, 3] - 0.025 - holder_z - 0.01,
            torch.full((2,), 0.001),
            rtol=0,
            atol=2e-6,
        )
        assert env._scene_randomization_metadata[0] is None
        assert env._scene_randomization_metadata[2] is None
        assert all(
            env._demo_episode_metadata[row]["scene_randomization"]["geometry_status"]
            == "passed"
            for row in (1, 3)
        )
        env.event_manager = None
        env = None
    finally:
        sim.destroy(exit_process=False)
        sim = None
        SimulationManager.flush_cleanup_queue()


def _native_gym_probe(backend: str) -> None:
    from tensordict import TensorDict
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.envs.managers import EventCfg, SceneEntityCfg
    from embodichain.lab.gym.envs.managers.randomization import RandomizeSceneLayout
    from embodichain.lab.sim.cfg import RigidObjectCfg, physics_cfg_for_backend
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    class ProbeEnvironment(EmbodiedEnv):
        def get_obs(self, **kwargs):
            return TensorDict(
                {"probe_pose": self.sim.get_rigid_object("cube").get_local_pose()},
                batch_size=[self.num_envs],
                device=self.device,
            )

    root = Path(__file__).resolve().parents[4]
    path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    raw = load_config(path)
    raw.pop("task_program", None)
    raw["id"] = "EmbodiedEnv-v1"
    cfg = config_to_cfg(raw, source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = 4
    cfg.sim_cfg.physics_cfg = physics_cfg_for_backend(backend)
    cfg.sim_cfg.device = "cpu"
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.seed = 9
    cfg.enable_sensor = False
    cfg.light.direct = []
    cfg.light.indirect = None
    cfg.dataset = None
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = False
    cfg.background.append(
        RigidObjectCfg(
            uid="bench",
            shape=CubeCfg(size=(1.4, 1.4, 0.1)),
            init_pos=(0.0, 0.0, -0.05),
            body_type="kinematic",
        )
    )
    cfg.events = SimpleNamespace(
        layout=EventCfg(
            func=RandomizeSceneLayout,
            mode="reset",
            params={
                "entity_cfg": SceneEntityCfg(uid="bench"),
                "entity_cfgs": [SceneEntityCfg(uid="cube")],
                "parent_by_uid": {"cube": "bench"},
                "candidate_count": 4,
                "max_attempts": 32,
            },
        )
    )
    env = None
    try:
        env = ProbeEnvironment(cfg=cfg)
        obs, info = env.reset(seed=9, options={"save_data": False})
        torch.testing.assert_close(
            obs["probe_pose"], env.sim.get_rigid_object("cube").get_local_pose()
        )
        before = obs["probe_pose"].clone()
        previous = [m["variant_id"] for m in info["scene_randomization"]]
        obs, info = env.reset(options={"save_data": False, "reset_ids": [3, 1]})
        torch.testing.assert_close(
            obs["probe_pose"][[0, 2]], before[[0, 2]], atol=0, rtol=0
        )
        torch.testing.assert_close(
            obs["probe_pose"], env.sim.get_rigid_object("cube").get_local_pose()
        )
        assert [info["scene_randomization"][i]["variant_id"] for i in (0, 2)] == [
            previous[i] for i in (0, 2)
        ]
        assert all(
            info["scene_randomization"][i]["variant_id"] != previous[i] for i in (1, 3)
        )
    finally:
        if env is not None:
            env.close(exit_process=False)
        env = None
        SimulationManager.flush_cleanup_queue()


@pytest.mark.requires_sim
@pytest.mark.parametrize("backend", ["default", "newton"])
@pytest.mark.parametrize("frontend", ["sim", "gym"])
def test_native_four_row_functor_publication_is_selective(
    backend: str, frontend: str
) -> None:
    command = (
        "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); "
        "m['_native_gym_probe' if sys.argv[3]=='gym' else '_native_layout_probe'](sys.argv[2]); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            str(Path(__file__).resolve()),
            backend,
            frontend,
        ],
        env={**os.environ, "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-12000:]
