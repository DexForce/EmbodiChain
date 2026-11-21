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

import os
from pathlib import Path
import subprocess
import sys
import pytest


def _probe(backend: str) -> None:
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg, physics_cfg_for_backend
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.sim.scene import (
        SceneView,
        capture_state,
        restore_state,
        compare_states,
    )

    cfg = SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=8,
        physics_cfg=physics_cfg_for_backend(backend),
    )
    cfg.render_cfg.renderer = "no-render"
    sim = SimulationManager(cfg)
    try:
        cube = sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube", shape=CubeCfg(size=(0.05, 0.05, 0.05)), init_pos=(0, 0, 0.5)
            )
        )
        sim.prepare()
        cube = sim.get_rigid_object("cube")
        poses = cube.get_local_pose().clone()
        poses[:, 0] = torch.arange(8) * 0.1
        cube.set_local_pose(poses, env_ids=list(range(8)))
        view = SceneView(sim, env_ids=(7, 1, 5))
        saved = capture_state(view)
        poses[:, 0] = 0.9
        cube.set_local_pose(poses, env_ids=list(range(8)))
        unchanged = cube.get_local_pose()[[0, 2, 3, 4, 6]].clone()
        from embodichain.lab.sim.scene.operations import (
            SceneChangeSet,
            apply_changes_batch,
        )

        changes = {}
        for env_id, x in ((7, 0.2), (1, 0.4), (5, 0.6)):
            pose = torch.eye(4)
            pose[0, 3], pose[2, 3] = x, 0.5
            changes[env_id] = SceneChangeSet({"cube": pose.tolist()}, ("cube",))
        apply_changes_batch(view, changes)
        torch.testing.assert_close(
            cube.get_local_pose()[[7, 1, 5], 0],
            torch.tensor([0.2, 0.4, 0.6]),
            atol=1e-6,
            rtol=0,
        )
        restore_state(view, saved)
        assert compare_states(saved, capture_state(view)).passed
        torch.testing.assert_close(
            cube.get_local_pose()[[0, 2, 3, 4, 6]], unchanged, atol=0, rtol=0
        )
        from embodichain.lab.sim.scene.physics import (
            capture_rigid_physics,
            apply_rigid_physics,
            restore_rigid_physics,
        )

        baseline = capture_rigid_physics(view, ("cube",))
        untouched = cube.get_mass()[[0, 2, 3, 4, 6]].clone()
        scales = torch.tensor([0.5, 2.0, 1.0])
        changed = apply_rigid_physics(
            view,
            baseline,
            mass_scale={"cube": scales},
            friction={"cube": torch.tensor([0.2, 0.4, 0.6])},
        )
        assert changed == apply_rigid_physics(
            view,
            baseline,
            mass_scale={"cube": scales},
            friction={"cube": torch.tensor([0.2, 0.4, 0.6])},
        )
        torch.testing.assert_close(cube.get_mass()[[0, 2, 3, 4, 6]], untouched)
        cube.clear_dynamics(env_ids=list(range(8)))
        cube.add_force_torque(
            force=torch.tensor([[1.0, 0, 0]]).repeat(3, 1), env_ids=[7, 1, 5]
        )
        sim.update(step=1)
        velocity = cube.body_state[[7, 1, 5], 7].clone()
        assert bool((velocity > 0).all()), velocity
        momentum = velocity * cube.get_mass(env_ids=[7, 1, 5])
        torch.testing.assert_close(
            momentum, momentum[0].expand_as(momentum), rtol=0.02, atol=1e-6
        )
        restore_rigid_physics(view, baseline)
        assert capture_rigid_physics(view, ("cube",)) == baseline
        # Sliding contact independently checks that friction changes dynamics.
        contact_pose = cube.get_local_pose().clone()[[7, 1, 5]]
        contact_pose[:, 2] = 0.025
        cube.set_local_pose(contact_pose, env_ids=[7, 1, 5])
        cube.clear_dynamics(env_ids=[7, 1, 5])
        sim.update(step=20)
        apply_rigid_physics(
            view, baseline, friction={"cube": torch.tensor([0.0, 1.0, 0.5])}
        )
        cube.set_velocity(
            lin_vel=torch.tensor([[3.0, 0.0, 0.0]]).repeat(3, 1),
            ang_vel=torch.zeros(3, 3),
            env_ids=[7, 1, 5],
        )
        sim.update(step=15)
        sliding = cube.body_state[[7, 1, 5], 7].clone()
        assert float(sliding[0] - sliding[1]) > 0.1, sliding
        restore_rigid_physics(view, baseline)

    finally:
        sim.destroy(exit_process=False)
        sim = None
        SimulationManager.flush_cleanup_queue()


@pytest.mark.requires_sim
@pytest.mark.parametrize("backend", ["default", "newton"])
def test_native_eight_arena_selective_state_restore(backend: str) -> None:
    command = "import os,runpy,sys; runpy.run_path(sys.argv[1])['_probe'](sys.argv[2]); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"
    result = subprocess.run(
        [sys.executable, "-c", command, str(Path(__file__).resolve()), backend],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
