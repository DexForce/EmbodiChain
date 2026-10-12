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

"""Fresh-process checks for the physics-only simulation configuration."""

from __future__ import annotations

import os
import faulthandler
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["default", "default-cuda", "newton"])
def test_no_render_world_steps_and_resets(backend: str, tmp_path: Path) -> None:
    urdf = tmp_path / "slider.urdf"
    urdf.write_text(
        """<robot name="slider">
        <link name="base">
          <inertial><mass value="1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
          </inertial>
          <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
        </link>
        <link name="child">
          <inertial><mass value="1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
          </inertial>
          <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
        </link>
        <joint name="slide" type="prismatic">
          <parent link="base"/><child link="child"/>
          <origin xyz="0 0 0.2"/><axis xyz="1 0 0"/>
          <limit lower="-0.2" upper="0.2" effort="10" velocity="1"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), backend, str(urdf)],
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            # Newton compiles kernels for the initial and rebuilt topology.
            timeout=180 if backend == "newton" else 90,
        )
    except subprocess.TimeoutExpired as error:
        output = b"\n".join((error.stdout or b"", error.stderr or b"")).decode(
            errors="replace"
        )
        pytest.fail(
            f"{backend} NoRender worker timed out after {error.timeout}s.\n{output}",
            pytrace=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_world(backend: str, urdf_path: str) -> None:
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        ArticulationCfg,
        DefaultPhysicsCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import CubeCfg

    physics = (
        NewtonPhysicsCfg(
            device="cuda:0",
            num_substeps=1,
            physics_dt=0.005,
            solver_cfg={
                "solver_type": "mujoco_warp",
                "iterations": 20,
                "ls_iterations": 50,
            },
        )
        if backend == "newton"
        else DefaultPhysicsCfg(
            device="cuda:0" if backend == "default-cuda" else "cpu", physics_dt=0.005
        )
    )
    cfg = SimulationManagerCfg(
        headless=True,
        num_envs=2,
        physics_cfg=physics,
        render_cfg=RenderCfg(renderer="no-render"),
        startup_summary="off",
    )
    sim = SimulationManager(cfg)
    body = articulation = None
    try:
        body = sim.add_rigid_object(
            RigidObjectCfg(
                uid="box",
                shape=CubeCfg(),
                init_pos=(0.0, 0.0, 1.0),
            )
        )
        sim.prepare()
        initial = body.get_local_pose(to_matrix=True).clone()
        for _ in range(10):
            sim.update(step=1)
        after = body.get_local_pose(to_matrix=True).clone()
        assert torch.isfinite(after).all()
        assert torch.all(after[:, 2, 3] < initial[:, 2, 3])
        body.reset(env_ids=[0])
        reset_pose = body.get_local_pose(to_matrix=True)
        torch.testing.assert_close(reset_pose[0], initial[0])
        torch.testing.assert_close(reset_pose[1], after[1])
        # The collision ground remains present, even though it has no material.
        assert sim._spawn_scene.handles("default_plane")
        assert not sim._world.is_window_initialized()
        assert not sim.has_native_renderer
        assert not sim._visual_materials
        from embodichain.lab.sim.sensors import CameraCfg

        with pytest.raises(RuntimeError, match="requires a native renderer"):
            sim.add_sensor(CameraCfg(uid="camera", width=32, height=32))
        assert not sim._sensors
        print(f"{backend}: rigid state and renderer checks passed", flush=True)

        # Exercise the URDF source resolver against the selected DexSim build.
        articulation = sim.add_articulation(
            ArticulationCfg(uid="slider", fpath=urdf_path, init_pos=(0.0, 0.0, 2.0))
        )
        sim.prepare()
        assert articulation.dof == 1
        # Root-only writes must move descendants immediately without a step.
        links_before = articulation.body_data.body_link_pose.clone()
        body_before = body.get_local_pose().clone()
        moved_root = articulation.get_local_pose()[1:2].clone()
        moved_root[:, 0] += 1.0
        articulation.set_state(env_ids=[1], root_pose=moved_root)
        links_after = articulation.body_data.body_link_pose.clone()
        expected_links = links_before.clone()
        expected_links[1, :, 0] += 1.0
        torch.testing.assert_close(links_after, expected_links)
        torch.testing.assert_close(body.get_local_pose(), body_before)
        print(f"{backend}: root-only descendant propagation passed", flush=True)
        if backend == "newton":
            # The published native API must not download command selections.
            with pytest.MonkeyPatch.context() as patch:

                def reject_host_selection(tensor, *args, **kwargs):
                    if tensor.dtype == torch.long and tensor.device.type == "cuda":
                        raise AssertionError("CUDA selection was copied to CPU")
                    return original_cpu(tensor, *args, **kwargs)

                original_cpu = torch.Tensor.cpu
                patch.setattr(torch.Tensor, "cpu", reject_host_selection)
                articulation.set_qpos(
                    torch.full((2, 1), 0.03, device=sim.device), env_ids=[0, 1]
                )
                articulation.set_qvel(
                    torch.full((1, 1), 0.04, device=sim.device),
                    env_ids=torch.tensor([1], device=sim.device),
                    joint_ids=torch.tensor([0], device=sim.device),
                )
            torch.testing.assert_close(
                articulation.body_data.target_qpos,
                torch.full((2, 1), 0.03, device=sim.device),
            )
            torch.testing.assert_close(
                articulation.body_data.target_qvel,
                torch.tensor([[0.0], [0.04]], device=sim.device),
            )
            print(f"{backend}: device-resident control selections passed", flush=True)
        state = torch.full((2, 1), 0.05, device=sim.device)
        articulation.set_qpos(state, target=False)
        torch.testing.assert_close(articulation.body_data.qpos, state)
        articulation.reset(env_ids=[0])
        torch.testing.assert_close(
            articulation.body_data.qpos[0], torch.zeros_like(state[0])
        )
        torch.testing.assert_close(articulation.body_data.qpos[1], state[1])
        sim.update(step=1)
        assert torch.isfinite(articulation.body_data.qpos).all()
        print(f"{backend}: articulation state checks passed", flush=True)
    finally:
        body = articulation = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        print(f"{backend}: cleanup complete", flush=True)


if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        _run_world(sys.argv[1], sys.argv[2])
    finally:
        faulthandler.cancel_dump_traceback_later()
