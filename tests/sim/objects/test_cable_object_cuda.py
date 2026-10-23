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

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    CableAttachmentCfg,
    CableObjectCfg,
    CablePhysicsCfg,
    DenoisingCfg,
    JointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    RenderCfg,
)

pytestmark = [
    pytest.mark.requires_sim,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CableObject requires CUDA"
    ),
]


class TestCableObjectCUDA:
    """Real replicated cable lifecycle through the EmbodiChain manager."""

    def setup_method(self, method) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CableObject requires CUDA")
        solver = {"solver_type": "dexuni"}
        if "expert_trajectory" in method.__name__:
            solver.update(
                {
                    "coupling": "two_way",
                    "mujoco_options": {
                        "enable_sleeping": True,
                        "update_data_interval": 1,
                        "iterations": 10,
                        "ls_iterations": 10,
                    },
                    "vbd_options": {"iterations": 10, "cable_dahl_eps_max": 0.2},
                }
            )
        cfg = SimulationManagerCfg(
            width=128,
            height=128,
            headless=True,
            device="cuda:0",
            num_envs=2,
            arena_space=3,
            physics_dt=1 / 120,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(
                renderer="hybrid",
                denoising=DenoisingCfg(window="optix", offscreen="optix"),
            ),
            physics_cfg=NewtonPhysicsCfg(
                solver_cfg=solver,
                num_substeps=2,
            ),
        )
        self.sim = SimulationManager(cfg)

    def teardown_method(self) -> None:
        if hasattr(self, "sim"):
            self.sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()

    def test_replicated_state_force_and_cleanup(self) -> None:
        frames = Rotation.from_euler("z", [[37], [-19]], degrees=True).as_quat()
        cable = self.sim.add_cable_object(
            CableObjectCfg(
                uid="wire",
                points=[[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2]],
                radius=0.01,
                init_pos=(0.5, 0.2, 1),
                segment_quaternions=frames,
                attrs=CablePhysicsCfg(bend_stiffness=0.1, twist_stiffness=0.1),
            )
        )
        assert cable.data is None and cable.num_instances == 2
        assert self.sim.get_asset("wire") is cable
        assert "wire" in self.sim.asset_uids
        self.sim.prepare()
        assert cable.is_spawn_bound and cable.data.num_segments == 2
        initial = cable.data.segment_state_w
        assert initial.shape == (2, 2, 13)
        assert torch.isfinite(initial).all()
        scene = self.sim.spawn_result
        for env_id, path in enumerate(cable.data.batch.paths):
            handle = scene.get_cable(path)
            matrices = handle.get_segment_poses()
            np.testing.assert_allclose(
                initial[env_id, :, :3].cpu(), matrices[:, :3, 3], atol=2e-6
            )
            np.testing.assert_allclose(
                Rotation.from_quat(initial[env_id, :, 3:7].cpu()).as_matrix(),
                matrices[:, :3, :3],
                atol=2e-6,
            )
        offsets = torch.tensor(
            np.asarray(scene.arenas.root_offsets)[list(cable.data.batch.arena_indices)],
            device=cable.device,
        )
        torch.testing.assert_close(
            initial[:, :, :3] - offsets[:, None, :],
            torch.tensor(
                [[0.5, 0.2, 1.05], [0.5, 0.2, 1.15]], device=cable.device
            ).expand(2, 2, 3),
        )
        self.sim.prepare()
        for _ in range(3):
            self.sim.update(step=1)
        before = cable.data.segment_state_w
        assert before[0, :, 2].mean() < initial[0, :, 2].mean()
        cable.set_force([0, 0.1, 0], env_ids=[1])
        self.sim.update(step=1)
        after = cable.data.segment_state_w
        displacement = after[:, -1, 1] - before[:, -1, 1]
        assert displacement[1] > displacement[0] + 1e-6
        cable.clear_wrenches(env_ids=[1])
        torch.testing.assert_close(cable.data.default_segment_state_w, initial)
        self.sim.reset_objects_state(env_ids=[0])
        reset = cable.data.segment_state_w
        torch.testing.assert_close(reset[0], initial[0], rtol=0, atol=0)
        torch.testing.assert_close(reset[1], after[1], rtol=0, atol=0)
        state = reset[[0]].clone()
        state[..., 0] += 0.025
        cable.write_segment_state_to_sim(state, env_ids=[0])
        torch.testing.assert_close(cable.data.segment_state_w[0], state[0])
        torch.testing.assert_close(
            cable.data.segment_state_w[1], after[1], rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="finite"):
            cable.write_segment_state_to_sim(torch.full_like(state, float("nan")), [0])
        torch.testing.assert_close(cable.data.segment_state_w[0], state[0])
        data = cable.data
        self.sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        assert cable.data is None
        with pytest.raises(RuntimeError, match="invalid"):
            _ = data.segment_state_w
        del self.sim

    def test_attached_articulation_expert_trajectory_repeats_after_reset(
        self, tmp_path
    ) -> None:
        """Public commands, seeded initial state, and recorded repeated episodes."""
        urdf = tmp_path / "linear_tool.urdf"
        urdf.write_text("""<robot name="linear_tool">
<link name="base"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="tool"><inertial><mass value=".1"/><inertia ixx=".001" iyy=".001" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/><axis xyz="0 1 0"/><limit lower="-.2" upper=".2" effort="100" velocity="1"/></joint>
</robot>""")
        robot = self.sim.add_articulation(
            ArticulationCfg(
                uid="expert_tool",
                fpath=str(urdf),
                init_pos=(0, 0, 1),
                init_qpos=[0.0],
                asset_physics_mode="overlay",
                joint_drive_props=JointDrivePropertiesCfg(
                    stiffness=100.0, damping=5.0, target_mode="position"
                ),
            )
        )
        cable = self.sim.add_cable_object(
            CableObjectCfg(
                uid="wire",
                points=[[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
                init_pos=(0, 0, 1),
                radius=0.005,
                start_attachment=CableAttachmentCfg(
                    target="expert_tool", link_name="tool"
                ),
                attrs=CablePhysicsCfg(bend_stiffness=0.01, twist_stiffness=0.01),
            )
        )
        self.sim.prepare()
        seed = 73
        generator = torch.Generator(device=cable.device).manual_seed(seed)
        initial = cable.data.default_segment_state_w
        # Small tail-only perturbation leaves the attached segment and anchor
        # unchanged. Rest geometry remains engine-owned.
        initial[:, 1:, :3] += 0.001 * (
            torch.rand((2, 2, 3), device=cable.device, generator=generator) - 0.5
        )
        initial[..., 7:] = 0
        frames = 40
        commands = (
            torch.linspace(0, 0.03, frames, device=cable.device)[:, None, None]
            .expand(-1, 2, 1)
            .contiguous()
        )
        episodes, joint_episodes = [], []
        for _ in range(2):
            self.sim.reset_objects_state()
            cable.write_segment_state_to_sim(initial)
            torch.testing.assert_close(
                robot.get_qpos(),
                torch.zeros_like(robot.get_qpos()),
                atol=1e-7,
                rtol=0,
            )
            torch.testing.assert_close(
                robot.get_qpos(target=True),
                torch.zeros_like(robot.get_qpos()),
                atol=1e-7,
                rtol=0,
            )
            states, joints = [], []
            for command in commands:
                robot.set_qpos(command, target=True)
                self.sim.update(step=1)
                states.append(cable.data.segment_state_w)
                joints.append(robot.get_qpos().clone())
            episodes.append(torch.stack(states))
            joint_episodes.append(torch.stack(joints))
        states = torch.stack(episodes)
        joints = torch.stack(joint_episodes)
        assert torch.isfinite(states).all() and torch.isfinite(joints).all()
        assert joints[0, -1, :, 0].min() > 0.01
        assert states[0, -1, :, 0, 1].min() > 0.01
        torch.testing.assert_close(states[0], states[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(joints[0], joints[1], atol=1e-6, rtol=1e-5)
        np.savez_compressed(
            tmp_path / "expert_trajectory.npz",
            seed=seed,
            physics_dt=self.sim.sim_config.physics_dt,
            commands=commands.cpu().numpy(),
            qpos=joints.cpu().numpy(),
            cable_state_w=states.cpu().numpy(),
        )

    def test_nonzero_joint_creation_defaults_and_targets_survive_reset(
        self, tmp_path
    ) -> None:
        """An initialization hook must seed defaults without driving later frames."""
        urdf = tmp_path / "linear_tool.urdf"
        urdf.write_text("""<robot name="linear_tool">
<link name="base"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="tool"><inertial><mass value=".1"/><inertia ixx=".001" iyy=".001" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/><axis xyz="0 1 0"/><limit lower="-.2" upper=".2" effort="100" velocity="1"/></joint>
</robot>""")

        robot = self.sim.add_articulation(
            ArticulationCfg(
                uid="tool",
                fpath=str(urdf),
                init_pos=(0, 0, 1),
                init_qpos=torch.tensor([0.025], device=self.sim.device),
                asset_physics_mode="overlay",
                joint_drive_props=JointDrivePropertiesCfg(
                    stiffness=100.0, damping=5.0, target_mode="position"
                ),
            )
        )
        self.sim.add_cable_object(
            CableObjectCfg(
                uid="wire",
                points=[[0, 0, 0], [0.1, 0, 0]],
                init_pos=(1, 0, 1),
                radius=0.005,
            )
        )
        self.sim.prepare()
        expected = torch.full_like(robot.get_qpos(), 0.025)
        self.sim.reset_objects_state()
        torch.testing.assert_close(robot.get_qpos(), expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(
            robot.get_qpos(target=True), expected, rtol=0, atol=1e-7
        )
        robot.set_qpos(torch.full_like(expected, 0.075), target=True)
        for _ in range(15):
            self.sim.update(step=1)
        moved = robot.get_qpos().clone()
        assert moved.min() > 0.03
        self.sim.reset_objects_state(env_ids=[0])
        torch.testing.assert_close(robot.get_qpos()[0], expected[0], rtol=0, atol=1e-7)
        torch.testing.assert_close(
            robot.get_qpos(target=True)[0], expected[0], rtol=0, atol=1e-7
        )
        torch.testing.assert_close(robot.get_qpos()[1], moved[1], rtol=0, atol=0)
