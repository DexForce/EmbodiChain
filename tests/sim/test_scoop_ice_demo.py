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

import numpy as np
import pytest
import torch

from examples.sim.demo import scoop_ice as demo


def test_ice_is_separated_and_stationary_before_first_step() -> None:
    class Ice:
        num_objects = 300
        pose = torch.eye(4).repeat(1, 300, 1, 1)
        cleared = False

        def set_local_pose(self, pose, obj_ids=None):
            if obj_ids is None:
                self.pose = pose
            else:
                self.pose[:, obj_ids] = pose

        def clear_dynamics(self):
            self.cleared = True

    ice = Ice()

    def update(step):
        positions = ice.pose[0, :, :3, 3].numpy()
        # Meshes fit inside a 29 mm box; leave contact-offset clearance.
        separations = np.abs(positions[:, None] - positions[None, :]).max(axis=-1)
        np.fill_diagonal(separations, np.inf)
        assert separations.min() >= 0.030
        assert ice.cleared

    sim = SimpleNamespace(
        device="cpu",
        num_envs=1,
        update=update,
        get_articulation=lambda uid: SimpleNamespace(
            get_local_pose=lambda **kwargs: torch.eye(4).unsqueeze(0)
        ),
    )
    demo.randomize_ice_positions(sim, ice)


def test_unreachable_arm_pose_is_rejected() -> None:
    robot = SimpleNamespace(
        compute_ik=lambda *args, **kwargs: (torch.tensor([False]), torch.zeros(1, 6))
    )
    with pytest.raises(RuntimeError, match="unreachable"):
        demo._solve_arm_ik(robot, torch.eye(4).unsqueeze(0), torch.zeros(1, 6))


def test_scoop_path_follows_container_and_measured_grasp() -> None:
    def plan(scene_pose, grasp_pose):
        commands = []
        scoop_pose = scene_pose.clone()
        scoop_pose[:, :3, 3] += torch.tensor([0.0, 0.0, 0.1])

        def compute_ik(pose, **kwargs):
            commands.append(pose @ torch.linalg.inv(grasp_pose))
            return torch.tensor([True]), torch.zeros(1, 6)

        robot = SimpleNamespace(
            get_joint_ids=lambda name: list(range(6)),
            get_qpos=lambda: torch.zeros(1, 6),
            compute_fk=lambda *args, **kwargs: scoop_pose @ grasp_pose,
            compute_ik=compute_ik,
            set_qpos=lambda *args, **kwargs: None,
        )
        sim = SimpleNamespace(
            device="cpu",
            update=lambda **kwargs: None,
            get_articulation=lambda uid: SimpleNamespace(
                get_local_pose=lambda **kwargs: scene_pose
            ),
        )
        scoop = SimpleNamespace(get_local_pose=lambda **kwargs: scoop_pose)
        demo.scoop_ice(sim, robot, scoop)
        return torch.stack(commands)

    identity = torch.eye(4).unsqueeze(0)
    moved_bin = identity.clone()
    moved_bin[:, :3, 3] = torch.tensor([0.1, -0.2, 0.05])
    tilted_grasp = identity.clone()
    tilted_grasp[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tilted_grasp[:, :3, 3] = torch.tensor([0.03, 0.02, 0.1])
    baseline = plan(identity, identity)
    # Approach outside the front wall (bin-local -X), then enter the bin.
    assert baseline[1, 0, 0, 3] < -0.21
    assert abs(baseline[-1, 0, 0, 3]) < 0.15
    torch.testing.assert_close(plan(moved_bin, tilted_grasp), moved_bin @ baseline)


@pytest.mark.gpu
@pytest.mark.requires_sim
@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_scoops_and_retains_ice(seed: int) -> None:
    """Qualify the physical grasp and retained ice after the final settling step."""
    sim = demo.initialize_simulation(
        demo.build_parser().parse_args(["--headless", "--physics", "default"])
    )
    try:
        robot = demo.create_robot(sim)
        demo.create_container(sim)
        padding = demo.create_padding_box(sim)
        scoop = demo.create_scoop(sim)
        heavy = demo.create_heave_ice(sim)
        ice = demo.create_ice_cubes(sim)
        sim.prepare()
        demo.randomize_ice_positions(sim, ice, seed=seed)
        demo.scoop_grasp(sim, robot, scoop, heavy, padding)
        demo.scoop_ice(sim, robot, scoop)

        assert scoop.get_local_pose()[0, 2] > 0.30  # Above the 22 cm bin rim.
        relative = torch.linalg.inv(scoop.get_local_pose(to_matrix=True))[
            :, None
        ] @ ice.get_local_pose(to_matrix=True)
        centers = relative[0, :, :3, 3]
        # Bowl bounds, including one cube radius above the upper surface.
        retained = (
            (centers[:, 0].abs() < 0.055)
            & (centers[:, 1] > -0.19)
            & (centers[:, 1] < -0.04)
            & (centers[:, 2] > -0.025)
            & (centers[:, 2] < 0.065)
        )
        count = int(retained.sum())
        print(f"seed={seed}: retained {count} ice cubes")
        assert count > 0
    finally:
        sim.destroy(exit_process=False)
        demo.SimulationManager.flush_cleanup_queue()
