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
# ----------------------------------------------------------------------------,

import pytest
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.sensors import StereoCamera, SensorCfg
import time
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RenderCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.sensors import (
    ContactSensorCfg,
    ArticulationContactFilterCfg,
    SensorCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg, Robot, RobotCfg
from embodichain.data import get_data_path

NUM_ENVS = 4


class ContactTest:
    def setup_simulation(self, sim_device):
        sim_cfg = SimulationManagerCfg(
            width=1920,
            height=1080,
            num_envs=2,
            headless=True,
            physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
            sim_device=sim_device,
        )

        # Create the simulation instance
        self.sim = SimulationManager(sim_cfg)

        # Add objects to the scene
        cube2 = self.create_cube("cube2", position=[0.0, 0.0, 0.09])
        self.robot = self.create_robot("UR10_PGI", position=[0.5, 0.0, 0.0])

        contact_filter_cfg = ContactSensorCfg()
        contact_filter_cfg.rigid_uid_list = ["cube2"]
        contact_filter_art_cfg = ArticulationContactFilterCfg()
        contact_filter_art_cfg.articulation_uid = "UR10_PGI"
        contact_filter_art_cfg.link_name_list = ["finger1_link", "finger2_link"]
        contact_filter_cfg.articulation_cfg_list = [contact_filter_art_cfg]
        contact_filter_cfg.filter_need_both_actor = True
        self.contact_sensor = self.sim.add_sensor(sensor_cfg=contact_filter_cfg)

        self.to_grasp_pose(cube2)

    def create_cube(self, uid: str, position: list = (0.0, 0.0, 0)) -> RigidObject:
        """create cube

        Args:
            sim (SimulationManager): simulation manager
            uid (str): uid of the rigid object
            position (list, optional): init position. Defaults to (0., 0., 0).

        Returns:
            RigidObject: rigid object
        """
        cube_size = (0.025, 0.025, 0.025)
        cube: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid=uid,
                shape=CubeCfg(size=cube_size),
                body_type="dynamic",
                attrs=RigidBodyAttributesCfg(
                    mass=0.1,
                    dynamic_friction=0.9,
                    static_friction=0.95,
                    restitution=0.01,
                    sleep_threshold=0.0,
                ),
                init_pos=position,
            )
        )
        return cube

    def create_robot(self, uid: str, position: list = (0.0, 0.0, 0)) -> Robot:
        """create robot

        Args:
            sim (SimulationManager): _description_
            uid (str): _description_
            position (list, optional): _description_. Defaults to (0., 0., 0).

        Returns:
            Robot: _description_
        """
        ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
        pgi_urdf_path = get_data_path("DH_PGC_140_50/DH_PGC_140_50.urdf")
        robot_cfg_dict = {
            "uid": "UR10_PGI",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "arm",
                        "urdf_path": ur10_urdf_path,
                        "transform": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                    {
                        "component_type": "hand",
                        "urdf_path": pgi_urdf_path,
                        "transform": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                ],
            },
            "init_pos": position,
            "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
            "drive_pros": {
                "stiffness": {"JOINT[1-6]": 1e4, "FINGER[1-2]_JOINT": 1e2},
                "damping": {"JOINT[1-6]": 1e3, "FINGER[1-2]_JOINT": 1e1},
                "max_effort": {"JOINT[1-6]": 1e5, "FINGER[1-2]_JOINT": 1e3},
            },
            "solver_cfg": {
                "arm": {
                    "class_type": "PytorchSolver",
                    "end_link_name": "ee_link",
                    "root_link_name": "base_link",
                    "tcp": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.13],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            },
            "control_parts": {"arm": ["JOINT[1-6]"], "hand": ["FINGER[1-2]_JOINT"]},
        }
        robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(robot_cfg_dict))
        return robot

    def to_grasp_pose(self, cube: RigidObject):
        self.sim.update(step=100)
        arm_ids = self.robot.get_joint_ids("arm")
        gripper_ids = self.robot.get_joint_ids("hand")
        rest_arm_qpos = self.robot.get_qpos()[:, arm_ids]
        ee_xpos = self.robot.compute_fk(qpos=rest_arm_qpos, name="arm", to_matrix=True)
        target_xpos = ee_xpos.clone()
        cube_xpos = cube.get_local_pose(to_matrix=True)
        cube_position = cube_xpos[:, :3, 3]

        target_xpos[:, :3, 3] = cube_position

        approach_xpos = target_xpos.clone()
        approach_xpos[:, 2, 3] += 0.1

        is_success, approach_qpos = self.robot.compute_ik(
            pose=approach_xpos, joint_seed=rest_arm_qpos, name="arm"
        )
        is_success, target_qpos = self.robot.compute_ik(
            pose=target_xpos, joint_seed=approach_qpos, name="arm"
        )
        self.robot.set_qpos(approach_qpos, joint_ids=arm_ids)
        self.sim.update(step=40)

        self.robot.set_qpos(target_qpos, joint_ids=arm_ids)
        self.sim.update(step=40)
        hand_close_qpos = (
            torch.tensor([0.025, 0.025], device=self.sim.device)
            .unsqueeze(0)
            .repeat(self.sim.num_envs, 1)
        )
        self.robot.set_qpos(hand_close_qpos, joint_ids=gripper_ids)
        self.sim.update(step=20)

    def test_fetch_contact(self):
        self.sim.update(step=1)
        self.contact_sensor.update()
        contact_report = self.contact_sensor.get_data()

        # Check that contact data has correct shape (num_envs, max_contacts_per_env, ...)
        assert contact_report["position"].shape[0] == self.sim.num_envs
        assert (
            contact_report["position"].dim() == 3
        )  # (num_envs, max_contacts_per_env, 3)

        # Check that is_valid field exists and has correct shape
        assert "is_valid" in contact_report.keys()
        assert contact_report["is_valid"].shape == (
            self.sim.num_envs,
            self.contact_sensor.cfg.max_contacts_per_env,
        )
        assert contact_report["is_valid"].dtype == torch.bool

        # Check that we have contacts in at least one environment
        total_contacts = self.contact_sensor.total_current_contacts
        assert total_contacts > 0, "No contact detected."

        # Check that is_valid correctly indicates valid contacts
        for env_id in range(self.sim.num_envs):
            num_contacts = self.contact_sensor._num_contacts_per_env[env_id].item()
            if num_contacts > 0:
                # First num_contacts slots should be True
                assert contact_report["is_valid"][env_id, :num_contacts].all()
                # Remaining slots should be False
                assert not contact_report["is_valid"][env_id, num_contacts:].any()

        cube2_user_ids = self.sim.get_rigid_object("cube2").get_user_ids()
        finger1_user_ids = (
            self.sim.get_robot("UR10_PGI").get_user_ids("finger1_link").reshape(-1)
        )
        filter_user_ids = torch.cat([cube2_user_ids, self.sim.get_robot("UR10_PGI").get_user_ids("finger1_link").reshape(-1), self.sim.get_robot("UR10_PGI").get_user_ids("finger2_link").reshape(-1)])
        filter_contact_report = self.contact_sensor.filter_by_user_ids(filter_user_ids)
        n_filtered_contact = filter_contact_report["position"].shape[0]
        assert n_filtered_contact > 0, "No contact detected between gripper and cube."
        # Check that filtered results also have is_valid field
        assert "is_valid" in filter_contact_report.keys()
        # All filtered contacts should be valid (True)
        assert filter_contact_report["is_valid"].all()

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestContactRaster(ContactTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestContactRasterCuda(ContactTest):
    def setup_method(self):
        self.setup_simulation("cuda")


class TestContactFastRT(ContactTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestContactFastRTCuda(ContactTest):
    def setup_method(self):
        self.setup_simulation("cuda")


def test_contact_sensor_from_dict():
    """Test ContactSensorCfg.from_dict converts list items correctly."""
    dict_config = {
        "sensor_type": "ContactSensor",
        "rigid_uid_list": ["cube1", "cube2"],
        "articulation_cfg_list": [
            {
                "articulation_uid": "robot1",
                "link_name_list": ["link1", "link2"],
            }
        ],
        "filter_need_both_actor": True,
        "max_contacts_per_env": 1000,
    }

    cfg = SensorCfg.from_dict(dict_config)

    assert cfg.sensor_type == "ContactSensor"
    assert cfg.rigid_uid_list == ["cube1", "cube2"]
    assert cfg.filter_need_both_actor is True
    assert cfg.max_contacts_per_env == 1000

    # Verify articulation_cfg_list items are properly converted
    assert len(cfg.articulation_cfg_list) == 1
    art_cfg = cfg.articulation_cfg_list[0]
    assert isinstance(art_cfg, ArticulationContactFilterCfg)
    assert art_cfg.articulation_uid == "robot1"
    assert art_cfg.link_name_list == ["link1", "link2"]


if __name__ == "__main__":
    test = TestContactRasterCuda()
    test.setup_simulation("cuda")
    test.test_fetch_contact()
