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

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions.actions import (
    MoveAction,
    MoveActionCfg,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
)
from embodichain.lab.sim.atomic_actions.core import AntipodalAffordance, ObjectSemantics
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import MoveType
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg


class DummyMotionGenerator:
    def __init__(self, robot: Robot) -> None:
        self.robot = robot


def create_robot(sim: SimulationManager, position: list[float] | None = None) -> Robot:
    if position is None:
        position = [0.0, 0.0, 0.0]

    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")

    cfg = RobotCfg(
        uid="UR10",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur10_urdf_path},
                {"component_type": "hand", "urdf_path": gripper_urdf_path},
            ]
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e2},
            damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e1},
            max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e3},
            drive_type="force",
        ),
        control_parts={
            "arm": ["JOINT[0-9]"],
            "hand": ["FINGER[1-2]"],
        },
        solver_cfg={
            "arm": PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=[
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.12],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        },
        init_qpos=[0.0, -np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_mug(sim: SimulationManager) -> RigidObject:
    mug_cfg = RigidObjectCfg(
        uid="table",
        shape=MeshCfg(
            fpath=get_data_path("CoffeeCup/cup.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.01,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[0.55, 0.0, 0.01],
        init_rot=[0.0, 0.0, -90],
        body_scale=(4, 4, 4),
    )
    return sim.add_rigid_object(cfg=mug_cfg)


class BaseActionTest:
    @classmethod
    def setup_class(cls) -> None:
        cls.sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=2)
        )
        cls.robot = create_robot(cls.sim)
        cls.mug = create_mug(cls.sim)
        cls.motion_generator = DummyMotionGenerator(cls.robot)
        cls.arm_joint_ids = cls.robot.get_joint_ids("arm")
        cls.hand_joint_ids = cls.robot.get_joint_ids("hand")
        cls.arm_dof = len(cls.arm_joint_ids)
        cls.hand_dof = len(cls.hand_joint_ids)
        cls.device = cls.robot.device
        cls.sim.update(step=1)

    @classmethod
    def teardown_class(cls) -> None:
        cls.sim.destroy()

    def _make_pose(
        self,
        translation: tuple[float, float, float],
    ) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32, device=self.device)
        pose[:3, 3] = torch.tensor(translation, dtype=torch.float32, device=self.device)
        return pose

    def _make_semantics(self) -> ObjectSemantics:
        return ObjectSemantics(
            label="mug",
            geometry={
                "mesh_vertices": torch.zeros(
                    (3, 3), dtype=torch.float32, device=self.device
                ),
                "mesh_triangles": torch.tensor(
                    [[0, 1, 2]], dtype=torch.int64, device=self.device
                ),
            },
            affordance=AntipodalAffordance(object_label="mug"),
            entity=self.mug,
        )


class TestActions(BaseActionTest):
    def test_move_action_execute_with_pose_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        action = MoveAction(
            motion_generator=self.motion_generator,
            cfg=MoveActionCfg(control_part="arm", sample_interval=6),
        )
        target_pose = self._make_pose((0.45, -0.05, 0.18))
        plan_call: dict[str, object] = {}

        def fake_plan_arm_trajectory(
            target_states_list,
            start_qpos,
            n_waypoints,
            arm_dof=None,
        ):
            actual_arm_dof = action.dof if arm_dof is None else arm_dof
            plan_call["target_states_list"] = target_states_list
            plan_call["start_qpos"] = start_qpos.clone()
            plan_call["n_waypoints"] = n_waypoints
            return torch.full(
                (action.n_envs, n_waypoints, actual_arm_dof),
                fill_value=0.5,
                dtype=torch.float32,
                device=action.device,
            )

        monkeypatch.setattr(action, "_plan_arm_trajectory", fake_plan_arm_trajectory)

        success, trajectory, joint_ids = action.execute(target=target_pose)

        assert success is True
        assert trajectory.shape == (
            action.n_envs,
            action.cfg.sample_interval,
            action.dof,
        )
        assert joint_ids == self.arm_joint_ids
        assert plan_call["n_waypoints"] == action.cfg.sample_interval
        assert torch.allclose(
            plan_call["start_qpos"],
            self.robot.get_qpos(name="arm"),
        )

        target_states_list = plan_call["target_states_list"]
        assert len(target_states_list) == action.n_envs
        for target_states in target_states_list:
            assert len(target_states) == 1
            assert target_states[0].move_type == MoveType.EEF_MOVE
            assert torch.allclose(target_states[0].xpos, target_pose)

    def test_move_action_resolve_start_qpos_repeats_single_configuration(self) -> None:
        action = MoveAction(
            motion_generator=self.motion_generator,
            cfg=MoveActionCfg(control_part="arm", sample_interval=6),
        )
        single_qpos = torch.zeros(self.arm_dof, dtype=torch.float32, device=self.device)

        resolved_qpos = action._resolve_start_qpos(single_qpos)

        assert resolved_qpos.shape == (action.n_envs, self.arm_dof)
        assert torch.allclose(
            resolved_qpos,
            torch.zeros(
                (action.n_envs, self.arm_dof), dtype=torch.float32, device=self.device
            ),
        )

    def test_pick_up_action_execute_builds_three_phase_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        action = PickUpAction(
            motion_generator=self.motion_generator,
            cfg=PickUpActionCfg(
                control_part="arm",
                hand_control_part="hand",
                hand_open_qpos=torch.tensor(
                    [0.0, 0.0], dtype=torch.float32, device=self.device
                ),
                hand_close_qpos=torch.tensor(
                    [0.025, 0.025], dtype=torch.float32, device=self.device
                ),
                approach_direction=torch.tensor(
                    [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device
                ),
                pre_grasp_distance=0.15,
                lift_height=0.2,
                sample_interval=11,
                hand_interp_steps=3,
            ),
        )
        grasp_pose = (
            self._make_pose((0.5, 0.02, 0.12)).unsqueeze(0).repeat(action.n_envs, 1, 1)
        )
        semantics = self._make_semantics()
        plan_calls: list[dict[str, object]] = []

        def fake_resolve_grasp_pose(_semantics):
            return (
                True,
                grasp_pose,
                torch.full(
                    (action.n_envs,), 0.025, dtype=torch.float32, device=self.device
                ),
            )

        def fake_plan_arm_trajectory(
            target_states_list,
            start_qpos,
            n_waypoints,
            arm_dof=None,
        ):
            actual_arm_dof = action.arm_dof if arm_dof is None else arm_dof
            fill_value = float(len(plan_calls) + 1)
            plan_calls.append(
                {
                    "target_states_list": target_states_list,
                    "start_qpos": start_qpos.clone(),
                    "n_waypoints": n_waypoints,
                    "arm_dof": actual_arm_dof,
                }
            )
            return torch.full(
                (action.n_envs, n_waypoints, actual_arm_dof),
                fill_value=fill_value,
                dtype=torch.float32,
                device=action.device,
            )

        monkeypatch.setattr(action, "_resolve_grasp_pose", fake_resolve_grasp_pose)
        monkeypatch.setattr(action, "_plan_arm_trajectory", fake_plan_arm_trajectory)

        start_qpos = self.robot.get_qpos(name="arm")
        success, trajectory, joint_ids = action.execute(
            target=semantics,
            start_qpos=start_qpos,
        )

        n_approach, n_close, n_lift = action._compute_three_phase_waypoints(
            action.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )

        assert success is True
        assert len(plan_calls) == 2
        assert trajectory.shape == (
            action.n_envs,
            n_approach + n_close + n_lift,
            action.dof,
        )
        assert joint_ids == action.joint_ids
        assert torch.allclose(plan_calls[0]["start_qpos"], start_qpos)
        assert torch.allclose(
            plan_calls[1]["start_qpos"],
            torch.ones(
                (action.n_envs, action.arm_dof),
                dtype=torch.float32,
                device=self.device,
            ),
        )

        approach_states = plan_calls[0]["target_states_list"]
        for env_id in range(action.n_envs):
            assert len(approach_states[env_id]) == 2
            pre_grasp_pose = approach_states[env_id][0].xpos
            final_grasp_pose = approach_states[env_id][1].xpos
            assert approach_states[env_id][0].move_type == MoveType.EEF_MOVE
            assert approach_states[env_id][1].move_type == MoveType.EEF_MOVE
            assert torch.allclose(final_grasp_pose, grasp_pose[env_id])
            assert pre_grasp_pose[2, 3] == pytest.approx(
                final_grasp_pose[2, 3].item() + action.cfg.pre_grasp_distance
            )

        lift_states = plan_calls[1]["target_states_list"]
        for env_id in range(action.n_envs):
            assert len(lift_states[env_id]) == 1
            assert lift_states[env_id][0].xpos[2, 3] > grasp_pose[env_id][2, 3]

        assert torch.allclose(
            trajectory[:, :n_approach, : action.arm_dof],
            torch.ones(
                (action.n_envs, n_approach, action.arm_dof),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        assert torch.allclose(
            trajectory[:, :n_approach, action.arm_dof :],
            action.hand_open_qpos.view(1, 1, -1).expand(action.n_envs, n_approach, -1),
        )

        expected_close_path = action._interpolate_hand_qpos(
            action.hand_open_qpos,
            action.hand_close_qpos,
            n_close,
        )
        assert torch.allclose(
            trajectory[:, n_approach : n_approach + n_close, : action.arm_dof],
            torch.ones(
                (action.n_envs, n_close, action.arm_dof),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        assert torch.allclose(
            trajectory[:, n_approach : n_approach + n_close, action.arm_dof :],
            expected_close_path.unsqueeze(0).expand(action.n_envs, -1, -1),
        )
        assert torch.allclose(
            trajectory[:, n_approach + n_close :, : action.arm_dof],
            torch.full(
                (action.n_envs, n_lift, action.arm_dof),
                fill_value=2.0,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        assert torch.allclose(
            trajectory[:, n_approach + n_close :, action.arm_dof :],
            action.hand_close_qpos.view(1, 1, -1).expand(action.n_envs, n_lift, -1),
        )

    def test_place_action_execute_builds_release_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        action = PlaceAction(
            motion_generator=self.motion_generator,
            cfg=PlaceActionCfg(
                control_part="arm",
                hand_control_part="hand",
                hand_open_qpos=torch.tensor(
                    [0.0, 0.0], dtype=torch.float32, device=self.device
                ),
                hand_close_qpos=torch.tensor(
                    [0.025, 0.025], dtype=torch.float32, device=self.device
                ),
                lift_height=0.2,
                sample_interval=11,
                hand_interp_steps=3,
            ),
        )
        place_pose = self._make_pose((0.42, 0.18, 0.1))
        start_qpos = self.robot.get_qpos(name="arm")
        plan_calls: list[dict[str, object]] = []

        def fake_plan_arm_trajectory(
            target_states_list,
            start_qpos_arg,
            n_waypoints,
            arm_dof=None,
        ):
            actual_arm_dof = action.arm_dof if arm_dof is None else arm_dof
            fill_value = float(len(plan_calls) + 1)
            plan_calls.append(
                {
                    "target_states_list": target_states_list,
                    "start_qpos": start_qpos_arg.clone(),
                    "n_waypoints": n_waypoints,
                    "arm_dof": actual_arm_dof,
                }
            )
            return torch.full(
                (action.n_envs, n_waypoints, actual_arm_dof),
                fill_value=fill_value,
                dtype=torch.float32,
                device=action.device,
            )

        monkeypatch.setattr(action, "_plan_arm_trajectory", fake_plan_arm_trajectory)

        success, trajectory, joint_ids = action.execute(
            target=place_pose,
            start_qpos=start_qpos,
        )

        n_down, n_open, n_lift = action._compute_three_phase_waypoints(
            action.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )

        assert success is True
        assert len(plan_calls) == 2
        assert trajectory.shape == (
            action.n_envs,
            n_down + n_open + n_lift,
            action.dof,
        )
        assert joint_ids == action.joint_ids
        assert torch.allclose(plan_calls[0]["start_qpos"], start_qpos)
        assert torch.allclose(
            plan_calls[1]["start_qpos"],
            torch.ones(
                (action.n_envs, action.arm_dof),
                dtype=torch.float32,
                device=self.device,
            ),
        )

        down_states = plan_calls[0]["target_states_list"]
        for env_id in range(action.n_envs):
            assert len(down_states[env_id]) == 2
            lifted_pose = down_states[env_id][0].xpos
            final_place_pose = down_states[env_id][1].xpos
            assert torch.allclose(final_place_pose, place_pose)
            assert lifted_pose[2, 3] > final_place_pose[2, 3]

        expected_open_path = action._interpolate_hand_qpos(
            action.hand_close_qpos,
            action.hand_open_qpos,
            n_open,
        )
        assert torch.allclose(
            trajectory[:, :n_down, : action.arm_dof],
            torch.ones(
                (action.n_envs, n_down, action.arm_dof),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        assert torch.allclose(
            trajectory[:, :n_down, action.arm_dof :],
            action.hand_close_qpos.view(1, 1, -1).expand(action.n_envs, n_down, -1),
        )
        assert torch.allclose(
            trajectory[:, n_down : n_down + n_open, : action.arm_dof],
            torch.ones(
                (action.n_envs, n_open, action.arm_dof),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        assert torch.allclose(
            trajectory[:, n_down : n_down + n_open, action.arm_dof :],
            expected_open_path.unsqueeze(0).expand(action.n_envs, -1, -1),
        )
        assert torch.allclose(
            trajectory[:, n_down + n_open :, : action.arm_dof],
            torch.full(
                (action.n_envs, n_lift, action.arm_dof),
                fill_value=2.0,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        assert torch.allclose(
            trajectory[:, n_down + n_open :, action.arm_dof :],
            action.hand_open_qpos.view(1, 1, -1).expand(action.n_envs, n_lift, -1),
        )
