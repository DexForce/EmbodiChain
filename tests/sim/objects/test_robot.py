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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.objects.backends.newton import _default_mujoco_mimic_solref
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim.cfg import physics_cfg_for_backend
from embodichain.data import get_data_path

# Define control parts
CONTROL_PARTS = {
    "left_arm": [
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",
    ],
    "right_arm": [
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",
    ],
}

W1_ACTIVE_DOF = 40  # Dexforce W1 v021 scalar active-DOF count.


@pytest.mark.no_sim
def test_default_mujoco_mimic_solref_preserves_damping_and_timestep_floor():
    np.testing.assert_allclose(
        _default_mujoco_mimic_solref(physics_dt=0.01, num_substeps=10),
        [2.0e-3, 1.0e1],
    )
    np.testing.assert_allclose(
        _default_mujoco_mimic_solref(physics_dt=1.0e-4, num_substeps=10),
        [1.0e-4, 1.0e1],
    )


def test_get_qf_selects_control_part_joint_efforts():
    full_qf = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
    )
    robot = object.__new__(Robot)
    robot._data = SimpleNamespace(qf=full_qf)
    robot.cfg = SimpleNamespace(control_parts={"arm": ["joint_3", "joint_1"]})
    robot._joint_ids = {"arm": [3, 1]}

    actual_qf = robot.get_qf(name="arm")

    assert torch.equal(actual_qf, full_qf[:, [3, 1]])


# Base test class for CPU and CUDA
class BaseRobotTest:
    @classmethod
    def setup_simulation(cls, device):
        if hasattr(cls, "sim"):
            return
        # Set up simulation with specified device (CPU or CUDA)
        config = SimulationManagerCfg(headless=True, device=device, num_envs=10)
        cls.sim = SimulationManager(config)

        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1",
                "version": "v021",
            }
        )

        cls.robot: Robot = cls.sim.add_robot(cfg=cfg)
        cls.sim.prepare()

    def test_get_joint_ids(self):
        left_joint_ids = self.robot.get_joint_ids("left_arm")
        right_joint_ids = self.robot.get_joint_ids("right_arm")

        assert left_joint_ids == [
            6,
            8,
            10,
            12,
            14,
            16,
            18,
        ], f"Unexpected left arm joint IDs: {left_joint_ids}"
        assert right_joint_ids == [
            7,
            9,
            11,
            13,
            15,
            17,
            19,
        ], f"Unexpected right arm joint IDs: {right_joint_ids}"

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_fk(self, arm_name: str):
        # Test forward kinematics (FK) for both to_matrix=True and to_matrix=False

        qpos = torch.randn(10, 7, device=self.sim.device)  # Random joint positions

        # Test with to_matrix=False (6D result: translation + Euler angles)
        result_7d = self.robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=False)

        # Check result shape for 6D output (batch, 6)
        assert result_7d.shape == (
            10,
            7,
        ), f"Expected shape (10, 7), got {result_7d.shape}"

        # Test with to_matrix=True (4x4 matrix result)
        result_matrix = self.robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)
        print("result_matrix:", result_matrix)
        # Check result shape for matrix output (batch, 4, 4)
        assert result_matrix.shape == (
            10,
            4,
            4,
        ), f"Expected shape (10, 4, 4), got {result_matrix.shape}"

    def test_compute_fk(self):
        torch.set_printoptions(precision=6, sci_mode=False)
        qpos = np.zeros(40)
        result = self.robot.compute_fk(qpos=qpos, link_names=["left_ee", "right_ee"])

        # Additional checks for specific values (if known)
        expected_values = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.791],
                    [0.0, -1.0, 0.0, 1.3648],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -0.791],
                    [0.0, 1.0, 0.0, 1.3648],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
            device=self.sim.device,
        ).unsqueeze_(0)

        assert torch.allclose(
            result, expected_values, atol=1e-4, rtol=1e-4
        ), f"FK result does not match expected values. Got {result}, expected {expected_values}."

    def test_compute_jacobian(self):
        qpos = np.full(7, 10 * np.pi / 180)

        left_ee_jacobian = self.robot.compute_jacobian(
            qpos=qpos, end_link_name="left_ee", root_link_name="left_arm_base"
        )
        right_ee_jacobian = self.robot.compute_jacobian(
            qpos=qpos, end_link_name="right_ee", root_link_name="right_arm_base"
        )

        assert left_ee_jacobian.shape == (
            1,
            6,
            7,
        ), f"Expected shape (1, 6, 7) for left EE Jacobian, got {left_ee_jacobian.shape}"
        assert right_ee_jacobian.shape == (
            1,
            6,
            7,
        ), f"Expected shape (1, 6, 7) for right EE Jacobian, got {right_ee_jacobian.shape}"

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed

        # Define a sample target pose as a 1x4x4 homogeneous matrix
        target_pose = torch.tensor(
            [
                [-0.3490, -0.6369, -0.6874, -0.4502],
                [0.2168, -0.7685, 0.6020, -0.0639],
                [-0.9117, 0.0611, 0.4063, 0.3361],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            dtype=torch.float32,
            device=self.sim.device,
        ).unsqueeze(0)

        # Define joint_seed as a tensor of ones with shape (1, 7) for initialization
        joint_seed = torch.ones(1, 7, device=self.sim.device)
        success_tensor, qpos_tensor = self.robot.compute_ik(
            pose=target_pose, name=arm_name, joint_seed=joint_seed, env_ids=[0]
        )
        print(f"Success: {success_tensor}, Qpos: {qpos_tensor}")

        # Check output shapes robustly
        assert success_tensor.shape == (
            1,
        ), f"Expected shape (1,), got {success_tensor.shape}"
        assert isinstance(
            qpos_tensor, torch.Tensor
        ), "qpos_tensor should be a torch.Tensor"
        # Accept both (1, 7) and (1, N, 7) shapes
        if qpos_tensor.ndim == 2:
            assert qpos_tensor.shape == (
                1,
                7,
            ), f"Expected shape (1, 7), got {qpos_tensor.shape}"
        elif qpos_tensor.ndim == 3:
            assert (
                qpos_tensor.shape[2] == 7
            ), f"Expected dof 7, got {qpos_tensor.shape[2]}"
            assert (
                qpos_tensor.shape[0] == 1
            ), f"Expected batch size 1, got {qpos_tensor.shape[0]}"
            assert (
                qpos_tensor.shape[1] >= 1
            ), f"Expected at least one solution, got {qpos_tensor.shape[1]}"
        else:
            raise AssertionError(f"Unexpected qpos_tensor shape: {qpos_tensor.shape}")

        # If success, check qpos is not all zeros
        if success_tensor.item():
            assert not torch.all(
                qpos_tensor == 0
            ), "IK returned all zeros for valid solution"

    def test_mimic(self):

        assert (
            len(self.robot.mimic_ids) == 8
        ), f"Expected 8 mimic IDs, got {len(self.robot.mimic_ids)}"

        left_eef_ids_without_mimic = self.robot.get_joint_ids(
            "left_eef", remove_mimic=False
        )
        right_eef_ids_without_mimic = self.robot.get_joint_ids(
            "right_eef", remove_mimic=False
        )
        assert (
            len(left_eef_ids_without_mimic) == 6
        ), f"Expected 6 left eef joint IDs without mimic, got {len(left_eef_ids_without_mimic)}"
        assert (
            len(right_eef_ids_without_mimic) == 6
        ), f"Expected 6 right eef joint IDs without mimic, got {len(right_eef_ids_without_mimic)}"

    def test_default_mimic_tracks_closed_hand_target(self):
        """Keep W1 hand mimic constraints equally stiff on CPU and CUDA."""
        self.robot.reset()
        open_target = torch.tensor(
            [[0.0, 1.5, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        close_target = torch.tensor(
            [[0.1, 1.5, 0.3, 0.2, 0.3, 0.3]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        for target in (open_target, close_target):
            self.robot.set_qpos(
                target.repeat(self.robot.num_instances, 1), name="right_eef"
            )
            self.sim.update(step=100)

        qpos = self.robot.body_data.qpos
        target_qpos = self.robot.body_data.target_qpos
        right_eef_ids = self.robot.get_joint_ids("right_eef")
        right_mimic_errors = []
        for mimic_id, parent_id, multiplier, offset in zip(
            self.robot.mimic_ids,
            self.robot.mimic_parents,
            self.robot.mimic_multipliers,
            self.robot.mimic_offsets,
            strict=True,
        ):
            if not self.robot.joint_names[parent_id].startswith("RIGHT_HAND"):
                continue
            right_mimic_errors.append(
                torch.abs(
                    qpos[:, mimic_id] - (qpos[:, parent_id] * multiplier + offset)
                )
            )

        assert torch.max(torch.stack(right_mimic_errors)).item() < 0.02
        assert (
            torch.max(
                torch.abs(qpos[:, right_eef_ids] - target_qpos[:, right_eef_ids])
            ).item()
            < 0.01
        )

    def test_setter_and_getter_with_control_part(self):
        left_arm_qpos = self.robot.get_qpos(name="left_arm")
        assert left_arm_qpos.shape == (10, 7)

        left_qpos_limits = self.robot.get_qpos_limits(name="left_arm")
        assert left_qpos_limits.shape == (10, 7, 2)

        dummy_qpos = torch.randn(10, 7, device=self.sim.device)
        # Clamp to limits
        dummy_qpos = torch.max(
            torch.min(dummy_qpos, left_qpos_limits[:, :, 1]), left_qpos_limits[:, :, 0]
        )
        self.robot.set_qpos(qpos=dummy_qpos, name="left_arm")

    def test_joint_limit_apis_support_control_parts_and_joint_ids(self):
        left_arm_joint_ids = self.robot.get_joint_ids("left_arm")
        selected_joint_ids = left_arm_joint_ids[:2]

        qpos_limits = self.robot.get_qpos_limits(joint_ids=selected_joint_ids)
        qvel_limits = self.robot.get_qvel_limits(joint_ids=selected_joint_ids)
        qf_limits = self.robot.get_qf_limits(joint_ids=selected_joint_ids)

        assert qpos_limits.shape == (10, len(selected_joint_ids), 2)
        assert qvel_limits.shape == (10, len(selected_joint_ids))
        assert qf_limits.shape == (10, len(selected_joint_ids))

        left_arm_qpos_limits = self.robot.get_qpos_limits(
            name="left_arm", joint_ids=[0]
        )
        left_arm_qvel_limits = self.robot.get_qvel_limits(
            name="left_arm", joint_ids=[0]
        )
        left_arm_qf_limits = self.robot.get_qf_limits(name="left_arm", joint_ids=[0])

        assert left_arm_qpos_limits.shape == (10, len(left_arm_joint_ids), 2)
        assert left_arm_qvel_limits.shape == (10, len(left_arm_joint_ids))
        assert left_arm_qf_limits.shape == (10, len(left_arm_joint_ids))

    def test_joint_limit_setters_ignore_joint_ids_when_name_is_provided(self):
        left_arm_joint_ids = self.robot.get_joint_ids("left_arm")
        selected_joint_ids = left_arm_joint_ids[:2]

        qpos_limits = self.robot.get_qpos_limits(name="left_arm").clone()
        qpos_limits[..., 0] = qpos_limits[..., 0] + 0.01
        qpos_limits[..., 1] = qpos_limits[..., 1] - 0.01
        self.robot.set_qpos_limits(
            qpos_limits,
            name="left_arm",
            joint_ids=[left_arm_joint_ids[0]],
        )
        assert torch.allclose(
            self.robot.get_qpos_limits(name="left_arm"),
            qpos_limits,
            atol=1e-5,
        )

        qvel_limits = torch.full(
            (10, len(left_arm_joint_ids)),
            0.5,
            device=self.sim.device,
        )
        qf_limits = torch.full(
            (10, len(left_arm_joint_ids)),
            1.5,
            device=self.sim.device,
        )
        self.robot.set_qvel_limits(
            qvel_limits,
            name="left_arm",
            joint_ids=[left_arm_joint_ids[0]],
        )
        self.robot.set_qf_limits(
            qf_limits,
            name="left_arm",
            joint_ids=[left_arm_joint_ids[0]],
        )

        assert torch.allclose(
            self.robot.get_qvel_limits(name="left_arm"),
            qvel_limits,
            atol=1e-5,
        )
        assert torch.allclose(
            self.robot.get_qf_limits(name="left_arm"),
            qf_limits,
            atol=1e-5,
        )

        joint_qpos_limits = self.robot.get_qpos_limits(
            joint_ids=selected_joint_ids
        ).clone()
        joint_qpos_limits[..., 0] = joint_qpos_limits[..., 0] + 0.02
        joint_qpos_limits[..., 1] = joint_qpos_limits[..., 1] - 0.02
        joint_qvel_limits = torch.full(
            (10, len(selected_joint_ids)),
            0.65,
            device=self.sim.device,
        )
        joint_qf_limits = torch.full(
            (10, len(selected_joint_ids)),
            1.65,
            device=self.sim.device,
        )
        self.robot.set_qpos_limits(joint_qpos_limits, joint_ids=selected_joint_ids)
        self.robot.set_qvel_limits(joint_qvel_limits, joint_ids=selected_joint_ids)
        self.robot.set_qf_limits(joint_qf_limits, joint_ids=selected_joint_ids)

        assert torch.allclose(
            self.robot.get_qpos_limits(joint_ids=selected_joint_ids),
            joint_qpos_limits,
            atol=1e-5,
        )
        assert torch.allclose(
            self.robot.get_qvel_limits(joint_ids=selected_joint_ids),
            joint_qvel_limits,
            atol=1e-5,
        )
        assert torch.allclose(
            self.robot.get_qf_limits(joint_ids=selected_joint_ids),
            joint_qf_limits,
            atol=1e-5,
        )

    def test_qpos_limits_update_solver_limits(self):
        """Test qpos limit updates are propagated to the control-part solver."""
        arm_name = "left_arm"
        solver = self.robot.get_solver(arm_name)
        assert solver is not None, "FAIL: expected left_arm solver to be initialized"

        asset_limits = self.robot.get_qpos_limits(name=arm_name).clone()
        updated_limits = asset_limits.clone()
        margin = 0.05
        updated_limits[..., 0] = torch.clamp(
            updated_limits[..., 0] + margin,
            asset_limits[..., 0],
            asset_limits[..., 1],
        )
        updated_limits[..., 1] = torch.clamp(
            updated_limits[..., 1] - margin,
            asset_limits[..., 0],
            asset_limits[..., 1],
        )

        self.robot.set_qpos_limits(updated_limits, name=arm_name)

        solver_limits = solver.get_qpos_limits()
        assert torch.allclose(
            torch.tensor(solver_limits["lower_qpos_limits"], device=self.sim.device),
            updated_limits[0, :, 0],
            atol=1e-5,
        ), "FAIL: solver lower_qpos_limits did not update with robot qpos limits"
        assert torch.allclose(
            torch.tensor(solver_limits["upper_qpos_limits"], device=self.sim.device),
            updated_limits[0, :, 1],
            atol=1e-5,
        ), "FAIL: solver upper_qpos_limits did not update with robot qpos limits"

    def test_configured_qpos_limits_sync_to_solver_after_initialization(self):
        """Test configured robot limits sync to the solver after it is created."""
        configured_limits = [-0.05, 0.05]
        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1_solver_limit_sync",
                "version": "v021",
                "qpos_limits": {"LEFT_J[1-7]": configured_limits},
            }
        )
        robot: Robot = self.sim.add_robot(cfg=cfg)

        solver = robot.get_solver("left_arm")
        assert solver is not None, "FAIL: expected left_arm solver to be initialized"

        solver_limits = solver.get_qpos_limits()
        expected_limits = robot.get_qpos_limits(name="left_arm")[0]
        assert torch.allclose(
            torch.tensor(solver_limits["lower_qpos_limits"], device=self.sim.device),
            expected_limits[:, 0],
            atol=1e-5,
        ), "FAIL: solver lower_qpos_limits did not sync configured robot limits"
        assert torch.allclose(
            torch.tensor(solver_limits["upper_qpos_limits"], device=self.sim.device),
            expected_limits[:, 1],
            atol=1e-5,
        ), "FAIL: solver upper_qpos_limits did not sync configured robot limits"

    def test_robot_cfg_merge(self):
        from copy import deepcopy
        from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg

        cfg = deepcopy(self.robot.cfg)

        cfg_dict = {
            "joint_drive_props": {
                "max_effort": {
                    "(LEFT|RIGHT)_HAND_(THUMB[12]|INDEX|MIDDLE|RING|PINKY)": 1.0,
                },
            },
            "solver_cfg": {
                "left_arm": {
                    "tcp": np.eye(4),
                }
            },
        }

        cfg = merge_robot_cfg(cfg, cfg_dict)

        assert (
            cfg.joint_drive_props.max_effort[
                "(LEFT|RIGHT)_HAND_(THUMB[12]|INDEX|MIDDLE|RING|PINKY)"
            ]
            == 1.0
        ), "Drive properties merge failed."

        assert np.allclose(
            cfg.solver_cfg["left_arm"].tcp, np.eye(4)
        ), "Solver config merge failed."

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        """Clean up resources after each test class."""
        if hasattr(cls, "sim"):
            cls.sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()
            del cls.sim
            import gc

            gc.collect()

    def test_set_physical_visible(self):
        self.robot.set_physical_visible(
            visible=True,
            rgba=(0.1, 0.1, 0.9, 0.4),
            control_part="left_arm",
        )
        self.robot.set_physical_visible(
            visible=True,
            control_part="left_arm",
        )
        self.robot.set_physical_visible(
            visible=False,
            control_part="left_arm",
        )


class TestRobotCPU(BaseRobotTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestRobotCUDA(BaseRobotTest):
    def setup_method(self):
        self.setup_simulation("cuda")


def _teardown_newton_physics() -> None:
    from dexsim.engine.newton_physics import teardown_newton_physics

    teardown_newton_physics()


class TestRobotNewton:
    """Focused Robot-on-Newton coverage (spawn, prepare, control surface).

    A robot is a URDF articulation; the Newton ``load_urdf`` patch builds a
    NewtonArticulation. This exercises the add_robot -> prepare -> control-part
    / qpos path end-to-end on Newton. It does NOT inherit the
    full BaseRobotTest suite because rebuilding the (complex, mimic-jointed)
    dexforce_w1 Newton model per test method is prohibitively slow; the
    default/CUDA classes already cover the shared control-part/FK/IK logic.
    """

    def setup_method(self):
        physics_cfg = physics_cfg_for_backend("newton")
        physics_cfg.solver_cfg = {
            "solver_type": "mujoco_warp",
            "njmax": 8192,
            "nconmax": 8192,
        }
        config = SimulationManagerCfg(
            headless=True, device="cuda", num_envs=1, physics_cfg=physics_cfg
        )
        self.sim = SimulationManager(config)
        cfg = DexforceW1Cfg.from_dict({"uid": "dexforce_w1", "version": "v021"})
        cfg.init_qpos = [0.0001 * (index + 1) for index in range(W1_ACTIVE_DOF)]
        self.robot: Robot = self.sim.add_robot(cfg=cfg)
        self.sim.prepare()

    def teardown_method(self):
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        _teardown_newton_physics()
        import gc

        gc.collect()

    def test_newton_robot_spawn_and_control(self):
        """Robot spawns on Newton, prepares, and exposes a working control surface."""
        assert self.sim.is_newton_backend
        assert self.robot.body_data.is_ready
        assert self.robot.dof > 0

        state_joint_names = self.robot.body_data.articulation_view.joint_names
        assert self.robot.joint_names == state_joint_names
        source_joint_names = self.robot._entities[0].get_actived_joint_names()
        initial_qpos_by_name = dict(
            zip(source_joint_names, self.robot.cfg.init_qpos, strict=True)
        )
        mimic_relations = list(
            zip(
                self.robot.mimic_ids,
                self.robot.mimic_parents,
                self.robot.mimic_multipliers,
                self.robot.mimic_offsets,
                strict=True,
            )
        )
        assert all(
            state_joint_names[mimic_id].endswith("_PIP")
            and "_HAND_" in state_joint_names[parent_id]
            for mimic_id, parent_id, _, _ in mimic_relations
        )
        initial_qpos = self.robot.body_data.qpos[0].detach().cpu().tolist()
        assert dict(zip(state_joint_names, initial_qpos, strict=True)) == pytest.approx(
            initial_qpos_by_name
        )

        binding = self.robot._entities[0]._physics_binding
        model = binding._runtime.model
        runtime_joints = {joint.name: joint for joint in binding.joints}
        mimic_joint0 = np.asarray(model.constraint_mimic_joint0.numpy()).reshape(-1)
        mimic_joint1 = np.asarray(model.constraint_mimic_joint1.numpy()).reshape(-1)
        row_by_pair = {
            (int(child), int(parent)): row
            for row, (child, parent) in enumerate(
                zip(mimic_joint0, mimic_joint1, strict=True)
            )
        }
        constraint_rows = []
        for mimic_id, parent_id, _, _ in mimic_relations:
            child = runtime_joints[state_joint_names[mimic_id]]
            parent = runtime_joints[state_joint_names[parent_id]]
            constraint_rows.append(
                row_by_pair[(int(child.joint_id), int(parent.joint_id))]
            )

        solver = binding._runtime.solver
        mapping = np.asarray(solver.mjc_eq_to_newton_mimic.numpy())
        selected_eq = np.isin(mapping, np.asarray(constraint_rows, dtype=np.int32))
        assert int(selected_eq.sum()) == len(mimic_relations)
        eq_solref = np.asarray(solver.mjw_model.eq_solref.numpy())
        np.testing.assert_allclose(
            eq_solref[selected_eq],
            np.broadcast_to([2.0e-3, 1.0e1], (len(mimic_relations), 2)),
        )
        target_ke = np.asarray(model.joint_target_ke.numpy())
        target_kd = np.asarray(model.joint_target_kd.numpy())
        target_mode = np.asarray(model.joint_target_mode.numpy())
        eq_active = np.asarray(solver.mjw_data.eq_active.numpy())
        assert np.all(eq_active[selected_eq])
        for mimic_id, parent_id, _, _ in mimic_relations:
            child = runtime_joints[state_joint_names[mimic_id]]
            parent = runtime_joints[state_joint_names[parent_id]]
            assert target_ke[child.qd_start] == pytest.approx(
                target_ke[parent.qd_start] * 1.0e-2
            )
            assert target_kd[child.qd_start] == pytest.approx(
                target_kd[parent.qd_start] * 1.0e-2
            )
            assert target_mode[child.qd_start] == target_mode[parent.qd_start]

        # This physical check covers both hands and keeps the native coupled
        # constraints bounded under the W1's self-contacts. Default can also
        # deflect these compliant joints by several tenths of a radian.
        self.sim.update(step=100)
        settled_qpos = self.robot.body_data.qpos
        settled_errors = []
        for mimic_id, parent_id, multiplier, offset in mimic_relations:
            settled_errors.append(
                torch.abs(
                    settled_qpos[:, mimic_id]
                    - (settled_qpos[:, parent_id] * multiplier + offset)
                )
            )
        assert torch.max(torch.stack(settled_errors)).item() < 0.5

        left_ids = self.robot.get_joint_ids("left_arm")
        right_ids = self.robot.get_joint_ids("right_arm")
        assert len(left_ids) > 0 and len(right_ids) > 0
        assert [
            state_joint_names[index] for index in left_ids
        ] == self.robot.control_parts["left_arm"]
        assert [
            state_joint_names[index] for index in right_ids
        ] == self.robot.control_parts["right_arm"]
        right_eef_ids = self.robot.get_joint_ids("right_eef")
        assert [
            state_joint_names[index] for index in right_eef_ids
        ] == self.robot.control_parts["right_eef"]

        right_qpos_limits = self.robot.get_qpos_limits(name="right_arm")
        requested_target = torch.full(
            (1, len(right_ids)), 0.1, dtype=torch.float32, device=self.sim.device
        )
        expected_target = requested_target.clamp(
            right_qpos_limits[..., 0], right_qpos_limits[..., 1]
        )
        self.robot.set_qpos(requested_target, name="right_arm")
        torch.testing.assert_close(
            self.robot.body_data.target_qpos[:, right_ids], expected_target
        )

        hand_target = torch.tensor(
            [[0.1, 1.0, 0.2, 0.3, 0.4, 0.5]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        self.robot.set_qpos(hand_target, name="right_eef")
        target_qpos = self.robot.body_data.target_qpos
        torch.testing.assert_close(target_qpos[:, right_eef_ids], hand_target)
        for mimic_id, parent_id, multiplier, offset in mimic_relations:
            torch.testing.assert_close(
                target_qpos[:, mimic_id],
                target_qpos[:, parent_id] * multiplier + offset,
            )
        hand_velocity_target = torch.tensor(
            [[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        self.robot.set_qvel(hand_velocity_target, name="right_eef")
        target_qvel = self.robot.body_data.target_qvel
        torch.testing.assert_close(target_qvel[:, right_eef_ids], hand_velocity_target)
        for mimic_id, parent_id, multiplier, _ in mimic_relations:
            torch.testing.assert_close(
                target_qvel[:, mimic_id],
                target_qvel[:, parent_id] * multiplier,
            )
        self.robot.set_qvel(torch.zeros_like(hand_velocity_target), name="right_eef")

        # State round-trip via the Newton articulation view.
        qpos = torch.zeros(
            (1, self.robot.dof), dtype=torch.float32, device=self.sim.device
        )
        self.robot.set_qpos(qpos, env_ids=None, target=False)
        assert torch.allclose(self.robot.body_data.qpos, qpos, atol=1e-5)


if __name__ == "__main__":
    # Run tests directly
    test_cpu = TestRobotCUDA()
    test_cpu.setup_method()
    test_cpu.test_compute_jacobian()
